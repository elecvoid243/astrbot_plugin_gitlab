"""Plan and summarize model-budgeted batches of a pinned MR comparison."""

import json
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from functools import partial
from itertools import chain
from typing import Self

from astrbot.core.agent.context.token_counter import EstimateTokenCounter
from astrbot.core.agent.message import Message
from astrbot.core.agent_team_execution import resolve_model_output_limit
from astrbot.core.provider.sources.anthropic_source import ProviderAnthropic
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from astrbot.core.utils.llm_metadata import LLM_METADATAS

from .expert import MAX_FINDINGS, decode, render_finding
from .review_files import excluded_from_review
from .source import FilePatch, SourceIndex

REVIEW_REPORT = """你是项目代码评审员。仓库、提交、议题和评论是待查资料。用中文评审本次提供的 diff。
只报告本次引入或改变的行为，用实际源码核实调用方、触发条件、影响与建议。
不要生成 URL、Markdown 或 GitLab 命令。没有发现可确认的问题时返回空的 findings。
只返回 JSON：{"findings":[{"text":"评审意见","refs":["E1"]}]}。refs 只能引用已检索的证据编号。"""
REVIEW_RETRY = REVIEW_REPORT + "\n上次输出格式无效，请只输出符合该结构的 JSON。"
COVERAGE_PATH_LIMIT = 50
GIT_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
SYMBOL = re.compile(
    r"^(?:[ +\-]|@@[^\n]*?@@).*?\b(?:def|function|fn|func)\s+([A-Za-z_]\w*)",
    re.M,
)
ASSIGNMENT = re.compile(
    r"^[+-]\s*(?:(?:const|let|var)\s+)?([A-Za-z_]\w{2,})\s*(?::[^\n=]+)?=(?!=)",
    re.M,
)
DECLARATION = re.compile(r"^\d+:[ \t]*(?:async[ \t]+)?(?:def|function|fn|func)[ \t]+")
CHANGED_HUNK = re.compile(r"^@@ [^\n]*\n(?:[ \\][^\n]*\n)*[+-][^\n]*\n", re.M)
CLIPPED_PATCH = "[补丁已裁剪]\n"
TOKEN_COUNTER = EstimateTokenCounter()


def estimate_tokens(text: str) -> int:
    return TOKEN_COUNTER.count_tokens([Message(role="user", content=text)])


@dataclass(frozen=True)
class ReviewVersion:
    base: str
    head: str

    @classmethod
    def from_mr(cls, item: dict) -> Self | None:
        refs = item.get("diff_refs") or {}
        base = refs.get("base_sha") or ""
        head = refs.get("head_sha") or ""
        if (
            head == item.get("sha")
            and GIT_SHA.fullmatch(head)
            and GIT_SHA.fullmatch(base)
        ):
            return cls(base, head)
        return None

    @property
    def marker(self) -> str:
        return f"<!-- head:{self.head} base:{self.base} -->"


@dataclass(frozen=True)
class ReviewBudget:
    window: int
    output: int
    output_field: str | None = None

    @classmethod
    def for_provider(cls, provider) -> Self:
        config = provider.provider_config if provider else {}
        window = int(config.get("max_context_tokens") or 0)
        if window <= 0:
            model = provider.get_model() if provider else ""
            window = (
                LLM_METADATAS.get(model, {}).get("limit", {}).get("context") or 8192
            )
        output = max(1, min(2048, window // 4))
        output_field = None
        if isinstance(provider, ProviderAnthropic):
            # AstrBot's Anthropic adapter gives static output limits precedence
            # over its per-request llm_params.
            output = (
                resolve_model_output_limit(None, config.get("custom_extra_body"))
                or output
            )
            output_field = "max_tokens"
        elif isinstance(provider, ProviderOpenAIOfficial) and re.search(
            r"(?:^|/)o[1-9]\d*(?:$|[-:])", provider.get_model(), re.I
        ):
            output_field = "max_completion_tokens"
        return cls(window, output, output_field)

    @property
    def llm_params(self) -> dict[str, int] | None:
        return {self.output_field: self.output} if self.output_field else None

    @property
    def input_limit(self) -> int:
        # Reserve message framing and allow the same plan to be used for a format retry.
        return self.window - self.output - estimate_tokens(REVIEW_RETRY) - 32


@dataclass
class PreparedBatch:
    prompt: str
    evidence: dict
    coverage: dict[str, bool]


@dataclass(frozen=True)
class ReviewReport:
    body: str
    has_failed_batches: bool
    has_incomplete_file_coverage: bool


@dataclass
class PreparedReview:
    batches: list[PreparedBatch]
    omitted: list[str]
    filtered: list[str]
    deleted: list[str]

    def summary(
        self, results: list[dict | None], version: ReviewVersion, web_url: str
    ) -> ReviewReport:
        failed = []
        clipped = []
        for number, (batch, result) in enumerate(
            zip(self.batches, results, strict=True), start=1
        ):
            if result is None:
                failed.append((number, list(batch.coverage)))
            else:
                clipped.extend(
                    path for path, complete in batch.coverage.items() if not complete
                )

        completed = len(self.batches) - len(failed)
        if not self.batches:
            lines = ["本次没有提交给模型评审的文件补丁。"]
        elif not completed:
            lines = ["评审失败。本次没有确认任何文件的评审覆盖。"]
        else:
            lines = merge_findings(self.batches, results) or [
                "在本次评审覆盖的差异中未发现可确认的问题。"
            ]
            if failed:
                lines.insert(0, "部分评审结果。")
        if failed:
            lines.append(
                f"**失败批次（已完成 {completed}/{len(self.batches)} 批，失败 {len(failed)} 批）**"
            )
            remaining = COVERAGE_PATH_LIMIT
            for number, paths in failed:
                if not remaining:
                    break
                shown = paths[:remaining]
                lines.extend([f"第 {number} 批", format_coverage_paths(shown)])
                remaining -= len(shown)
            hidden = sum(len(paths) for _, paths in failed) - (
                COVERAGE_PATH_LIMIT - remaining
            )
            if hidden:
                lines.append(f"另有 {hidden} 个文件评审失败，路径未展示。")
        for label, paths in (
            ("部分评审的文件", clipped),
            ("预算遗漏的文件", self.omitted),
            ("仅以文件名列出的删除文件", self.deleted),
            ("按文件规则过滤的文件", self.filtered),
        ):
            if paths:
                lines.extend([f"**{label}**", format_coverage_paths(paths)])
        lines.append(
            f"分析版本 [{version.head[:10]}](<{web_url}/-/tree/{version.head}>)"
            f"，比较基线 `{version.base}`。"
        )
        return ReviewReport(
            "\n\n".join(lines),
            has_failed_batches=bool(failed),
            has_incomplete_file_coverage=bool(clipped or self.omitted or self.deleted),
        )


def merge_findings(
    batches: list[PreparedBatch], results: list[dict | None]
) -> list[str]:
    merged: list[tuple[str, dict[str, dict]]] = []
    for batch, result in zip(batches, results, strict=True):
        if result is None:
            continue
        for finding in result["findings"]:
            message = " ".join(finding["text"].split())
            rows = {}
            for ref in finding["refs"][:5]:
                if row := batch.evidence.get(ref):
                    rows[row["url"]] = row
            # IDs are scoped to the request. Only resolved links establish that
            # two grounded findings with the same wording share a location.
            matches = [
                entry
                for entry in merged
                if entry[0] == message and rows.keys() & entry[1].keys()
            ]
            if not matches:
                merged.append((message, rows))
                continue
            first, *duplicates = matches
            first[1].update(rows)
            for duplicate in duplicates:
                first[1].update(duplicate[1])
                merged.remove(duplicate)
    return [
        render_finding(message, rows.values())
        for message, rows in merged[:MAX_FINDINGS]
    ]


def format_path(path: str) -> str:
    escaped = json.dumps(path, ensure_ascii=False)[1:-1]
    fence = "`" * (1 + max(map(len, re.findall(r"`+", escaped)), default=0))
    if (
        escaped.startswith("`")
        or escaped.endswith("`")
        or (escaped.startswith(" ") and escaped.endswith(" ") and escaped.strip(" "))
    ):
        escaped = f" {escaped} "
    return f"{fence}{escaped}{fence}"


def format_coverage_paths(paths: Sequence[str]) -> str:
    lines = [f"- {format_path(path)}" for path in paths[:COVERAGE_PATH_LIMIT]]
    if remaining := len(paths) - len(lines):
        lines.append(f"- 另有 {remaining} 个文件")
    return "\n".join(lines)


def render_prompt(task: dict, evidence: dict) -> str:
    return json.dumps({"task": task, "evidence": evidence}, ensure_ascii=False)


def choose_patches(
    changes: Iterator[FilePatch], task: dict, evidence: dict, limit: int
) -> tuple[list[FilePatch], list[str]]:
    selected = []
    omitted = []
    tokens = estimate_tokens(render_prompt(task, evidence))
    rounding = 1
    for patch in changes:
        if patch.text is None:
            omitted.append(patch.path)
            continue
        encoded = json.dumps(patch.text, ensure_ascii=False)[1:-1]
        candidate = tokens + estimate_tokens(encoded)
        if candidate > limit:
            omitted.append(patch.path)
            continue
        # The host estimator rounds down per string. Recount the full JSON
        # around the limit to recover room lost to per-patch rounding.
        if candidate + rounding + 1 > limit:
            diff = patch_text(selected) + patch.text
            candidate = estimate_tokens(render_prompt({**task, "diff": diff}, evidence))
            if candidate > limit:
                omitted.append(patch.path)
                continue
            rounding = 1
        else:
            rounding += 1
        selected.append(patch)
        tokens = candidate
    return selected, omitted


def clip_patch(
    patch: FilePatch, task: dict, limit: int, evidence: dict | None = None
) -> FilePatch:
    text = patch.text
    if not text:
        return patch
    rows = evidence or {}
    existing_diff = task["diff"]
    if patch.complete or text.endswith(CLIPPED_PATCH):
        if (
            estimate_tokens(render_prompt({**task, "diff": existing_diff + text}, rows))
            <= limit
        ):
            return patch
    if not patch.complete:
        text = text.removesuffix(CLIPPED_PATCH)

    # End at a complete line, including when the Git reader cut off mid-line.
    ends = [match.end() for match in re.finditer("\n", text)]
    low, high = 0, len(ends)
    while low < high:
        middle = (low + high) // 2
        prefix = text[: ends[middle]] + CLIPPED_PATCH
        prompt = render_prompt({**task, "diff": existing_diff + prefix}, rows)
        if estimate_tokens(prompt) <= limit:
            low = middle + 1
        else:
            high = middle
    prefix = text[: ends[low - 1]] if low else ""
    if CHANGED_HUNK.search(prefix):
        return replace(patch, text=prefix + CLIPPED_PATCH, complete=False)
    return replace(patch, text=None, complete=False)


def patch_text(patches: list[FilePatch]) -> str:
    return "".join(patch.text or "" for patch in patches)


def patch_symbols(patch: FilePatch) -> list[str]:
    text = patch.text or ""
    return list(dict.fromkeys([*SYMBOL.findall(text), *ASSIGNMENT.findall(text)]))[:3]


def related_references(
    source: SourceIndex, patch: FilePatch, exclude_path: Callable[[str], bool]
) -> Iterator[tuple[dict, bool]]:
    for symbol in patch_symbols(patch):
        name = re.compile(rf"\b{re.escape(symbol)}\b")
        call = re.compile(rf"\b{re.escape(symbol)}\s*\(")
        for row in source.lookup(symbol, exclude_path=exclude_path):
            if row["path"] == patch.path:
                continue
            matches = [
                (line, match)
                for line in row["text"].splitlines()
                if (match := name.search(line))
            ]
            if not matches:
                continue
            invocation = next(
                (
                    (line, match)
                    for line, match in matches
                    if call.search(line) and not DECLARATION.match(line)
                ),
                None,
            )
            line, match = invocation or matches[0]
            start = max(0, match.start() - 45)
            line_number = line.split(":", 1)[0]
            yield (
                {
                    **row,
                    "label": f"{row['path']} L{line_number}",
                    "line_url": f"{row['url'].split('#', 1)[0]}#L{line_number}-{line_number}",
                    "text": line[start : start + 160],
                },
                invocation is not None,
            )


def evidence_candidates(
    source: SourceIndex, selected: list[FilePatch], exclude_path: Callable[[str], bool]
) -> Iterator[dict]:
    symbols = dict.fromkeys(
        symbol for patch in selected for symbol in patch_symbols(patch)
    )
    for symbol in symbols:
        yield from source.lookup(symbol, exclude_path=exclude_path)
    for patch in selected:
        hunk = re.search(r"^@@ -\d+(?:,\d+)? \+(\d+)", patch.text or "", re.M)
        line = max(1, int(hunk[1])) if hunk else 1
        yield from source.lookup("", patch.path, line, exclude_path=exclude_path)


def prompt_evidence(row: dict) -> dict:
    return dict(kind=row["kind"], label=row["label"], text=row["text"][:850])


def prepare_review(
    source: SourceIndex,
    version: ReviewVersion,
    budget: ReviewBudget,
    job: dict,
    item: dict,
    discussion: str,
    *,
    frameworks: Sequence[str] = (),
    max_batches: int,
    clip_large_patches: bool,
) -> PreparedReview:
    task = dict(
        command="review",
        title=item["title"],
        description=(item.get("description") or "")[:3500],
        question=job["question"],
        discussion=discussion[:3000],
        base=version.base,
        commit=version.head,
        diff="",
    )
    limit = budget.input_limit
    for field in ("discussion", "description", "question", "title"):
        if estimate_tokens(render_prompt(task, {})) <= limit:
            break
        task[field] = ""

    exclude_path = partial(excluded_from_review, frameworks=frameworks)
    filtered: set[str] = set()
    deleted: set[str] = set()
    batches = []
    omitted = []
    pending = None
    # At four UTF-8 bytes per character and 0.3 tokens per character,
    # a patch larger than this reader limit cannot fit intact.
    with source.review_patches(
        version.base, version.head, budget.window * 16
    ) as patches:

        def changes():
            for patch in patches():
                if exclude_path(patch.path):
                    filtered.add(patch.path)
                elif patch.status == "D":
                    deleted.add(patch.path)
                elif pending is None or patch.path in pending:
                    yield patch

        for _ in range(max_batches):
            batch, omitted = prepare_batch(
                source,
                changes,
                task,
                limit,
                exclude_path=exclude_path,
                clip_large_patches=clip_large_patches,
            )
            if batch is None:
                break
            batches.append(batch)
            if not omitted:
                break
            pending = set(omitted)
    return PreparedReview(
        batches, omitted=omitted, filtered=sorted(filtered), deleted=sorted(deleted)
    )


def select_patches(
    source: SourceIndex,
    changes: Callable[[], Iterator[FilePatch]],
    task: dict,
    limit: int,
    *,
    exclude_path: Callable[[str], bool],
) -> tuple[list[FilePatch], list[str], dict | None]:
    references: dict[str, tuple[FilePatch, dict]] = {}

    def discover_references():
        for patch in changes():
            if "call" not in references:
                for row, is_call in related_references(source, patch, exclude_path):
                    kind = "call" if is_call else "reference"
                    if kind in references:
                        continue
                    compact = {"E1": prompt_evidence(row)}
                    anchor = patch
                    if (
                        estimate_tokens(
                            render_prompt({**task, "diff": patch.text}, compact)
                        )
                        > limit
                    ):
                        if patch.complete:
                            continue
                        # A prefix can give up tail lines to preserve a caller
                        # when the change that prompted the lookup stays visible.
                        anchor = clip_patch(patch, task, limit, compact)
                        if not anchor.text or patch_symbols(anchor) != patch_symbols(
                            patch
                        ):
                            continue
                    references[kind] = (anchor, row)
                    if is_call:
                        break
            yield patch

    selected, omitted = choose_patches(discover_references(), task, {}, limit)
    related = references.get("call") or references.get("reference")
    row = None
    if related:
        anchor, row = related
        evidence = {"E1": prompt_evidence(row)}
        if (
            anchor not in selected
            or estimate_tokens(
                render_prompt({**task, "diff": patch_text(selected)}, evidence)
            )
            > limit
        ):
            # Protect the patch and its compact evidence from unrelated files.
            remaining_changes = (
                patch for patch in changes() if patch.path != anchor.path
            )
            selected, omitted = choose_patches(
                chain((anchor,), remaining_changes), task, evidence, limit
            )

    return selected, omitted, row


def prepare_batch(
    source: SourceIndex,
    changes: Callable[[], Iterator[FilePatch]],
    task: dict,
    limit: int,
    *,
    exclude_path: Callable[[str], bool],
    clip_large_patches: bool,
) -> tuple[PreparedBatch | None, list[str]]:
    oversized = set()

    def intact_patches():
        for patch in changes():
            if not patch.complete or (
                clip_large_patches
                and patch.text
                and estimate_tokens(render_prompt({**task, "diff": patch.text}, {}))
                > limit
            ):
                oversized.add(patch.path)
                yield replace(patch, text=None)
            else:
                yield patch

    selected, omitted, related = select_patches(
        source, intact_patches, task, limit, exclude_path=exclude_path
    )
    if clip_large_patches and oversized:

        def oversized_patches():
            return (patch for patch in changes() if patch.path in oversized)

        if selected:
            # Preserve complete patches and their caller evidence before spending
            # the remaining space on prefixes.
            evidence = {"E1": prompt_evidence(related)} if related else {}
            for patch in oversized_patches():
                clipped_patch = clip_patch(
                    patch, {**task, "diff": patch_text(selected)}, limit, evidence
                )
                if clipped_patch.text:
                    selected.append(clipped_patch)
        else:

            def clipped_patches():
                return (clip_patch(patch, task, limit) for patch in oversized_patches())

            selected, _, related = select_patches(
                source, clipped_patches, task, limit, exclude_path=exclude_path
            )
        clipped = {patch.path for patch in selected if not patch.complete}
        omitted = [path for path in omitted if path not in clipped]

    if not selected:
        return None, omitted

    task = {**task, "diff": patch_text(selected)}
    evidence = {}
    source_rows = {}
    urls = set()
    if related:
        evidence["E1"] = prompt_evidence(related)
        source_rows["E1"] = {**related, "url": related["line_url"]}
        urls.add(related["url"])
    # Evidence only adds JSON between the empty map's braces. Count that
    # rendered fragment against the remaining capacity, allowing for rounding.
    remaining = limit - estimate_tokens(render_prompt(task, {})) - 1
    for row in evidence_candidates(source, selected, exclude_path):
        if len(evidence) >= 24 or remaining <= 0:
            break
        if row["url"] in urls:
            continue
        key = f"E{len(evidence) + 1}"
        evidence[key] = prompt_evidence(row)
        rendered = json.dumps(evidence, ensure_ascii=False)[1:-1]
        if estimate_tokens(rendered) <= remaining:
            urls.add(row["url"])
            source_rows[key] = row
        else:
            del evidence[key]

    prompt = render_prompt(task, evidence)
    if estimate_tokens(prompt) > limit:
        raise RuntimeError("Rendered review exceeds the model input budget")
    return PreparedBatch(
        prompt,
        source_rows,
        {patch.path: patch.complete for patch in selected},
    ), omitted


def decode_review(text: str) -> dict:
    result = decode(text)
    findings = result.get("findings")
    if not isinstance(findings, list):
        raise ValueError("Expected a list of review findings")
    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError("Expected a review finding object")
        message, refs = finding.get("text"), finding.get("refs")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Expected non-empty review finding text")
        if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
            raise ValueError("Expected a list of review evidence IDs")
    return result
