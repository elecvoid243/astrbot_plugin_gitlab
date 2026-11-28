"""Plan one model-budgeted review of a pinned MR comparison."""

import json
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Self

from astrbot.core.agent.context.token_counter import EstimateTokenCounter
from astrbot.core.agent.message import Message
from astrbot.core.agent_team_execution import resolve_model_output_limit
from astrbot.core.provider.sources.anthropic_source import ProviderAnthropic
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from astrbot.core.utils.llm_metadata import LLM_METADATAS

from .expert import decode, render_findings
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
class PreparedReview:
    prompt: str | None
    evidence: dict
    omitted: list[str]
    filtered: list[str]
    deleted: list[str]

    def summary(self, result: dict, version: ReviewVersion, web_url: str) -> str:
        lines = render_findings(result["findings"], self.evidence)
        if not lines:
            lines.append(
                "在本次评审覆盖的差异中未发现可确认的问题。"
                if self.prompt
                else "本次没有提交给模型评审的文件补丁。"
            )
        for label, paths in (
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
        return "\n\n".join(lines)


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
            yield {**row, "text": line[start : start + 160]}, invocation is not None


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

    # Even at four UTF-8 bytes per character the estimator charges at least
    # 0.3 tokens per character, so longer raw patches cannot fit this window.
    def changes():
        for patch in source.review_patches(
            version.base, version.head, budget.window * 16
        ):
            if exclude_path(patch.path):
                filtered.add(patch.path)
            elif patch.status == "D":
                deleted.add(patch.path)
            else:
                yield patch

    references: dict[str, tuple[FilePatch, dict]] = {}

    def discover_references():
        for patch in changes():
            if "call" not in references:
                for row, is_call in related_references(source, patch, exclude_path):
                    kind = "call" if is_call else "reference"
                    if kind in references:
                        continue
                    # Keep the initial plan when a patch cannot coexist even
                    # with a single compact reference.
                    minimal = render_prompt(
                        {**task, "diff": patch.text}, {"E1": prompt_evidence(row)}
                    )
                    if estimate_tokens(minimal) > limit:
                        continue
                    references[kind] = (patch, row)
                    if is_call:
                        break
            yield patch

    selected, omitted = choose_patches(discover_references(), task, {}, limit)
    evidence = {}
    source_rows = {}
    related = references.get("call") or references.get("reference")
    if related:
        anchor, row = related
        evidence["E1"] = prompt_evidence(row)
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
        source_rows["E1"] = row

    if not selected:
        return PreparedReview(
            None,
            {},
            omitted=omitted,
            filtered=sorted(filtered),
            deleted=sorted(deleted),
        )

    task["diff"] = patch_text(selected)
    urls = {row["url"] for row in source_rows.values()}
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
    return PreparedReview(
        prompt,
        source_rows,
        omitted=omitted,
        filtered=sorted(filtered),
        deleted=sorted(deleted),
    )


def decode_review(text: str) -> dict:
    result = decode(text)
    findings = result.get("findings")
    if not isinstance(findings, list):
        raise ValueError("Expected a list of review findings")
    return result
