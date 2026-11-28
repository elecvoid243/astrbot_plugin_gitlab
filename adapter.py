"""GitLab platform adapter with durable, project-scoped command dispatch."""

import asyncio
import hashlib
import hmac
import json
import re
import sqlite3
import time
from contextlib import closing
from urllib.parse import quote

import aiohttp
from astrbot.api import logger
from astrbot.api.message_components import Plain
from astrbot.api.platform import Platform, PlatformMetadata, register_platform_adapter
from astrbot.api.star import Context, StarTools
from astrbot.core.agent_team_execution import model_delivery
from astrbot.core.provider.sources.anthropic_source import ProviderAnthropic

from .events import project_job
from .expert import HELP, ProjectExpert
from .gitlab import APIError, GitLab
from .review import (
    REVIEW_REPORT,
    REVIEW_RETRY,
    ReviewBudget,
    ReviewVersion,
    decode_review,
    prepare_review,
)
from .source import SourceIndex

DEFAULTS = dict(
    gitlab_url="",
    gitlab_project_id=0,
    gitlab_token="",
    gitlab_webhook_secret="",
    gitlab_provider="",
    gitlab_auto_issue=True,
    gitlab_auto_review=True,
    publish_review_as_thread=False,
    final_update_message=True,
    reply_to_trigger_comment=False,
    unified_webhook_mode=True,
    webhook_uuid="",
)
FIELDS = {
    "gitlab_url": {
        "type": "string",
        "description": "GitLab 实例地址",
        "hint": "例如 https://gitlab.example.com",
    },
    "gitlab_project_id": {"type": "int", "description": "GitLab 项目数字 ID"},
    "gitlab_token": {
        "type": "string",
        "description": "机器人 API Token",
        "secret": True,
    },
    "gitlab_webhook_secret": {
        "type": "string",
        "description": "Webhook Secret token",
        "secret": True,
    },
    "gitlab_provider": {
        "type": "string",
        "description": "项目分析模型",
        "_special": "select_provider",
    },
    "gitlab_auto_issue": {"type": "bool", "description": "Issue 自动回复"},
    "gitlab_auto_review": {"type": "bool", "description": "MR 自动评审"},
    "publish_review_as_thread": {
        "type": "bool",
        "description": "将新评审摘要发布为可解决讨论",
    },
    "final_update_message": {"type": "bool", "description": "发送评审摘要更新提示"},
    "reply_to_trigger_comment": {
        "type": "bool",
        "description": "在命令所在讨论发送评审更新提示",
    },
}


@register_platform_adapter(
    "gitlab",
    "GitLab 项目机器人",
    default_config_tmpl=DEFAULTS.copy(),
    adapter_display_name="GitLab",
    config_metadata=FIELDS,
    support_streaming_message=False,
)
class GitLabPlatform(Platform):
    context: Context

    def __init__(self, platform_config, platform_settings, event_queue):
        super().__init__(platform_config, event_queue)
        self.bot = None
        self.session = None
        self.db = None
        self.wake = asyncio.Event()
        self.stopping = False
        self.worker = None
        if not self.config.get("webhook_uuid"):
            self.config["webhook_uuid"] = hashlib.sha256(
                ("gitlab:" + self.config["id"]).encode()
            ).hexdigest()[:32]

    def meta(self):
        return PlatformMetadata(
            "gitlab",
            "GitLab 项目机器人",
            self.config["id"],
            adapter_display_name="GitLab",
            support_streaming_message=False,
        )

    async def run(self):
        if not all(
            self.config.get(k)
            for k in (
                "gitlab_url",
                "gitlab_project_id",
                "gitlab_token",
                "gitlab_webhook_secret",
                "gitlab_provider",
            )
        ):
            raise ValueError(
                "Configure GitLab instance, project, both credentials and provider"
            )
        if not self.unified_webhook():
            raise ValueError("Enable unified webhook mode for GitLab")
        self.worker = asyncio.current_task()
        try:
            async with aiohttp.ClientSession(
                headers={"PRIVATE-TOKEN": self.config["gitlab_token"]}
            ) as session:
                self.session = session
                self.api = GitLab(
                    session, self.config["gitlab_url"], self.config["gitlab_project_id"]
                )
                identity = await self.api.request("GET", "/user")
                project = await self.api.get()
                namespace = quote(project["path_with_namespace"], safe="/")
                self.api.web = self.api.instance + "/" + namespace
                binding = hashlib.sha256(
                    f"{self.api.instance}|{self.api.project_id}|{self.config['id']}".encode()
                ).hexdigest()[:32]
                directory = StarTools.get_data_dir("astrbot_plugin_gitlab") / binding
                directory.mkdir(parents=True, exist_ok=True)
                self.source = SourceIndex(directory, self.api.web)
                self.expert = ProjectExpert(self.api, self.source, self.generate)
                with closing(sqlite3.connect(directory / "events.sqlite")) as db:
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT, attempts INTEGER DEFAULT 0, due REAL DEFAULT 0, state TEXT DEFAULT 'queued')"
                    )
                    self.db = db
                    self.bot = identity
                    logger.info(
                        "GitLab platform %s ready at /api/v1/webhooks/platforms/%s",
                        self.config["id"],
                        self.config["webhook_uuid"],
                    )
                    await self.consume()
        finally:
            self.db = self.bot = self.session = self.worker = None

    async def terminate(self):
        self.stopping = True
        self.wake.set()
        if self.worker and self.worker != asyncio.current_task():
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)

    async def generate(self, system, prompt, *, output_limit=None, **kwargs):
        with model_delivery(None, output_limit=output_limit):
            reply = await self.context.llm_generate(
                chat_provider_id=self.config["gitlab_provider"],
                system_prompt=system,
                prompt=prompt,
                **kwargs,
            )
        return reply.completion_text

    async def webhook_callback(self, request):
        """Authenticate the hook and store accepted work before acknowledging it.

        Args:
            request: AstrBot's unified webhook request.
        Returns:
            A JSON body and HTTP status code understood by the dashboard.
        """
        secret = self.config["gitlab_webhook_secret"]
        provided = request.headers.get("X-Gitlab-Token", "")
        if not secret or not hmac.compare_digest(provided.encode(), secret.encode()):
            return {"error": "Invalid webhook secret"}, 403
        if not self.bot or self.stopping:
            return {"error": "Platform starting or stopping"}, 503
        payload = await request.get_json(silent=True)
        if not isinstance(payload, dict):
            return {"error": "Invalid event"}, 400
        try:
            job = project_job(payload, self.api.project_id, self.bot, self.config)
        except (ValueError, TypeError, AttributeError):
            return {"error": "Invalid event fields"}, 400
        if job:
            # Distinct actors cannot consume one another's pending review trigger.
            delivery = job["key"]
            if job["sha"]:
                attrs = payload["object_attributes"]
                delivery += ":" + str(
                    attrs.get("updated_at") or attrs.get("oldrev") or ""
                )
            with self.db:
                self.db.execute(
                    "INSERT INTO jobs(id,payload) VALUES (?,?) ON CONFLICT(id) DO UPDATE SET "
                    "payload=COALESCE(jobs.payload, excluded.payload), attempts=0, due=0, "
                    "state='queued' WHERE jobs.state='failed'",
                    (f"{delivery}:{job['actor']}", json.dumps(job, ensure_ascii=False)),
                )
            self.wake.set()
        return {"accepted": bool(job)}, 200

    async def consume(self):
        """Process durable jobs serially so two versions cannot update one note concurrently."""
        while not self.stopping:
            self.wake.clear()
            row = self.db.execute(
                "SELECT id,payload,attempts FROM jobs WHERE state='queued' AND due<=? ORDER BY rowid LIMIT 1",
                (time.time(),),
            ).fetchone()
            if not row:
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=5)
                except asyncio.TimeoutError:
                    pass
                continue
            key, payload, attempts = row
            try:
                await self.process(json.loads(payload), key)
            except Exception as exc:
                attempts += 1
                state = "failed" if attempts >= 5 else "queued"
                logger.warning(
                    "GitLab job %s attempt %d: %s (%s)",
                    key,
                    attempts,
                    str(exc) if isinstance(exc, APIError) else type(exc).__name__,
                    state,
                )
                with self.db:
                    self.db.execute(
                        "UPDATE jobs SET attempts=?,due=?,state=? WHERE id=?",
                        (attempts, time.time() + min(300, 5**attempts), state, key),
                    )
            else:
                with self.db:
                    self.db.execute(
                        "UPDATE jobs SET state='done',payload=NULL WHERE id=?", (key,)
                    )

    async def review_mr(self, path, version, job, item, discussion):
        provider = self.context.get_provider_by_id(self.config["gitlab_provider"])
        budget = ReviewBudget.for_provider(provider)
        prepared = await asyncio.to_thread(
            prepare_review, self.source, version, budget, job, item, discussion
        )
        if ReviewVersion.from_mr(await self.api.get(path)) != version:
            return None

        result = {"findings": []}
        if prepared.prompt:
            llm_params = (
                {"max_tokens": budget.output}
                if isinstance(provider, ProviderAnthropic)
                else None
            )
            for system in (REVIEW_REPORT, REVIEW_RETRY):
                completion = await self.generate(
                    system,
                    prepared.prompt,
                    output_limit=budget.output,
                    llm_params=llm_params,
                )
                try:
                    result = decode_review(completion)
                    break
                except (TypeError, ValueError):
                    if system == REVIEW_RETRY:
                        raise
        return prepared.summary(result, version, self.api.web)

    def note_marker(self, identifier):
        digest = hashlib.sha256(
            (self.config["id"] + ":" + identifier).encode()
        ).hexdigest()[:24]
        return f"<!-- astrbot-gitlab:{digest} -->"

    async def trigger_discussion(self, path, job, *, optional):
        thread = job["thread"]
        if thread or not job["key"].startswith("note:"):
            return thread
        # Some GitLab versions omit discussion_id in the Note Hook.
        note_id = int(job["key"].split(":")[1])
        try:
            return await self.api.find_discussion(path, note_id)
        except (APIError, aiohttp.ClientError, TimeoutError):
            if not optional:
                raise
            return ""

    async def finish_review_update(
        self, path, job, delivery, summary_id, bot_id, *, thread=None
    ):
        if not self.config.get("final_update_message", True):
            return
        notice_marker = self.note_marker(f"review-update:{delivery}")
        if job.get("review_update_note_id"):
            if await self.api.marker_note(path, notice_marker, bot_id):
                return
        else:
            # A newer review can update the summary while the notification waits to retry.
            db = self.db
            if db is None:
                raise RuntimeError("GitLab job store is unavailable")
            job["review_update_note_id"] = summary_id
            with db:
                db.execute(
                    "UPDATE jobs SET payload=? WHERE id=?",
                    (json.dumps(job, ensure_ascii=False), delivery),
                )
        url = f"{self.api.web}/-{path}#note_{int(summary_id)}"
        body = f"已更新[评审摘要]({url})。\n\n{notice_marker}"
        if self.config.get("reply_to_trigger_comment", False):
            if thread is None:
                thread = await self.trigger_discussion(path, job, optional=True)
        else:
            thread = ""
        if thread:
            try:
                await self.api.publish(path, thread, body)
            except (APIError, aiohttp.ClientError, TimeoutError):
                # A server error may arrive after GitLab has already saved the reply.
                if await self.api.marker_note(path, notice_marker, bot_id):
                    return
            else:
                return
        await self.api.publish(path, "", body)

    async def process(self, job, delivery):
        """Resolve membership and pinned source before publishing one result.

        Args:
            job: Durable, project-scoped webhook work item.
            delivery: Queue ID used to identify a retried review.
        """
        project = await self.api.get()
        if not await self.api.member(job["actor"], project):
            return
        path = f"/{job['resource']}/{job['iid']}"
        if self.bot is None:
            raise RuntimeError("GitLab bot identity is unavailable")
        bot_id = self.bot["id"]
        if summary_id := job.get("review_update_note_id"):
            await self.finish_review_update(path, job, delivery, summary_id, bot_id)
            return
        item = await self.api.get(path)
        is_mr = job["resource"] == "merge_requests"
        automatic = bool(job["sha"])
        if automatic and (
            item["state"] != "opened"
            or item.get("draft")
            or item.get("work_in_progress")
            or job["sha"] != item["sha"]
        ):
            return
        reviewing = is_mr and job["command"] == "review"
        marker = self.note_marker("review-summary" if reviewing else job["key"])
        current = await self.api.marker_note(path, marker, bot_id)
        delivery_marker = self.note_marker(f"review-run:{delivery}")
        updated_marker = "<!-- review-summary-updated -->"
        if current and not reviewing:
            return
        if current and delivery_marker in current["body"]:
            # Recover a save whose response was lost before we recorded it locally.
            if updated_marker in current["body"]:
                await self.finish_review_update(
                    path, job, delivery, current["id"], bot_id
                )
            return
        diff_version = ReviewVersion.from_mr(item) if reviewing else None
        if reviewing and not diff_version:
            raise ValueError("MR diff version is not ready")
        version = (
            diff_version.marker
            if diff_version
            else f"<!-- head:{item.get('sha', '')} -->"
        )
        thread = await self.trigger_discussion(path, job, optional=reviewing)
        if not thread and not reviewing and job["key"].startswith("note:"):
            return
        if job["command"] == "help":
            body = HELP
        elif not is_mr and (
            not project.get("default_branch") or project.get("empty_repo")
        ):
            self.source.sha = ""
            discussion = await self.api.discussion(path, thread)
            body = await self.expert.analyze(job, item, discussion=discussion)
        else:
            ref = (
                f"refs/merge-requests/{job['iid']}/head"
                if is_mr
                else "refs/heads/" + project["default_branch"]
            )
            # Finish a filesystem transaction before a hot reload can open the same mirror.
            sync = asyncio.create_task(
                asyncio.to_thread(self.source.sync, ref, self.config["gitlab_token"])
            )
            try:
                sha = await asyncio.shield(sync)
            except asyncio.CancelledError:
                await sync
                raise
            if is_mr and sha != item["sha"]:
                raise ValueError("MR head changed during fetch")
            try:
                discussion = await self.api.discussion(path, thread)
            except (APIError, aiohttp.ClientError, TimeoutError):
                if not reviewing:
                    raise
                discussion = ""
            if diff_version:
                body = await self.review_mr(path, diff_version, job, item, discussion)
                if body is None:
                    return
            else:
                diff = ""
                if is_mr:
                    refs = ReviewVersion.from_mr(item)
                    if not refs:
                        raise ValueError("MR diff version is not ready")
                    diff = await asyncio.to_thread(self.source.diff, refs.base)
                body = await self.expert.analyze(job, item, diff, discussion)
            if is_mr:
                latest = await self.api.get(path)
                if diff_version and ReviewVersion.from_mr(latest) != diff_version:
                    return
                if latest["sha"] != sha:
                    if automatic:
                        return
                    raise ValueError("MR changed during analysis")
                if automatic and (
                    latest["state"] != "opened"
                    or latest.get("draft")
                    or latest.get("work_in_progress")
                ):
                    return
        body = f"{body}\n\n{marker}\n{version}"
        if reviewing:
            body += f"\n{delivery_marker}"
            if current:
                body += f"\n{updated_marker}"
            try:
                await self.api.publish_review(
                    path,
                    body,
                    current,
                    as_thread=self.config.get("publish_review_as_thread", False),
                )
            except (APIError, aiohttp.ClientError, TimeoutError):
                if not await self.api.marker_note(path, delivery_marker, bot_id):
                    raise
            if current:
                await self.finish_review_update(
                    path, job, delivery, current["id"], bot_id, thread=thread
                )
        else:
            await self.api.publish(path, thread, body)

    async def send_by_session(self, session, message_chain):
        """Deliver text to a session in the bound issue or MR.

        Args:
            session: AstrBot destination with resource/IID[/discussion] session ID.
            message_chain: Outgoing AstrBot text components.
        Returns:
            Whether GitLab accepted the message.
        """
        dest = re.fullmatch(
            r"(issues|merge_requests)/([1-9]\d*)(?:/([a-zA-Z0-9_-]+))?",
            session.session_id,
        )
        if not dest or not self.session or session.platform_id != self.config["id"]:
            return False
        text = "\n".join(
            part.text for part in message_chain.chain if isinstance(part, Plain)
        )
        if not text.strip():
            return False
        text = re.sub(r"(?m)^([ \t]*)/", r"\1\\/", text)
        await self.api.publish(f"/{dest[1]}/{dest[2]}", dest[3], text)
        return True
