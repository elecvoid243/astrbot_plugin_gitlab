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
from astrbot.api.star import StarTools

from .events import project_job
from .expert import HELP, ProjectExpert
from .gitlab import APIError, GitLab
from .source import SourceIndex


DEFAULTS = dict(
    gitlab_url="",
    gitlab_project_id=0,
    gitlab_token="",
    gitlab_webhook_secret="",
    gitlab_provider="",
    gitlab_auto_issue=True,
    gitlab_auto_review=True,
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
    context = None

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

    async def generate(self, system, prompt):
        reply = await self.context.llm_generate(
            chat_provider_id=self.config["gitlab_provider"],
            system_prompt=system,
            prompt=prompt,
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
                    "payload=excluded.payload, attempts=0, due=0, state='queued' WHERE jobs.state='failed'",
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
                await self.process(json.loads(payload))
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

    async def process(self, job):
        """Resolve membership and pinned source before publishing one result.

        Args:
            job: Durable, project-scoped webhook work item.
        """
        project = await self.api.get()
        if not await self.api.member(job["actor"], project):
            return
        path = f"/{job['resource']}/{job['iid']}"
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
        identifier = "review-summary" if automatic else job["key"]
        digest = hashlib.sha256(
            (self.config["id"] + ":" + identifier).encode()
        ).hexdigest()[:24]
        marker = f"<!-- astrbot-gitlab:{digest} -->"
        current = await self.api.marker_note(path, marker, self.bot["id"])
        version = f"<!-- head:{item.get('sha', '')} -->"
        if current and (not automatic or version in current["body"]):
            return
        if job["key"].startswith("note:") and not job["thread"]:
            # Some GitLab versions omit discussion_id in the Note Hook.
            note_id = int(job["key"].split(":")[1])
            page = 1
            while not job["thread"]:
                groups = await self.api.get(
                    f"{path}/discussions", per_page=100, page=page
                )
                for group in groups:
                    if any(note["id"] == note_id for note in group.get("notes", [])):
                        job["thread"] = group["id"]
                        break
                if not job["thread"] and len(groups) < 100:
                    return
                page += 1
        if job["command"] == "help":
            body = HELP
        elif not is_mr and (
            not project.get("default_branch") or project.get("empty_repo")
        ):
            self.source.sha = ""
            thread = await self.api.discussion(path, job["thread"])
            body = await self.expert.analyze(job, item, discussion=thread)
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
            diff = ""
            if is_mr:
                refs = item.get("diff_refs") or {}
                if refs.get("head_sha") != sha or not re.fullmatch(
                    r"[a-f0-9]{40,64}", refs.get("base_sha") or ""
                ):
                    raise ValueError("MR diff version is not ready")
                diff = await asyncio.to_thread(self.source.diff, refs["base_sha"])
            thread = await self.api.discussion(path, job["thread"])
            body = await self.expert.analyze(job, item, diff, thread)
            if is_mr:
                latest = await self.api.get(path)
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
        await self.api.publish(
            path, job["thread"], f"{body}\n\n{marker}\n{version}", current
        )

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
