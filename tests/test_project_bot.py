"""Exercise the actual AstrBot adapter against Git and a local GitLab API double."""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import httpx
from aiohttp import web
from anthropic import AsyncAnthropic
from astrbot.core.agent.context.token_counter import EstimateTokenCounter
from astrbot.core.agent.message import Message
from astrbot.core.agent_team_execution import current_model_output_limit
from astrbot.core.provider.sources.anthropic_source import ProviderAnthropic
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from astrbot.dashboard.api.platform import router
from astrbot.dashboard.services.platform_service import PlatformService
from fastapi import FastAPI
from openai import AsyncOpenAI

from ..adapter import DEFAULTS, GitLabPlatform, StarTools
from ..events import project_job
from ..gitlab import GitLab
from ..main import GitLabPlugin


class TestProjectBot(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.git("init", "-b", "main")
        (self.work / "caller.py").write_text(
            "from service import find_order\nprint(find_order(42).upper())\n"
        )
        (self.work / "service.py").write_text(
            "def find_order(key):\n    return 'order'\n"
        )
        self.git("add", ".")
        self.git("commit", "-m", "Fix timeout when loading order")
        self.base = self.git("rev-parse", "HEAD")
        (self.work / "service.py").write_text("def find_order(key):\n    return None\n")
        self.git("commit", "-am", "Change order fallback")
        self.head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/merge-requests/2/head", self.head)
        self.notes = {}
        self.review_threads = {}
        self.reject_review_reply = False
        self.reject_review_discussion_lookup = False
        self.review_notes_second_page = False
        self.lose_review_update_response = False
        self.reject_next_review_notice = False
        self.calls = []
        self.reports = []
        self.model_requests = []
        self.change_during_review = False
        self.advance_head_on_discussion = False
        self.base_after_review = None
        self.draft = False
        app = web.Application()
        app.router.add_route("*", "/api/v4/{path:.*}", self.route)
        runner = web.AppRunner(app)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        self.origin = f"http://127.0.0.1:{runner.addresses[0][1]}"
        # Redirect only this fixture's Git transport to its on-disk repository.
        git_config = self.root / "gitconfig"
        subprocess.run(
            [
                "git",
                "config",
                "--file",
                str(git_config),
                f"url.{self.work.as_uri()}.insteadOf",
                self.origin + "/team/demo.git",
            ],
            check=True,
        )
        original = os.environ.get("GIT_CONFIG_GLOBAL")
        os.environ["GIT_CONFIG_GLOBAL"] = str(git_config)
        if original is None:
            self.addCleanup(os.environ.pop, "GIT_CONFIG_GLOBAL", None)
        else:
            self.addCleanup(os.environ.__setitem__, "GIT_CONFIG_GLOBAL", original)
        self.botdata = self.root / "botdata"
        data = patch.object(
            StarTools,
            "get_data_dir",
            return_value=self.botdata,
        )
        data.start()
        self.addCleanup(data.stop)
        self.config: dict[str, Any] = dict(
            DEFAULTS,
            id="gitlab-demo",
            gitlab_url=self.origin,
            gitlab_project_id=8,
            gitlab_token="test-api",
            gitlab_webhook_secret="test-hook",
            gitlab_provider="test-model",
        )
        self.provider = SimpleNamespace(provider_config={"max_context_tokens": 24000})
        ctx = SimpleNamespace(
            llm_generate=self.generate,
            get_provider_by_id=lambda _: self.provider,
        )
        GitLabPlugin(ctx)
        self.adapter = GitLabPlatform(self.config, {}, asyncio.Queue())
        self.runner = asyncio.create_task(self.adapter.run())
        self.addAsyncCleanup(self.adapter.terminate)
        await self.wait(lambda: self.adapter.bot)
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        core = SimpleNamespace(
            platform_manager=SimpleNamespace(platform_insts=[self.adapter])
        )
        app.state.services = SimpleNamespace(platforms=PlatformService(core))
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)

    def git(self, *args):
        return (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(self.work),
                    "-c",
                    "user.name=Tester",
                    "-c",
                    "user.email=test@example.invalid",
                    *args,
                ]
            )
            .decode()
            .strip()
        )

    async def wait(self, condition):
        for _ in range(400):
            if self.runner.done():
                await self.runner
            if condition():
                return
            await asyncio.sleep(0.025)
        self.fail(f"Timed out: calls={self.calls}")

    async def route(self, req):
        self.assertEqual(req.headers.get("PRIVATE-TOKEN"), "test-api")
        path = "/" + req.match_info["path"]
        self.calls.append((req.method, path))
        if path == "/user":
            return web.json_response({"id": 77, "username": "review-bot"})
        if path == "/projects/8":
            return web.json_response(
                dict(
                    id=8,
                    path_with_namespace="team/demo",
                    default_branch="main",
                    visibility="private",
                    repository_access_level="enabled",
                )
            )
        path = path.removeprefix("/projects/8")
        if path == "/members/all/3":
            return web.json_response(dict(access_level=30, state="active"))
        if path.startswith("/members/"):
            raise web.HTTPNotFound()
        if path == "/issues/1":
            return web.json_response(
                dict(
                    iid=1,
                    title="order timeout",
                    description="find_order results",
                    state="opened",
                )
            )
        if path == "/issues/8":
            return web.json_response(
                dict(iid=8, title="Earlier issue", state="closed", project_id=8)
            )
        if path == "/merge_requests/2":
            return web.json_response(
                dict(
                    iid=2,
                    title="Change order lookup",
                    state="opened",
                    draft=self.draft,
                    sha=self.head,
                    diff_refs={"head_sha": self.head, "base_sha": self.base},
                )
            )
        if path in ("/issues", "/merge_requests"):
            if path == "/issues":
                data = [
                    dict(
                        iid=i,
                        title="Previous order timeout",
                        state="closed",
                        project_id=8,
                        confidential=i == 9,
                        description="Retry fails after find_order returns None",
                    )
                    for i in (1, 8, 9)
                ]
            else:
                data = [
                    dict(
                        iid=5,
                        title="Earlier fix",
                        state="merged",
                        project_id=8,
                        description="find_order timeout",
                    )
                ]
            return web.json_response(data)
        if path == "/merge_requests/2/discussions" and req.method == "POST":
            payload = await req.json()
            notes = self.notes.setdefault("/merge_requests/2", [])
            note = dict(
                id=len(notes) + 1,
                author={"id": 77},
                body=payload["body"],
                type="DiscussionNote",
                resolvable=True,
                resolved=False,
            )
            thread = dict(id="summary-thread", individual_note=False, notes=[note])
            self.review_threads[thread["id"]] = thread
            notes.append(note)
            return web.json_response(thread, status=201)
        if path.endswith("/discussions"):
            groups = [dict(id="abc", notes=[dict(id=101)])]
            if path == "/merge_requests/2/discussions":
                if self.reject_review_discussion_lookup:
                    raise web.HTTPForbidden()
                groups.extend(self.review_threads.values())
            return web.json_response(groups)
        if path.endswith("/discussions/abc"):
            if self.advance_head_on_discussion:
                self.advance_head_on_discussion = False
                self.git(
                    "commit", "--allow-empty", "-m", "Push during review collection"
                )
                self.head = self.git("rev-parse", "HEAD")
                self.git("update-ref", "refs/merge-requests/2/head", self.head)
            return web.json_response(
                dict(
                    notes=[
                        dict(body="Earlier thread context"),
                        dict(body="hidden note", internal=True),
                    ]
                )
            )
        if (
            self.reject_review_reply
            and req.method == "POST"
            and path == "/merge_requests/2/discussions/abc/notes"
        ):
            raise web.HTTPNotFound()
        if "/notes" in path:
            scope = "/".join(path.split("/")[:3])
            notes = self.notes.setdefault(scope, [])
            if req.method == "GET":
                if self.review_notes_second_page and path == "/merge_requests/2/notes":
                    page = req.query.get("page", "1")
                    if page == "1":
                        return web.json_response(
                            notes[:1], headers={"X-Next-Page": "2"}
                        )
                    return web.json_response(notes[1:], headers={"X-Next-Page": ""})
                return web.json_response(notes)
            payload = await req.json()
            if req.method == "POST":
                if self.reject_next_review_notice and "#note_" in payload["body"]:
                    self.reject_next_review_notice = False
                    raise web.HTTPBadGateway()
                note = dict(id=len(notes) + 1, author={"id": 77}, body=payload["body"])
                notes.append(note)
            elif req.method == "PUT":
                note_id = int(path.split("/")[-1])
                note = next(note for note in notes if note["id"] == note_id)
                note["body"] = payload["body"]
                if self.lose_review_update_response and scope == "/merge_requests/2":
                    self.lose_review_update_response = False
                    raise web.HTTPBadGateway()
            return web.json_response(note)
        raise web.HTTPNotFound()

    async def generate(self, **kwargs):
        self.assertEqual(kwargs["chat_provider_id"], "test-model")
        prompt = json.loads(kwargs["prompt"])
        self.model_requests.append(kwargs)
        if "requests" in kwargs["system_prompt"]:
            result = {
                "requests": [
                    dict(tool="code", query="find_order"),
                    dict(tool="read", path="caller.py", line=2),
                    dict(tool="history", query="timeout"),
                ]
            }
        else:
            self.reports.append(prompt)
            index = prompt["evidence"]
            src = [k for k, r in index.items() if "caller.py" in r["label"]]
            self.assertFalse(any(r["label"] == "#9" for r in index.values()))
            result = dict(
                findings=[dict(text="检查调用方是否对空值调用 upper", refs=src)]
            )
            if prompt["task"]["command"] != "review":
                issue = next(k for k, r in index.items() if r["label"] == "#8")
                result["relations"] = [
                    dict(
                        issue=issue,
                        status="duplicate"
                        if prompt["task"]["command"] == "diagnose"
                        else "related",
                        reason="同一超时条件",
                    )
                ]
            if self.change_during_review:
                self.head = "d" * 40
            if self.base_after_review:
                self.base = self.base_after_review
            if prompt["task"]["command"] == "review":
                output_limit = current_model_output_limit.get()
                assert output_limit is not None
                self.assertGreater(output_limit, 0)
                tokens = EstimateTokenCounter().count_tokens(
                    [
                        Message(role="system", content=kwargs["system_prompt"]),
                        Message(role="user", content=kwargs["prompt"]),
                    ]
                )
                self.assertLessEqual(
                    tokens + output_limit,
                    self.provider.provider_config["max_context_tokens"],
                )
        return SimpleNamespace(completion_text=json.dumps(result))

    async def deliver(self, payload, secret="test-hook"):
        response = await self.deliver_native(payload, secret=secret)
        return response.json(), response.status_code

    async def deliver_native(self, payload, secret: str | None = "test-hook"):
        path = "/api/v1/webhooks/platforms/" + self.config["webhook_uuid"]
        headers = {"X-Gitlab-Token": secret} if secret else {}
        return await self.client.post(path, json=payload, headers=headers)

    def review_command(self, note_id, **attrs):
        event = self.event(
            "note",
            noteable_type="MergeRequest",
            action="create",
            id=note_id,
            note="/review",
            **attrs,
        )
        event["merge_request"] = {"iid": 2}
        return event

    def use_native_provider(self, provider):
        self.provider = provider

        async def generate_native(*, chat_provider_id, **kwargs):
            self.assertEqual(chat_provider_id, "test-model")
            return await provider.text_chat(**kwargs)

        self.adapter.context.llm_generate = generate_native

    async def use_native_chat(self, model, *, reject_context=False):
        provider = ProviderOpenAIOfficial(
            {
                "id": "test-model",
                "model": model,
                "key": ["offline"],
                "api_base": "https://example.invalid/v1",
                "max_context_tokens": 24000,
            },
            {},
        )
        await provider.client.close()
        requests = []

        def record(request):
            payload = json.loads(request.content)
            requests.append(payload)
            if reject_context and any(m["role"] == "user" for m in payload["messages"]):
                return httpx.Response(
                    400, json={"error": {"message": "maximum context length"}}
                )
            findings = [{"text": "模型返回的发现", "refs": ["E1"]}]
            return httpx.Response(
                200,
                json={
                    "id": "chat_1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"findings": findings}),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 21,
                        "total_tokens": 71,
                    },
                },
            )

        provider.client = AsyncOpenAI(
            api_key="offline",
            base_url="https://example.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(record)),
        )
        self.addAsyncCleanup(provider.terminate)
        self.use_native_provider(provider)
        return requests

    def event(self, kind, **attrs) -> dict[str, Any]:
        return dict(
            object_kind=kind, project={"id": 8}, user={"id": 3}, object_attributes=attrs
        )

    async def wait_done(self):
        await self.wait(
            lambda: (
                self.adapter.db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE state!='done'"
                ).fetchone()
                == (0,)
            )
        )

    async def test_issue_triage_history_and_replay(self):
        event = self.event("issue", action="open", iid=1)
        self.assertEqual((await self.deliver(event, "wrong"))[1], 403)
        await self.deliver(event)
        await self.deliver(event)
        await self.wait_done()
        notes = self.notes["/issues/1"]
        self.assertEqual(len(notes), 1)
        self.assertIn("疑似重复", notes[0]["body"])
        self.assertIn("/-/issues/8", notes[0]["body"])
        self.assertIn(f"/-/blob/{self.head}/caller.py", notes[0]["body"])
        self.assertEqual(
            {r["kind"] for r in self.reports[0]["evidence"].values()},
            {"issue", "mr", "commit", "code"},
        )
        self.assertTrue(any(self.botdata.rglob("source.sqlite")))

    async def test_update_summary_and_find_unchanged_caller(self):
        event = self.event(
            "merge_request", action="open", iid=2, last_commit={"id": self.head}
        )
        await self.deliver_native(event)
        await self.wait_done()
        self.assertNotIn("+++ b/caller.py", self.reports[0]["task"]["diff"])
        self.assertIn("caller.py", self.notes["/merge_requests/2"][0]["body"])
        self.assertNotIn("**疑似重复**", self.notes["/merge_requests/2"][0]["body"])
        previous = self.head
        (self.work / "service.py").write_text("def find_order(key):\n    return 0\n")
        self.git("commit", "-am", "Change fallback to int")
        self.head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/merge-requests/2/head", self.head)
        event["object_attributes"].update(
            action="update", oldrev=previous, last_commit={"id": self.head}
        )
        await self.deliver_native(event)
        await self.wait_done()
        await self.deliver_native(event)
        await self.wait_done()
        self.assertEqual(len(self.reports), 2)
        notes = self.notes["/merge_requests/2"]
        self.assertEqual(len(notes), 2)
        self.assertIn(self.head, notes[0]["body"])
        self.assertIn(
            f"{self.origin}/team/demo/-/merge_requests/2#note_1", notes[1]["body"]
        )
        self.assertIn(("PUT", "/projects/8/merge_requests/2/notes/1"), self.calls)

    async def test_manual_review_updates_automatic_summary_and_replay_is_idempotent(
        self,
    ):
        opened = self.event(
            "merge_request", action="open", iid=2, last_commit={"id": self.head}
        )
        self.assertEqual((await self.deliver_native(opened)).json(), {"accepted": True})
        await self.wait_done()
        notes = self.notes["/merge_requests/2"]
        self.assertEqual(len(notes), 1)
        self.assertIn("return None", self.reports[0]["task"]["diff"])
        self.assertIn("caller.py", notes[0]["body"])
        self.adapter.config["publish_review_as_thread"] = True

        command = self.review_command(101, discussion_id="abc")
        await self.deliver_native(command)
        await self.wait_done()
        await self.deliver_native(command)
        await self.wait_done()
        self.assertEqual(len(self.reports), 2)
        self.assertEqual(
            self.reports[1]["task"]["diff"], self.reports[0]["task"]["diff"]
        )
        self.assertEqual(
            self.calls.count(("PUT", "/projects/8/merge_requests/2/notes/1")), 1
        )
        self.assertFalse(
            any(
                method == "POST" and "/discussions" in path
                for method, path in self.calls
            )
        )
        self.assertEqual(len(notes), 2)
        self.assertIn(
            f"{self.origin}/team/demo/-/merge_requests/2#note_1", notes[1]["body"]
        )
        self.assertNotIn("caller.py", notes[1]["body"])

    async def test_resolvable_summary_keeps_its_discussion_after_setting_changes(self):
        self.adapter.config["publish_review_as_thread"] = True
        await self.deliver_native(self.review_command(101, discussion_id="abc"))
        await self.wait_done()
        self.assertEqual(list(self.review_threads), ["summary-thread"])
        summary = self.review_threads["summary-thread"]["notes"][0]
        self.assertTrue(summary["resolvable"])
        self.assertIn("caller.py", summary["body"])
        original_body = summary["body"]
        self.adapter.config.update(
            publish_review_as_thread=False, reply_to_trigger_comment=True
        )
        self.reject_review_discussion_lookup = True

        await self.deliver_native(self.review_command(102))
        await self.wait_done()
        update = self.calls.index(
            ("PUT", f"/projects/8/merge_requests/2/notes/{summary['id']}")
        )
        self.assertEqual(
            self.calls[update - 1], ("GET", "/projects/8/merge_requests/2")
        )
        self.assertNotEqual(summary["body"], original_body)
        self.assertEqual(len(self.review_threads), 1)
        self.assertEqual(len(self.reports), 2)
        self.assertIn(
            f"{self.origin}/team/demo/-/merge_requests/2#note_{summary['id']}",
            self.notes["/merge_requests/2"][1]["body"],
        )

    async def test_update_link_goes_to_the_trigger_discussion(self):
        await self.deliver_native(self.review_command(100))
        await self.wait_done()
        self.adapter.config["reply_to_trigger_comment"] = True

        # This Note Hook has no discussion_id. GitLab lists the command in abc.
        await self.deliver_native(self.review_command(101))
        await self.wait_done()
        self.assertIn(
            ("POST", "/projects/8/merge_requests/2/discussions/abc/notes"), self.calls
        )
        self.assertEqual(
            self.calls.count(("POST", "/projects/8/merge_requests/2/notes")), 1
        )
        self.assertIn(
            f"{self.origin}/team/demo/-/merge_requests/2#note_1",
            self.notes["/merge_requests/2"][1]["body"],
        )

    async def test_update_link_falls_back_when_lookup_or_reply_fails(self):
        await self.deliver_native(self.review_command(100))
        await self.wait_done()
        self.adapter.config["reply_to_trigger_comment"] = True

        self.reject_review_discussion_lookup = True
        await self.deliver_native(self.review_command(101))
        await self.wait_done()
        self.assertNotIn(
            ("POST", "/projects/8/merge_requests/2/discussions/abc/notes"), self.calls
        )

        self.reject_review_discussion_lookup = False
        self.reject_review_reply = True
        await self.deliver_native(self.review_command(102, discussion_id="abc"))
        await self.wait_done()
        self.assertIn(
            ("POST", "/projects/8/merge_requests/2/discussions/abc/notes"), self.calls
        )
        self.assertEqual(
            self.calls.count(("POST", "/projects/8/merge_requests/2/notes")), 3
        )
        notes = self.notes["/merge_requests/2"]
        self.assertEqual(len(notes), 3)
        for notice in notes[1:]:
            self.assertIn(
                f"{self.origin}/team/demo/-/merge_requests/2#note_1", notice["body"]
            )

    async def test_members_can_review_drafts_with_automation_and_notices_disabled(self):
        self.adapter.config.update(
            gitlab_auto_review=False,
            final_update_message=False,
            reply_to_trigger_comment=True,
        )
        self.draft = True
        opened = self.event(
            "merge_request",
            action="open",
            iid=2,
            last_commit={"id": self.head},
            draft=True,
        )
        self.assertEqual(
            (await self.deliver_native(opened)).json(), {"accepted": False}
        )
        manual = self.review_command(101, discussion_id="abc")
        manual["user"]["id"] = 99
        await self.deliver_native(manual)
        await self.wait_done()
        self.assertFalse(self.reports)
        self.assertFalse(self.notes.get("/merge_requests/2"))

        manual["user"]["id"] = 3
        await self.deliver_native(manual)
        await self.wait_done()
        await self.deliver_native(self.review_command(102, discussion_id="abc"))
        await self.wait_done()
        self.assertEqual(len(self.reports), 2)
        self.assertEqual(len(self.notes["/merge_requests/2"]), 1)
        self.assertIn(("PUT", "/projects/8/merge_requests/2/notes/1"), self.calls)

    async def test_marking_a_reviewed_draft_ready_runs_automatic_review_at_same_head(
        self,
    ):
        self.draft = True
        await self.deliver_native(self.review_command(101, discussion_id="abc"))
        await self.wait_done()
        self.draft = False
        ready = self.event(
            "merge_request", action="update", iid=2, last_commit={"id": self.head}
        )
        ready["changes"] = {"draft": {"previous": True, "current": False}}
        self.assertEqual((await self.deliver_native(ready)).json(), {"accepted": True})
        await self.wait_done()

        self.assertEqual(len(self.reports), 2)
        self.assertIn(("PUT", "/projects/8/merge_requests/2/notes/1"), self.calls)
        self.assertEqual(len(self.notes["/merge_requests/2"]), 2)

    async def test_updates_only_the_verified_legacy_summary_on_a_later_page(self):
        legacy_marker = "<!-- astrbot-gitlab:92040a3ead789155be366e81 -->"
        forged: dict[str, Any] = dict(
            id=1, author={"id": 3}, body="Copied marker\n" + legacy_marker
        )
        legacy: dict[str, Any] = dict(
            id=2, author={"id": 77}, body="Earlier automatic review\n" + legacy_marker
        )
        notes = self.notes["/merge_requests/2"] = [forged, legacy]
        self.review_notes_second_page = True
        await self.deliver_native(self.review_command(101, discussion_id="abc"))
        await self.wait_done()

        self.assertEqual(forged["body"], "Copied marker\n" + legacy_marker)
        self.assertIn("caller.py", legacy["body"])
        self.assertEqual(len(notes), 3)
        self.assertIn(
            f"{self.origin}/team/demo/-/merge_requests/2#note_2", notes[2]["body"]
        )

    async def test_saved_summary_is_not_rewritten_after_gitlab_loses_update_response(
        self,
    ):
        await self.deliver_native(self.review_command(100))
        await self.wait_done()
        self.lose_review_update_response = True
        await self.deliver_native(self.review_command(101, discussion_id="abc"))
        await self.wait_done()

        self.assertEqual(len(self.reports), 2)
        self.assertEqual(
            self.calls.count(("PUT", "/projects/8/merge_requests/2/notes/1")), 1
        )
        notes = self.notes["/merge_requests/2"]
        self.assertEqual(len(notes), 2)
        self.assertIn(
            f"{self.origin}/team/demo/-/merge_requests/2#note_1", notes[1]["body"]
        )

    async def test_notice_retry_does_not_rerun_a_review_after_a_newer_update(self):
        await self.deliver_native(self.review_command(100))
        await self.wait_done()
        self.reject_next_review_notice = True
        await self.deliver_native(self.review_command(101))
        await self.wait(lambda: not self.reject_next_review_notice)

        newer = self.review_command(102)
        newer["object_attributes"]["note"] = "/review 检查新的评审请求"
        await self.deliver_native(newer)
        notes = self.notes["/merge_requests/2"]
        await self.wait(lambda: len(notes) == 2)
        newer_body = notes[0]["body"]

        await self.adapter.terminate()
        self.adapter = GitLabPlatform(self.config, {}, asyncio.Queue())
        self.runner = asyncio.create_task(self.adapter.run())
        self.addAsyncCleanup(self.adapter.terminate)
        await self.wait(lambda: self.adapter.bot)
        await self.wait_done()
        self.assertEqual(len(self.reports), 3)
        self.assertEqual(self.reports[-1]["task"]["question"], "检查新的评审请求")
        self.assertEqual(notes[0]["body"], newer_body)
        self.assertEqual(
            self.calls.count(("PUT", "/projects/8/merge_requests/2/notes/1")), 2
        )
        self.assertEqual(len(notes), 3)

    async def test_stale_result_and_unauthorized_actor(self):
        event = self.event(
            "merge_request", action="open", iid=2, last_commit={"id": self.head}
        )
        event["user"]["id"] = 99
        await self.deliver(event)
        await self.wait_done()
        self.assertFalse(self.reports)
        event["user"]["id"] = 3
        self.change_during_review = True
        await self.deliver(event)
        await self.wait_done()
        self.assertFalse(self.notes.get("/merge_requests/2"))

    async def test_review_includes_all_patches_past_old_limit_in_one_request(self):
        (self.work / "a_feature.txt").write_text(
            "".join(f"account event {i:03} ready for processing\n" for i in range(260))
        )
        (self.work / "z_feature.py").write_text("last_change = True\n")
        self.git("add", ".")
        self.git("commit", "-m", "Add two features")
        self.head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/merge-requests/2/head", self.head)
        event = self.event(
            "merge_request", action="open", iid=2, last_commit={"id": self.head}
        )

        self.assertEqual(await self.deliver(event), ({"accepted": True}, 200))
        await self.wait_done()

        self.assertEqual(len(self.reports), 1)
        diff = self.reports[0]["task"]["diff"]
        self.assertIn("+account event 259 ready for processing", diff)
        self.assertIn("+    return None", diff)
        self.assertIn("+last_change = True", diff)
        self.assertEqual(len(self.model_requests), 1)
        body = self.notes["/merge_requests/2"][0]["body"]
        self.assertIn(f"/-/blob/{self.head}/caller.py", body)
        self.assertNotIn("预算遗漏", body)

    async def test_review_filters_patches_and_source_evidence_by_default(self):
        self.provider.provider_config["max_context_tokens"] = 1500
        excluded = {
            "vendor/order_helper.py": "vendor_marker = find_order(17)",
            "ui/package-lock.json": "package_lock_marker",
            "yarn.lock": "yarn_lock_marker",
            "pnpm-lock.yaml": "pnpm_lock_marker",
            "go.sum": "go_sum_marker",
            "uv.lock": "uv_lock_marker",
            "public/widget.MIN.JS": "minified_js_marker",
            "public/widget.min.css": "minified_css_marker",
            "public/widget.ts.map": "source_map_marker",
            "icon.svg": "<svg></svg>",
        }
        for filename, content in excluded.items():
            file = self.work / filename
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(content + "\n")
        (self.work / "a_large_source.txt").write_text("x" * 18000 + "\n")
        self.git("add", ".")
        self.git(
            "commit", "-m", "Add dependencies, build artifacts, and project source"
        )
        self.head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/merge-requests/2/head", self.head)
        event = self.event(
            "merge_request", action="open", iid=2, last_commit={"id": self.head}
        )

        self.assertEqual(await self.deliver(event), ({"accepted": True}, 200))
        await self.wait_done()

        self.assertEqual(len(self.model_requests), 1)
        prompt = self.model_requests[0]["prompt"]
        self.assertIn("+    return None", self.reports[0]["task"]["diff"])
        self.assertIn("print(find_order(42).upper())", prompt)
        for filename, content in excluded.items():
            with self.subTest(path=filename):
                self.assertNotIn(filename, prompt)
                self.assertNotIn(content, prompt)
        body = self.notes["/merge_requests/2"][0]["body"]
        self.assertIn("预算遗漏的文件**\n\n- `a_large_source.txt`", body)
        budget_section, filtered_section = body.split("**按文件规则过滤的文件**")
        for filename in excluded:
            with self.subTest(path=filename):
                self.assertNotIn(filename, budget_section)
                self.assertIn(f"`{filename}`", filtered_section)

    async def test_review_summarizes_excess_filtered_paths(self):
        vendor = self.work / "vendor"
        vendor.mkdir()
        for index in range(53):
            (vendor / f"file_{index:02d}.py").touch()
        self.git("add", ".")
        self.git("commit", "-m", "Import a vendor tree")
        self.head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/merge-requests/2/head", self.head)
        event = self.event(
            "merge_request", action="open", iid=2, last_commit={"id": self.head}
        )

        self.assertEqual(await self.deliver(event), ({"accepted": True}, 200))
        await self.wait_done()

        body = self.notes["/merge_requests/2"][0]["body"]
        self.assertIn("按文件规则过滤的文件", body)
        self.assertEqual(body.count("\n- `vendor/"), 50)
        self.assertIn("- `vendor/file_00.py`", body)
        self.assertIn("- `vendor/file_49.py`", body)
        self.assertNotIn("vendor/file_50.py", body)
        self.assertIn("- 另有 3 个文件", body)

    async def test_project_review_filters_only_enabled_generated_frameworks(self):
        for filename in ("schema_pb2.py", "api/__generated__/client.py"):
            file = self.work / filename
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("generated_call = find_order(21)\n")
        self.git("add", ".")
        self.git("commit", "-m", "Add two kinds of generated clients")
        self.head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/merge-requests/2/head", self.head)

        await self.deliver(
            self.event(
                "merge_request", action="open", iid=2, last_commit={"id": self.head}
            )
        )
        await self.wait_done()

        diff = self.reports[0]["task"]["diff"]
        self.assertIn("schema_pb2.py", diff)
        self.assertIn("api/__generated__/client.py", diff)

        self.config["ignore_language_framework"] = ["protobuf"]
        event = self.event(
            "note",
            noteable_type="MergeRequest",
            action="create",
            id=101,
            note="/review",
        )
        event["merge_request"] = {"iid": 2}
        self.assertEqual(await self.deliver(event), ({"accepted": True}, 200))
        await self.wait_done()

        self.assertEqual(len(self.model_requests), 2)
        self.assertNotIn("schema_pb2.py", self.model_requests[1]["prompt"])
        self.assertIn("api/__generated__/client.py", self.reports[1]["task"]["diff"])
        body = self.notes["/merge_requests/2"][0]["body"]
        self.assertIn("按文件规则过滤的文件**\n\n- `schema_pb2.py`", body)
        self.assertNotIn("api/__generated__/client.py", body)

    async def test_review_reports_only_whole_file_deletions_as_filename_only(self):
        self.git("switch", "--detach", self.base)
        (self.work / "retired.py").write_text("def retired_order():\n    return 17\n")
        (self.work / "service.py").write_text(
            "def find_order(key):\n    print('temporary trace')\n    return 'order'\n"
        )
        (self.work / "old_name.py").write_text("rename_marker = True\n")
        (self.work / "move_into_vendor.py").write_text("vendor_rename_marker = True\n")
        (self.work / "vendor").mkdir()
        (self.work / "vendor" / "retired.py").write_text(
            "vendor_delete_marker = True\n"
        )
        (self.work / "switch_type.py").write_text("original_type = 'regular file'\n")
        self.git("add", ".")
        self.git("commit", "-m", "Prepare files for the comparison")
        self.base = self.git("rev-parse", "HEAD")

        self.git("rm", "retired.py", "vendor/retired.py")
        (self.work / "service.py").write_text(
            "def find_order(key):\n    return 'order'\n"
        )
        self.git("add", "service.py")
        self.git("mv", "old_name.py", "new_name.py")
        (self.work / "vendor").mkdir(exist_ok=True)
        self.git("mv", "move_into_vendor.py", "vendor/moved.py")
        # Stage a symlink directly so this also exercises Git status T on Windows.
        target_file = self.work / "link_target"
        target_file.write_bytes(b"new_name.py")
        target = self.git("hash-object", "-w", "link_target")
        self.git("update-index", "--cacheinfo", f"120000,{target},switch_type.py")
        self.git("commit", "-m", "Delete, rename, and change the type of files")
        self.head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/merge-requests/2/head", self.head)
        event = self.event(
            "merge_request", action="open", iid=2, last_commit={"id": self.head}
        )

        self.assertEqual(await self.deliver(event), ({"accepted": True}, 200))
        await self.wait_done()

        self.assertEqual(len(self.model_requests), 1)
        prompt = self.model_requests[0]["prompt"]
        self.assertNotIn("retired_order", prompt)
        self.assertNotIn("vendor_delete_marker", prompt)
        self.assertNotIn("vendor/moved.py", prompt)
        diff = self.reports[0]["task"]["diff"]
        self.assertIn("-    print('temporary trace')", diff)
        self.assertIn("rename to new_name.py", diff)
        self.assertIn("-original_type = 'regular file'", diff)
        self.assertIn("+new_name.py", diff)
        body = self.notes["/merge_requests/2"][0]["body"]
        self.assertIn("仅以文件名列出的删除文件**\n\n- `retired.py`\n\n", body)
        self.assertIn(
            "按文件规则过滤的文件**\n\n- `vendor/moved.py`\n- `vendor/retired.py`", body
        )
        self.assertNotIn("预算遗漏", body)
        for filename in ("service.py", "new_name.py", "switch_type.py"):
            self.assertNotIn(f"`{filename}`", body)

    async def test_review_with_only_whole_file_deletions_skips_the_model(self):
        self.git("switch", "--detach", self.base)
        self.git("rm", "service.py")
        self.git("commit", "-m", "Remove the order service")
        self.head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/merge-requests/2/head", self.head)
        event = self.event(
            "merge_request", action="open", iid=2, last_commit={"id": self.head}
        )

        self.assertEqual(await self.deliver(event), ({"accepted": True}, 200))
        await self.wait_done()

        self.assertFalse(self.model_requests)
        body = self.notes["/merge_requests/2"][0]["body"]
        self.assertIn("本次没有提交给模型评审的文件补丁。", body)
        self.assertIn("仅以文件名列出的删除文件**\n\n- `service.py`", body)
        self.assertNotIn("预算遗漏", body)

    async def test_manual_review_drops_result_when_head_changes_during_collection(self):
        event = self.event(
            "note",
            noteable_type="MergeRequest",
            action="create",
            id=101,
            note="/review",
        )
        event["merge_request"] = {"iid": 2}
        self.advance_head_on_discussion = True

        self.assertEqual(await self.deliver(event), ({"accepted": True}, 200))
        await self.wait_done()

        self.assertFalse(self.model_requests)
        self.assertFalse(self.notes.get("/merge_requests/2"))

    async def test_review_prefers_complete_patches_to_extra_source_excerpts(self):
        self.provider.provider_config["max_context_tokens"] = 1800
        for filename, repeated in (("a_more.txt", "a"), ("z_more.txt", "b")):
            (self.work / filename).write_text(
                "".join(f"row {i:02d}: {repeated * 40}\n" for i in range(25))
            )
        self.git("add", ".")
        self.git("commit", "-m", "Add two different data files")
        self.head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/merge-requests/2/head", self.head)
        event = self.event(
            "merge_request", action="open", iid=2, last_commit={"id": self.head}
        )

        await self.deliver(event)
        await self.wait_done()

        self.assertEqual(len(self.model_requests), 1)
        diff = self.reports[0]["task"]["diff"]
        self.assertIn("+row 24: " + "a" * 40, diff)
        self.assertIn("+row 24: " + "b" * 40, diff)
        body = self.notes["/merge_requests/2"][0]["body"]
        self.assertIn(f"/-/blob/{self.head}/caller.py", body)
        self.assertNotIn("预算遗漏", body)

    async def test_review_reserves_room_for_the_unchanged_call_site(self):
        self.provider.provider_config["max_context_tokens"] = 1500
        self.git("switch", "--detach", self.base)
        (self.work / "caller.py").write_text(
            "from service import find_order\n"
            + "# Existing call-site documentation for order processing\n" * 22
            + "print(find_order(42).upper())\n"
        )
        self.git("commit", "-am", "Document the existing caller")
        self.base = self.git("rev-parse", "HEAD")
        (self.work / "service.py").write_text("def find_order(key):\n    return None\n")
        for filename, repeated in (("a_padding.txt", "a"), ("z_padding.txt", "z")):
            (self.work / filename).write_text(
                "".join(f"row {i:02d}: {repeated * 40}\n" for i in range(25))
            )
        self.git("add", ".")
        self.git("commit", "-m", "Change order fallback with additional data")
        self.head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/merge-requests/2/head", self.head)
        event = self.event(
            "merge_request", action="open", iid=2, last_commit={"id": self.head}
        )

        self.assertEqual(await self.deliver(event), ({"accepted": True}, 200))
        await self.wait_done()

        self.assertEqual(len(self.model_requests), 1)
        call_sites = [
            row["text"]
            for row in self.reports[0]["evidence"].values()
            if "caller.py" in row["label"]
        ]
        self.assertTrue(
            any("print(find_order(42).upper())" in text for text in call_sites)
        )
        diff = self.reports[0]["task"]["diff"]
        self.assertIn("+    return None", diff)
        self.assertIn("+row 24: " + "a" * 40, diff)
        self.assertNotIn("z_padding.txt", diff)
        body = self.notes["/merge_requests/2"][0]["body"]
        self.assertIn(f"/-/blob/{self.head}/caller.py", body)
        self.assertIn("预算遗漏的文件**\n\n- `z_padding.txt`", body)

    async def test_review_sends_the_o_series_limit_in_the_native_chat_payload(self):
        requests = await self.use_native_chat("o3-2025-04-16")
        await self.deliver(
            self.event(
                "merge_request", action="open", iid=2, last_commit={"id": self.head}
            )
        )
        await self.wait_done()

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["max_completion_tokens"], 2048)
        self.assertNotIn("max_tokens", requests[0])

    async def test_review_prioritizes_a_patch_with_evidence_after_an_early_large_file(
        self,
    ):
        self.provider.provider_config["max_context_tokens"] = 1500
        (self.work / "a_padding.txt").write_text("a" * 2800 + "\n")
        self.git("add", "a_padding.txt")
        self.git("commit", "-m", "Add a large data file")
        self.head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/merge-requests/2/head", self.head)
        await self.deliver(
            self.event(
                "merge_request", action="open", iid=2, last_commit={"id": self.head}
            )
        )
        await self.wait_done()

        self.assertEqual(len(self.model_requests), 1)
        self.assertIn("+    return None", self.reports[0]["task"]["diff"])
        self.assertNotIn("a_padding.txt", self.reports[0]["task"]["diff"])
        body = self.notes["/merge_requests/2"][0]["body"]
        self.assertIn(f"/-/blob/{self.head}/caller.py", body)
        self.assertIn("预算遗漏的文件**\n\n- `a_padding.txt`", body)

    async def test_review_keeps_an_intact_patch_when_its_evidence_cannot_fit(self):
        self.provider.provider_config["max_context_tokens"] = 380
        await self.deliver(
            self.event(
                "merge_request", action="open", iid=2, last_commit={"id": self.head}
            )
        )
        await self.wait_done()

        self.assertEqual(len(self.model_requests), 1)
        self.assertIn("+    return None", self.reports[0]["task"]["diff"])
        self.assertNotIn("预算遗漏", self.notes["/merge_requests/2"][0]["body"])

    async def test_review_keeps_the_vllm_limit_in_the_native_chat_payload(self):
        requests = await self.use_native_chat("Qwen/Qwen3-8B")
        await self.deliver(
            self.event(
                "merge_request", action="open", iid=2, last_commit={"id": self.head}
            )
        )
        await self.wait_done()

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["max_tokens"], 2048)
        self.assertNotIn("max_completion_tokens", requests[0])

    async def test_review_reports_failure_after_native_chat_discards_the_diff(self):
        requests = await self.use_native_chat("Qwen/Qwen3-8B", reject_context=True)
        await self.deliver(
            self.event(
                "merge_request", action="open", iid=2, last_commit={"id": self.head}
            )
        )
        await self.wait_done()

        self.assertEqual(len(requests), 2)
        self.assertIn(
            "+    return None",
            json.loads(requests[0]["messages"][-1]["content"])["task"]["diff"],
        )
        self.assertEqual([m["role"] for m in requests[1]["messages"]], ["system"])
        body = self.notes["/merge_requests/2"][0]["body"]
        self.assertIn("评审失败", body)
        self.assertNotIn("模型返回的发现", body)

    async def test_issue_analysis_keeps_the_provider_recovery_behavior(self):
        requests = await self.use_native_chat("Qwen/Qwen3-8B", reject_context=True)
        await self.deliver(self.event("issue", action="open", iid=1))
        await self.wait_done()

        self.assertEqual(len(requests), 4)
        self.assertIn("模型返回的发现", self.notes["/issues/1"][0]["body"])

    async def test_review_keeps_both_halves_of_a_git_file_type_change(self):
        target = self.work / "target.txt"
        target.write_text("caller.py")
        blob = self.git("hash-object", "-w", "target.txt")
        self.git("update-index", "--cacheinfo", f"120000,{blob},service.py")
        self.git("commit", "-m", "Replace the service with a symbolic link")
        self.base = self.head
        self.head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/merge-requests/2/head", self.head)
        event = self.event(
            "merge_request", action="open", iid=2, last_commit={"id": self.head}
        )

        await self.deliver(event)
        await self.wait_done()

        self.assertEqual(len(self.reports), 1)
        diff = self.reports[0]["task"]["diff"]
        self.assertIn("-def find_order(key):", diff)
        self.assertIn("+caller.py", diff)

    async def test_review_reserves_the_effective_anthropic_output_limit(self):
        provider = ProviderAnthropic(
            {
                "id": "test-model",
                "model": "claude-test",
                "max_context_tokens": 5000,
                "custom_extra_body": {"max_output_tokens": 1800},
            },
            {},
            use_api_key=False,
        )
        requests = []

        def record(request):
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {"findings": [{"text": "检查调用方", "refs": ["E1"]}]}
                            ),
                        }
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 200, "output_tokens": 20},
                },
            )

        provider.client = AsyncAnthropic(
            api_key="offline",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(record)),
        )
        self.addAsyncCleanup(provider.terminate)
        self.use_native_provider(provider)
        event = self.event(
            "merge_request", action="open", iid=2, last_commit={"id": self.head}
        )

        await self.deliver(event)
        await self.wait_done()

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["max_tokens"], 1800)
        self.assertNotIn("max_output_tokens", requests[0])
        self.assertIn("/-/blob/", self.notes["/merge_requests/2"][0]["body"])

    async def test_manual_draft_review_omits_whole_files_to_fit_the_selected_model(
        self,
    ):
        self.provider.provider_config["max_context_tokens"] = 1500
        self.config["gitlab_auto_review"] = False
        self.draft = True
        (self.work / "a  large.txt").write_text(
            "".join(f"account event {i:03} ready for processing\n" for i in range(110))
        )
        (self.work / "b_huge.txt").write_text(("a" * 70000) + "\n")
        (self.work / "z_feature.py").write_text("last_change = True\n")
        self.git("add", ".")
        self.git("commit", "-m", "Add large and small features")
        self.head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/merge-requests/2/head", self.head)
        event = self.event(
            "note",
            noteable_type="MergeRequest",
            action="create",
            id=101,
            note="/review",
        )
        event["merge_request"] = {"iid": 2}

        self.assertEqual(await self.deliver(event), ({"accepted": True}, 200))
        await self.wait_done()

        self.assertEqual(len(self.model_requests), 1)
        task = self.reports[0]["task"]
        self.assertEqual((task["base"], task["commit"]), (self.base, self.head))
        self.assertNotIn("a  large.txt", task["diff"])
        self.assertIn("+last_change = True", task["diff"])
        self.assertIn("+    return None", task["diff"])
        body = self.notes["/merge_requests/2"][0]["body"]
        self.assertIn(f"/-/blob/{self.head}/caller.py", body)
        self.assertIn("预算遗漏的文件**\n\n- `a  large.txt`\n- `b_huge.txt`", body)
        self.assertNotIn("z_feature.py", body)

    async def test_review_keeps_existing_summary_when_only_base_changes_before_publish(
        self,
    ):
        event = self.event(
            "merge_request", action="open", iid=2, last_commit={"id": self.head}
        )
        await self.deliver(event)
        await self.wait_done()
        notes = self.notes["/merge_requests/2"]
        original = notes[0]["body"]
        original_base = self.base
        previous_head = self.head
        self.git("commit", "--allow-empty", "-m", "Push a new MR head")
        self.head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/merge-requests/2/head", self.head)
        self.base_after_review = previous_head
        event["object_attributes"].update(
            action="update", oldrev=previous_head, last_commit={"id": self.head}
        )

        await self.deliver(event)
        await self.wait_done()

        self.assertEqual(len(self.model_requests), 2)
        self.assertEqual(self.reports[-1]["task"]["base"], original_base)
        self.assertEqual(notes[0]["body"], original)
        self.assertNotIn(("PUT", "/projects/8/merge_requests/2/notes/1"), self.calls)

    async def test_thread_and_filters(self):
        event = self.event(
            "note",
            noteable_type="Issue",
            action="create",
            id=100,
            note="@review-bot /related timeout",
            discussion_id="abc",
        )
        event["issue"] = {"iid": 1}
        await self.deliver(event)
        await self.wait_done()
        self.assertIn(
            ("POST", "/projects/8/issues/1/discussions/abc/notes"), self.calls
        )
        self.assertEqual(
            self.reports[0]["task"]["discussion"], "Earlier thread context"
        )
        self.assertIn("有关联", self.notes["/issues/1"][0]["body"])
        bot = self.adapter.bot
        for note in ("/unknown", "ordinary text", "@review-bot-else hello"):
            event["object_attributes"]["note"] = note
            self.assertIsNone(project_job(event, 8, bot, self.config))
        event["object_attributes"]["note"] = "@review-bot"
        self.assertEqual(project_job(event, 8, bot, self.config)["command"], "ask")
        event["user"]["id"] = 77
        self.assertIsNone(project_job(event, 8, bot, self.config))
        self.assertIsNone(project_job(event, 9, bot, self.config))
        mr = self.event(
            "merge_request",
            iid=2,
            action="update",
            title="Edited",
            last_commit={"id": self.head},
        )
        self.assertIsNone(project_job(mr, 8, bot, self.config))
        mr["changes"] = {"draft": {"previous": True, "current": False}}
        self.assertEqual(project_job(mr, 8, bot, self.config)["command"], "review")
        self.assertIsNone(
            project_job(mr, 8, bot, dict(self.config, gitlab_auto_review=False))
        )

    async def test_native_webhook_and_panel_registration(self):
        from astrbot.core.platform.register import platform_registry

        registered = next(
            platform for platform in platform_registry if platform.name == "gitlab"
        )
        metadata = registered.config_metadata
        defaults = registered.default_config_tmpl
        assert metadata is not None and defaults is not None
        self.assertEqual(metadata["gitlab_provider"]["_special"], "select_provider")
        self.assertTrue(defaults["gitlab_auto_review"])
        self.assertTrue(defaults["final_update_message"])
        for key in ("publish_review_as_thread", "reply_to_trigger_comment"):
            self.assertFalse(defaults[key])
            self.assertEqual(metadata[key]["type"], "bool")
        event = self.event(
            "note", noteable_type="Issue", action="create", id=101, note="/help"
        )
        event["issue"] = {"iid": 1}
        self.assertEqual(
            (await self.deliver_native(event, secret=None)).status_code, 403
        )
        res = await self.deliver_native(event)
        self.assertEqual((res.status_code, res.json()), (200, {"accepted": True}))
        await self.wait_done()
        self.assertIn(
            ("POST", "/projects/8/issues/1/discussions/abc/notes"), self.calls
        )
        self.assertIn("/related", self.notes["/issues/1"][0]["body"])
        self.assertFalse(self.reports)

    async def test_resume_after_restart_and_ambiguous_post(self):
        event = self.event("issue", action="open", iid=1)
        # A crash after POST leaves the job queued locally and the note present remotely.
        await self.deliver(event)
        await self.wait_done()
        job = project_job(event, 8, self.adapter.bot, self.config)
        with self.adapter.db:
            self.adapter.db.execute(
                "UPDATE jobs SET state='queued',payload=?", (json.dumps(job),)
            )
        await self.adapter.terminate()
        self.adapter = GitLabPlatform(self.config, {}, asyncio.Queue())
        self.runner = asyncio.create_task(self.adapter.run())
        self.addAsyncCleanup(self.adapter.terminate)
        await self.wait(lambda: self.adapter.bot)
        await self.wait_done()
        self.assertEqual(len(self.notes["/issues/1"]), 1)
        self.assertEqual(len(self.reports), 1)

    async def test_history_details_and_issue_comments(self):
        self.notes["/issues/8"] = [
            dict(id=11, body="Internal-only root cause", internal=True),
            dict(id=12, body="Resolved by changing the retry condition"),
        ]
        rows = await self.adapter.api.search("issues", "#8", ("issues", 1))
        self.assertEqual([row["kind"] for row in rows], ["issue", "note"])
        self.assertTrue(rows[-1]["url"].endswith("/issues/8#note_12"))
        source = self.adapter.source
        await asyncio.to_thread(
            source.sync, "refs/heads/main", self.config["gitlab_token"]
        )
        self.assertEqual(source.history("", "caller.py")[0]["label"], self.base[:10])
        self.assertIn("fallback", source.history(self.head)[0]["text"])
        self.assertTrue(
            source.lookup("", "caller.py", 2)[0]["text"].startswith("2: print")
        )

    async def test_missing_member_fails_closed(self):
        api = GitLab(self.adapter.session, self.origin, 8)
        self.assertFalse(
            await api.member(
                999, {"visibility": "public", "repository_access_level": "enabled"}
            )
        )
