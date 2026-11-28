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
from unittest.mock import Mock, patch

import httpx
from aiohttp import web
from astrbot.core.agent.context.token_counter import EstimateTokenCounter
from astrbot.core.agent.message import Message
from astrbot.core.agent_team_execution import current_model_output_limit
from astrbot.core.provider.sources.anthropic_source import ProviderAnthropic
from astrbot.dashboard.api.platform import router
from astrbot.dashboard.services.platform_service import PlatformService
from fastapi import FastAPI

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
        if path.endswith("/discussions"):
            return web.json_response([dict(id="abc", notes=[dict(id=101)])])
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
        if "/notes" in path:
            scope = "/".join(path.split("/")[:3])
            notes = self.notes.setdefault(scope, [])
            if req.method == "GET":
                return web.json_response(notes)
            payload = await req.json()
            if req.method == "POST":
                note = dict(id=len(notes) + 1, author={"id": 77}, body=payload["body"])
                notes.append(note)
            elif req.method == "PUT":
                note = notes[int(path.split("/")[-1]) - 1]
                note["body"] = payload["body"]
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
                if isinstance(self.provider, ProviderAnthropic):
                    self.assertEqual(
                        kwargs.get("llm_params"), {"max_tokens": output_limit}
                    )
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
        response = await self.client.post(
            "/api/v1/webhooks/platforms/" + self.config["webhook_uuid"],
            json=payload,
            headers={"X-Gitlab-Token": secret},
        )
        return response.json(), response.status_code

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
        await self.deliver(event)
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
        await self.deliver(event)
        await self.wait_done()
        self.assertEqual(len(self.notes["/merge_requests/2"]), 1)
        self.assertIn(self.head, self.notes["/merge_requests/2"][0]["body"])
        self.assertIn(("PUT", "/projects/8/merge_requests/2/notes/1"), self.calls)

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
        self.provider = Mock(
            spec=ProviderAnthropic,
            provider_config={
                "max_context_tokens": 5000,
                "custom_extra_body": {"max_output_tokens": 1800},
            },
        )
        event = self.event(
            "merge_request", action="open", iid=2, last_commit={"id": self.head}
        )

        await self.deliver(event)
        await self.wait_done()

        self.assertEqual(len(self.model_requests), 1)
        self.assertEqual(self.model_requests[0]["llm_params"], {"max_tokens": 1800})
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
        self.assertEqual(
            registered.config_metadata["gitlab_provider"]["_special"], "select_provider"
        )
        self.assertTrue(registered.default_config_tmpl["gitlab_auto_review"])
        event = self.event(
            "note", noteable_type="Issue", action="create", id=101, note="/help"
        )
        event["issue"] = {"iid": 1}
        self.assertEqual((await self.deliver(event, secret=""))[1], 403)
        self.assertEqual(await self.deliver(event), ({"accepted": True}, 200))
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
