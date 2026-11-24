"""Exercise the actual AstrBot adapter against Git and a local GitLab API double."""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from aiohttp import web

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
        self.change_during_review = False
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
        self.config = dict(
            DEFAULTS,
            id="gitlab-demo",
            gitlab_url=self.origin,
            gitlab_project_id=8,
            gitlab_token="test-api",
            gitlab_webhook_secret="test-hook",
            gitlab_provider="test-model",
        )
        ctx = SimpleNamespace(llm_generate=self.generate)
        GitLabPlugin(ctx)
        self.adapter = GitLabPlatform(self.config, {}, asyncio.Queue())
        self.runner = asyncio.create_task(self.adapter.run())
        self.addAsyncCleanup(self.adapter.terminate)
        await self.wait(lambda: self.adapter.bot)

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
            issue = next(k for k, r in index.items() if r["label"] == "#8")
            self.assertFalse(any(r["label"] == "#9" for r in index.values()))
            result = dict(
                findings=[dict(text="检查调用方是否对空值调用 upper", refs=src)],
                relations=[
                    dict(
                        issue=issue,
                        status="duplicate"
                        if prompt["task"]["command"] in ("diagnose", "review")
                        else "related",
                        reason="同一超时条件",
                    )
                ],
            )
            if self.change_during_review:
                self.head = "d" * 40
        return SimpleNamespace(completion_text=json.dumps(result))

    async def deliver(self, payload, secret="test-hook"):
        async def get_json(**kwargs):
            return payload

        return await self.adapter.webhook_callback(
            SimpleNamespace(headers={"X-Gitlab-Token": secret}, get_json=get_json)
        )

    def event(self, kind, **attrs):
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
        import httpx
        from fastapi import FastAPI
        from astrbot.core.platform.register import platform_registry
        from astrbot.dashboard.api.platform import router
        from astrbot.dashboard.services.platform_service import PlatformService

        registered = next(
            platform for platform in platform_registry if platform.name == "gitlab"
        )
        self.assertEqual(
            registered.config_metadata["gitlab_provider"]["_special"], "select_provider"
        )
        self.assertTrue(registered.default_config_tmpl["gitlab_auto_review"])
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        core = SimpleNamespace(
            platform_manager=SimpleNamespace(platform_insts=[self.adapter])
        )
        app.state.services = SimpleNamespace(platforms=PlatformService(core))
        event = self.event(
            "note", noteable_type="Issue", action="create", id=101, note="/help"
        )
        event["issue"] = {"iid": 1}
        path = "/api/v1/webhooks/platforms/" + self.config["webhook_uuid"]
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            self.assertEqual((await client.post(path, json=event)).status_code, 403)
            res = await client.post(
                path, json=event, headers={"X-Gitlab-Token": "test-hook"}
            )
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
