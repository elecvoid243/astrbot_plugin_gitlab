"""Project-scoped GitLab REST operations."""

import re
from urllib.parse import quote, urlsplit

import aiohttp


class APIError(RuntimeError):
    def __init__(self, status):
        super().__init__(f"GitLab API returned HTTP {status}")
        self.status = status


class GitLab:
    def __init__(self, session, instance, project_id):
        url = urlsplit(instance)
        if (
            url.scheme not in ("https", "http")
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "Configure a GitLab HTTP(S) origin or installation subpath"
            )
        self.session = session
        self.instance = instance.rstrip("/")
        self.root = f"/projects/{int(project_id)}"
        self.project_id = int(project_id)
        self.web = ""

    async def request(self, method, path, *, params=None, body=None):
        async with self.session.request(
            method,
            self.instance + "/api/v4" + path,
            params=params,
            json=body,
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=25),
        ) as response:
            if not 200 <= response.status < 300:
                raise APIError(response.status)
            return await response.json()

    async def get(self, path="", **params):
        return await self.request("GET", self.root + path, params=params)

    async def member(self, actor, project):
        """Check effective direct or inherited membership, failing closed.

        Args:
            actor: GitLab user ID.
            project: Current project settings, including repository visibility.
        Returns:
            Whether this member may trigger source-backed replies.
        """
        try:
            member = await self.get(f"/members/all/{int(actor)}")
        except APIError as exc:
            if exc.status == 404:
                return False
            raise
        access = project.get("repository_access_level")
        if (
            access == "disabled"
            or member.get("state", "active") != "active"
            or member.get("membership_state", "active") != "active"
        ):
            return False
        level = member.get("access_level", 0)
        if level >= 20 or (
            level >= 10
            and project.get("visibility") == "public"
            and access == "enabled"
        ):
            return True
        if level == 15:
            version = await self.request("GET", "/version")
            number = re.match(r"(\d+)\.(\d+)", str(version.get("version", "")))
            if number and tuple(map(int, number.groups())) >= (18, 7):
                return True
        role = member.get("member_role") or {}
        return role.get("read_code") is True

    async def search(self, resource, term, exclude):
        """Search open and closed items without exposing confidential candidates.

        Args:
            resource: Issues or merge requests collection.
            term: A title, keyword, or #IID/!IID reference.
            exclude: The current (resource, IID).
        Returns:
            Linkable, project-local historical evidence.
        """
        prefix = "#" if resource == "issues" else "!"
        direct = re.fullmatch(re.escape(prefix) + r"[1-9]\d{0,9}", term)
        if direct:
            try:
                rows = [await self.get(f"/{resource}/{int(term[1:])}")]
            except APIError as exc:
                if exc.status == 404:
                    return []
                raise
        else:
            rows = await self.get(
                "/" + resource, search=term[:120], scope="all", state="all", per_page=12
            )
        result = []
        for row in rows:
            iid = row["iid"]
            if (
                (resource, iid) == exclude
                or row.get("confidential")
                or row.get("project_id") != self.project_id
            ):
                continue
            result.append(
                dict(
                    kind="issue" if resource == "issues" else "mr",
                    label=f"{prefix}{iid}",
                    url=f"{self.web}/-/{resource}/{iid}",
                    text=f"{row['title']} ({row['state']})\n{(row.get('description') or '')[:1800]}",
                )
            )
            if direct:
                notes = await self.get(
                    f"/{resource}/{iid}/notes",
                    per_page=20,
                    sort="desc",
                    order_by="created_at",
                )
                for note in notes:
                    body = note.get("body") or ""
                    if (
                        note.get("internal")
                        or note.get("confidential")
                        or note.get("system")
                        or "<!-- astrbot-gitlab:" in body
                    ):
                        continue
                    result.append(
                        dict(
                            kind="note",
                            label=f"{prefix}{iid} 评论 {note['id']}",
                            url=f"{self.web}/-/{resource}/{iid}#note_{int(note['id'])}",
                            text=body[:1500],
                        )
                    )
                    if len(result) >= 5:
                        break
        return result[:6]

    async def discussion(self, path, thread):
        if not thread:
            return ""
        item = await self.get(f"{path}/discussions/{quote(thread, safe='')}")
        return "\n".join(
            note.get("body", "")[:600]
            for note in item.get("notes", [])[-8:]
            if not (
                note.get("system") or note.get("internal") or note.get("confidential")
            )
        )

    async def marker_note(self, path, marker, bot_id):
        page = 1
        while True:
            notes = await self.get(f"{path}/notes", per_page=100, page=page)
            for note in notes:
                if note.get("author", {}).get("id") == bot_id and marker in note.get(
                    "body", ""
                ):
                    return note
            if len(notes) < 100:
                return None
            page += 1

    async def publish(self, path, thread, body, existing=None):
        if existing:
            return await self.request(
                "PUT",
                f"{self.root}{path}/notes/{int(existing['id'])}",
                body={"body": body},
            )
        endpoint = (
            f"{path}/discussions/{quote(thread, safe='')}/notes"
            if thread
            else f"{path}/notes"
        )
        return await self.request("POST", self.root + endpoint, body={"body": body})
