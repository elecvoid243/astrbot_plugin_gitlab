"""Translate GitLab webhooks into project-specific jobs."""

import re


def project_job(payload, project, bot, config):
    """Select an actionable event and identify its destination.

    Args:
        payload: Authenticated GitLab webhook body.
        project: Bound numeric project ID.
        bot: Current GitLab user returned by the API.
        config: Platform instance settings.

    Returns:
        A serializable job, or None for an irrelevant event.
    """
    attrs = payload.get("object_attributes") or {}
    actor = (payload.get("user") or {}).get("id")
    if (
        (payload.get("project") or {}).get("id") != project
        or not isinstance(actor, int)
        or actor == bot["id"]
    ):
        return None
    kind = payload.get("object_kind")
    action = attrs.get("action")
    thread = ""
    sha = ""
    question = ""
    if kind == "issue" and action == "open" and config["gitlab_auto_issue"]:
        resource, iid, command = "issues", attrs.get("iid"), "diagnose"
        key = f"issue:{iid}:open"
    elif kind == "merge_request" and config["gitlab_auto_review"]:
        changes = payload.get("changes") or {}
        title = str(attrs.get("title", ""))
        draft = attrs.get("draft", attrs.get("work_in_progress", False)) or re.match(
            r"^(draft:|wip:|\[draft\])", title, re.I
        )
        ready = any(
            (changes.get(field) or {}).get("previous") is True
            and (changes.get(field) or {}).get("current") is False
            for field in ("draft", "work_in_progress")
        )
        previous_title = (changes.get("title") or {}).get("previous") or ""
        ready = ready or bool(
            re.match(r"^(draft:|wip:|\[draft\])", previous_title, re.I)
        )
        if draft or not (
            action in ("open", "reopen")
            or action == "update"
            and (attrs.get("oldrev") or ready)
        ):
            return None
        sha = (attrs.get("last_commit") or {}).get("id", "")
        if not re.fullmatch(r"[0-9a-f]{40,64}", sha):
            return None
        resource, iid, command = "merge_requests", attrs.get("iid"), "review"
        key = f"review:{iid}:{sha}"
    elif kind == "note" and action in (None, "create"):
        if attrs.get("system") or attrs.get("internal") or attrs.get("confidential"):
            return None
        resource = {"Issue": "issues", "MergeRequest": "merge_requests"}.get(
            attrs.get("noteable_type")
        )
        if not resource:
            return None
        item = payload.get("issue" if resource == "issues" else "merge_request") or {}
        iid = item.get("iid")
        note = str(attrs.get("note", "")).strip()
        mention = re.compile(
            r"(?<![\w@.-])@" + re.escape(bot["username"]) + r"(?![\w.-])", re.I
        )
        question = mention.sub("", note).strip()
        parts = question.split(maxsplit=1)
        first, tail = (parts[0] if parts else ""), parts[1:]
        if first in ("/review", "/related", "/ask", "/help"):
            command = first[1:]
            question = tail[0] if tail else ""
        elif mention.search(note) and not question.startswith("/"):
            command = "ask"
        else:
            return None
        if command == "review" and resource != "merge_requests":
            command = "help"
        if not isinstance(attrs.get("id"), int):
            return None
        key = f"note:{attrs['id']}"
        thread = attrs.get("discussion_id") or ""
        if thread and not re.fullmatch(r"[a-zA-Z0-9_-]+", thread):
            return None
    else:
        return None
    if not isinstance(iid, int) or iid <= 0:
        return None
    return dict(
        key=key,
        resource=resource,
        iid=iid,
        actor=actor,
        command=command,
        question=question[:4000],
        thread=thread,
        sha=sha,
    )
