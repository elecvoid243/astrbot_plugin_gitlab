"""Index regular text blobs directly from a bare Git repository."""

import base64
import os
import re
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from urllib.parse import quote


class SourceIndex:
    def __init__(self, directory: Path, web_url: str):
        self.repo = directory / "source.git"
        self.database = directory / "source.sqlite"
        self.web_url = web_url
        self.sha = ""

    def git(self, *args, data=None, env=None):
        proc = subprocess.run(
            ["git", "-C", str(self.repo), *args],
            input=data,
            capture_output=True,
            timeout=180,
            env=env,
            check=False,
        )
        if proc.returncode:
            # Git diagnostics can contain credentials or remote-controlled text.
            raise RuntimeError(f"Git {args[0]} failed with status {proc.returncode}")
        return proc.stdout

    def sync(self, ref: str, token: str):
        """Fetch the requested ref and update the index atomically.

        Args:
            ref: Fully qualified default branch or GitLab merge request head.
            token: Credential scoped to the configured repository.
        Returns:
            The fetched commit ID.
        """
        if not self.repo.exists():
            self.repo.mkdir(parents=True)
        self.git("init", "--bare", "--quiet")
        credential = base64.b64encode(f"oauth2:{token}".encode()).decode()
        env = dict(
            os.environ,
            GIT_TERMINAL_PROMPT="0",
            ASTRBOT_GIT_AUTH=f"Authorization: Basic {credential}",
        )
        self.git(
            "--config-env=http.extraHeader=ASTRBOT_GIT_AUTH",
            "-c",
            "http.followRedirects=false",
            "-c",
            "credential.helper=",
            "fetch",
            "--quiet",
            "--no-tags",
            "--no-recurse-submodules",
            "--",
            self.web_url + ".git",
            "+" + ref + ":refs/astrbot/current",
            env=env,
        )
        sha = self.git("rev-parse", "refs/astrbot/current").decode().strip()
        self.build(sha)
        return sha

    def build(self, sha):
        """Index changed blobs while retaining unchanged file chunks.

        Args:
            sha: Commit in the local bare repository.
        """
        if not re.fullmatch("[a-f0-9]{40,64}", sha):
            raise ValueError("Invalid Git object ID")
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS version (sha TEXT)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, blob TEXT)"
            )
            db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(path, body, line UNINDEXED)"
            )
            if db.execute("SELECT sha FROM version").fetchone() != (sha,):
                old = dict(db.execute("SELECT path, blob FROM files"))
                pending = []
                listing = self.git("ls-tree", "-rlz", sha)
                for record in listing.split(b"\0"):
                    if not record:
                        continue
                    meta, raw_path = record.split(b"\t", 1)
                    mode, kind, blob, size = meta.split()
                    if (
                        kind != b"blob"
                        or mode not in (b"100644", b"100755")
                        or int(size) > 256 * 1024
                    ):
                        continue
                    path = raw_path.decode("utf-8", "replace")
                    if old.get(path) == blob.decode():
                        old.pop(path)
                        continue
                    pending.append((path, blob))
                for path in old:
                    db.execute("DELETE FROM passages WHERE path=?", (path,))
                    db.execute("DELETE FROM files WHERE path=?", (path,))
                for offset in range(0, len(pending), 32):
                    batch = pending[offset : offset + 32]
                    raw = self.git(
                        "cat-file",
                        "--batch",
                        data=b"".join(blob + b"\n" for _, blob in batch),
                    )
                    cursor = 0
                    for path, blob in batch:
                        end = raw.index(b"\n", cursor)
                        size = int(raw[cursor:end].split()[-1])
                        content = raw[end + 1 : end + 1 + size]
                        cursor = end + size + 2
                        db.execute(
                            "INSERT INTO files VALUES (?,?)", (path, blob.decode())
                        )
                        if b"\0" in content:
                            continue
                        lines = content.decode("utf-8", "replace").splitlines()
                        db.executemany(
                            "INSERT INTO passages(path,body,line) VALUES (?,?,?)",
                            (
                                (
                                    path,
                                    "\n".join(
                                        line[:1500] for line in lines[i : i + 25]
                                    ),
                                    i + 1,
                                )
                                for i in range(0, len(lines), 25)
                            ),
                        )
                db.execute("DELETE FROM version")
                db.execute("INSERT INTO version VALUES (?)", (sha,))
        self.sha = sha

    def lookup(self, query, path="", line=1):
        """Return source passages with immutable links to the observed lines.

        Args:
            query: Literal tokens to search for when no path is supplied.
            path: Exact repository-relative file to read.
            line: First requested line.
        Returns:
            Evidence records containing excerpts and GitLab blob links.
        """
        with closing(sqlite3.connect(self.database)) as db:
            if path:
                rows = db.execute(
                    "SELECT path,line,body FROM passages WHERE path=? AND line<=? AND line+24>=? ORDER BY line LIMIT 2",
                    (path, line + 24, line),
                ).fetchall()
            else:
                terms = re.findall(r"\w+", query, re.UNICODE)[:6]
                if not terms:
                    return []
                expr = " OR ".join('"' + term + '"' for term in terms)
                rows = db.execute(
                    "SELECT path,line,body FROM passages WHERE passages MATCH ? ORDER BY bm25(passages, 0.3, 1) LIMIT 6",
                    (expr,),
                ).fetchall()
                if not rows:
                    term = max(terms, key=len)
                    rows = db.execute(
                        "SELECT path,line,body FROM passages WHERE instr(lower(body),lower(?)) OR instr(lower(path),lower(?)) LIMIT 6",
                        (term, term),
                    ).fetchall()
        result = []
        for file, start, body in rows:
            lines = body.splitlines() or [""]
            if path:
                first = max(line, start)
                lines = lines[first - start : line + 25 - start]
                start = first
                if not lines:
                    continue
            end = start + len(lines) - 1
            result.append(
                dict(
                    kind="code",
                    label=f"{file} L{start}–{end}",
                    url=f"{self.web_url}/-/blob/{self.sha}/{quote(file)}#L{start}-{end}",
                    text="\n".join(f"{n}: {s}" for n, s in enumerate(lines, start))[
                        :3200
                    ],
                )
            )
        return result

    def history(self, query, path=""):
        """Find reachable commits by literal subject, file history or commit ID.

        Args:
            query: Search text or hexadecimal commit ID.
            path: Optional repository-relative file path.
        Returns:
            Commit evidence with stable links.
        """
        if re.fullmatch("[0-9a-f]{7,64}", query):
            try:
                sha = (
                    self.git("rev-parse", "--verify", query + "^{commit}")
                    .decode()
                    .strip()
                )
                self.git("merge-base", "--is-ancestor", sha, self.sha)
            except RuntimeError:
                return []
            details = self.git("show", "--no-patch", "--format=%B", sha).decode(
                "utf-8", "replace"
            )
            return [
                dict(
                    kind="commit",
                    label=sha[:10],
                    url=f"{self.web_url}/-/commit/{sha}",
                    text=details[:2000],
                )
            ]
        records = self.git(
            "--literal-pathspecs",
            "log",
            "-8",
            "--fixed-strings",
            "-i",
            *([f"--grep={query[:150]}"] if query else []),
            "--format=%H%x09%s",
            self.sha,
            "--",
            *([path] if path else []),
        )
        result = []
        for line in records.decode("utf-8", "replace").splitlines():
            sha, _, title = line.partition("\t")
            result.append(
                dict(
                    kind="commit",
                    label=sha[:10],
                    url=f"{self.web_url}/-/commit/{sha}",
                    text=title[:500],
                )
            )
        return result

    def diff(self, base):
        """Read the beginning of the patch without buffering the entire MR.

        Args:
            base: Common ancestor returned by the GitLab diff version.
        Returns:
            A bounded patch annotated when truncated.
        """
        args = [
            "git",
            "-C",
            str(self.repo),
            "-c",
            "core.quotePath=false",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            base,
            self.sha,
            "--",
        ]
        with subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        ) as proc:
            raw = proc.stdout.read(9001)
            if len(raw) > 9000:
                proc.kill()
                return raw[:8910].decode("utf-8", "replace") + "\n[diff truncated]"
            if proc.wait(timeout=30):
                raise RuntimeError("Git diff failed")
            return raw.decode("utf-8", "replace")
