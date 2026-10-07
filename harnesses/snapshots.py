"""Snapshot materialisation and digests for harness experience state.

A snapshot is a directory tree of a harness's durable experience. Two rules make
the digest meaningful for the frozen comparison:

- the digest covers sorted relative paths and file bytes, so two exports of the
  same read-only experience are byte-identical regardless of write order;
- exports are filtered by an explicit include/exclude list, so raw conversations,
  session databases and answer caches never enter the digest or a later run.

`DEFAULT_EXCLUDED_NAMES` and `DEFAULT_EXCLUDED_SUFFIXES` are the shared exclusion
baseline; a harness worker narrows the include set further.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

__all__ = [
    "DEFAULT_EXCLUDED_NAMES", "DEFAULT_EXCLUDED_SUFFIXES", "collect_files",
    "directory_digest", "materialize_files",
]

# Raw conversations, transcripts, session stores and answer caches.
DEFAULT_EXCLUDED_NAMES = frozenset({
    "sessions", "transcripts", "credentials", "logs", "cache", "caches",
    "answers", "answer_cache", "history",
})
DEFAULT_EXCLUDED_SUFFIXES = (".sqlite", ".sqlite3", ".db", ".db-wal", ".db-shm",
                             ".log", ".wal", ".jsonl")


def collect_files(
    root: Path,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    excluded_names: Iterable[str] = DEFAULT_EXCLUDED_NAMES,
    excluded_suffixes: Sequence[str] = DEFAULT_EXCLUDED_SUFFIXES,
) -> dict[str, dict]:
    """Return `{relative path: {"encoding", "content"}}` for a directory tree.

    `include` limits the walk to the listed relative subpaths when it is not
    empty. Exclusion wins over inclusion: a path is skipped when any of its parts
    is an excluded directory name or when it ends with an excluded suffix.
    """
    root = Path(root)
    if not root.is_dir():
        return {}
    names = set(excluded_names)
    files: dict[str, dict] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        parts = relative.split("/")
        if any(part in names for part in parts[:-1]):
            continue
        if path.suffix in excluded_suffixes:
            continue
        if include and not any(
            relative == item or relative.startswith(item.rstrip("/") + "/")
            for item in include
        ):
            continue
        raw = path.read_bytes()
        try:
            files[relative] = {"encoding": "utf-8", "content": raw.decode("utf-8")}
        except UnicodeDecodeError:
            files[relative] = {
                "encoding": "base64",
                "content": base64.b64encode(raw).decode("ascii"),
            }
    return files


def materialize_files(destination: Path, files: Mapping[str, Mapping[str, str]]) -> list[str]:
    """Write a collected file map under `destination`; return written paths."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for relative, entry in sorted(files.items()):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry.get("encoding") == "base64":
            target.write_bytes(base64.b64decode(entry["content"]))
        else:
            target.write_text(entry["content"], encoding="utf-8")
        written.append(relative)
    return written


def directory_digest(root: Path) -> str:
    """Digest of a materialised tree: sorted paths, contents and modes."""
    root = Path(root)
    digest = hashlib.sha256()
    if not root.is_dir():
        return "sha256:" + digest.hexdigest()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            digest.update(f"link {relative} {path.readlink()}\n".encode())
        elif path.is_dir():
            digest.update(f"dir {relative}\n".encode())
        else:
            digest.update(f"file {relative} {path.stat().st_mode & 0o777}\n".encode())
            digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()
