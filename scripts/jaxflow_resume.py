#!/usr/bin/env python3
"""Resume checkpoint helpers for jaxflow managed builder retry (MOA-473 P1)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from pathlib import Path

import jaxflow_settings as jset
from jax_init import Refusal, _is_secret_path
from jaxflow_hook import redact

FILE_READ_CAP = 16 << 20
SCAN_CAP = 64 << 20
CHECKPOINT_CAP = 128 << 10
CHECKPOINT_VERSION = 1
_RENAME_CODES = "RC"
_UNMERGED = {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}
_CHECKPOINT_KEYS = (
    "version", "run_id", "root_build_run_id", "repo", "worktree", "branch", "base",
    "outcome", "plan_revision", "work_state",
)
_STATE_KEYS = ("head", "plan_sha256", "fingerprint")
_RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_OUTCOMES = frozenset({"success", "failure", "blocked"})


def require_same_work_state(expected, actual):
    if expected != actual:
        raise Refusal("resume-state-changed")


def worktree_claim(control_repo, worktree):
    control = Path(control_repo)
    key = hashlib.sha256(str(Path(worktree).resolve()).encode()).hexdigest()
    lock_dir = control / ".local" / "runs" / "locks"
    return jset.configuration_claim(lock_dir / key, create=True)


def write_checkpoint(path, checkpoint):
    path = Path(path)
    payload = json.dumps(checkpoint, separators=(",", ":")).encode("utf-8")
    parent_fd = jset._open_directory_nofollow(path.parent, create=False)
    tmp_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    fd = None
    try:
        fd = os.open(
            tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600,
            dir_fd=parent_fd,
        )
        os.fchmod(fd, 0o600)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("checkpoint write made no progress")
            remaining = remaining[written:]
        os.fsync(fd)
        os.close(fd)
        fd = None
        jset._require_same_directory(parent_fd, path.parent)
        try:
            os.link(
                tmp_name, path.name,
                src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise Refusal("resume-ineligible") from exc
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp_name, dir_fd=parent_fd)
        except OSError:
            pass
        os.close(parent_fd)


def read_checkpoint(path):
    path = Path(path)
    parent_fd = jset._open_directory_nofollow(path.parent, create=False)
    try:
        try:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        except OSError as exc:
            raise Refusal("resume-ineligible") from exc
    finally:
        os.close(parent_fd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o600:
            raise Refusal("resume-ineligible")
        raw = _read_capped(fd, CHECKPOINT_CAP)
    finally:
        os.close(fd)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Refusal("resume-ineligible") from exc
    try:
        data = json.loads(
            text,
            object_pairs_hook=jset._reject_duplicates,
            parse_constant=jset._reject_constant,
        )
    except Refusal:
        raise Refusal("resume-ineligible")
    except (json.JSONDecodeError, ValueError, TypeError, RecursionError) as exc:
        raise Refusal("resume-ineligible") from exc
    return _require_checkpoint(data)


def _hex(value, pattern):
    if type(value) is not str or not pattern.fullmatch(value):
        raise Refusal("resume-ineligible")


def _require_checkpoint(data):
    if type(data) is not dict or set(data) != set(_CHECKPOINT_KEYS):
        raise Refusal("resume-ineligible")
    if type(data["version"]) is not int or data["version"] != CHECKPOINT_VERSION:
        raise Refusal("resume-ineligible")
    for key in ("repo", "worktree", "branch"):
        if type(data[key]) is not str:
            raise Refusal("resume-ineligible")
    _hex(data["run_id"], _RUN_ID_RE)
    _hex(data["root_build_run_id"], _RUN_ID_RE)
    _hex(data["base"], _SHA1_RE)
    if type(data["outcome"]) is not str or data["outcome"] not in _OUTCOMES:
        raise Refusal("resume-ineligible")
    _hex(data["plan_revision"], jset.SHA256_RE)
    state = data["work_state"]
    if type(state) is not dict or set(state) != set(_STATE_KEYS):
        raise Refusal("resume-ineligible")
    _hex(state["head"], _SHA1_RE)
    _hex(state["plan_sha256"], jset.SHA256_RE)
    _hex(state["fingerprint"], jset.SHA256_RE)
    return data


def checkpoint_status(path):
    path = Path(path)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "unreadable"
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        return "unreadable"
    try:
        read_checkpoint(path)
    except Exception:
        return "unreadable"
    return "present"


def capture_work_state(worktree, plan_path, *, run):
    worktree = Path(worktree)
    head = _head_sha(worktree, run)
    plan_sha = _hash_file(Path(plan_path))
    fingerprint = _worktree_fingerprint(worktree, run)
    return {"head": head, "plan_sha256": plan_sha, "fingerprint": fingerprint}


def _head_sha(worktree, run):
    probe = run(["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=worktree)
    sha = (probe.stdout or "").strip()
    if probe.returncode != 0 or len(sha) != 40:
        raise Refusal("resume-ineligible")
    return sha


def _hash_file(path):
    fd = _open_nofollow(path)
    try:
        data = _read_capped(fd, FILE_READ_CAP)
    finally:
        os.close(fd)
    return hashlib.sha256(data).hexdigest()


def _open_nofollow(path):
    try:
        return os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise Refusal("resume-ineligible") from exc


def _read_capped(fd, cap):
    chunks = []
    total = 0
    while True:
        data = os.read(fd, min(65536, cap + 1 - total))
        if not data:
            break
        total += len(data)
        if total > cap:
            raise Refusal("resume-ineligible")
        chunks.append(data)
    return b"".join(chunks)


def _runtime_artifact(rel):
    rel = rel.replace("\\", "/")
    return (
        rel == ".local" or rel.startswith(".local/")
        or rel == ".jax-os" or rel.startswith(".jax-os/")
    )


def _parse_porcelain_z(blob):
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8", "surrogateescape")
    entries = []
    i = 0
    n = len(blob)
    while i < n:
        if i + 3 > n or blob[i + 2] != " ":
            raise Refusal("resume-ineligible")
        xy = blob[i:i + 2]
        i += 3
        end = blob.find("\0", i)
        if end < 0:
            raise Refusal("resume-ineligible")
        path = blob[i:end]
        i = end + 1
        orig = ""
        if xy[0] in _RENAME_CODES or xy[1] in _RENAME_CODES:
            end = blob.find("\0", i)
            if end < 0:
                raise Refusal("resume-ineligible")
            orig = blob[i:end]
            i = end + 1
        entries.append((xy, path, orig))
    return entries


def _index_map(worktree, run):
    probe = run(["git", "ls-files", "-s", "-z"], cwd=worktree)
    if probe.returncode != 0:
        raise Refusal("resume-ineligible")
    blob = probe.stdout or ""
    mapping = {}
    i = 0
    n = len(blob)
    while i < n:
        end = blob.find("\0", i)
        if end < 0:
            raise Refusal("resume-ineligible")
        rec = blob[i:end]
        i = end + 1
        try:
            meta, path = rec.split("\t", 1)
            mode, obj, stage = meta.split(" ")
        except ValueError as exc:
            raise Refusal("resume-ineligible") from exc
        mapping[path] = (mode, obj, stage)
    return mapping


def _worktree_fingerprint(worktree, run):
    status = run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=worktree,
    )
    if status.returncode != 0:
        raise Refusal("resume-ineligible")
    index = _index_map(worktree, run)
    records = []
    scanned = 0
    for xy, path, orig in _parse_porcelain_z(status.stdout or ""):
        if xy == "!!" or _runtime_artifact(path) or (orig and _runtime_artifact(orig)):
            continue
        if xy in _UNMERGED or "U" in xy:
            raise Refusal("resume-ineligible")
        for rel in (path, orig):
            if rel and _is_secret_path(rel):
                raise Refusal(f"secret-detected: {rel}")
        idx = index.get(path)
        if idx and (idx[0] == "160000" or idx[2] != "0"):
            raise Refusal("resume-ineligible")
        wtype, wmode, digest, scanned = _worktree_identity(worktree / path, scanned)
        idx_mode, idx_blob, idx_stage = idx if idx else ("", "", "")
        records.append("\0".join((
            xy, path, orig, idx_mode, idx_blob, idx_stage, wtype, wmode, digest,
        )).encode("utf-8", "surrogateescape"))
    records.sort()
    buf = bytearray()
    for rec in records:
        buf.extend(f"{len(rec)}\0".encode("ascii"))
        buf.extend(rec)
    return hashlib.sha256(buf).hexdigest()


def _worktree_identity(full, scanned):
    try:
        info = os.lstat(full)
    except FileNotFoundError:
        return "missing", "", "", scanned
    mode = format(info.st_mode & 0o777, "o")
    if stat.S_ISREG(info.st_mode):
        if info.st_size > FILE_READ_CAP or scanned + info.st_size > SCAN_CAP:
            raise Refusal("resume-ineligible")
        fd = _open_nofollow(full)
        try:
            data = _read_capped(fd, FILE_READ_CAP)
        finally:
            os.close(fd)
        scanned += len(data)
        if scanned > SCAN_CAP:
            raise Refusal("resume-ineligible")
        text = data.decode("utf-8", "surrogateescape")
        if redact(text) != text:
            raise Refusal("secret-detected")
        return "file", mode, hashlib.sha256(data).hexdigest(), scanned
    if stat.S_ISLNK(info.st_mode):
        target = os.readlink(full)
        raw = target.encode("utf-8", "surrogateescape")
        scanned += len(raw)
        if scanned > SCAN_CAP:
            raise Refusal("resume-ineligible")
        return "symlink", mode, hashlib.sha256(raw).hexdigest(), scanned
    raise Refusal("resume-ineligible")
