#!/usr/bin/env python3
"""Tests for jaxflow resume checkpoints (MOA-473 Part 1 Task 4)."""
import fcntl
import hashlib
import json
import multiprocessing
import os
import signal
import stat
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import jaxflow_resume as jresume
from jax_init import Refusal
from jaxflow_resume import (
    capture_work_state,
    checkpoint_status,
    read_checkpoint,
    require_same_work_state,
    worktree_claim,
    write_checkpoint,
)


def _payload(**overrides):
    data = {
        "version": 1,
        "run_id": "aaaabbbbcccc",
        "root_build_run_id": "aaaabbbbcccc",
        "repo": "/tmp/repo",
        "worktree": "/tmp/wt",
        "branch": "feat/x",
        "base": "a" * 40,
        "outcome": "failure",
        "plan_revision": "b" * 64,
        "work_state": {"head": "c" * 40, "plan_sha256": "d" * 64, "fingerprint": "e" * 64},
    }
    data.update(overrides)
    return data


def _tmps(folder):
    return [p for p in folder.iterdir() if p.name.endswith(".tmp")]


def _refuse_read(path, code="resume-ineligible"):
    with pytest.raises(Refusal) as exc:
        read_checkpoint(path)
    assert exc.value.code == code
    return exc


def _git(cwd, *args):
    import subprocess
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _run(argv, cwd=None):
    import subprocess
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)


def _init_repo(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "t@t.test")
    _git(root, "config", "user.name", "t")
    (root / "committed.py").write_text("orig\n", encoding="utf-8")
    (root / ".gitignore").write_text(".local/\n.jax-os/\n*.tmp\n", encoding="utf-8")
    _git(root, "add", "committed.py", ".gitignore")
    _git(root, "commit", "-m", "init")
    return root / "plan.md"


def _capture(root, plan):
    return capture_work_state(root, plan, run=_run)


def test_capture_matches_until_one_byte_or_staging_changes():
    with TemporaryDirectory() as raw:
        root = Path(raw) / "repo"
        plan = _init_repo(root)
        (root / "committed.py").write_text("edited\n", encoding="utf-8")
        (root / "staged.py").write_text("staged\n", encoding="utf-8")
        _git(root, "add", "staged.py")
        untracked = root / "untracked.py"
        untracked.write_text("new\n", encoding="utf-8")
        plan.write_text("# plan\n", encoding="utf-8")

        state = _capture(root, plan)
        untracked.write_text("nex\n", encoding="utf-8")
        with pytest.raises(Refusal) as exc:
            require_same_work_state(state, _capture(root, plan))
        assert exc.value.code == "resume-state-changed"
        untracked.write_text("new\n", encoding="utf-8")
        require_same_work_state(state, _capture(root, plan))

        _git(root, "add", "committed.py")
        with pytest.raises(Refusal) as exc:
            require_same_work_state(state, _capture(root, plan))
        assert exc.value.code == "resume-state-changed"


def test_capture_clean_deletion_rename_and_ignored_runtime_paths():
    with TemporaryDirectory() as raw:
        root = Path(raw) / "repo"
        plan = _init_repo(root)
        plan.write_text("# plan\n", encoding="utf-8")
        clean = _capture(root, plan)

        (root / "noise.tmp").write_text("ignored\n", encoding="utf-8")
        (root / ".local").mkdir()
        (root / ".local" / "report.md").write_text("private\n", encoding="utf-8")
        (root / ".jax-os").mkdir()
        (root / ".jax-os" / "status.md").write_text("now\n", encoding="utf-8")
        require_same_work_state(clean, _capture(root, plan))

        (root / "committed.py").unlink()
        deleted = _capture(root, plan)
        with pytest.raises(Refusal) as exc:
            require_same_work_state(clean, deleted)
        assert exc.value.code == "resume-state-changed"
        (root / "committed.py").write_text("orig\n", encoding="utf-8")
        require_same_work_state(clean, _capture(root, plan))

        _git(root, "mv", "committed.py", "renamed.py")
        renamed = _capture(root, plan)
        with pytest.raises(Refusal) as exc:
            require_same_work_state(clean, renamed)
        assert exc.value.code == "resume-state-changed"


def test_capture_refuses_secrets_without_hashing_and_skips_symlink_targets(monkeypatch):
    with TemporaryDirectory() as raw:
        root = Path(raw) / "repo"
        plan = _init_repo(root)
        plan.write_text("# plan\n", encoding="utf-8")
        (root / ".env").write_text("not-hashed\n", encoding="utf-8")
        with pytest.raises(Refusal) as exc:
            _capture(root, plan)
        assert exc.value.code.startswith("secret-detected")

        (root / ".env").unlink()
        (root / "open.py").write_text("api_key=supersecretvalue\n", encoding="utf-8")
        with pytest.raises(Refusal) as exc:
            _capture(root, plan)
        assert exc.value.code.startswith("secret-detected")

        (root / "open.py").unlink()
        hidden = root / "hidden.tmp"
        hidden.write_text("api_key=supersecretvalue\n", encoding="utf-8")
        (root / "alias.py").symlink_to(hidden)
        state = _capture(root, plan)
        assert "fingerprint" in state
        hidden.write_text("api_key=changedsecret\n", encoding="utf-8")
        require_same_work_state(state, _capture(root, plan))

        monkeypatch.setattr(jresume, "FILE_READ_CAP", 4)
        (root / "alias.py").unlink()
        (root / "big.py").write_text("12345", encoding="utf-8")
        with pytest.raises(Refusal) as exc:
            _capture(root, plan)
        assert exc.value.code == "resume-ineligible"


def test_write_checkpoint_is_exclusive_owner_only_and_roundtrips():
    with TemporaryDirectory() as raw:
        dest = Path(raw) / "resume-checkpoint.json"
        payload = {
            "version": 1,
            "run_id": "aaaabbbbcccc",
            "root_build_run_id": "aaaabbbbcccc",
            "repo": "/tmp/repo",
            "worktree": "/tmp/wt",
            "branch": "feat/x",
            "base": "a" * 40,
            "outcome": "failure",
            "plan_revision": "b" * 64,
            "work_state": {"head": "c" * 40, "plan_sha256": "d" * 64, "fingerprint": "e" * 64},
        }
        write_checkpoint(dest, payload)
        assert stat.S_IMODE(dest.stat().st_mode) == 0o600
        assert read_checkpoint(dest) == payload
        assert checkpoint_status(dest) == "present"
        with pytest.raises(Refusal):
            write_checkpoint(dest, payload)
        dest.write_text("{", encoding="utf-8")
        assert checkpoint_status(dest) == "unreadable"
        dest.unlink()
        assert checkpoint_status(dest) == "absent"


def test_write_checkpoint_completes_short_writes_and_cleans_failures(tmp_path, monkeypatch):
    dest = tmp_path / "resume-checkpoint.json"
    payload = _payload()
    real_write = os.write
    real_fsync = os.fsync

    def short_write(fd, data):
        view = memoryview(data)
        return real_write(fd, view[:7])

    monkeypatch.setattr(os, "write", short_write)
    write_checkpoint(dest, payload)
    assert read_checkpoint(dest) == payload
    dest.unlink()

    monkeypatch.setattr(os, "write", lambda fd, data: 0)
    with pytest.raises(OSError):
        write_checkpoint(dest, payload)
    assert not dest.exists()
    assert _tmps(tmp_path) == []

    calls = {"n": 0}

    def second_raises(fd, data):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_write(fd, memoryview(data)[:7])
        raise OSError("boom")

    monkeypatch.setattr(os, "write", second_raises)
    with pytest.raises(OSError):
        write_checkpoint(dest, payload)
    assert not dest.exists()
    assert _tmps(tmp_path) == []

    monkeypatch.setattr(os, "write", real_write)

    def boom_fsync(fd):
        raise OSError("fsync")

    monkeypatch.setattr(os, "fsync", boom_fsync)
    with pytest.raises(OSError):
        write_checkpoint(dest, payload)
    assert not dest.exists()
    assert _tmps(tmp_path) == []
    monkeypatch.setattr(os, "fsync", real_fsync)

    write_checkpoint(dest, payload)
    original = dest.read_bytes()
    monkeypatch.setattr(os, "write", lambda fd, data: 0)
    with pytest.raises(OSError):
        write_checkpoint(dest, payload)
    assert dest.read_bytes() == original
    assert _tmps(tmp_path) == []
    monkeypatch.setattr(os, "write", real_write)
    with pytest.raises(Refusal) as exc:
        write_checkpoint(dest, payload)
    assert exc.value.code == "resume-ineligible"
    assert dest.read_bytes() == original
    assert _tmps(tmp_path) == []


def test_read_checkpoint_refuses_malformed_and_special_files(tmp_path, monkeypatch):
    dest = tmp_path / "resume-checkpoint.json"

    def plant(raw, mode=0o600):
        dest.write_bytes(raw)
        dest.chmod(mode)

    for outcome in ("success", "failure", "blocked"):
        other = tmp_path / f"{outcome}.json"
        write_checkpoint(other, _payload(outcome=outcome))
        assert read_checkpoint(other)["outcome"] == outcome

    plant(json.dumps(_payload()).encode("utf-8"))
    assert read_checkpoint(dest)["run_id"] == "aaaabbbbcccc"

    plant(b"null")
    _refuse_read(dest)
    plant(json.dumps(_payload(version=True)).encode("utf-8"))
    _refuse_read(dest)
    plant(json.dumps(_payload(run_id="AAAABBBBCCCC")).encode("utf-8"))
    _refuse_read(dest)
    plant(json.dumps(_payload(base="a" * 39)).encode("utf-8"))
    _refuse_read(dest)
    plant(json.dumps(_payload(outcome="cancelled")).encode("utf-8"))
    _refuse_read(dest)
    plant(json.dumps(_payload(outcome=["failure"])).encode("utf-8"))
    _refuse_read(dest)
    plant(json.dumps(_payload(outcome={"result": "failure"})).encode("utf-8"))
    _refuse_read(dest)
    plant(json.dumps(_payload(extra=1)).encode("utf-8"))
    _refuse_read(dest)
    missing = _payload()
    del missing["branch"]
    plant(json.dumps(missing).encode("utf-8"))
    _refuse_read(dest)
    plant(json.dumps(_payload(work_state={"head": "c" * 40})).encode("utf-8"))
    _refuse_read(dest)
    plant(b'{"version":1, "version":1}')
    _refuse_read(dest)
    plant(b'{"version": NaN}')
    _refuse_read(dest)
    plant(b"\xff\xfe{")
    _refuse_read(dest)
    plant(b"x" * (jresume.CHECKPOINT_CAP + 1))
    _refuse_read(dest)
    plant(b"[" * 60000 + b"]" * 60000)
    _refuse_read(dest)
    plant(json.dumps(_payload()).encode("utf-8"), 0o644)
    _refuse_read(dest)

    real_fstat = os.fstat

    def wrong_owner(fd):
        st = real_fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return st
        return os.stat_result((
            st.st_mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid + 1, st.st_gid,
            st.st_size, int(st.st_atime), int(st.st_mtime), int(st.st_ctime),
        ))

    plant(json.dumps(_payload()).encode("utf-8"))
    monkeypatch.setattr(os, "fstat", wrong_owner)
    _refuse_read(dest)
    monkeypatch.setattr(os, "fstat", real_fstat)

    dest.unlink()
    dest.symlink_to(tmp_path / "success.json")
    _refuse_read(dest)

    linked = tmp_path / "via-link" / "resume-checkpoint.json"
    linked.parent.mkdir()
    write_checkpoint(linked, _payload())
    alias = tmp_path / "alias-parent"
    alias.symlink_to(linked.parent)
    _refuse_read(alias / "resume-checkpoint.json", "agent-settings-symlink")

    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    os.chmod(fifo, 0o600)

    def _blocked(signum, frame):
        raise AssertionError("read_checkpoint blocked on FIFO")

    previous = (signal.signal(signal.SIGALRM, _blocked),
                signal.setitimer(signal.ITIMER_REAL, 2))
    try:
        _refuse_read(fifo)
    finally:
        handler, (delay, interval) = previous
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        signal.signal(signal.SIGALRM, handler)


def _lock_identities(root):
    lock_dir = Path(root) / ".local" / "runs" / "locks"
    if not lock_dir.is_dir():
        return set()
    return {(p.stat().st_ino, p.stat().st_dev) for p in lock_dir.iterdir()}


def test_worktree_claim_holds_kernel_lock():
    with TemporaryDirectory() as raw:
        control = Path(raw) / "repo"
        worktree = Path(raw) / "wt"
        worktree.mkdir()
        with worktree_claim(control, worktree):
            lock_dir = control / ".local" / "runs" / "locks"
            locks = list(lock_dir.iterdir())
            assert len(locks) == 1
            expected = hashlib.sha256(str(worktree.resolve()).encode()).hexdigest()
            assert locks[0].name == expected
            fd = os.open(str(locks[0]), os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
        assert locks[0].exists()


@pytest.mark.parametrize("kind", [".local", "runs", "locks"])
def test_worktree_claim_refuses_symlink_at_local_runs_and_locks(kind):
    with TemporaryDirectory() as raw:
        base = Path(raw)
        worktree = base / "wt"
        worktree.mkdir()
        real = base / "real"
        (real / ".local" / "runs" / "locks").mkdir(parents=True)
        control = base / "control"
        if kind == ".local":
            control.mkdir()
            (control / ".local").symlink_to(real / ".local")
        elif kind == "runs":
            (control / ".local").mkdir(parents=True)
            (control / ".local" / "runs").symlink_to(real / ".local" / "runs")
        else:
            (control / ".local" / "runs").mkdir(parents=True)
            (control / ".local" / "runs" / "locks").symlink_to(real / ".local" / "runs" / "locks")
        with pytest.raises(Refusal) as exc:
            with worktree_claim(control, worktree):
                pass
        assert exc.value.code == "agent-settings-symlink"
        assert _lock_identities(real) == set()
        assert _lock_identities(control) == set()


def test_worktree_claim_refuses_ancestor_switch_between_acquisitions():
    with TemporaryDirectory() as raw:
        base = Path(raw)
        worktree = base / "wt"
        worktree.mkdir()
        control = base / "repo"
        control.mkdir()
        with worktree_claim(control, worktree):
            first = _lock_identities(control)
        assert len(first) == 1
        other = base / "other"
        (other / ".local" / "runs" / "locks").mkdir(parents=True)
        (control / ".local").rename(base / "repo-local-real")
        (control / ".local").symlink_to(other / ".local")
        with pytest.raises(Refusal) as exc:
            with worktree_claim(control, worktree):
                pass
        assert exc.value.code == "agent-settings-symlink"
        assert _lock_identities(other) == set()
        leftover = base / "repo-local-real" / "runs" / "locks"
        assert {(p.stat().st_ino, p.stat().st_dev) for p in leftover.iterdir()} == first


def _c2_claim_child(control, worktree, ready, allow, release, result_path, gated):
    import fcntl
    import json
    import os
    from pathlib import Path

    from jax_init import Refusal
    from jaxflow_resume import worktree_claim

    if gated:
        real = fcntl.flock

        def paused(fd, op):
            if op & fcntl.LOCK_EX:
                ready.set()
                if not allow.wait(15):
                    raise TimeoutError("allow")
            return real(fd, op)

        fcntl.flock = paused
    try:
        with worktree_claim(Path(control), Path(worktree)):
            Path(result_path).write_text(json.dumps({"ok": True}), encoding="utf-8")
            release.wait(15)
    except Refusal as exc:
        Path(result_path).write_text(
            json.dumps({"ok": False, "code": exc.code}), encoding="utf-8",
        )
    except Exception as exc:
        Path(result_path).write_text(
            json.dumps({"ok": False, "error": type(exc).__name__}), encoding="utf-8",
        )


def test_worktree_claim_refuses_parent_switch_before_yield(tmp_path):
    control = tmp_path / "repo"
    worktree = tmp_path / "wt"
    worktree.mkdir()
    ctx = multiprocessing.get_context("spawn")
    ready, allow, release = ctx.Event(), ctx.Event(), ctx.Event()
    a_path, b_path = tmp_path / "a.json", tmp_path / "b.json"
    holder = ctx.Process(
        target=_c2_claim_child,
        args=(str(control), str(worktree), ready, allow, release, str(a_path), True),
    )
    holder.start()
    assert ready.wait(15)
    real_local = control / ".local"
    moved = tmp_path / "repo-local-old"
    real_local.rename(moved)
    (control / ".local").mkdir()
    waiter = ctx.Process(
        target=_c2_claim_child,
        args=(str(control), str(worktree), ready, allow, release, str(b_path), False),
    )
    waiter.start()
    allow.set()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not (a_path.is_file() and b_path.is_file()):
        time.sleep(0.05)
    a_res = json.loads(a_path.read_text(encoding="utf-8")) if a_path.is_file() else {}
    b_res = json.loads(b_path.read_text(encoding="utf-8")) if b_path.is_file() else {}
    release.set()
    for proc in (holder, waiter):
        proc.join(5)
        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join(2)
    accepted = [row for row in (a_res, b_res) if row.get("ok")]
    assert len(accepted) <= 1


def test_write_checkpoint_refuses_parent_switch(tmp_path, monkeypatch):
    parent = tmp_path / "run"
    parent.mkdir()
    dest = parent / "resume-checkpoint.json"
    outside = tmp_path / "outside"
    real_open = os.open
    swapped = {"n": False}

    def gated(path, flags, mode=0o777, *args, dir_fd=None, **kwargs):
        name = os.fspath(path)
        if str(name).endswith(".tmp") and not swapped["n"]:
            swapped["n"] = True
            if parent.exists() and not outside.exists():
                parent.rename(outside)
                parent.mkdir()
        if dir_fd is None:
            return real_open(path, flags, mode, *args, **kwargs)
        return real_open(path, flags, mode, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(os, "open", gated)
    with pytest.raises(Refusal):
        write_checkpoint(dest, _payload())
    assert not dest.exists()
    assert list(parent.iterdir()) == []
    assert _tmps(outside) == []
    assert not (outside / "resume-checkpoint.json").exists()
