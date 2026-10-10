"""jaxflow review tests (P2 split of test_jaxflow.py)."""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import pytest
import general_settings
import jaxflow_cli
import jaxflow_worker
import jaxflow_build
import jaxflow_review
import jaxflow_workerkit
import jaxflow_common
import uuid
import shlex
import jaxflow_hook
import jaxflow_run as jr
import jax_init as ji

from testkit import (  # noqa: F401  (autouse fixtures included on purpose)
    FakePopen,
    FakeTmux,
    _CAPTURED_THREAD,
    _E2EBuilderPopen,
    _TEST_CLAUDE_SESSION_ID,
    _agents,
    _agents_setting,
    _build_args,
    _diff_args,
    _fixed_now,
    _forbidden_tmux,
    _fresh_db,
    _git,
    _init_repo,
    _init_worktree,
    _insert,
    _integration_on_by_default,
    _isolate_agent_settings,
    _isolate_callbacks,
    _ledger_post,
    _plan_file,
    _restore_signal_handlers,
    _review_args,
    _run_diff_worker_and_capture_prompt,
    _run_real,
    _run_with_tmux,
    _run_with_tmux_and_log,
    _seed_finished_build,
    _seed_resumable,
    _spec_file,
    _stub_launch_paths,
    _write_manifest_for_diff_worker,
    _write_manifest_for_worker,
    _write_plan,
)


# ---- nested cwd derives repo from the git toplevel (fixes cold review F1) ----

def test_dispatch_derives_repo_from_toplevel_when_cwd_is_nested(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        nested = root / "a" / "b"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)
        fake = FakeTmux()
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        run_id = jaxflow_review.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now, allowlist_root=allow_root,
        )
        payload = events[0]["payload"]
        assert payload["repo"] == str(root)
        assert payload["session"] == f"jax-demo-spec-{run_id}"


# ---- doc-review test-output evidence (acceptance smoke 2026-09-06, fix 2) ----

def test_dispatch_writes_doc_review_test_marker_before_dispatch(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()

        run_id = jaxflow_review.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        tests_file = root / ".local" / "reports" / f"{run_id}.tests.txt"
        assert tests_file.read_text(encoding="utf-8") == jaxflow_review.DOC_REVIEW_TEST_MARKER
        assert jaxflow_review.DOC_REVIEW_TEST_MARKER == "No test/build command was required by this handoff.\n"


def test_dispatch_hub_unreachable_removes_manifest_and_skips_tmux(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()

        def post(event):
            raise RuntimeError("event post failed")

        try:
            jaxflow_review.dispatch_review(
                _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root), post=post,
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now, allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert exc.code == "hub-unreachable"
        else:
            raise AssertionError("hub-unreachable not raised")
        assert [c[1] for c in fake.calls] == ["has-session"]
        assert not (root / ".local" / "runs").exists() or not any((root / ".local" / "runs").iterdir())
        reports_dir = root / ".local" / "reports"
        assert not any(reports_dir.glob("*.tests.txt"))


def test_dispatch_tmux_failed_posts_cancelled_row_first(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        fake.new_session_fails = True
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        try:
            jaxflow_review.dispatch_review(
                _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root), post=post,
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now, allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert exc.code == "tmux-failed"
        else:
            raise AssertionError("tmux-failed not raised")
        assert [e["type"] for e in events] == ["run-started", "run-finished"]
        cancelled = events[1]["payload"]
        assert cancelled["contract_status"] == "cancelled"
        assert cancelled["exit_code"] is None
        assert cancelled["report_path"] is None
        assert cancelled["summary"] == "tmux new-session failed at dispatch"
        assert not (root / ".local" / "runs").exists() or not any((root / ".local" / "runs").iterdir())
        reports_dir = root / ".local" / "reports"
        assert not any(reports_dir.glob("*.tests.txt"))


def test_dispatch_caller_pane_present_reaches_manifest_and_payload(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        run_id = jaxflow_review.dispatch_review(
            _review_args(plan=str(target)), run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"CODEX_THREAD_ID": "1", "TMUX_PANE": "%3"}, now=_fixed_now, allowlist_root=allow_root,
        )
        assert events[0]["payload"]["caller_pane"] == "%3"
        import json
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert "caller_pane" not in manifest
        assert "caller_incarnation" not in manifest
        assert manifest["model"] == "sonnet"
        assert manifest["runtime"] == "claude"


# ---- --phase default from stem, --model/--effort override ----

def test_dispatch_phase_defaults_to_stem_and_overrides_win(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root, name="2026-05-01-my-cool-spec.md")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        jaxflow_review.dispatch_review(
            _review_args(spec=str(target), model="custom-model", effort="high"),
            run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now, allowlist_root=allow_root,
        )
        payload = events[0]["payload"]
        assert payload["phase"] == "2026-05-01-my-cool-spec"
        assert payload["model"] == "custom-model"
        assert payload["effort"] == "high"


def test_dispatch_writes_pointer_for_claude_caller(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        run_id = jaxflow_review.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID},
            now=_fixed_now, allowlist_root=allow_root,
        )
        pointer_path = jaxflow_common.CALLBACKS_ROOT / _TEST_CLAUDE_SESSION_ID / f"{run_id}.json"
        data = json.loads(pointer_path.read_text(encoding="utf-8"))
        assert data == {"run_id": run_id, "kind": "spec"}


def test_dispatch_writes_pointer_for_build(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        run_id = jaxflow_build.dispatch_build(
            _build_args(plan=str(plan)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID},
            now=_fixed_now, allowlist_root=allow_root,
        )
        pointer_path = jaxflow_common.CALLBACKS_ROOT / _TEST_CLAUDE_SESSION_ID / f"{run_id}.json"
        data = json.loads(pointer_path.read_text(encoding="utf-8"))
        assert data == {"run_id": run_id, "kind": "build"}


def test_dispatch_writes_pointer_for_build_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    _E2EBuilderPopen.launches = 0
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        _stub_launch_paths(tmp_path, monkeypatch)
        db = Path(raw) / "jaxos.db"
        _fresh_db(db).close()
        monkeypatch.setattr(jr, "DB_PATH", db)
        fake = FakeTmux()
        events = []
        post = _ledger_post(db, events)
        env = {
            "CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID,
            "TMUX_PANE": "%3", "PATH": "/bin", "HOME": str(tmp_path),
        }
        first_id = jaxflow_build.dispatch_build(
            _build_args(plan=str(plan), branch="feat/x", whitelist="a.py", verify="false"),
            run=_run_with_tmux(fake, real_cwd=root),
            post=post, env=env, now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        worktree = allow_root / "demo-feat-x"
        jaxflow_worker.run_worker(
            str(root / ".local" / "runs" / first_id / "manifest.json"),
            run=_run_with_tmux(fake, real_cwd=worktree),
            post=post, popen=_E2EBuilderPopen, allowlist_root=allow_root, env=env,
        )
        second_id = jaxflow_build.dispatch_build(
            _build_args(resume=first_id), run=_run_with_tmux(fake, real_cwd=root),
            post=post, env=env, now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        pointer_path = jaxflow_common.CALLBACKS_ROOT / _TEST_CLAUDE_SESSION_ID / f"{second_id}.json"
        data = json.loads(pointer_path.read_text(encoding="utf-8"))
        assert data == {"run_id": second_id, "kind": "build"}


def test_dispatch_writes_pointer_for_diff_review(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        monkeypatch.chdir(root)
        fake = FakeTmux()
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        pointer_path = jaxflow_common.CALLBACKS_ROOT / _TEST_CLAUDE_SESSION_ID / f"{run_id}.json"
        data = json.loads(pointer_path.read_text(encoding="utf-8"))
        assert data == {"run_id": run_id, "kind": "diff"}


def test_dispatch_no_pointer_on_no_callback(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        jaxflow_review.dispatch_review(
            _review_args(spec=str(target), no_callback=True),
            run=_run_with_tmux(fake, real_cwd=root), post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID},
            now=_fixed_now, allowlist_root=allow_root,
        )
        assert not jaxflow_common.CALLBACKS_ROOT.exists()


def test_dispatch_no_pointer_for_codex_caller(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        jaxflow_review.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CODEX_THREAD_ID": _CAPTURED_THREAD},
            now=_fixed_now, allowlist_root=allow_root,
        )
        assert not jaxflow_common.CALLBACKS_ROOT.exists()


def test_dispatch_no_pointer_for_noncanonical_session(monkeypatch, capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        run_id = jaxflow_review.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "not-a-uuid"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        assert not jaxflow_common.CALLBACKS_ROOT.exists()
        err = capsys.readouterr().err
        # Cold review round 2 F4: run_id itself is a valid, canonical run id here (only caller_session is
        # forged) -- `ascii(run_id)[:80]` still applies and simply adds the repr quotes.
        assert (
            f"callback pointer write failed: {ascii(run_id)[:80]} path-unsafe; "
            "use jaxflow status/result"
        ) in err


def test_dispatch_review_writes_no_pointer_on_tmux_failure(monkeypatch):
    # Cold review round 1 F2/F5: the pointer is written only AFTER a successful launch
    # (see Step 2 below), so a nonzero tmux result never reaches that write -- there is
    # nothing to clean up.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        fake.new_session_fails = True
        with pytest.raises(ji.Refusal, match="tmux-failed"):
            jaxflow_review.dispatch_review(
                _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True},
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID},
                now=_fixed_now, allowlist_root=allow_root,
            )
        session_dir = jaxflow_common.CALLBACKS_ROOT / _TEST_CLAUDE_SESSION_ID
        assert not session_dir.exists() or not any(session_dir.glob("*.json"))


def test_dispatch_build_writes_no_pointer_on_tmux_failure(monkeypatch):
    # Same proof through `_dispatch_run` (shared by build/build-resume/diff-review),
    # not `dispatch_review`'s own inline copy.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        fake.new_session_fails = True
        with pytest.raises(ji.Refusal, match="tmux-failed"):
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan)), run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True},
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID},
                now=_fixed_now, allowlist_root=allow_root,
            )
        session_dir = jaxflow_common.CALLBACKS_ROOT / _TEST_CLAUDE_SESSION_ID
        assert not session_dir.exists() or not any(session_dir.glob("*.json"))


def test_dispatch_review_refuses_on_a_tmux_launch_exception_same_as_a_nonzero_exit(monkeypatch):
    # Spec: Clean error on missing CLI -- `dispatch_review`'s own `tmux new-session` call
    # reuses `_dispatch_run`'s exact try/except pattern, so a launcher exception becomes
    # the same `Refusal("tmux-failed")` a nonzero exit already produced (and, before this,
    # propagated OSError out of the dispatcher uncaught). No pointer is ever written.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)

        def raising_run(argv, cwd=None):
            if argv[:2] == ["tmux", "new-session"]:
                raise OSError("tmux binary not found")
            return _run_with_tmux(FakeTmux(), real_cwd=root)(argv, cwd=cwd)

        events = []
        code = jaxflow_cli.main(
            ["review", "--spec", str(target)], run=raising_run,
            post=lambda e: events.append(e) or {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID},
            allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert any(
            e["type"] == "run-finished" and e["payload"]["contract_status"] == "cancelled"
            for e in events
        )
        session_dir = jaxflow_common.CALLBACKS_ROOT / _TEST_CLAUDE_SESSION_ID
        assert not session_dir.exists() or not any(session_dir.glob("*.json"))


def test_dispatch_build_writes_no_pointer_when_launcher_raises(monkeypatch):
    # Same proof through `_dispatch_run`, which DOES catch a launcher exception
    # (pre-existing, cold review F7 from an earlier round) and turns it into the same
    # `Refusal("tmux-failed")` a nonzero exit code produces.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)

        def raising_run(argv, cwd=None):
            if argv[:2] == ["tmux", "new-session"]:
                raise OSError("tmux binary not found")
            return _run_with_tmux(FakeTmux(), real_cwd=root)(argv, cwd=cwd)

        with pytest.raises(ji.Refusal, match="tmux-failed"):
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan)), run=raising_run, post=lambda e: {"ok": True},
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID},
                now=_fixed_now, allowlist_root=allow_root,
            )
        session_dir = jaxflow_common.CALLBACKS_ROOT / _TEST_CLAUDE_SESSION_ID
        assert not session_dir.exists() or not any(session_dir.glob("*.json"))


def test_dispatch_codex_manifest_has_no_pane_incarnation_fields(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        run_id = jaxflow_review.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CODEX_THREAD_ID": _CAPTURED_THREAD, "TMUX_PANE": "%3"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        manifest = json.loads(
            (root / ".local" / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
        assert "caller_incarnation" not in manifest
        assert "caller_pane" not in manifest


def test_dispatch_warns_when_hook_missing(monkeypatch, tmp_path, capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        missing = tmp_path / "settings.json"  # never written -- read raises OSError
        monkeypatch.setattr(jaxflow_common, "CLAUDE_SETTINGS_PATH", missing)
        fake = FakeTmux()
        run_id = jaxflow_review.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID},
            now=_fixed_now, allowlist_root=allow_root,
        )
        err = capsys.readouterr().err
        assert str(missing) in err
        assert run_id in err
        assert "claude-callback" in err


def test_dispatch_silent_when_hook_present(monkeypatch, capsys):
    # F6: the autouse `_isolate_callbacks` fixture already points CLAUDE_SETTINGS_PATH
    # at a file containing a STRUCTURALLY valid claude-callback PostToolUse/Bash hook
    # entry -- this is the default-path regression test: present -> no warning.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        jaxflow_review.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID},
            now=_fixed_now, allowlist_root=allow_root,
        )
        assert "no claude-callback hook installed" not in capsys.readouterr().err


def test_dispatch_warns_when_hook_settings_malformed_json(monkeypatch, tmp_path, capsys):
    # F6: an unreadable/malformed settings file counts as missing -- never raises.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        bad_settings = tmp_path / "settings.json"
        bad_settings.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(jaxflow_common, "CLAUDE_SETTINGS_PATH", bad_settings)
        fake = FakeTmux()
        run_id = jaxflow_review.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID},
            now=_fixed_now, allowlist_root=allow_root,
        )
        err = capsys.readouterr().err
        assert "no claude-callback hook installed" in err
        assert run_id in err


@pytest.mark.parametrize("settings", [
    {"hooks": None},
    {"hooks": {"PostToolUse": None}},
    {"hooks": {"PostToolUse": [{"matcher": "Bash", "hooks": None}]}},
    {"hooks": {"PostToolUse": [{"matcher": "Bash", "hooks": 5}]}},
    [],
])
def test_dispatch_warns_when_hook_settings_have_a_malformed_shape(monkeypatch, tmp_path, capsys, settings):
    # D9: readable JSON with an unexpected shape counts as missing and never aborts dispatch.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps(settings), encoding="utf-8")
        monkeypatch.setattr(jaxflow_common, "CLAUDE_SETTINGS_PATH", settings_path)
        fake = FakeTmux()
        run_id = jaxflow_review.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID},
            now=_fixed_now, allowlist_root=allow_root,
        )
        err = capsys.readouterr().err
        assert "no claude-callback hook installed" in err
        assert run_id in err


@pytest.mark.parametrize("unsafe_hex", ["AAAABBBBCCCC" + "0" * 20, "aaaabbbbccc\n" + "0" * 20])
def test_dispatch_rejects_unsafe_run_id_for_pointer(monkeypatch, capsys, unsafe_hex):
    # §5.1: dispatch run ids come from uuid4, so this gate is defense in depth -- an
    # injected non-canonical id still dispatches, writes no pointer, and warns once.
    monkeypatch.setattr(uuid, "uuid4", lambda: SimpleNamespace(hex=unsafe_hex))
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        run_id = jaxflow_review.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID},
            now=_fixed_now, allowlist_root=allow_root,
        )
        assert run_id == unsafe_hex[:12]
        assert not jaxflow_common.CALLBACKS_ROOT.exists()
        err = capsys.readouterr().err
        warning = f"callback pointer write failed: {ascii(run_id)[:80]} path-unsafe; use jaxflow status/result"
        assert err.count(warning) == 1
        assert "\n" not in warning


def test_dispatch_warns_when_hook_present_but_not_structurally(monkeypatch, tmp_path, capsys):
    # F6: the literal text "claude-callback" appears in the file (in a Stop hook, and
    # in a PostToolUse/Bash command missing asyncRewake) but no entry satisfies the full
    # structural shape -- a substring match must not silence the warning.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps({
            "hooks": {
                "Stop": [
                    {"hooks": [{"type": "command", "command": "notify claude-callback"}]}
                ],
                "PostToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "python3 /x/jaxflow_hook.py claude-callback",
                                # no "asyncRewake": true -- structurally incomplete
                            }
                        ],
                    }
                ],
            }
        }), encoding="utf-8")
        monkeypatch.setattr(jaxflow_common, "CLAUDE_SETTINGS_PATH", settings_path)
        fake = FakeTmux()
        run_id = jaxflow_review.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID},
            now=_fixed_now, allowlist_root=allow_root,
        )
        err = capsys.readouterr().err
        assert "no claude-callback hook installed" in err
        assert run_id in err


# ---- tmux command quoting (fixes cold review F7/G16) ----

def test_dispatch_quotes_tmux_command_for_space_and_semicolon_in_repo_path(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo app; two"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()

        run_id = jaxflow_review.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        new_session_call = next(c for c in fake.calls if c[1] == "new-session")
        cmd = new_session_call[-1]
        manifest_path = root / ".local" / "runs" / run_id / "manifest.json"
        # the raw ";" is present in the string but only inside a shlex-quoted token --
        # round-tripping through shlex.split proves it can never act as a shell separator.
        assert ";" in cmd
        # fixes cold review F4: the worker PROCESS is sealed from birth via an `env`
        # prefix in the launch command itself, not by writing os.environ after the fact.
        assert shlex.split(cmd) == [
            "env", "HONCHO_ENABLED=false", "JAXFLOW_ONESHOT=1",
            sys.executable, str(jaxflow_common.SCRIPT_PATH), "--run-worker", str(manifest_path),
        ]


def test_diff_dispatch_on_a_legacy_build_run_without_a_build_key(monkeypatch):
    """Spec §7.1 #36. A run-started payload recorded before MOA-454 has `verify` and NO
    `build` key. `review --diff` must re-run only the test command and must not raise.
    A subscript read (`started_payload["build"]`) fails this test with a KeyError."""
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        # no `build=` -- the seeded payload has the pre-MOA-454 shape
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        monkeypatch.chdir(root)
        fake = FakeTmux()
        jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        assert fake.calls != []  # a reviewer WAS dispatched
        text = (worktree / ".local" / "reports" / "b1.tests.txt").read_text(encoding="utf-8")
        assert text == "COMMAND: true\n\nEXIT: 0\n"  # exactly one frame


def test_diff_dispatch_reviews_a_finished_build_whose_report_was_invalid(monkeypatch):
    """A build that finished with `contract_status: invalid` (report missing its
    frontmatter) and no `result` still has a real diff and a recorded head_sha; `review
    --diff` must dispatch a reviewer instead of refusing `unknown-run` (MOA-471)."""
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, head_sha=head, contract_status="invalid")
        con.close()
        monkeypatch.chdir(root)
        fake = FakeTmux()
        jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        assert fake.calls != []  # a reviewer WAS dispatched


def test_diff_review_eligible_on_an_interrupted_builder_row_with_a_real_head_sha(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(
            con, "b1", root, worktree, verify="true", head_sha=head,
            contract_status="interrupted",
        )
        con.close()
        monkeypatch.chdir(root)
        # Must NOT raise -- no "unknown-run" Refusal with the "finished without a
        # head_sha" hint, since head_sha is real and non-empty regardless of
        # contract_status (dispatch_diff_review's own check only reads head_sha).
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        assert run_id is not None


def test_diff_dispatch_reruns_both_commands_and_rewrites_the_evidence(monkeypatch):
    """Spec §4.3 step 2 (cold-review F2): the file the reviewer reads must be the evidence
    of THIS re-run -- both frames -- never the builder's older single frame."""
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", build="true", head_sha=head)
        con.close()
        stale = worktree / ".local" / "reports" / "b1.tests.txt"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("COMMAND: STALE\nold\nEXIT: 0\n", encoding="utf-8")
        monkeypatch.chdir(root)
        jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        text = stale.read_text(encoding="utf-8")
        assert "STALE" not in text
        assert text.count("COMMAND: ") == 2
        assert text.count("EXIT: 0") == 2


def test_decide_verify_reuse_pure_conditions():
    ok_frames = [{"exit_code": 0}]
    assert jaxflow_common._decide_verify_reuse(
        head_matches=False, porcelain_clean=True, commands_match=True,
        frames=ok_frames, parse_error=None,
    ) == (False, "head-sha-changed")
    assert jaxflow_common._decide_verify_reuse(
        head_matches=True, porcelain_clean=False, commands_match=True,
        frames=ok_frames, parse_error=None,
    ) == (False, "worktree-dirty")
    assert jaxflow_common._decide_verify_reuse(
        head_matches=True, porcelain_clean=True, commands_match=False,
        frames=ok_frames, parse_error=None,
    ) == (False, "verify-command-changed")
    assert jaxflow_common._decide_verify_reuse(
        head_matches=True, porcelain_clean=True, commands_match=True,
        frames=None, parse_error="frame-2-missing",
    ) == (False, "frame-2-missing")
    assert jaxflow_common._decide_verify_reuse(
        head_matches=True, porcelain_clean=True, commands_match=True,
        frames=[{"exit_code": 0}, {"exit_code": 1}], parse_error=None,
    ) == (False, "prior-verify-failed")
    assert jaxflow_common._decide_verify_reuse(
        head_matches=True, porcelain_clean=True, commands_match=True,
        frames=ok_frames, parse_error=None,
    ) == (True, "all-conditions-met")


def test_diff_dispatch_reuses_verify_when_all_conditions_hold(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", build="true", head_sha=head)
        con.close()
        tests_path = worktree / ".local" / "reports" / "b1.tests.txt"
        tests_path.parent.mkdir(parents=True, exist_ok=True)
        tests_path.write_text("COMMAND: true\n\nEXIT: 0\nCOMMAND: true\n\nEXIT: 0\n", encoding="utf-8")
        before = tests_path.read_bytes()
        monkeypatch.chdir(root)
        seen = []
        run_id = jaxflow_review.dispatch_diff_review(
            # F5 fix: the logging runner must be PASSED to dispatch (an unused
            # `logging_run` here previously left `seen` empty, vacuous assertion).
            _diff_args("b1"), run=_run_with_tmux_and_log(FakeTmux(), seen, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        assert tests_path.read_bytes() == before  # untouched
        assert not any(a[:3] == ["/bin/sh", "-c", "true"] for a in seen)  # zero verify/build invocations
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["verify"]["mode"] == "reused"
        assert manifest["verify"]["source_build_run_id"] == "b1"


def test_diff_dispatch_reverify_forces_rerun_even_when_conditions_hold(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        tests_path = worktree / ".local" / "reports" / "b1.tests.txt"
        tests_path.parent.mkdir(parents=True, exist_ok=True)
        tests_path.write_text("COMMAND: true\n\nEXIT: 0\n", encoding="utf-8")
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1", reverify=True), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["verify"]["mode"] == "rerun"
        assert manifest["verify"]["reason"] == "reverify-forced"


def test_diff_dispatch_dirty_worktree_reruns_verify(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        tests_path = worktree / ".local" / "reports" / "b1.tests.txt"
        tests_path.parent.mkdir(parents=True, exist_ok=True)
        tests_path.write_text("COMMAND: true\n\nEXIT: 0\n", encoding="utf-8")
        (worktree / "dirty.txt").write_text("x\n", encoding="utf-8")  # untracked -> dirty
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["verify"] == {
            "mode": "rerun", "reason": "worktree-dirty",
            "source_build_run_id": "b1", "head_sha": head,
        }


def test_diff_dispatch_head_moved_past_builder_sha_reruns_verify(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        base_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        (worktree / "extra.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "extra.txt")
        _git(worktree, "commit", "-m", "fix commit")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=base_sha)
        con.close()
        tests_path = worktree / ".local" / "reports" / "b1.tests.txt"
        tests_path.parent.mkdir(parents=True, exist_ok=True)
        tests_path.write_text("COMMAND: true\n\nEXIT: 0\n", encoding="utf-8")
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["verify"]["mode"] == "rerun"
        assert manifest["verify"]["reason"] == "head-sha-changed"


def test_diff_dispatch_missing_tests_file_reruns_verify(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()  # no pre-existing .tests.txt at all
        monkeypatch.chdir(root)
        seen = []
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux_and_log(FakeTmux(), seen, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["verify"]["mode"] == "rerun"
        assert manifest["verify"]["reason"] == "tests-file-missing"
        assert sum(1 for a in seen if a[:3] == ["/bin/sh", "-c", "true"]) >= 1  # at least one real invocation


def test_diff_dispatch_command_mismatch_only_inside_redacted_token_still_reuses(monkeypatch):
    # redact() masks secret-shaped tokens (spec item 8): evidence COMMAND lines are
    # compared against redact(requested_cmd). A verify command with a real *_KEY=...
    # token proves it -- a raw-string comparison would MISMATCH (evidence never has
    # the secret); this only reuses if the comparison is redaction-normalized.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        verify_cmd = "API_KEY=aaaaaaaa1 true"
        assert jaxflow_hook.redact(verify_cmd) != verify_cmd  # sanity: the token really masks
        _seed_finished_build(con, "b1", root, worktree, verify=verify_cmd, head_sha=head)
        con.close()
        tests_path = worktree / ".local" / "reports" / "b1.tests.txt"
        tests_path.parent.mkdir(parents=True, exist_ok=True)
        tests_path.write_text(f"COMMAND: {jaxflow_hook.redact(verify_cmd)}\n\nEXIT: 0\n", encoding="utf-8")
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["verify"]["mode"] == "reused"


def test_diff_dispatch_empty_evidence_file_reruns_verify(monkeypatch):
    # F5 (diff review 4884f63bdd16): an empty `.tests.txt` must trigger a real rerun
    # (not silently "reuse") -- asserts the verify command actually ran AND that the
    # persisted reason is the one `jr.parse_tests_frames` emits for empty text.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        tests_path = worktree / ".local" / "reports" / "b1.tests.txt"
        tests_path.parent.mkdir(parents=True, exist_ok=True)
        tests_path.write_text("", encoding="utf-8")
        monkeypatch.chdir(root)
        seen = []
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux_and_log(FakeTmux(), seen, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        assert any(a[:3] == ["/bin/sh", "-c", "true"] for a in seen)  # verify actually re-ran
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["verify"]["mode"] == "rerun"
        assert manifest["verify"]["reason"] == "frame-1-missing"


def test_diff_dispatch_one_frame_present_but_two_commands_expected_reruns_verify(monkeypatch):
    # F5: evidence has exactly one COMMAND frame (from the builder's own single-command
    # run) but this dispatch expects TWO (verify + build) -- must rerun, not reuse.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", build="true", head_sha=head)
        con.close()
        tests_path = worktree / ".local" / "reports" / "b1.tests.txt"
        tests_path.parent.mkdir(parents=True, exist_ok=True)
        tests_path.write_text("COMMAND: true\n\nEXIT: 0\n", encoding="utf-8")  # only frame 1
        monkeypatch.chdir(root)
        seen = []
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux_and_log(FakeTmux(), seen, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        assert any(a[:3] == ["/bin/sh", "-c", "true"] for a in seen)  # verify actually re-ran
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["verify"]["mode"] == "rerun"
        assert manifest["verify"]["reason"] == "frame-2-missing"


def test_diff_dispatch_malformed_frame_without_exit_reruns_verify(monkeypatch):
    # F5: a frame with no `EXIT: <n>` line at all (truncated/corrupted evidence) must
    # rerun, not crash and not reuse.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        tests_path = worktree / ".local" / "reports" / "b1.tests.txt"
        tests_path.parent.mkdir(parents=True, exist_ok=True)
        tests_path.write_text("COMMAND: true\nno exit line here\n", encoding="utf-8")
        monkeypatch.chdir(root)
        seen = []
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux_and_log(FakeTmux(), seen, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        assert any(a[:3] == ["/bin/sh", "-c", "true"] for a in seen)  # verify actually re-ran
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["verify"]["mode"] == "rerun"
        assert manifest["verify"]["reason"] == "frame-1-no-exit"


def _seed_diff_review(con, run_id, root, worktree, *, target="feat/x", verdict="approve",
                       contract_status="ok", finished=True, ts="t", session=None, project="demo"):
    """Seeds a `kind: diff` reviewer run-started (+ optional run-finished) pair matching
    `dispatch_diff_review`/its worker write, plus that run's own manifest -- everything
    the item 7 guard and item 9 chain walk read. `project` defaults to "demo"; another
    value seeds a review in a DIFFERENT project (F3's test)."""
    session = session or f"jax-demo-diff-{run_id}"
    _insert(con, run_id, project, "reviewer", "run-started", {
        "phase": "PHASE", "runtime": "codex", "kind": "diff", "target": target,
        "caller": "claude", "caller_session": "s", "model": "gpt-5.6-luna", "effort": "xhigh",
        "session": session, "repo": str(root),
    }, ts=ts)
    if finished:
        payload = {
            "phase": "PHASE", "exit_code": 0, "contract_status": contract_status,
            "report_path": str(root / ".local" / "reports" / f"{run_id}.md"), "summary": "reviewed",
        }
        if contract_status == "ok":
            payload["verdict"] = verdict
        _insert(con, run_id, project, "reviewer", "run-finished", payload, ts=ts)
    manifest_dir = root / ".local" / "runs" / run_id
    manifest_dir.mkdir(parents=True, exist_ok=True)


def test_guard_no_prior_review_dispatches_unguarded(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["guard"] == {"prior_review_run_id": None, "prior_verdict": None, "override": "none"}


def test_guard_no_prior_review_with_full_flag_still_records_override_full(monkeypatch):
    # F4 (diff review 4884f63bdd16): `guard.override` must reflect the flag actually
    # passed even when there is no terminal prior review to bypass (a first-ever diff
    # review dispatched with `--full`). A `--since` counterpart is skipped: `--since`
    # requires a TERMINAL prior review on the same branch, which -- with a single seeded
    # review -- always becomes `selected`, so "selected is None and --since is valid" has
    # no cheap fixture; that combination is exercised indirectly by the existing
    # `test_guard_reject_proceeds_with_or_without_since`/`..._approve_blocks_and_full_or_
    # since_bypasses` tests, which cover --since with a non-None `selected`.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1", full="first review ever, no prior verdict to bypass"),
            run=_run_with_tmux(FakeTmux(), real_cwd=root), post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["guard"] == {"prior_review_run_id": None, "prior_verdict": None, "override": "full"}
        assert manifest["full_reason"] == "first review ever, no prior verdict to bypass"


def test_guard_filters_by_project_same_branch_other_project_does_not_block(monkeypatch):
    # F3: an approved review in a DIFFERENT project must never guard THIS project's dispatch.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)  # project "demo"
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve", project="other")
        con.close()
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["guard"] == {"prior_review_run_id": None, "prior_verdict": None, "override": "none"}


def test_guard_running_review_refuses_unconditionally(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        fake = FakeTmux()
        fake.existing.add("jax-demo-diff-aaaaaaaaaaa0")
        _seed_diff_review(con, "aaaaaaaaaaa0", root, worktree, finished=False, session="jax-demo-diff-aaaaaaaaaaa0")
        con.close()
        monkeypatch.chdir(root)
        for extra in ({}, {"full": "an audited reason"}, {"since": "aaaaaaaaaaa0"}):
            try:
                jaxflow_review.dispatch_diff_review(
                    _diff_args("b1", **extra), run=_run_with_tmux(fake, real_cwd=root),
                    post=lambda e: {"ok": True},
                    env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                    now=_fixed_now, allowlist_root=allow_root, db_path=db,
                )
            except ji.Refusal as exc:
                assert exc.code == "review-running"
            else:
                raise AssertionError("review-running not raised")
        lines = (root / ".local" / "runs" / "refusals.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        for line in lines:
            row = json.loads(line)
            assert row["reason"] == "review-running"
            assert row["verb"] == "review" and row["kind"] == "diff"
            assert row["target"] == "feat/x" and row["phase"] == "PHASE"
            assert row["builder_run_id"] == "b1" and row["prior_review_run_id"] == "aaaaaaaaaaa0"
            assert row["ts"]


def test_guard_approve_blocks_and_full_or_since_bypasses(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve", ts="2026-01-01T00:00:00+00:00")
        con.close()
        (jaxflow_common._manifest_dir(root, "aaaaaaaaaaa1")).mkdir(parents=True, exist_ok=True)
        (jaxflow_common._manifest_dir(root, "aaaaaaaaaaa1") / "manifest.json").write_text(
            json.dumps({"head_sha": head, "base_sha": head}), encoding="utf-8",
        )
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True},
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "prior-review-accepted"
        else:
            raise AssertionError("prior-review-accepted not raised")
        line = (root / ".local" / "runs" / "refusals.jsonl").read_text(encoding="utf-8").strip()
        row = json.loads(line)
        assert row == {
            "ts": row["ts"], "verb": "review", "kind": "diff", "reason": "prior-review-accepted",
            "target": "feat/x", "phase": "PHASE", "builder_run_id": "b1", "prior_review_run_id": "aaaaaaaaaaa1",
        }

        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1", full="planned re-review after an unrelated hotfix"),
            run=_run_with_tmux(FakeTmux(), real_cwd=root), post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["guard"]["override"] == "full"
        assert manifest["full_reason"] == "planned re-review after an unrelated hotfix"


def test_guard_approve_with_changes_blocks_same_as_approve(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve-with-changes")
        con.close()
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "prior-review-accepted"
        else:
            raise AssertionError("prior-review-accepted not raised")


def test_guard_latest_invalid_but_earlier_approve_still_applies(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve", ts="2026-01-01T00:00:00+00:00")
        _seed_diff_review(con, "aaaaaaaaaaa2", root, worktree, contract_status="invalid", ts="2026-02-01T00:00:00+00:00")
        con.close()
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "prior-review-accepted"
        else:
            raise AssertionError("prior-review-accepted not raised (the earlier approve must still apply)")


def test_guard_reject_proceeds_with_or_without_since(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="reject")
        con.close()
        (jaxflow_common._manifest_dir(root, "aaaaaaaaaaa1")).mkdir(parents=True, exist_ok=True)
        (jaxflow_common._manifest_dir(root, "aaaaaaaaaaa1") / "manifest.json").write_text(
            # "kind": "diff" (F1): a real dispatch_diff_review manifest always has it.
            json.dumps({"kind": "diff", "head_sha": head, "base_sha": head, "builder_run_id": "b1"}),
            encoding="utf-8",
        )
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["guard"]["override"] == "none"
        assert manifest["guard"]["prior_verdict"] == "reject"

        run_id2 = jaxflow_review.dispatch_diff_review(
            _diff_args("b1", since="aaaaaaaaaaa1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest2 = json.loads((root / ".local" / "runs" / run_id2 / "manifest.json").read_text())
        assert manifest2["guard"]["override"] == "since"


def test_guard_since_and_full_together_is_usage_error(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", since="aaaaaaaaaaa1", full="x"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "since-full-conflict"
        else:
            raise AssertionError("since-full-conflict not raised")


def test_guard_since_empty_string_and_full_together_is_still_usage_error():
    # cr 99ec95af61a2 F4: `since=""` is falsy but a real direct-call value -- the check
    # compares to `None`, not truthiness, and fires before any repo/DB access at all,
    # so no fixture setup is needed here.
    try:
        jaxflow_review.dispatch_diff_review(
            _diff_args("b1", since="", full="reason"), run=None, post=None, env={}, now=_fixed_now,
        )
    except ji.Refusal as exc:
        assert exc.code == "since-full-conflict"
    else:
        raise AssertionError("since-full-conflict not raised for since=''")


def test_guard_full_with_blank_reason_is_usage_error(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", full="   "), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "full-reason-blank"
        else:
            raise AssertionError("full-reason-blank not raised")


def test_guard_refusal_log_write_failure_still_raises_the_refusal(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        fake = FakeTmux()
        fake.existing.add("jax-demo-diff-aaaaaaaaaaa0")
        _seed_diff_review(con, "aaaaaaaaaaa0", root, worktree, finished=False, session="jax-demo-diff-aaaaaaaaaaa0")
        con.close()
        # strip write perm on the existing runs dir so `_log_refusal`'s os.open fails;
        # confirm the refusal still fires.
        runs_dir = root / ".local" / "runs"
        os.chmod(runs_dir, 0o500)
        monkeypatch.chdir(root)
        try:
            try:
                jaxflow_review.dispatch_diff_review(
                    _diff_args("b1"), run=_run_with_tmux(fake, real_cwd=root),
                    post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                    now=_fixed_now, allowlist_root=allow_root, db_path=db,
                )
            except ji.Refusal as exc:
                assert exc.code == "review-running"
            else:
                raise AssertionError("review-running not raised despite the unwritable refusals log dir")
            assert not (runs_dir / "refusals.jsonl").exists()
        finally:
            os.chmod(runs_dir, 0o700)  # TemporaryDirectory cleanup needs write back


def _write_diff_manifest(root, run_id, *, base_sha, head_sha, builder_run_id="b1",
                          since_review_run_id=None, kind="diff"):
    """`kind` defaults to "diff" -- a real `dispatch_diff_review` manifest always has it
    (F1's regression test overrides it to prove a mismatch refuses like a missing node)."""
    manifest_dir = root / ".local" / "runs" / run_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    data = {"base_sha": base_sha, "head_sha": head_sha, "builder_run_id": builder_run_id, "kind": kind}
    if since_review_run_id:
        data["since_review_run_id"] = since_review_run_id
    (manifest_dir / "manifest.json").write_text(json.dumps(data), encoding="utf-8")


def _commit(worktree, name):
    (worktree / name).write_text("x\n", encoding="utf-8")
    _git(worktree, "add", name)
    _git(worktree, "commit", "-m", name)
    return _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()


def test_since_root_only_success_narrows_range_and_records_manifest(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        merge_base = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        root_head = _commit(worktree, "c1.txt")
        new_head = _commit(worktree, "c2.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=new_head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve-with-changes")
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa1", base_sha=merge_base, head_sha=root_head)
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1", since="aaaaaaaaaaa1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["base_sha"] == root_head
        assert manifest["head_sha"] == new_head
        assert manifest["since_review_run_id"] == "aaaaaaaaaaa1"
        assert manifest["since_verdict"] == "approve-with-changes"


def test_since_delta_node_root_mismatch_but_continuity_holds_still_proceeds(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        merge_base = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        root_head = _commit(worktree, "c1.txt")
        delta_head = _commit(worktree, "c2.txt")
        new_head = _commit(worktree, "c3.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=new_head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve", ts="2026-01-01T00:00:00+00:00")
        _seed_diff_review(con, "aaaaaaaaaaa2", root, worktree, verdict="approve-with-changes", ts="2026-02-01T00:00:00+00:00")
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa1", base_sha=merge_base, head_sha=root_head)
        _write_diff_manifest(root, "aaaaaaaaaaa2", base_sha=root_head, head_sha=delta_head, since_review_run_id="aaaaaaaaaaa1")
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1", since="aaaaaaaaaaa2"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["base_sha"] == delta_head
        assert manifest["since_review_run_id"] == "aaaaaaaaaaa2"


def test_since_non_ancestor_head_refuses(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        merge_base = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        new_head = _commit(worktree, "c1.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=new_head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve")
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa1", base_sha=merge_base, head_sha="f" * 40)
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", since="aaaaaaaaaaa1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "since-not-ancestor"
        else:
            raise AssertionError("since-not-ancestor not raised")


def test_since_root_base_sha_drifted_from_current_merge_base_refuses(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        stale_merge_base = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        root_head = _commit(worktree, "c1.txt")
        _commit(root, "advance-main.txt")
        _git(worktree, "merge", "--no-ff", "-m", "merge main", "main")
        current_head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=current_head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve")
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa1", base_sha=stale_merge_base, head_sha=root_head)
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", since="aaaaaaaaaaa1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "since-root-stale"
        else:
            raise AssertionError("since-root-stale not raised")


def test_since_non_root_continuity_broken_refuses(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        merge_base = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        root_head = _commit(worktree, "c1.txt")
        stray = _commit(worktree, "c2.txt")
        delta_head = _commit(worktree, "c3.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=delta_head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve", ts="2026-01-01T00:00:00+00:00")
        _seed_diff_review(con, "aaaaaaaaaaa2", root, worktree, verdict="approve", ts="2026-02-01T00:00:00+00:00")
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa1", base_sha=merge_base, head_sha=root_head)
        _write_diff_manifest(root, "aaaaaaaaaaa2", base_sha=stray, head_sha=delta_head, since_review_run_id="aaaaaaaaaaa1")
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", since="aaaaaaaaaaa2"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "since-continuity-broken"
        else:
            raise AssertionError("since-continuity-broken not raised")


def test_since_node_ancestry_broken_refuses(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        root_head = _commit(worktree, "c1.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=root_head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve")
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa1", base_sha="f" * 40, head_sha=root_head)
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", since="aaaaaaaaaaa1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "since-ancestry-broken"
        else:
            raise AssertionError("since-ancestry-broken not raised")


def test_since_missing_intermediate_manifest_refuses(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        root_head = _commit(worktree, "c1.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=root_head)
        _seed_diff_review(con, "aaaaaaaaaaa2", root, worktree, verdict="approve")
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa2", base_sha=root_head, head_sha=root_head, since_review_run_id="aaaaaaaaaaa1")
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", since="aaaaaaaaaaa2"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "since-chain-missing"
        else:
            raise AssertionError("since-chain-missing not raised")


def test_since_cycle_in_chain_refuses(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        root_head = _commit(worktree, "c1.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=root_head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve")
        _seed_diff_review(con, "aaaaaaaaaaa2", root, worktree, verdict="approve")
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa1", base_sha=root_head, head_sha=root_head, since_review_run_id="aaaaaaaaaaa2")
        _write_diff_manifest(root, "aaaaaaaaaaa2", base_sha=root_head, head_sha=root_head, since_review_run_id="aaaaaaaaaaa1")
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", since="aaaaaaaaaaa1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "since-chain-cycle"
        else:
            raise AssertionError("since-chain-cycle not raised")


def test_since_reaches_root_within_bounded_depth(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        merge_base = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        shas = [merge_base] + [_commit(worktree, f"c{i}.txt") for i in range(5)]
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=shas[-1])
        ids = [f"aaaaaaaaaaa{i}" for i in range(1, 6)]
        for i, rid in enumerate(ids):
            _seed_diff_review(con, rid, root, worktree, verdict="approve-with-changes",
                             ts=f"2026-01-0{i+1}T00:00:00+00:00")
        con.close()
        _write_diff_manifest(root, ids[0], base_sha=shas[0], head_sha=shas[1])
        for i in range(1, 5):
            _write_diff_manifest(
                root, ids[i], base_sha=shas[i], head_sha=shas[i + 1], since_review_run_id=ids[i - 1],
            )
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1", since=ids[-1]), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["base_sha"] == shas[-1]


def test_since_naming_non_diff_run_refuses(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", since="bbbbbbbbbbbb"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "since-chain-missing"
        else:
            raise AssertionError("since-chain-missing not raised (b1 is a build, not a diff review)")


def test_since_manifest_kind_mismatch_refuses_since_chain_missing(monkeypatch):
    # F1 (diff review 4884f63bdd16): the LEDGER row says `kind: diff` (via
    # `_seed_diff_review`), but the run's OWN manifest says a different kind -- must
    # refuse exactly like a missing/unreadable node, not be accepted as the chain root.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        root_head = _commit(worktree, "c1.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=root_head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve")
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa1", base_sha=root_head, head_sha=root_head, kind="build")
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", since="aaaaaaaaaaa1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "since-chain-missing"
        else:
            raise AssertionError("since-chain-missing not raised (manifest kind mismatch)")


def test_since_naming_non_terminal_prior_refuses(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        root_head = _commit(worktree, "c1.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=root_head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, contract_status="invalid")
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa1", base_sha=root_head, head_sha=root_head)
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", since="aaaaaaaaaaa1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "since-not-terminal"
        else:
            raise AssertionError("since-not-terminal not raised")


def test_since_naming_other_branch_target_and_lineage_mismatch_refuse(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        root_head = _commit(worktree, "c1.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=root_head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, target="feat/other", verdict="approve")
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa1", base_sha=root_head, head_sha=root_head)
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", since="aaaaaaaaaaa1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "since-lineage-mismatch"
        else:
            raise AssertionError("since-lineage-mismatch not raised (target mismatch)")

        con2 = sqlite3.connect(db)
        con2.row_factory = sqlite3.Row
        _insert(con2, "b2", "demo", "builder", "run-started", {
            "phase": "PHASE", "runtime": "opencode-grok", "kind": "build", "target": "feat/other",
            "caller": "claude", "caller_session": "s", "model": "xai/grok-4.6", "effort": "n/a",
            "session": "jax-demo-build-b2", "repo": str(root), "verify": "true",
        })
        _seed_diff_review(con2, "aaaaaaaaaaa2", root, worktree, target="feat/x", verdict="approve")
        con2.close()
        _write_diff_manifest(root, "aaaaaaaaaaa2", base_sha=root_head, head_sha=root_head, builder_run_id="b2")
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", since="aaaaaaaaaaa2"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "since-lineage-mismatch"
        else:
            raise AssertionError("since-lineage-mismatch not raised (builder target mismatch)")


def test_since_path_traversal_and_symlink_escape_refuse(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        monkeypatch.chdir(root)
        runs_dir_before = (root / ".local" / "runs")
        existed_before = set(runs_dir_before.iterdir()) if runs_dir_before.is_dir() else set()
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", since="../x"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "since-invalid"
        else:
            raise AssertionError("since-invalid not raised")
        after = set(runs_dir_before.iterdir()) if runs_dir_before.is_dir() else set()
        assert after - existed_before == set()

        outside = Path(raw) / "outside-secret"
        outside.mkdir()
        (outside / "manifest.json").write_text(
            json.dumps({"base_sha": head, "head_sha": head, "builder_run_id": "b1"}), encoding="utf-8",
        )
        runs_dir = root / ".local" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        escape_id = "a" * 12
        os.symlink(outside, runs_dir / escape_id)
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        _seed_diff_review(con, escape_id, root, worktree, verdict="approve")
        con.close()
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", since=escape_id), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "since-chain-missing"
        else:
            raise AssertionError("symlinked manifest dir escape not refused")


@pytest.mark.parametrize("bad_field,bad_value", [
    ("since_review_run_id", 123),  # non-string chain link -- crashes in _safe_run_subpath
    ("head_sha", 123),  # non-string SHA -- crashes building the `git merge-base` argv
])
def test_since_malformed_chain_data_refuses_since_chain_missing_never_crashes(monkeypatch, bad_field, bad_value):
    # F3 (diff review 4884f63bdd16): malformed chain metadata (a non-string link id or
    # SHA) must refuse with the SAME code as a broken/unreadable chain node, never raise
    # a raw TypeError/AttributeError out of dispatch_diff_review.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        root_head = _commit(worktree, "c1.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=root_head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve")
        con.close()
        kwargs = {"base_sha": root_head, "head_sha": root_head, bad_field: bad_value}
        _write_diff_manifest(root, "aaaaaaaaaaa1", **kwargs)
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1", since="aaaaaaaaaaa1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "since-chain-missing"
        else:
            raise AssertionError(f"since-chain-missing not raised for malformed {bad_field}={bad_value!r}")


def test_guard_bypasses_terminal_verdict_only_never_the_running_lock(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        merge_base = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        root_head = _commit(worktree, "c1.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=root_head)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve-with-changes")
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa1", base_sha=merge_base, head_sha=root_head)
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1", full="approved bypass"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        assert json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())["guard"]["override"] == "full"

        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        fake = FakeTmux()
        fake.existing.add("jax-demo-diff-aaaaaaaaaaa2")
        _seed_diff_review(con, "aaaaaaaaaaa2", root, worktree, finished=False, session="jax-demo-diff-aaaaaaaaaaa2")
        con.close()
        for extra in ({"full": "x"}, {"since": "aaaaaaaaaaa1"}):
            try:
                jaxflow_review.dispatch_diff_review(
                    _diff_args("b1", **extra), run=_run_with_tmux(fake, real_cwd=root),
                    post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                    now=_fixed_now, allowlist_root=allow_root, db_path=db,
                )
            except ji.Refusal as exc:
                assert exc.code == "review-running"
            else:
                raise AssertionError("review-running not raised despite --full/--since")


def test_chain_block_single_node_and_three_node_render_and_broken_chain(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        merge_base = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        root_head = _commit(worktree, "c1.txt")
        delta_head = _commit(worktree, "c2.txt")
        delta2_head = _commit(worktree, "c3.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        (root / ".local" / "reports").mkdir(parents=True, exist_ok=True)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=delta2_head)

        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve", ts="2026-01-01T00:00:00+00:00")
        _write_diff_manifest(root, "aaaaaaaaaaa1", base_sha=merge_base, head_sha=root_head)
        assert jaxflow_cli.cmd_status("aaaaaaaaaaa1", run=_run_with_tmux(FakeTmux()), db_path=db) == (
            f"chain:\n  aaaaaaaaaaa1 {merge_base[:12]}..{root_head[:12]} approve\nfinished(approve)"
        )

        _seed_diff_review(con, "aaaaaaaaaaa2", root, worktree, verdict="approve-with-changes", ts="2026-02-01T00:00:00+00:00")
        _write_diff_manifest(root, "aaaaaaaaaaa2", base_sha=root_head, head_sha=delta_head, since_review_run_id="aaaaaaaaaaa1")
        _seed_diff_review(con, "aaaaaaaaaaa3", root, worktree, verdict="reject", ts="2026-03-01T00:00:00+00:00")
        _write_diff_manifest(root, "aaaaaaaaaaa3", base_sha=delta_head, head_sha=delta2_head, since_review_run_id="aaaaaaaaaaa2")
        con.close()
        status = jaxflow_cli.cmd_status("aaaaaaaaaaa3", run=_run_with_tmux(FakeTmux()), db_path=db)
        expected = "\n".join([
            "chain:",
            f"  aaaaaaaaaaa1 {merge_base[:12]}..{root_head[:12]} approve",
            f"  aaaaaaaaaaa2 {root_head[:12]}..{delta_head[:12]} approve-with-changes [since aaaaaaaaaaa1]",
            f"  aaaaaaaaaaa3 {delta_head[:12]}..{delta2_head[:12]} reject [since aaaaaaaaaaa2]",
            "finished(reject)",
        ])
        assert status == expected

        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        _seed_diff_review(con, "aaaaaaaaaaa4", root, worktree, verdict="approve")
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa4", base_sha=delta2_head, head_sha=delta2_head, since_review_run_id="deadbeef0000")
        status4 = jaxflow_cli.cmd_status("aaaaaaaaaaa4", run=_run_with_tmux(FakeTmux()), db_path=db)
        assert status4 == "chain: broken at deadbeef0000 (missing)\nfinished(approve)"


def test_chain_block_missing_head_sha_renders_broken_not_raise(monkeypatch):
    # F7: a manifest that parses but is MISSING `head_sha`/`base_sha` must never crash the display path.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve")
        con.close()
        manifest_dir = root / ".local" / "runs" / "aaaaaaaaaaa1"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "manifest.json").write_text(
            # "kind": "diff" (F1): a real manifest always has it -- this fixture is
            # exercising the MISSING head_sha field, not a kind mismatch.
            json.dumps({"kind": "diff", "base_sha": head, "builder_run_id": "b1"}), encoding="utf-8",  # no head_sha key
        )
        status = jaxflow_cli.cmd_status("aaaaaaaaaaa1", run=_run_with_tmux(FakeTmux()), db_path=db)
        assert status == "chain: broken at aaaaaaaaaaa1 (missing-sha)\nfinished(approve)"


def test_chain_block_invalid_manifest_json_renders_broken_not_raise(monkeypatch):
    # cr F7: invalid manifest JSON must never raise out of cmd_status/cmd_result --
    # `_load_diff_review_node` swallows it and returns None, walk reports "missing".
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve")
        con.close()
        manifest_dir = root / ".local" / "runs" / "aaaaaaaaaaa1"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "manifest.json").write_text("{not valid json", encoding="utf-8")
        status = jaxflow_cli.cmd_status("aaaaaaaaaaa1", run=_run_with_tmux(FakeTmux()), db_path=db)
        assert status == "chain: broken at aaaaaaaaaaa1 (missing)\nfinished(approve)"


def test_chain_block_malformed_data_renders_broken_malformed_not_raise():
    # cr 99ec95af61a2 F3: the ONE outer guard in `_chain_block` catches every crash
    # shape below as "chain: broken (malformed)". (a) malformed ledger JSON payload,
    # (b) non-string since_review_run_id reaching _RUN_ID_RE.match, (c) non-string
    # head_sha sliced during rendering -- one test, not three, per the ponytail guard.
    with TemporaryDirectory() as raw:
        root = Path(raw) / "demo"
        root.mkdir(parents=True)
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)

        # (a) malformed ledger JSON payload on the run-started row itself.
        con.execute(
            "INSERT INTO workflow_events (ts, run_id, project, role, type, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("t", "aaaaaaaaaaa1", "demo", "reviewer", "run-started", "{not valid json"),
        )
        con.commit()
        assert jaxflow_cli._chain_block("aaaaaaaaaaa1", db_path=db) == "chain: broken (malformed)"

        # (b) non-string since_review_run_id in an otherwise well-formed manifest.
        # "kind": "diff" (F1): a real manifest always has it -- this fixture is
        # exercising the non-string since_review_run_id, not a kind mismatch.
        _seed_diff_review(con, "aaaaaaaaaaa2", root, None, verdict="approve")
        (root / ".local" / "runs" / "aaaaaaaaaaa2" / "manifest.json").write_text(
            json.dumps({
                "kind": "diff", "base_sha": "f" * 40, "head_sha": "e" * 40, "since_review_run_id": 123,
            }),
            encoding="utf-8",
        )
        assert jaxflow_cli._chain_block("aaaaaaaaaaa2", db_path=db) == "chain: broken (malformed)"

        # (c) non-string head_sha in an otherwise well-formed root manifest.
        _seed_diff_review(con, "aaaaaaaaaaa3", root, None, verdict="approve")
        (root / ".local" / "runs" / "aaaaaaaaaaa3" / "manifest.json").write_text(
            json.dumps({"kind": "diff", "base_sha": "f" * 40, "head_sha": 12345}), encoding="utf-8",
        )
        con.close()
        assert jaxflow_cli._chain_block("aaaaaaaaaaa3", db_path=db) == "chain: broken (malformed)"


def test_chain_block_absent_for_non_diff_runs(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        root.mkdir(parents=True)
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _insert(con, "s1", "demo", "reviewer", "run-started", {"session": "jax-demo-spec-s1"})
        _insert(con, "s1", "demo", "reviewer", "run-finished", {"contract_status": "ok"})
        con.close()
        assert jaxflow_cli.cmd_status("s1", run=_run_with_tmux(FakeTmux()), db_path=db) == "finished(no verdict)"


def test_result_shows_chain_block_for_a_diff_review():
    # cmd_status above exercises _chain_block's rendering; this is the dedicated
    # regression for cmd_result's own wiring of it (item 9 acceptance criterion).
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        merge_base = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        root_head = _commit(worktree, "c1.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        (root / ".local" / "reports").mkdir(parents=True, exist_ok=True)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve")
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa1", base_sha=merge_base, head_sha=root_head)
        report = root / ".local" / "reports" / "aaaaaaaaaaa1.md"
        report.write_text("REPORT BODY\n", encoding="utf-8")
        out = jaxflow_cli.cmd_result("aaaaaaaaaaa1", allowlist_root=allow_root, db_path=db)
        expected_chain = f"chain:\n  aaaaaaaaaaa1 {merge_base[:12]}..{root_head[:12]} approve"
        assert out == f"{expected_chain}\n{report}\nREPORT BODY\n"


def test_diff_dispatch_refuses_verify_failed_when_only_the_build_command_fails(monkeypatch):
    """A failure of EITHER command refuses, before any reviewer runtime is dispatched --
    and both frames are still written, so the tech lead can see which half broke."""
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", build="false", head_sha=head)
        con.close()
        monkeypatch.chdir(root)
        fake = FakeTmux()
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1"), run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "verify-failed"
        else:
            raise AssertionError("verify-failed not raised")
        assert fake.calls == []  # zero reviewer-runtime invocations
        text = (worktree / ".local" / "reports" / "b1.tests.txt").read_text(encoding="utf-8")
        assert text.count("COMMAND: ") == 2
        assert "EXIT: 1\n" in text


def test_diff_dispatch_refuses_when_the_declared_spec_fragment_heading_is_missing(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        spec_src = root / ".local" / "docs" / "specs" / "spec.md"
        spec_src.parent.mkdir(parents=True)
        spec_src.write_text("# Spec\n\n## Real Heading\ntext\n", encoding="utf-8")
        plan_path = root / ".local" / "docs" / "plans" / "plan.md"
        plan_path.parent.mkdir(parents=True)
        plan_path.write_text(
            f"# Plan\n\n**Goal:** g\n\n**Spec:** `{spec_src}#No Such Heading`\n\n"
            "### Task 1: t\n",
            encoding="utf-8",
        )
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head, plan_path=plan_path)
        con.close()
        monkeypatch.chdir(root)
        fake = FakeTmux()
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1"), run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "spec-fragment-not-found"
        else:
            raise AssertionError("spec-fragment-not-found not raised")
        assert fake.calls == []


def test_worker_diff_review_prompt_carries_the_spec_fragment_when_the_plan_declares_one():
    # Regression for the Path-vs-str comparison hazard at `scripts/jaxflow_review.py` (diff-review worker) --
    # under the OLD `if spec_path != derived_spec:` this fails with path-outside-allowlist
    # even though every value agrees, because spec_path is always a Path and derived_spec
    # is a str whenever a fragment is present.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        (worktree / "changed.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "change")
        base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        spec_src = root / ".local" / "docs" / "specs" / "spec.md"
        spec_src.parent.mkdir(parents=True)
        spec_src.write_text("# Spec\n\n## Heading A\ntext\n", encoding="utf-8")
        plan_src = root / ".local" / "docs" / "plans" / "plan.md"
        plan_src.parent.mkdir(parents=True)
        plan_src.write_text(
            f"# Plan\n\n**Goal:** g\n\n**Spec:** `{spec_src}#Heading A`\n\n### Task 1: t\n",
            encoding="utf-8",
        )
        tests_path = worktree / ".local" / "reports" / "b1b1b1b1b1b1.tests.txt"
        tests_path.parent.mkdir(parents=True)
        tests_path.write_text("COMMAND: true\n\nEXIT: 0\n", encoding="utf-8")
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, base_sha=base_sha, head_sha=head_sha,
            plan_path=str(plan_src), spec_path=f"{spec_src}#Heading A", tests_path=str(tests_path),
        )
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["stdin_bytes"] = self.stdin_bytes

        code = jaxflow_worker.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
        )
        assert code == 0
        prompt = captured["stdin_bytes"].decode("utf-8")
        assert f"  spec: {spec_src}#Heading A" in prompt


def test_diff_dispatch_refuses_unknown_run(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        monkeypatch.chdir(root)
        db = Path(raw) / "jaxos.db"
        _fresh_db(db).close()
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("nope"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "unknown-run"
            # fixes cold review round 2 F4: §4.3 requires a second stderr line naming WHY.
            # fixes Part 2 diff-review F3: the hint now comes from a separate unfiltered
            # lookup and names the run id explicitly -- no `run-started` row at all here.
            assert "no run-started row for nope" in exc.hint
        else:
            raise AssertionError("unknown-run not raised")


def test_diff_dispatch_refuses_non_build_run_and_unfinished_build(monkeypatch):
    # fixes cold review F1: a spec/plan review's own run_id, or a build that has not
    # finished yet, must refuse exactly like an unknown run -- not crash on a missing
    # "verify" key, and not review a diff that is not actually final.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        monkeypatch.chdir(root)
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _insert(con, "s1", "demo", "reviewer", "run-started", {
            "phase": "P", "runtime": "codex", "kind": "spec", "target": str(root / "spec.md"),
            "caller": "claude", "caller_session": "s", "model": "m", "effort": "e",
            "session": "jax-demo-spec-s1", "repo": str(root),
        })
        _insert(con, "b1", "demo", "builder", "run-started", {
            "phase": "PHASE", "runtime": "opencode-grok", "kind": "build", "target": "feat/x",
            "caller": "claude", "caller_session": "s", "model": "xai/grok-4.6", "effort": "n/a",
            "session": "jax-demo-build-b1", "repo": str(root), "verify": "true",
        })  # no matching run-finished row -- still running.
        con.close()
        # fixes cold review round 2 F4: each case's own reason, not one shared message.
        for run_id, expect, hint_substr in (
            ("s1", "unknown-run", "not a build"),
            ("b1", "unknown-run", "not finished"),
        ):
            try:
                jaxflow_review.dispatch_diff_review(
                    _diff_args(run_id), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                    post=lambda e: {"ok": True},
                    env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                    now=_fixed_now, allowlist_root=allow_root, db_path=db,
                )
            except ji.Refusal as exc:
                assert exc.code == expect, run_id
                assert hint_substr in exc.hint, run_id
            else:
                raise AssertionError(f"{expect} not raised for {run_id}")


def test_diff_dispatch_refuses_mixed_role_run_id_as_unknown_run(monkeypatch):
    # fixes Part 2 diff-review F3: a reviewer's own `run-started` row -- forged with
    # `kind: build` -- paired with an UNRELATED builder `run-finished` row sharing the
    # same run id, must never be accepted as a build. `started` is now filtered to
    # `role = 'builder'`, so this mixed pair refuses `unknown-run` exactly like a
    # completely absent run, instead of being treated as a finished build.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        monkeypatch.chdir(root)
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _insert(con, "mix1", "demo", "reviewer", "run-started", {
            "phase": "P", "runtime": "codex", "kind": "build", "target": "feat/x",
            "caller": "claude", "caller_session": "s", "model": "m", "effort": "e",
            "session": "jax-demo-diff-mix1", "repo": str(root),
        })
        _insert(con, "mix1", "demo", "builder", "run-finished", {
            "phase": "PHASE", "exit_code": 0, "contract_status": "ok",
            "report_path": str(worktree / ".local" / "reports" / "mix1.md"),
            "summary": "built the thing", "head_sha": "a" * 40, "result": "success",
        })
        con.close()
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("mix1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "unknown-run"
            assert "run mix1 is a reviewer run, not a build" in exc.hint
        else:
            raise AssertionError("unknown-run not raised for a mixed-role run id")


def test_diff_dispatch_refuses_missing_worktree(monkeypatch):
    # fixes cold review F2: a removed/never-created worktree must refuse cleanly, not
    # raise an uncaught FileNotFoundError from the verify subprocess.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        monkeypatch.chdir(root)
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        # a finished build whose worktree was never created / already removed.
        _seed_finished_build(con, "b1", root, allow_root / "demo-feat-x", verify="true")
        con.close()
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "worktree-missing"
        else:
            raise AssertionError("worktree-missing not raised")


def test_diff_dispatch_ignores_builder_report_path_and_derives_tests_path_from_branch(monkeypatch):
    # fixes cold review round 2 F6: seeds a deliberately WRONG/outside report_path in the
    # builder's own run-finished row -- a wrong implementation that read and reused that
    # path (instead of deriving tests_path from the branch/worktree, spec §2.6) would still
    # pass every other test above, since they all seed a valid report_path.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _insert(con, "b1", "demo", "builder", "run-started", {
            "phase": "PHASE", "runtime": "opencode-grok", "kind": "build", "target": "feat/x",
            "caller": "claude", "caller_session": "s", "model": "xai/grok-4.6", "effort": "n/a",
            "session": "jax-demo-build-b1", "repo": str(root), "verify": "true",
        })
        _insert(con, "b1", "demo", "builder", "run-finished", {
            "phase": "PHASE", "exit_code": 0, "contract_status": "ok",
            "report_path": "/outside/not-the-worktree/report.md",  # deliberately wrong
            "summary": "built the thing", "head_sha": head, "result": "success",
        })
        con.close()
        # fixes S2: this test seeds the builder's DB rows by hand (not via
        # `_seed_finished_build`), so it also needs the build's own manifest.json --
        # `review --diff` now reads it for `plan_path`/`spec_path`.
        _write_plan(worktree)
        plan_path = worktree / ".local" / "docs" / "plans" / "plan.md"
        manifest_dir = root / ".local" / "runs" / "b1"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "manifest.json").write_text(
            json.dumps({"plan_path": str(plan_path)}), encoding="utf-8",
        )
        monkeypatch.chdir(root)
        fake = FakeTmux()
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        assert run_id
        # tests_path derived from the WORKTREE, never from the (ignored, wrong) report_path.
        tests_file = worktree / ".local" / "reports" / "b1.tests.txt"
        assert tests_file.is_file()
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["tests_path"] == str(tests_file)


def test_diff_dispatch_verify_failed_refuses_before_any_reviewer_dispatch(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="false", head_sha=head)
        con.close()
        monkeypatch.chdir(root)
        fake = FakeTmux()
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1"), run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "verify-failed"
        else:
            raise AssertionError("verify-failed not raised")
        # zero reviewer-runtime invocations -- no tmux new-session, no post at all.
        assert fake.calls == []
        # the FRESH re-run's own tests.txt is still written on a failure (spec §4.3 step
        # 2/3 — "fresher... since it was just re-run" carries no pass/fail qualifier).
        tests_file = worktree / ".local" / "reports" / "b1.tests.txt"
        assert "COMMAND: false" in tests_file.read_text(encoding="utf-8")
        assert "EXIT: 1" in tests_file.read_text(encoding="utf-8")


def test_diff_dispatch_ancestor_head_gate_pass_and_reject(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        base_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        # a fix commit on top -- the ancestor gate (spec §2.10 #3) must still pass.
        (worktree / "extra.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "extra.txt")
        _git(worktree, "commit", "-m", "fix commit")
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()

        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=base_sha)
        con.close()
        monkeypatch.chdir(root)
        fake = FakeTmux()
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
            allowlist_root=allow_root, db_path=db,
        )
        payload = events[0]["payload"]
        assert payload["kind"] == "diff"
        assert payload["phase"] == "PHASE"  # inherited from the builder run (spec §2.1 gap, this plan's header)
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["base_sha"] == base_sha
        assert manifest["head_sha"] == head_sha
        assert manifest["builder_run_id"] == "b1"
        assert manifest["worktree"] == str(worktree)
        # fixes cold review F7: the successful (passing) dispatch path also freshly
        # rewrote tests.txt -- not just the failure path (already covered above).
        tests_file = worktree / ".local" / "reports" / "b1.tests.txt"
        assert "COMMAND: true" in tests_file.read_text(encoding="utf-8")
        assert "EXIT: 0" in tests_file.read_text(encoding="utf-8")

        # a HEAD that is NOT a descendant of the recorded head_sha refuses, unmapped,
        # with the library's own message (spec §7.1 test #17).
        db2 = Path(raw) / "jaxos2.db"
        con2 = _fresh_db(db2)
        _seed_finished_build(con2, "b2", root, worktree, verify="true", head_sha="f" * 40)
        con2.close()
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b2"), run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db2,
            )
        except ji.Refusal as exc:
            assert exc.code == "reviewer head mismatch"
        else:
            raise AssertionError("reviewer head mismatch not raised")
        # fixes Part 2 diff-review F5: this refusal fires AFTER verify already passed and
        # wrote FRESH evidence for "b2" -- since none existed before, that new file is
        # removed entirely rather than left behind unlogged.
        assert not (worktree / ".local" / "reports" / "b2.tests.txt").exists()


def test_diff_dispatch_caller_session_missing_restores_preexisting_evidence(monkeypatch):
    # fixes Part 2 diff-review F5: `caller-session-missing` fires AFTER the verify re-run
    # already overwrote `.tests.txt` (verify itself PASSED here) -- the pre-existing
    # evidence must be restored byte-for-byte, never left as the freshly-verified (but
    # never-logged, since dispatch never happened) content.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        tests_file = worktree / ".local" / "reports" / "b1.tests.txt"
        tests_file.parent.mkdir(parents=True, exist_ok=True)
        old_evidence = "COMMAND: old-verify\nstale output from a previous round\nEXIT: 0\n"
        tests_file.write_text(old_evidence, encoding="utf-8")
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1"},  # no CLAUDE_CODE_SESSION_ID
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "caller-session-missing"
        else:
            raise AssertionError("caller-session-missing not raised")
        assert tests_file.read_text(encoding="utf-8") == old_evidence


def test_diff_dispatch_refuses_symlinked_evidence_path_before_any_verify_or_read(monkeypatch):
    # fixes F3 (HIGH, NEW): a builder-planted symlink at the evidence path must be
    # refused BEFORE it is ever snapshotted, verified through, or restored -- the fake
    # `run` asserts the verify command is never even invoked, and the outside target the
    # link points at is proven untouched.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        tests_file = worktree / ".local" / "reports" / "b1.tests.txt"
        tests_file.parent.mkdir(parents=True, exist_ok=True)
        outside = Path(raw).resolve() / "outside-evidence.txt"
        outside.write_text("attacker-controlled\n", encoding="utf-8")
        tests_file.symlink_to(outside)

        real_run = _run_with_tmux(FakeTmux(), real_cwd=root)

        def run_and_forbid_verify(argv, cwd=None):
            if argv[:1] == ["/bin/sh"]:
                raise AssertionError("verify must never run for a symlinked evidence path")
            return real_run(argv, cwd)

        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1"), run=run_and_forbid_verify,
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "path-outside-allowlist"
            assert "symlink" in exc.hint
        else:
            raise AssertionError("path-outside-allowlist not raised")
        assert tests_file.is_symlink()
        assert outside.read_text(encoding="utf-8") == "attacker-controlled\n"


def test_diff_dispatch_refuses_when_evidence_write_races_a_symlink_after_verify(monkeypatch):
    # fixes F3: a False return from _write_verify_tests_file (a symlink raced in between
    # the upfront guard and the write itself, simulated here via the injected `run`)
    # refuses the same way -- and the restore never follows the now-symlinked path.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        tests_file = worktree / ".local" / "reports" / "b1.tests.txt"
        tests_file.parent.mkdir(parents=True, exist_ok=True)
        old_evidence = "COMMAND: old-verify\nstale output\nEXIT: 0\n"
        tests_file.write_text(old_evidence, encoding="utf-8")
        outside = Path(raw).resolve() / "raced-evidence.txt"

        real_run = _run_with_tmux(FakeTmux(), real_cwd=root)

        def racing_run(argv, cwd=None):
            result = real_run(argv, cwd)
            if argv[:1] == ["/bin/sh"]:
                # An attacker swaps the file for a symlink the instant after verify runs,
                # before jaxflow's own evidence write.
                tests_file.unlink()
                tests_file.symlink_to(outside)
            return result

        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1"), run=racing_run,
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "path-outside-allowlist"
        else:
            raise AssertionError("path-outside-allowlist not raised")
        # the symlink is left exactly as the race planted it -- restore never follows it.
        assert tests_file.is_symlink()


# ---- review --diff: the handoff names the build's own plan/spec (fixes S2, real smoke
# b12a3db3da4b: a diff handoff with no plan/spec lines makes the reviewer reject
# `handoff:spec` -- required inputs missing) ----

def test_diff_dispatch_missing_builder_manifest_refuses_before_verify(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        # DB rows only -- deliberately no `.local/runs/b1/manifest.json` on the control
        # repo side (matches the shape a hand-crafted/forged run id or a since-cleaned-up
        # build manifest would leave behind).
        _insert(con, "b1", "demo", "builder", "run-started", {
            "phase": "PHASE", "runtime": "opencode-grok", "kind": "build", "target": "feat/x",
            "caller": "claude", "caller_session": "s", "model": "xai/grok-4.6", "effort": "n/a",
            "session": "jax-demo-build-b1", "repo": str(root), "verify": "true",
        })
        _insert(con, "b1", "demo", "builder", "run-finished", {
            "phase": "PHASE", "exit_code": 0, "contract_status": "ok",
            "report_path": str(worktree / ".local" / "reports" / "b1.md"),
            "summary": "built the thing", "head_sha": head, "result": "success",
        })
        con.close()
        monkeypatch.chdir(root)
        log = []
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1"), run=_run_with_tmux_and_log(FakeTmux(), log, real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "unknown-run"
            manifest_path = root / ".local" / "runs" / "b1" / "manifest.json"
            assert f"build manifest for b1 missing or invalid: {manifest_path}" in exc.hint
        else:
            raise AssertionError("unknown-run not raised")
        # zero verify invocations, and `.tests.txt` is never even touched for a build
        # manifest this dispatch cannot use.
        assert not any(argv[:2] == ["/bin/sh", "-c"] for argv in log)
        assert not (worktree / ".local" / "reports" / "b1.tests.txt").exists()


def test_diff_dispatch_missing_plan_path_in_builder_manifest_refuses_unknown_run(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        # Corrupts the build's own manifest -- valid JSON, but no usable plan_path.
        manifest_path = root / ".local" / "runs" / "b1" / "manifest.json"
        manifest_path.write_text(json.dumps({"plan_path": None}), encoding="utf-8")
        monkeypatch.chdir(root)
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "unknown-run"
            assert f"build manifest for b1 missing or invalid: {manifest_path}" in exc.hint
        else:
            raise AssertionError("unknown-run not raised")


def test_diff_dispatch_handoff_stub_and_manifest_name_plan_before_test_output(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        monkeypatch.chdir(root)

        captured = {}
        real_preflight = jr.preflight

        def spy_preflight(args_ns, **kw):
            if args_ns.prompt_file:
                captured["stub"] = Path(args_ns.prompt_file).read_text(encoding="utf-8")
            return real_preflight(args_ns, **kw)

        monkeypatch.setattr(jr, "preflight", spy_preflight)

        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        plan_path = worktree / ".local" / "docs" / "plans" / "plan.md"
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["plan_path"] == str(plan_path)
        # the default plan (`_write_plan`'s own text) has no **Spec:** line -- falls back
        # to the plan copy itself for BOTH paths (§4.3 amended, matching `build`'s own
        # `_resolve_handoff_spec_path` fallback).
        assert manifest["spec_path"] == str(plan_path)

        stub = captured["stub"]
        spec_idx = stub.index("  spec:")
        plan_idx = stub.index("  plan:")
        test_idx = stub.index("  test-output:")
        assert spec_idx < plan_idx < test_idx
        assert stub.splitlines()[stub.count("\n", 0, spec_idx)] == f"  spec: {plan_path}"
        assert stub.splitlines()[stub.count("\n", 0, plan_idx)] == f"  plan: {plan_path}"


def test_diff_dispatch_original_plan_flows_into_diff_manifest_without_copies(monkeypatch):
    # MOA-467: a NEW-style build manifest stores the ORIGINAL plan path (control repo),
    # and a declared **Spec:** is resolved to its ORIGINAL too -- the diff manifest names
    # those originals and copies nothing into the worktree. (The legacy copied-path
    # shape is covered by the tests above and below, which seed plan_path inside the
    # worktree.)
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        spec_src = root / ".local" / "docs" / "specs" / "the-spec.md"
        spec_src.parent.mkdir(parents=True)
        spec_src.write_text("# The spec\n", encoding="utf-8")
        plan_src = root / ".local" / "docs" / "plans" / "plan.md"
        plan_src.parent.mkdir(parents=True)
        plan_src.write_text(
            f"# Ship the thing\n\n**Goal:** make it spin.\n\n**Spec:** `{spec_src}`\n\n"
            "### Task 1: do it\n", encoding="utf-8",
        )
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head, plan_path=plan_src)
        con.close()
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["plan_path"] == str(plan_src)
        assert manifest["spec_path"] == str(spec_src)
        assert not (worktree / ".local" / "docs" / "specs").exists()
        assert not (worktree / ".local" / "docs" / "plans").exists()


def test_diff_dispatch_resolves_declared_spec_original_and_refuses_unresolvable_declaration(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()

        spec_src_dir = allow_root / "other"
        spec_src_dir.mkdir(parents=True, exist_ok=True)
        spec_src = spec_src_dir / "the-spec.md"
        spec_src.write_text("# The spec\n", encoding="utf-8")
        _write_plan(worktree, text=(
            f"# Ship the thing\n\n**Goal:** make it spin.\n\n**Spec:** `{spec_src}`\n\n"
            "### Task 1: do it\n"
        ))
        plan_path = worktree / ".local" / "docs" / "plans" / "plan.md"
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head, plan_path=plan_path)
        con.close()
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        # MOA-467: the spec is resolved to its ORIGINAL validated file, never copied.
        assert manifest["spec_path"] == str(spec_src)
        assert not (worktree / ".local" / "docs" / "specs").exists()

        # A declared **Spec:** that does not resolve to a real file refuses loudly
        # (`_resolve_handoff_spec_path`'s own rule, reused unchanged here) instead of
        # silently falling back to the plan.
        _write_plan(worktree, text=(
            "# Ship the thing\n\n**Goal:** make it spin.\n\n"
            f"**Spec:** `{allow_root / 'other' / 'missing-spec.md'}`\n\n### Task 1: do it\n"
        ))
        db2 = Path(raw) / "jaxos2.db"
        con2 = _fresh_db(db2)
        _seed_finished_build(con2, "b2", root, worktree, verify="true", head_sha=head, plan_path=plan_path)
        con2.close()
        try:
            jaxflow_review.dispatch_diff_review(
                _diff_args("b2"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db2,
            )
        except ji.Refusal as exc:
            assert exc.code == "spec-reference-invalid"
        else:
            raise AssertionError("spec-reference-invalid not raised")


# ---------------------------------------------------------------- threat model (D28)
# Blueprint decision D28: AGENTS.md declares a threat model once; `jaxflow review` injects
# one line into every reviewer handoff instead of the tech lead re-typing `--focus`.

_THREAT_MODEL_BLOCK = (
    "## Threat model  <!-- CHOOSE ONE MODE -->\n"
    "mode: internal-single-user\n"
    "<!-- internal-single-user: a single user on a VPS behind Tailscale. Only the owner or the\n"
    "     agents he dispatched write files and the DB. A hostile local process is OUT of scope.\n"
    "     Path traversal, symlink escape outside the allowlist and secret leaks stay IN scope.\n"
    "     public-app: anything reached by users you do not control (app with sign-ups, public\n"
    "     API). Full scope: the whole OWASP surface. -->\n"
)


_NO_THREAT_MODEL_NOTE = "jaxflow: no threat model in AGENTS.md — reviewer gets no threat-model line"


def test_parse_threat_model_reads_both_known_modes():
    assert jaxflow_review.parse_threat_model(_THREAT_MODEL_BLOCK) == "internal-single-user"
    public_text = _THREAT_MODEL_BLOCK.replace("internal-single-user", "public-app")
    assert jaxflow_review.parse_threat_model(public_text) == "public-app"


def test_parse_threat_model_unknown_mode_is_none():
    assert jaxflow_review.parse_threat_model("## Threat model\nmode: something-else\n") is None


def test_parse_threat_model_missing_section_is_none():
    assert jaxflow_review.parse_threat_model("# AGENTS.md\n\nNo threat model here.\n") is None
    assert jaxflow_review.parse_threat_model("") is None


def test_parse_threat_model_heading_with_trailing_comment_case_insensitive():
    text = "## THREAT MODEL  <!-- CHOOSE ONE MODE -->\nmode: public-app\n"
    assert jaxflow_review.parse_threat_model(text) == "public-app"


def test_parse_threat_model_mode_line_surrounding_spaces():
    text = "## Threat model\n   mode:    internal-single-user   \n"
    assert jaxflow_review.parse_threat_model(text) == "internal-single-user"


def test_parse_threat_model_stops_at_the_next_heading():
    # A `mode:` line belonging to a LATER section must not be borrowed (same idea as
    # `_resolve_delivery_target`'s Deploy policy block boundary).
    text = "## Threat model\n\n## Another section\nmode: public-app\n"
    assert jaxflow_review.parse_threat_model(text) is None


def test_threat_model_line_exact_text():
    assert jaxflow_workerkit.threat_model_line("internal-single-user") == (
        "threat-model: internal-single-user — hostile local writer out of scope; "
        "traversal, symlink escape and secrets in scope."
    )
    assert jaxflow_workerkit.threat_model_line("public-app") == (
        "threat-model: public-app — untrusted users reach this app; full OWASP scope."
    )


def test_threat_model_for_reads_valid_mode_from_agents_md(tmp_path):
    repo = _agents(tmp_path, _THREAT_MODEL_BLOCK)
    assert jaxflow_review._threat_model_for(repo) == "internal-single-user"


def test_threat_model_for_missing_agents_md_prints_note_and_returns_none(tmp_path, capsys):
    assert jaxflow_review._threat_model_for(tmp_path) is None
    assert capsys.readouterr().err.strip() == _NO_THREAT_MODEL_NOTE


def test_threat_model_for_no_valid_mode_prints_note_and_returns_none(tmp_path, capsys):
    repo = _agents(tmp_path, "## Threat model\nmode: something-else\n")
    assert jaxflow_review._threat_model_for(repo) is None
    assert capsys.readouterr().err.strip() == _NO_THREAT_MODEL_NOTE


def test_dispatch_review_manifest_carries_threat_model_from_agents_md(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        (root / "AGENTS.md").write_text(_THREAT_MODEL_BLOCK, encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        run_id = jaxflow_review.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CODEX_THREAD_ID": "1"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["threat_model"] == "internal-single-user"


def test_dispatch_review_no_agents_md_manifest_threat_model_none_with_stderr_note(monkeypatch, capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        run_id = jaxflow_review.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CODEX_THREAD_ID": "1"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["threat_model"] is None
        assert _NO_THREAT_MODEL_NOTE in capsys.readouterr().err


def test_dispatch_diff_review_manifest_carries_threat_model_from_agents_md(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        (root / "AGENTS.md").write_text(_THREAT_MODEL_BLOCK, encoding="utf-8")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        tests_path = worktree / ".local" / "reports" / "b1.tests.txt"
        tests_path.parent.mkdir(parents=True, exist_ok=True)
        tests_path.write_text("COMMAND: true\n\nEXIT: 0\n", encoding="utf-8")
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["threat_model"] == "internal-single-user"


def test_worker_doc_review_handoff_threat_model_line_before_focus():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        manifest_path, manifest = _write_manifest_for_worker(
            root, target, runtime="codex", focus="extra scrutiny",
            threat_model="internal-single-user",
        )
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["stdin_bytes"] = self.stdin_bytes

        code = jaxflow_worker.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()), post=lambda e: {"ok": True},
            popen=CapturePopen, allowlist_root=root,
        )
        assert code == 0
        prompt = captured["stdin_bytes"].decode("utf-8")
        threat_line = jaxflow_workerkit.threat_model_line("internal-single-user")
        assert threat_line in prompt
        assert "--focus: extra scrutiny" in prompt
        assert prompt.index(threat_line) < prompt.index("--focus: extra scrutiny")


def test_worker_doc_review_handoff_omits_threat_model_line_when_absent():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        manifest_path, manifest = _write_manifest_for_worker(root, target, runtime="codex")
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["stdin_bytes"] = self.stdin_bytes

        jaxflow_worker.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()), post=lambda e: {"ok": True},
            popen=CapturePopen, allowlist_root=root,
        )
        prompt = captured["stdin_bytes"].decode("utf-8")
        assert "threat-model:" not in prompt


def test_worker_diff_review_handoff_threat_model_line_before_focus():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        (worktree / "changed.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "change")
        base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, base_sha=base_sha, head_sha=head_sha,
            focus="extra scrutiny", threat_model="public-app",
        )
        prompt = _run_diff_worker_and_capture_prompt(manifest_path, worktree, root)
        threat_line = jaxflow_workerkit.threat_model_line("public-app")
        assert threat_line in prompt
        assert "--focus: extra scrutiny" in prompt
        assert prompt.index(threat_line) < prompt.index("--focus: extra scrutiny")


def test_worker_diff_review_handoff_omits_threat_model_line_when_absent():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        (worktree / "changed.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "change")
        base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, base_sha=base_sha, head_sha=head_sha,
        )
        prompt = _run_diff_worker_and_capture_prompt(manifest_path, worktree, root)
        assert "threat-model:" not in prompt


# ---- review --diff reviews the build's own recorded base (MOA-503) ----

def _stacked_diff_dispatch(monkeypatch, capsys, recorded):
    """main -> feat/p1 (a.txt, b.txt) -> feat/x (c.txt, built `--base feat/p1`). `recorded`
    is a callable (p1_tip, main_sha) -> the seeded manifest base_sha, or None for a legacy
    manifest with no key. Returns (diff manifest, p1_tip, main_sha, p2_head, worktree, stdout)."""
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        main_sha = _run_real(["git", "rev-parse", "HEAD"], root).stdout.strip()
        p1 = _init_worktree(root, "feat/p1")
        _commit(p1, "a.txt")
        p1_tip = _commit(p1, "b.txt")
        worktree = root.parent / "demo-feat-x"
        _git(root, "worktree", "add", "-b", "feat/x", str(worktree), "feat/p1")
        p2_head = _commit(worktree, "c.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        base = recorded(p1_tip, main_sha) if recorded else None
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=p2_head, base_sha=base)
        con.close()
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        names = _run_real(["git", "diff", "--name-only", f"{manifest['base_sha']}..{manifest['head_sha']}"], worktree).stdout.split()
        return manifest, p1_tip, main_sha, p2_head, names, capsys.readouterr().out


def test_diff_review_range_starts_at_the_recorded_stacked_base(monkeypatch, capsys):
    m, p1_tip, _, p2_head, names, out = _stacked_diff_dispatch(monkeypatch, capsys, lambda p1, main: p1)
    assert (m["base_sha"], m["head_sha"]) == (p1_tip, p2_head)
    assert names == ["c.txt"]
    assert "hint: recorded build base" not in out


def test_diff_review_without_a_recorded_base_keeps_the_merge_base(monkeypatch, capsys):
    m, _, main_sha, _, names, out = _stacked_diff_dispatch(monkeypatch, capsys, None)
    assert m["base_sha"] == main_sha
    assert names == ["a.txt", "b.txt", "c.txt"]
    assert "hint:" not in out


def test_diff_review_default_base_build_matches_the_merge_base(monkeypatch, capsys):
    m, _, main_sha, _, _, out = _stacked_diff_dispatch(monkeypatch, capsys, lambda p1, main: main)
    assert m["base_sha"] == main_sha
    assert "hint: recorded build base" not in out


def test_diff_review_unusable_recorded_base_falls_back_with_a_hint(monkeypatch, capsys):
    # not an ancestor of HEAD: a real commit that is on no branch of the stack
    m, _, main_sha, _, _, out = _stacked_diff_dispatch(monkeypatch, capsys, lambda p1, main: "f" * 40)
    assert m["base_sha"] == main_sha
    assert f"hint: recorded build base {'f' * 40} unusable" in out and "not an ancestor" in out
    # pre-F5 shape: a branch name, not a SHA
    m, _, main_sha, _, _, out = _stacked_diff_dispatch(monkeypatch, capsys, lambda p1, main: "feat/p1")
    assert m["base_sha"] == main_sha
    assert "hint: recorded build base feat/p1 unusable" in out and "not a 40-hex SHA" in out


def test_since_root_on_a_stacked_build_passes_the_lock(monkeypatch, capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        p1 = _init_worktree(root, "feat/p1")
        p1_tip = _commit(p1, "a.txt")
        worktree = root.parent / "demo-feat-x"
        _git(root, "worktree", "add", "-b", "feat/x", str(worktree), "feat/p1")
        root_head = _commit(worktree, "c.txt")
        new_head = _commit(worktree, "d.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=new_head, base_sha=p1_tip)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve")
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa1", base_sha=p1_tip, head_sha=root_head)
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1", since="aaaaaaaaaaa1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["base_sha"] == root_head


_CLAUDE_ENV = {"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID}


def test_dispatch_review_runs_on_the_callers_own_runtime_when_the_opposite_agent_is_off(monkeypatch, capsys):
    _agents_setting(monkeypatch, codex=False)
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_review(
            _review_args(spec=str(target), no_callback=True), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env=_CLAUDE_ENV, now=_fixed_now, allowlist_root=allow_root,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["runtime"] == "claude" and manifest["fallback"] == "codex off"
        assert "reviewer_runtime: claude (fallback: codex off)\n" in capsys.readouterr().out


def test_dispatch_review_without_fallback_prints_no_line_and_stores_no_fallback(monkeypatch, capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_review(
            _review_args(spec=str(target), no_callback=True), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env=_CLAUDE_ENV, now=_fixed_now, allowlist_root=allow_root,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["runtime"] == "codex" and "fallback" not in manifest
        assert "reviewer_runtime" not in capsys.readouterr().out


@pytest.mark.parametrize("agents", [dict(claude=False, codex=False), dict(claude=False, codex=False, opencode=False)])
def test_dispatch_review_refuses_without_a_reviewer_agent_and_reserves_nothing(monkeypatch, agents):
    _agents_setting(monkeypatch, **agents)
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        posts = []
        with pytest.raises(ji.Refusal) as caught:
            jaxflow_review.dispatch_review(
                _review_args(spec=str(target)), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: posts.append(e), env=_CLAUDE_ENV, now=_fixed_now, allowlist_root=allow_root,
            )
        assert caught.value.code == "no-reviewer-agent"
        assert posts == [] and not (root / ".local" / "runs").exists()


@pytest.mark.parametrize("error", ["settings-malformed", "settings-unreadable"])
def test_review_and_build_refuse_on_unreadable_settings_and_reserve_nothing(monkeypatch, error):
    monkeypatch.setattr(general_settings, "read_settings", lambda: {"ok": False, "error": error})
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        for call in (
            lambda: jaxflow_review.dispatch_review(_review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
                                            post=lambda e: {"ok": True}, env=_CLAUDE_ENV, now=_fixed_now, allowlist_root=allow_root),
            lambda: jaxflow_build.dispatch_build(_build_args(plan=str(plan)), run=_run_with_tmux(fake, real_cwd=root),
                                           post=lambda e: {"ok": True}, env=_CLAUDE_ENV, now=_fixed_now, allowlist_root=allow_root),
        ):
            with pytest.raises(ji.Refusal) as caught:
                call()
            assert caught.value.code == error
            assert caught.value.hint == "hint: fix settings.json or save /settings"
        assert fake.calls == [] and not (root / ".local" / "runs").exists()
        assert not (allow_root / "demo-feat-demo").exists()


def test_dispatch_diff_review_falls_back_and_prints_the_line(monkeypatch, capsys):
    _agents_setting(monkeypatch, codex=False)
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _seed_finished_build(con, "b1", root, worktree, verify="true", head_sha=head)
        con.close()
        tests_path = worktree / ".local" / "reports" / "b1.tests.txt"
        tests_path.parent.mkdir(parents=True, exist_ok=True)
        tests_path.write_text("COMMAND: true\n\nEXIT: 0\n", encoding="utf-8")
        monkeypatch.chdir(root)
        run_id = jaxflow_review.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root), post=lambda e: {"ok": True},
            env=_CLAUDE_ENV, now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["runtime"] == "claude" and manifest["fallback"] == "codex off"
        assert "reviewer_runtime: claude (fallback: codex off)\n" in capsys.readouterr().out


_FALLBACK_LINE = "reviewer_runtime: claude (fallback: codex off)\n"


def _seed_fallback_review(con, root, run_id, *, runtime="claude", write_report=True, **finished):
    """Reviewer run whose caller is claude: runtime claude = fallback (codex is the opposite), codex = none."""
    report = root / ".local" / "reports" / f"{run_id}.md"
    if write_report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("BODY\n", encoding="utf-8")
    _insert(con, run_id, "demo", "reviewer", "run-started", {"repo": str(root), "caller": "claude", "runtime": runtime})
    _insert(con, run_id, "demo", "reviewer", "run-finished",
            {"contract_status": "ok", "report_path": str(report), "summary": "ok", **finished})
    return report


def test_cmd_result_prints_the_fallback_line_once_and_first_with_a_tally(capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        report = _seed_fallback_review(con, root, "r1", tally="tally: 1 findings — 0 repeated, 1 new")
        _seed_fallback_review(con, root, "r2", runtime="codex")
        con.close()
        returned = jaxflow_cli.cmd_result("r1", allowlist_root=allow_root, db_path=db)
        printed = capsys.readouterr().out
        assert printed == _FALLBACK_LINE + "tally: 1 findings — 0 repeated, 1 new\n\n"  # line BEFORE the tally (cmd_result prints `tally + "\n"` with print)
        assert returned == f"{report}\nBODY\n"
        assert (printed + returned).count("reviewer_runtime") == 1
        assert "reviewer_runtime" not in jaxflow_cli.cmd_result("r2", allowlist_root=allow_root, db_path=db)
        assert capsys.readouterr().out == ""  # a non-fallback review prints nothing


def test_cmd_result_prints_the_fallback_line_before_the_review_chain(capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        merge_base = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        head = _commit(worktree, "c1.txt")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        (root / ".local" / "reports").mkdir(parents=True, exist_ok=True)
        _seed_diff_review(con, "aaaaaaaaaaa1", root, worktree, verdict="approve")
        con.execute("UPDATE workflow_events SET payload = json_set(payload, '$.runtime', 'claude') "
                    "WHERE run_id = 'aaaaaaaaaaa1' AND type = 'run-started'")  # caller claude + runtime claude
        con.commit()
        con.close()
        _write_diff_manifest(root, "aaaaaaaaaaa1", base_sha=merge_base, head_sha=head)
        report = root / ".local" / "reports" / "aaaaaaaaaaa1.md"
        report.write_text("REPORT BODY\n", encoding="utf-8")
        returned = jaxflow_cli.cmd_result("aaaaaaaaaaa1", allowlist_root=allow_root, db_path=db)
        printed = capsys.readouterr().out
        assert printed == _FALLBACK_LINE  # printed first; the chain heads the RETURNED text
        assert returned.startswith("chain:\n") and "reviewer_runtime" not in returned


def test_cmd_result_still_prints_the_fallback_line_when_the_report_is_missing(capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_fallback_review(con, root, "r1", write_report=False)
        con.close()
        with pytest.raises(ji.Refusal) as caught:
            jaxflow_cli.cmd_result("r1", allowlist_root=allow_root, db_path=db)
        assert caught.value.code == "report-missing"
        assert capsys.readouterr().out == _FALLBACK_LINE


def test_send_callback_appends_the_fallback_suffix_as_the_last_segment():
    manifest = dict(run_id="aaaabbbbcccc", caller="claude", caller_session=_TEST_CLAUDE_SESSION_ID, fallback="codex off")
    jaxflow_workerkit._send_callback(manifest, run=_forbidden_tmux, kind="spec", outcome="approve", summary="x", report_path="/report.md")
    line_path = jaxflow_common.CALLBACKS_ROOT / _TEST_CLAUDE_SESSION_ID / "aaaabbbbcccc.line"
    assert line_path.read_text(encoding="utf-8") == (
        "[JAXFLOW] spec aaaabbbbcccc finished — approve — /report.md (fallback: codex off)\n")
    jaxflow_workerkit._send_callback(manifest, run=_forbidden_tmux, kind="spec", outcome="approve", summary="x",
                           report_path="/report.md", ledger_pending=True)
    assert line_path.read_text(encoding="utf-8") == (
        "[JAXFLOW] spec aaaabbbbcccc finished — approve — /report.md · ledger pending (fallback: codex off)\n")


def test_build_refuses_when_opencode_is_off_before_reserving_anything(monkeypatch):
    _agents_setting(monkeypatch, opencode=False)
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        fake, posts = FakeTmux(), []
        with pytest.raises(ji.Refusal) as caught:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan)), run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: posts.append(e), env=_CLAUDE_ENV, now=_fixed_now, allowlist_root=allow_root,
            )
        assert caught.value.code == "agent-disabled: opencode"
        assert caught.value.hint == "hint: turn OpenCode on in /settings -> General"
        assert posts == [] and fake.calls == []
        assert not (root / ".local" / "runs").exists() and not (allow_root / "demo-feat-demo").exists()


def test_build_with_opencode_on_is_unchanged_by_the_other_switches(monkeypatch):
    _agents_setting(monkeypatch, claude=False, codex=False)
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        run_id = jaxflow_build.dispatch_build(
            _build_args(plan=str(plan)), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env=_CLAUDE_ENV, now=_fixed_now, allowlist_root=allow_root,
        )
        assert len(run_id) == 12


def test_build_resume_refuses_when_opencode_is_off_before_claiming_the_worktree(monkeypatch, tmp_path):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        seeded = _seed_resumable(monkeypatch, tmp_path, allow_root, root)
        _agents_setting(monkeypatch, opencode=False)  # AFTER seeding: the seed may patch settings itself
        fake, posts = FakeTmux(), []
        with pytest.raises(ji.Refusal) as caught:
            jaxflow_build.dispatch_build(
                _build_args(resume=seeded.prior), run=_run_with_tmux_and_log(fake, [], real_cwd=root),
                post=lambda e: posts.append(e), env=_CLAUDE_ENV, now=_fixed_now, allowlist_root=allow_root,
            )
        assert caught.value.code == "agent-disabled: opencode"
        assert posts == [] and fake.calls == []


def test_merge_help_documents_checks_reuse_and_recheck(capsys):
    with pytest.raises(SystemExit):
        jaxflow_cli.parse_args(["merge", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "--recheck" in out
    assert "unless an up-to-date diff review already proved the same command on this exact SHA" in out
