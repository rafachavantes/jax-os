"""jaxflow build tests (P2 split of test_jaxflow.py)."""
import json
import multiprocessing
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import pytest
import jaxflow
import jaxflow_cli
import jaxflow_worker
import jaxflow_build
import jaxflow_review
import jaxflow_merge
import jaxflow_common
import shlex
import jaxflow_run as jr
import jaxflow_settings as jset
import jax_init as ji

from testkit import (  # noqa: F401  (autouse fixtures included on purpose)
    FakeTmux,
    _E2EBuilderPopen,
    _MergeArgs,
    _assert_no_reservation,
    _build_args,
    _checkpoint_path,
    _fixed_now,
    _fresh_db,
    _git,
    _init_repo,
    _insert,
    _install_settings,
    _integration_on_by_default,
    _isolate_agent_settings,
    _isolate_callbacks,
    _ledger_post,
    _merge_env,
    _plan_file,
    _probe_dispatch_build,
    _restore_signal_handlers,
    _review_args,
    _rewrite_json,
    _run_real,
    _run_with_tmux,
    _run_with_tmux_and_log,
    _seed_resumable,
    _spec_file,
    _stub_launch_paths,
    _switch_aware_runner,
)


# ---- build: not-a-git-toplevel (fixes cold review round 2 F8 -- build-native and
# reachable via the CLI's very first check, but had no test of its own) ----

def test_build_refuses_not_a_git_toplevel(capsys, monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        outside = Path(raw) / "not-a-repo"
        outside.mkdir(parents=True)
        monkeypatch.chdir(outside)
        fake = FakeTmux()
        code = jaxflow_cli.main(
            ["build", "--plan", "plan.md", "--phase", "P", "--branch", "feat/x",
             "--whitelist", "a.py", "--verify", "true"],
            run=_run_with_tmux(fake, real_cwd=outside), post=lambda e: {"ok": True},
            env={}, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "not-a-git-toplevel"
        assert fake.calls == []


# ---- build: caller/session/path checks fire BEFORE any worktree reservation ----

def test_build_refuses_caller_unknown_with_no_side_effects(capsys, monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        code = jaxflow_cli.main(
            ["build", "--plan", str(plan), "--phase", "P", "--branch", "feat/x",
             "--whitelist", "a.py", "--verify", "true"],
            run=_run_with_tmux(fake, real_cwd=root), post=lambda e: {"ok": True},
            env={}, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "caller-unknown"
        assert fake.calls == []
        assert not (allow_root / "demo-feat-x").exists()


def test_build_refuses_caller_session_missing_before_worktree_reservation(capsys, monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        code = jaxflow_cli.main(
            ["build", "--plan", str(plan), "--phase", "P", "--branch", "feat/x",
             "--whitelist", "a.py", "--verify", "true", "--from", "codex"],
            run=_run_with_tmux(fake, real_cwd=root), post=lambda e: {"ok": True},
            env={}, allowlist_root=allow_root,
        )
        err = capsys.readouterr().err.strip().splitlines()
        assert code == jaxflow_common.REFUSED
        assert err[0] == "caller-session-missing"
        assert "CODEX_THREAD_ID" in err[1]
        assert fake.calls == []
        assert not (allow_root / "demo-feat-x").exists()


def test_build_refuses_plan_path_outside_allowlist(capsys, monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        outside = Path(raw) / "outside"
        root = allow_root / "demo"
        _init_repo(root)
        outside.mkdir()
        plan = outside / "plan.md"
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        code = jaxflow_cli.main(
            ["build", "--plan", str(plan), "--phase", "P", "--branch", "feat/x",
             "--whitelist", "a.py", "--verify", "true"],
            run=_run_with_tmux(fake, real_cwd=root), post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"
        assert fake.calls == []
        assert not (allow_root / "demo-feat-x").exists()


def test_build_refuses_secret_plan_path_variants(capsys, monkeypatch):
    # fixes cold review F3: dispatch_build never checked _is_secret_path at all in the
    # first draft. Direct, case-variant, and suffix forms, mirroring dispatch_review's own
    # existing secret-target tests.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        for name in (".env.local", "id_rsa", "foo.pem", ".ENV"):
            plan = root / name
            plan.write_text("SECRET=1\n", encoding="utf-8")
            code = jaxflow_cli.main(
                ["build", "--plan", str(plan), "--phase", "P", "--branch", "feat/x",
                 "--whitelist", "a.py", "--verify", "true"],
                run=_run_with_tmux(fake, real_cwd=root), post=lambda e: {"ok": True},
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, allowlist_root=allow_root,
            )
            assert code == jaxflow_common.REFUSED
            assert capsys.readouterr().err.strip() == f"secret-detected: {plan}"
        assert fake.calls == []
        assert not (allow_root / "demo-feat-x").exists()


def test_build_refuses_secret_plan_path_via_symlink(capsys, monkeypatch):
    # fixes cold review F3, same symlink-alias concern branch review F3 already caught for
    # dispatch_review: the plan path must be judged on its RESOLVED target.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        secret = root / ".env"
        secret.write_text("SECRET=1\n", encoding="utf-8")
        alias = root / "plan.md"
        alias.symlink_to(secret)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        code = jaxflow_cli.main(
            ["build", "--plan", str(alias), "--phase", "P", "--branch", "feat/x",
             "--whitelist", "a.py", "--verify", "true"],
            run=_run_with_tmux(fake, real_cwd=root), post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == f"secret-detected: {secret.resolve()}"
        assert fake.calls == []
        assert not (allow_root / "demo-feat-x").exists()


def test_build_refuses_effort_override_for_opencode_runtime_before_reservation(capsys, monkeypatch):
    # fixes cold review round 2 F3, amended 2026-09-07 (MOA-451): an OpenCode builder's
    # reasoning effort belongs to its RUNTIME definition (`jr.VARIANT_BY_RUNTIME` ->
    # `opencode run --variant`), not to the call -- picking the runtime IS picking the
    # effort, exactly as it is picking the model id. Accepting `--effort` here would
    # silently drop it at the one call site that launches the builder, so refuse loudly,
    # before any reservation, for BOTH OpenCode runtimes.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        for builder in (None, "opencode-deepseek"):
            argv = ["build", "--plan", str(plan), "--phase", "P", "--branch", "feat/x",
                    "--whitelist", "a.py", "--verify", "true", "--effort", "high"]
            if builder:
                argv += ["--builder", builder]
            code = jaxflow_cli.main(
                argv, run=_run_with_tmux(fake, real_cwd=root), post=lambda e: {"ok": True},
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, allowlist_root=allow_root,
            )
            expected_runtime = builder or jaxflow_build.BUILDER_DEFAULT
            assert code == jaxflow_common.REFUSED
            assert capsys.readouterr().err.strip() == f"effort-not-supported: {expected_runtime}"
        assert fake.calls == []
        assert not (allow_root / "demo-feat-x").exists()


@pytest.mark.parametrize("flags,install,code", [
    ({"fallback": True}, False, "agent-settings-uninitialized"),
    ({"builder": "opencode-builder"}, False, "agent-settings-uninitialized"),
    ({"builder": "opencode-deepseek"}, True, "agent-settings-override"),
    ({"model": "xai/another"}, True, "agent-settings-override"),
    ({"effort": "low"}, True, "agent-settings-override"),
])
def test_managed_build_flags_refuse_before_reservation(monkeypatch, tmp_path, flags, install, code):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        if install:
            _install_settings(tmp_path, monkeypatch)
        _run_id, events, git_calls, mkdir_calls, exc = _probe_dispatch_build(
            monkeypatch, root, allow_root, plan, **flags,
        )
        assert exc is not None and exc.code == code
        _assert_no_reservation(allow_root, git_calls, mkdir_calls, events)


def test_managed_build_uses_settings_profile(monkeypatch, tmp_path):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        _install_settings(tmp_path, monkeypatch)
        for flags, profile, branch in (
            ({}, "default", "feat/managed-default"),
            ({"fallback": True}, "fallback", "feat/managed-fallback"),
            ({"builder": "opencode-builder"}, "default", "feat/managed-explicit"),
        ):
            run_id, events, _git, _mkdir, exc = _probe_dispatch_build(
                monkeypatch, root, allow_root, plan, branch=branch, **flags,
            )
            assert exc is None
            payload = events[0]["payload"]
            assert payload["runtime"] == "opencode-builder"
            assert payload["requested_profile"] == profile
            assert payload["root_build_run_id"] == run_id
            assert "resumes_run_id" not in payload
            assert payload["model"] == "fixture/wire-real-model"
            assert payload["effort"] == "high"
            manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
            assert manifest["requested_profile"] == profile
            assert manifest["root_build_run_id"] == run_id
            assert manifest["reservation_owned"] is True
            assert "resumes_run_id" not in manifest
            assert manifest["model"] == "fixture/wire-real-model"
            assert manifest["effort"] == "high"


def test_build_fallback_cli_uninitialized(capsys, monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        code = jaxflow_cli.main(
            ["build", "--plan", str(plan), "--phase", "P", "--branch", "feat/x",
             "--whitelist", "a.py", "--verify", "true", "--fallback"],
            run=_run_with_tmux(FakeTmux(), real_cwd=root), post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "agent-settings-uninitialized"
        assert not (allow_root / "demo-feat-x").exists()


def test_dispatch_build_legacy_namespace_without_fallback_attr(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        events = []
        args = SimpleNamespace(
            plan=str(plan), phase="P", branch="feat/legacy-ns", whitelist="a.py",
            verify="true", builder=None, model=None, effort=None, from_caller=None,
            no_callback=False, base=None, build=None,
        )
        run_id = jaxflow_build.dispatch_build(
            args, run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: events.append(e) or {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
            allowlist_root=allow_root,
        )
        assert events[0]["payload"]["runtime"] == "opencode-grok"
        assert "requested_profile" not in events[0]["payload"]
        assert run_id


def test_review_invalid_override_does_not_launch(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        for kwargs, code in (
            ({"model": "-bad"}, "reviewer-model-invalid"),
            ({"effort": "nope"}, "reviewer-effort-invalid"),
            ({"model": "x" * 385}, "reviewer-model-invalid"),
        ):
            fake = FakeTmux()
            events = []
            try:
                jaxflow_review.dispatch_review(
                    _review_args(spec=str(target), **kwargs),
                    run=_run_with_tmux(fake, real_cwd=root),
                    post=lambda e: events.append(e) or {"ok": True},
                    env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                    now=_fixed_now, allowlist_root=allow_root,
                )
            except ji.Refusal as exc:
                assert exc.code == code
            else:
                raise AssertionError(code)
            assert fake.calls == []
            assert events == []


def test_build_refuses_model_over_hub_limit_before_reservation(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        _run_id, events, git_calls, mkdir_calls, exc = _probe_dispatch_build(
            monkeypatch, root, allow_root, plan, model="x" * 385,
        )
        assert exc is not None and exc.code == "model-invalid"
        _assert_no_reservation(allow_root, git_calls, mkdir_calls, events)


def test_build_accepts_combined_id_and_unicode_model_within_hub_limit(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        for branch, model in (
            ("feat/unicode", "é" * 384),
            ("feat/combined", "fixture/" + "m" * 200),
        ):
            _run_id, events, _git, _mkdir, exc = _probe_dispatch_build(
                monkeypatch, root, allow_root, plan, branch=branch, model=model,
            )
            assert exc is None
            assert events[0]["payload"]["model"] == model


def test_build_refuses_codex_as_a_builder_runtime(capsys, monkeypatch):
    # MOA-451: `codex` is a REVIEWER runtime only. Its `--sandbox workspace-write` cannot
    # write a LINKED worktree's git dir (which lives in <main-repo>/.git/worktrees/<name>,
    # outside the sandbox), so a Codex builder implemented ARC's B4 and could never commit
    # it; a second run then could not spawn `next build`'s TypeScript child. Refused at the
    # CLI, before any reservation.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        code = jaxflow_cli.main(
            ["build", "--plan", str(plan), "--phase", "P", "--branch", "feat/x",
             "--whitelist", "a.py", "--verify", "true", "--builder", "codex"],
            run=_run_with_tmux(fake, real_cwd=root), post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert "runtime-not-allowed" in capsys.readouterr().err
        assert fake.calls == []
        assert not (allow_root / "demo-feat-x").exists()


# ---- build: worktree reservation (spec §4.2 step 1, fixes cold review G8/RECURRENCE(F19)) ----

def test_build_refuses_branch_exists_when_worktree_path_already_a_directory(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        # Pre-existing EMPTY directory at the exact worktree path -- `git worktree add -b`
        # alone does not reliably refuse this (probed live per spec).
        (allow_root / "demo-feat-demo").mkdir()
        fake = FakeTmux()
        git_calls = []
        try:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan)),
                run=_run_with_tmux_and_log(fake, git_calls, real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert exc.code == "branch-exists"
        else:
            raise AssertionError("branch-exists not raised")
        assert fake.calls == []
        # fixes Part 1 diff-review F2: `os.mkdir` is the reservation BEFORE any git call
        # for `_default_branch()` -- an EEXIST here makes zero `_default_branch()` git
        # probes; only the toplevel resolve every verb needs ran at all.
        assert git_calls == [["git", "rev-parse", "--show-toplevel"]]
        assert (allow_root / "demo-feat-demo").is_dir()


def test_build_base_stacks_the_new_branch_on_the_named_ref(monkeypatch):
    """MOA-452: without --base every build starts from the default branch, so a plan split
    in two could not be built at all -- part 2 had to be dispatched outside jaxflow. The new
    branch must start from the named ref's commit, not from main."""
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        # part 1: a branch with a commit main does not have
        _git(root, "switch", "-c", "feat/p1")
        (root / "p1.txt").write_text("part one\n", encoding="utf-8")
        _git(root, "add", "p1.txt")
        _git(root, "commit", "-m", "feat: part one")
        p1_sha = _run_real(["git", "rev-parse", "HEAD"], root).stdout.strip()
        _git(root, "switch", "main")

        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        git_calls = []
        run_id = jaxflow_build.dispatch_build(
            _build_args(plan=str(plan), branch="feat/p2", base="feat/p1"),
            run=_run_with_tmux_and_log(FakeTmux(), git_calls, real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
            allowlist_root=allow_root,
        )
        assert run_id
        worktree = allow_root / "demo-feat-p2"
        assert (worktree / "p1.txt").is_file()  # part 1's work came along
        parent = _run_real(["git", "rev-parse", "HEAD~0"], worktree).stdout.strip()
        assert parent == p1_sha
        # The security invariant, not just the outcome (fixes base diff-review F1): the
        # positional base argument carries the RESOLVED sha, never the caller's own ref
        # string -- `git worktree add` reads that slot positionally, with no `--` guard.
        add = [c for c in git_calls if c[:3] == ["git", "worktree", "add"]]
        assert len(add) == 1, git_calls
        assert add[0][-1] == p1_sha
        assert "feat/p1" not in add[0]


def test_build_refuses_a_malformed_base_without_handing_it_to_git_at_all(monkeypatch):
    """The shape gate must refuse BEFORE the value reaches git. `git rev-parse` happens to
    error on most of these too, so asserting only the refusal would pass with the gate
    deleted -- what proves the gate is that no `rev-parse` call is ever made. A leading `-`
    matters because the base reaches `git worktree add` positionally with no `--` guard;
    the length bound matters because rev-parse would happily resolve a 513-char ref name."""
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        for bad in ("--foo", "-x", "has space", "", "a" * 513):
            fake = FakeTmux()
            git_calls = []
            try:
                jaxflow_build.dispatch_build(
                    _build_args(plan=str(plan), branch="feat/x", base=bad),
                    run=_run_with_tmux_and_log(fake, git_calls, real_cwd=root),
                    post=lambda e: {"ok": True},
                    env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
                    allowlist_root=allow_root,
                )
            except ji.Refusal as exc:
                assert exc.code == "base-invalid", bad
            else:
                raise AssertionError(f"base-invalid not raised for {bad!r}")
            assert git_calls == [["git", "rev-parse", "--show-toplevel"]], bad
            assert fake.calls == []
            assert not (allow_root / "demo-feat-x").exists(), bad


def test_build_refuses_a_well_shaped_base_that_does_not_resolve(monkeypatch):
    """A ref that passes the shape gate but names nothing must refuse on the `rev-parse`
    verdict -- and still before any reservation exists to unwind."""
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        git_calls = []
        try:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan), branch="feat/x", base="no/such/ref"),
                run=_run_with_tmux_and_log(fake, git_calls, real_cwd=root),
                post=lambda e: {"ok": True},
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
                allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert exc.code == "base-invalid"
        else:
            raise AssertionError("base-invalid not raised")
        # This one DID reach git -- that is the difference from the shape-gated cases.
        assert git_calls == [
            ["git", "rev-parse", "--show-toplevel"],
            ["git", "rev-parse", "--verify", "no/such/ref^{commit}"],
        ]
        assert fake.calls == []
        assert not (allow_root / "demo-feat-x").exists()


def test_build_without_base_still_starts_from_the_default_branch(monkeypatch):
    """The default path must be untouched: no --base means the repo's default branch, which
    is what every build did before MOA-452."""
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        main_sha = _run_real(["git", "rev-parse", "HEAD"], root).stdout.strip()
        _git(root, "switch", "-c", "feat/elsewhere")
        (root / "other.txt").write_text("unrelated\n", encoding="utf-8")
        _git(root, "add", "other.txt")
        _git(root, "commit", "-m", "feat: unrelated")
        _git(root, "switch", "main")

        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        jaxflow_build.dispatch_build(
            _build_args(plan=str(plan), branch="feat/plain"),
            run=_run_with_tmux(FakeTmux(), real_cwd=root), post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
            allowlist_root=allow_root,
        )
        worktree = allow_root / "demo-feat-plain"
        assert _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip() == main_sha
        assert not (worktree / "other.txt").exists()


def test_build_refuses_plan_path_that_is_a_directory_before_reservation(monkeypatch):
    # fixes Part 1 diff-review F1: a directory-valued --plan must refuse before ANY
    # worktree reservation, with the engine-style plain message, no git call beyond the
    # toplevel resolve.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan_dir = root / ".local" / "docs" / "plans"
        plan_dir.mkdir(parents=True)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        try:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan_dir)), run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert exc.code == "plan is not a readable file"
        else:
            raise AssertionError("refusal not raised")
        assert fake.calls == []
        assert not (allow_root / "demo-feat-demo").exists()


def test_build_refuses_plan_invalid_unresolvable_spec_line(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text(
            "# Plan\n\n**Goal:** g\n\n**Spec:** `missing-spec.md`\n\n### Task 1: t\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(root)
        fake = FakeTmux()
        try:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan)), run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert exc.code == "plan-invalid"
            assert "unresolvable **Spec:** line" in exc.hint
        else:
            raise AssertionError("plan-invalid not raised")
        assert fake.calls == []
        assert not (allow_root / "demo-feat-demo").exists()


def test_build_refuses_plan_invalid_spec_fragment_not_found(monkeypatch):
    # Closes spec item 2's own acceptance bullet ("a fragment naming no heading refuses:
    # plan-invalid at build") with a literal fragment case, not just a missing-file one --
    # `_validate_plan_structure` folds `spec-fragment-not-found` into its defects the same
    # way it folds `spec-reference-invalid`, via the same `_resolve_handoff_spec_path` call.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        spec = root / ".local" / "docs" / "specs" / "spec.md"
        spec.parent.mkdir(parents=True)
        spec.write_text("# Spec\n\n## Real Heading\ntext\n", encoding="utf-8")
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text(
            f"# Plan\n\n**Goal:** g\n\n**Spec:** `{spec}#No Such Heading`\n\n### Task 1: t\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(root)
        fake = FakeTmux()
        try:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan)), run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert exc.code == "plan-invalid"
            assert "unresolvable **Spec:** line: spec-fragment-not-found" in exc.hint
        else:
            raise AssertionError("plan-invalid not raised")
        assert fake.calls == []
        assert not (allow_root / "demo-feat-demo").exists()


def test_build_refuses_plan_invalid_lists_every_defect_not_just_the_first(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("no goal, no task\n\n**Spec:** `missing-spec.md`\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        try:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan)), run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert exc.code == "plan-invalid"
            # the missing goal/task headings are no longer defects at all (decision 7)
            # -- only the unresolvable **Spec:** line is.
            assert exc.hint == "hint: unresolvable **Spec:** line: spec-reference-invalid"
        else:
            raise AssertionError("plan-invalid not raised")
        assert fake.calls == []
        assert not (allow_root / "demo-feat-demo").exists()


def test_build_dispatches_a_plan_with_no_goal_or_task_heading(monkeypatch):
    """AC 2: a plan using `## Task - 1`, no **Goal:**, no H1 dispatches -- no
    plan-invalid for missing goal/task structure."""
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("## Task - 1\n\nDo the thing.\n", encoding="utf-8")
        monkeypatch.chdir(root)
        run_id = jaxflow_build.dispatch_build(
            _build_args(plan=str(plan)), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        assert run_id
        assert (allow_root / "demo-feat-demo").is_dir()


def test_build_handoff_omits_goal_tasks_acceptance_and_names_the_plan_path():
    handoff = jaxflow_worker._build_builder_handoff(
        plan_dest=Path("/p/plan.md"), spec_dest=Path("/p/spec.md"),
        agents_path=Path("/p/AGENTS.md"), branch="feat/x", head_sha="a" * 40,
        whitelist=["src"], verify_cmd="pnpm test", build_cmd=None,
    )
    assert "goal:" not in handoff
    assert "tasks:" not in handoff
    assert "acceptance:" not in handoff
    assert "  plan: /p/plan.md\n" in handoff


def test_build_accepts_a_structurally_valid_plan_unchanged(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)  # "# Plan\n\n**Goal:** g\n\n### Task 1: t\n"
        monkeypatch.chdir(root)
        run_id = jaxflow_build.dispatch_build(
            _build_args(plan=str(plan)), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        assert run_id
        assert (allow_root / "demo-feat-demo").is_dir()


def test_build_default_branch_probe_failure_after_reservation_removes_empty_dir(monkeypatch):
    # fixes Part 1 diff-review F2: an exception raised while resolving the default branch,
    # AFTER the mkdir reservation but BEFORE `git worktree add -b`, removes the empty
    # directory this dispatch itself just reserved and refuses -- no `git worktree add`
    # ever ran, so there is no branch or admin entry to undo (only the bare dir).
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()

        def run(argv, cwd=None):
            if argv and argv[0] == "tmux":
                return fake.handle(argv)
            if argv[:2] == ["git", "symbolic-ref"]:
                raise OSError("boom")
            return _run_real(argv, cwd if cwd is not None else root)

        try:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan)), run=run, post=lambda e: {"ok": True},
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
                allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert "boom" in exc.code
        else:
            raise AssertionError("refusal not raised")
        assert not (allow_root / "demo-feat-demo").exists()
        assert fake.calls == []


def test_build_post_add_exception_cleans_up_worktree_branch_and_admin_entry(monkeypatch):
    # fixes Part 1 diff-review F1: an exception AFTER `git worktree add -b` succeeded
    # (here, the AGENTS.md copy `read_text` call -- MOA-467 stopped copying the plan, and
    # the AGENTS.md copy is the only control-repo file still read after the add) must undo
    # everything that call created -- the
    # worktree directory, the branch ref, and Git's own `.git/worktrees/<id>` admin entry
    # -- not just the `Refusal`/`ValueError` cases already covered.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        (root / "AGENTS.md").write_text("# rules\n", encoding="utf-8")
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        agents_resolved = (root / "AGENTS.md").resolve()

        real_read_text = Path.read_text

        def failing_read_text(self, *a, **kw):
            if self == agents_resolved:
                raise OSError("disk exploded")
            return real_read_text(self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", failing_read_text)

        try:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan)), run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert "disk exploded" in exc.code
        else:
            raise AssertionError("refusal not raised")
        assert not (allow_root / "demo-feat-demo").exists()
        assert _run_real(["git", "rev-parse", "--verify", "feat/demo"], root).returncode != 0
        admin = _run_real(["git", "worktree", "list", "--porcelain"], root).stdout
        assert "demo-feat-demo" not in admin


def test_build_worktree_add_failure_cleans_up_reserved_dir_and_never_reaches_preflight(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        # Pre-create the BRANCH (not the directory) so `git worktree add -b` itself fails --
        # the mkdir reservation succeeds (fresh path), then git refuses because the branch
        # already exists. This never reaches preflight (the worktree is never populated).
        _git(root, "branch", "feat/demo")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        try:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan)), run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert exc.code == "branch-exists"
            assert exc.hint  # git's stderr is surfaced as a hint
        else:
            raise AssertionError("branch-exists not raised")
        assert not (allow_root / "demo-feat-demo").exists()
        # the pre-existing branch is untouched. Fixes cold review round 2 F5: add-failure
        # cleanup never force-deletes ANY branch -- dropping the pre-add `git rev-parse
        # <branch>` probe (itself a check-then-act) means this dispatch can no longer
        # safely tell a branch it just half-created apart from one that already existed, so
        # neither case triggers a delete.
        assert _run_real(["git", "rev-parse", "--verify", "feat/demo"], root).returncode == 0
        assert fake.calls == []


# ---- build: other reachable common refusals, cleanup verified (fixes cold review round 2
# F8 -- malformed phase and run-path collision were reachable but had no build test) ----

def test_build_refuses_malformed_phase_via_preflight_and_cleans_up_worktree(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        try:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan), branch="feat/x", phase="not a token!"),
                run=_run_with_tmux(fake, real_cwd=root), post=lambda e: {"ok": True},
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
                allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert exc.code == "malformed phase"
        else:
            raise AssertionError("malformed phase not raised")
        # This refusal fires AFTER a successful `git worktree add -b` (preflight runs
        # against the new worktree) -- directory, branch, AND Git's admin record must all
        # be gone (fixes cold review round 2 F1).
        assert not (allow_root / "demo-feat-x").exists()
        assert _run_real(["git", "rev-parse", "--verify", "feat/x"], root).returncode != 0
        admin = _run_real(["git", "worktree", "list", "--porcelain"], root).stdout
        assert "demo-feat-x" not in admin


def test_build_refuses_run_path_collision_and_cleans_up_worktree(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)

        def run(argv, cwd=None):
            if argv[:2] == ["tmux", "has-session"]:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return _run_real(argv, cwd if cwd is not None else root)

        try:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan), branch="feat/x"), run=run,
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert exc.code == "run path collision"
        else:
            raise AssertionError("run path collision not raised")
        assert not (allow_root / "demo-feat-x").exists()
        assert _run_real(["git", "rev-parse", "--verify", "feat/x"], root).returncode != 0
        admin = _run_real(["git", "worktree", "list", "--porcelain"], root).stdout
        assert "demo-feat-x" not in admin


# ---- build: plan copy + target-is-branch (spec §4.2 step 2) ----

def test_build_keeps_original_plan_path_and_copies_only_agents_md(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        (root / "AGENTS.md").write_text("# rules\n", encoding="utf-8")
        plan = root / ".local" / "docs" / "plans" / "my-plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# The Plan\ndo the thing\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        run_id = jaxflow_build.dispatch_build(
            _build_args(plan=str(plan), branch="feat/my-phase"),
            run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
            allowlist_root=allow_root,
        )
        worktree = allow_root / "demo-feat-my-phase"
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        # MOA-467: the manifest stores the ORIGINAL plan path, never a worktree copy,
        # and no plan/spec copy is created anywhere in the worktree.
        assert manifest["plan_path"] == str(plan)
        assert not (worktree / ".local" / "docs" / "plans").exists()
        assert not (worktree / ".local" / "docs" / "specs").exists()
        # AGENTS.md is still copied into the worktree root (stays gitignored there too).
        assert (worktree / "AGENTS.md").read_text(encoding="utf-8") == "# rules\n"
        payload = events[0]["payload"]
        assert payload["kind"] == "build"
        assert payload["target"] == "feat/my-phase"
        assert payload["verify"] == "true"
        assert manifest["whitelist"] == ["a.py", "b.py"]


def test_build_model_effort_flags_reach_manifest_and_run_started_payload(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        # Amended 2026-09-07 (MOA-451): `--effort` is now refused for EVERY builder runtime,
        # since both remaining ones are OpenCode and their variant is part of the runtime
        # definition. So a build's ledger `effort` is always the `n/a` sentinel, and only
        # `--model` can still be overridden per call.
        run_id = jaxflow_cli.main(
            ["build", "--plan", str(plan), "--phase", "P", "--branch", "feat/ov",
             "--whitelist", "a.py", "--verify", "true", "--model", "custom-model"],
            run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, allowlist_root=allow_root,
        )
        payload = events[0]["payload"]
        assert payload["model"] == "custom-model"
        assert payload["effort"] == "n/a"

        # No override -- the ledger records the sentinel-shaped default (fixes cold review
        # F4; the sentinels are converted back to None in Task 3, never in the ledger).
        events.clear()
        jaxflow_cli.main(
            ["build", "--plan", str(plan), "--phase", "P", "--branch", "feat/nov",
             "--whitelist", "a.py", "--verify", "true"],
            run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, allowlist_root=allow_root,
        )
        payload = events[0]["payload"]
        assert payload["model"] == "xai/grok-4.6"  # MODEL_BY_RUNTIME["opencode-grok"]
        assert payload["effort"] == "n/a"


def test_build_records_the_build_command_in_manifest_and_started_payload(monkeypatch):
    """MOA-454 spec §2.6. The build command rides alongside `verify` in BOTH places: the
    manifest (read by the builder worker) and the run-started payload (read later by
    `review --diff`, which never sees the manifest)."""
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        events = []
        run_id = jaxflow_build.dispatch_build(
            _build_args(plan=str(plan), verify="pnpm test", build="pnpm build"),
            run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: events.append(e) or {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        started = [e for e in events if e["type"] == "run-started"][0]
        assert started["payload"]["verify"] == "pnpm test"
        assert started["payload"]["build"] == "pnpm build"
        manifest = json.loads(
            (root / ".local" / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["verify"] == "pnpm test"
        assert manifest["build"] == "pnpm build"


def test_build_omits_the_build_key_entirely_when_no_build_command_is_given(monkeypatch):
    """Spec §2.6: ABSENT, never `null`, never the literal "none". The ingress validator
    treats a `null` as a type error, and a stored "none" would later be re-run as a shell
    command named `none`."""
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        events = []
        run_id = jaxflow_build.dispatch_build(
            _build_args(plan=str(plan), verify="pnpm test"),
            run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: events.append(e) or {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        started = [e for e in events if e["type"] == "run-started"][0]
        assert "build" not in started["payload"]
        manifest = json.loads(
            (root / ".local" / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
        assert "build" not in manifest


def test_build_rejects_a_blank_verify_or_build_command():
    """Spec §7.1 #37 — RECURRENCE of the `--checks` defect fixed 2026-09-07. argparse's
    `required=True` demands only that the FLAG be present; an empty string reaches
    `/bin/sh -c ""`, exits 0, and satisfies verification having verified nothing. This
    mirrors `test_merge_rejects_a_blank_checks_command` exactly."""
    base = ["build", "--plan", "p.md", "--phase", "P", "--branch", "feat/x",
            "--whitelist", "src"]
    for blank in ("", "   ", "\t", "\n", " \t\n "):
        with pytest.raises(SystemExit):
            jaxflow_cli.parse_args(base + ["--verify", blank])
        with pytest.raises(SystemExit):
            jaxflow_cli.parse_args(base + ["--verify", "true", "--build", blank])
    # A command that merely LOOKS trivial is the caller's business, not the parser's.
    args = jaxflow_cli.parse_args(base + ["--verify", "true", "--build", "true"])
    assert (args.verify, args.build) == ("true", "true")
    # --build is OPTIONAL: omitting it parses, and lands as None.
    assert jaxflow_cli.parse_args(base + ["--verify", "true"]).build is None


def test_build_help_describes_managed_builder_without_override_flags():
    result = subprocess.run(
        [sys.executable, str(Path(jaxflow.__file__).resolve()), "build", "--help"],
        capture_output=True, text=True, check=True,
    )
    help_text = result.stdout + result.stderr
    normalized = " ".join(help_text.split())
    assert "opencode-builder" in "".join(help_text.split())
    assert "profiles saved via /tools Agents" in normalized
    assert "--builder" not in help_text
    assert "--model" not in help_text
    assert "--effort" not in help_text


def _dispatch_resume(seeded, fake, **extra):
    events = []
    git_calls = []
    run_id = jaxflow_build.dispatch_build(
        _build_args(resume=seeded.prior, **extra),
        run=_run_with_tmux_and_log(fake, git_calls, real_cwd=seeded.root),
        post=lambda e: events.append(e) or {"ok": True},
        env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
        now=_fixed_now, allowlist_root=seeded.allow_root,
    )
    return run_id, events, git_calls


@pytest.mark.parametrize("missing", ["plan", "phase", "branch", "whitelist", "verify", "all"])
def test_fresh_build_refuses_missing_required_flags(monkeypatch, tmp_path, capsys, missing):
    plan = _plan_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    flags = dict(plan=str(plan), phase="P", branch="feat/x", whitelist="a.py", verify="true")
    argv = ["build"]
    for name, value in flags.items():
        if missing not in (name, "all"):
            argv.extend([f"--{name}", value])

    def run(cmd, **kwargs):
        assert cmd == ["git", "rev-parse", "--show-toplevel"]
        return SimpleNamespace(returncode=0, stdout=str(tmp_path))

    monkeypatch.setattr(os, "mkdir", lambda *a, **k: pytest.fail("unexpected reservation"))
    code = jaxflow_cli.main(
        argv, run=run, post=lambda e: pytest.fail("unexpected event"),
        env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
        now=_fixed_now, allowlist_root=tmp_path,
    )
    assert code == jaxflow_common.REFUSED
    captured = capsys.readouterr()
    assert captured.err.strip() == "build-missing-required-flags"
    assert captured.out == ""


def test_build_resume_parser_accepts_id_without_fresh_flags():
    args = jaxflow_cli.parse_args(["build", "--resume", "aaaaaaaaaaaa"])
    assert args.resume == "aaaaaaaaaaaa"
    assert args.plan is None
    assert args.phase is None
    assert args.branch is None
    assert args.whitelist is None
    assert args.verify is None
    assert args.fallback is False
    args = jaxflow_cli.parse_args([
        "build", "--resume", "aaaaaaaaaaaa", "--fallback", "--from", "claude", "--no-callback",
    ])
    assert args.fallback is True
    assert args.from_caller == "claude"
    assert args.no_callback is True


def test_build_resume_refuses_conflicting_fresh_flags(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        kwargs = dict(
            run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        extras = (
            {"plan": str(plan)}, {"phase": "P"}, {"branch": "feat/x"},
            {"whitelist": "a.py"}, {"verify": "true"}, {"build": "true"},
            {"base": "main"}, {"builder": "opencode-grok"}, {"model": "x"},
            {"effort": "high"},
        )
        for extra in extras:
            with pytest.raises(ji.Refusal) as caught:
                jaxflow_build.dispatch_build(_build_args(resume="aaaaaaaaaaaa", **extra), **kwargs)
            assert caught.value.code == "resume-ineligible"
            assert not (allow_root / "demo-feat-demo").exists()
            assert not (allow_root / "demo-feat-x").exists()
        assert fake.calls == []


def test_build_resume_refuses_invalid_run_id_before_paths(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        with pytest.raises(ji.Refusal) as caught:
            jaxflow_build.dispatch_build(
                _build_args(resume="../escape"),
                run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True},
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root,
            )
        assert caught.value.code == "resume-ineligible"
        assert fake.calls == []
        assert list(allow_root.iterdir()) == [root]


def test_build_resume_reuses_worktree_without_cleanup(monkeypatch, tmp_path):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        seeded = _seed_resumable(monkeypatch, tmp_path, allow_root, root)
        fake = FakeTmux()
        mkdir_calls = []
        real_mkdir = os.mkdir

        def wrapped_mkdir(path, *a, **k):
            mkdir_calls.append(path)
            return real_mkdir(path, *a, **k)

        monkeypatch.setattr(os, "mkdir", wrapped_mkdir)
        run_id, events, git_calls = _dispatch_resume(seeded, fake)
        assert run_id != seeded.prior
        assert seeded.worktree.is_dir()
        assert not any(c[:3] == ["git", "worktree", "add"] for c in git_calls)
        assert not any(c[:3] == ["git", "worktree", "remove"] for c in git_calls)
        assert str(seeded.worktree) not in {str(p) for p in mkdir_calls}
        payload = events[0]["payload"]
        assert payload["resumes_run_id"] == seeded.prior
        assert payload["root_build_run_id"] == seeded.prior
        assert payload["requested_profile"] == "default"
        assert payload["target"] == "feat/x"
        assert payload["verify"] == "true"
        manifest = json.loads(
            (root / ".local" / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["reservation_owned"] is False
        assert manifest["resumes_run_id"] == seeded.prior
        assert manifest["root_build_run_id"] == seeded.prior
        assert manifest["resume_start"] == seeded.state
        assert manifest["worktree"] == str(seeded.worktree)
        assert manifest["plan_path"] == str(seeded.plan)
        assert manifest["base_sha"] == seeded.base_sha
        assert manifest["whitelist"] == ["a.py"]


def test_resume_is_eligible_on_an_interrupted_row_with_a_valid_checkpoint(monkeypatch, tmp_path):
    # MOA-474 §5.1 conflict 2: line 999's `contract_status == "cancelled"` exclusion is
    # DROPPED -- eligibility rests on `result not in ("failure", "blocked")` (line 1000)
    # plus the checkpoint match alone. An interrupted row's result is always "failure",
    # so it is eligible under the exact same predicate a plain failure/blocked row is.
    # Reuses `_seed_resumable`/`_dispatch_resume` VERBATIM, then overwrites the seeded
    # run-finished row to the interrupted shape -- the checkpoint `_seed_resumable`
    # already wrote has `outcome: "failure"`, which stays valid unchanged (F2's own
    # checkpoint-outcome fix belongs to the reaper's write path, Task 2 -- this seeded
    # checkpoint is written directly by the test helper, already correct).
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        seeded = _seed_resumable(monkeypatch, tmp_path, allow_root, root)
        con = sqlite3.connect(seeded.db)
        con.execute(
            "UPDATE workflow_events SET payload = ? WHERE run_id = ? AND type = 'run-finished'",
            (json.dumps({
                "contract_status": "interrupted", "exit_code": None, "report_path": None,
                "summary": "worker interrupted by SIGTERM", "stage": "worker",
                "diagnostic": "worker interrupted by SIGTERM", "head_sha": seeded.base_sha,
                "result": "failure", "phase": "P",
            }), seeded.prior),
        )
        con.commit()
        con.close()
        fake = FakeTmux()
        run_id, events, git_calls = _dispatch_resume(seeded, fake)
        # No Refusal raised -- dispatch proceeded past line 999's (now-dropped) clause,
        # same as the "cancelled" row it used to also (redundantly) block.
        assert run_id != seeded.prior
        assert events[0]["payload"]["resumes_run_id"] == seeded.prior
        assert events[0]["payload"]["root_build_run_id"] == seeded.prior


def test_resume_is_ineligible_on_an_interrupted_row_with_no_checkpoint(monkeypatch, tmp_path):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        seeded = _seed_resumable(monkeypatch, tmp_path, allow_root, root)
        con = sqlite3.connect(seeded.db)
        con.execute(
            "UPDATE workflow_events SET payload = ? WHERE run_id = ? AND type = 'run-finished'",
            (json.dumps({
                "contract_status": "interrupted", "exit_code": None, "report_path": None,
                "summary": "worker interrupted by SIGTERM", "stage": "worker",
                "diagnostic": "worker interrupted by SIGTERM", "head_sha": seeded.base_sha,
                "result": "failure", "phase": "P",
            }), seeded.prior),
        )
        con.commit()
        con.close()
        _checkpoint_path(root, seeded.prior).unlink()
        fake = FakeTmux()
        with pytest.raises(ji.Refusal) as caught:
            jaxflow_build.dispatch_build(
                _build_args(resume=seeded.prior),
                run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True},
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root,
            )
        assert caught.value.code == "resume-ineligible"


def test_build_resume_has_no_force_flag():
    with pytest.raises(SystemExit):
        jaxflow_cli.parse_args(["build", "--resume", "aaaaaaaaaaaa", "--force"])


def test_build_resume_fallback_uses_current_fallback_profile(monkeypatch, tmp_path):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        seeded = _seed_resumable(monkeypatch, tmp_path, allow_root, root)
        fake = FakeTmux()
        run_id, events, git_calls = _dispatch_resume(seeded, fake, fallback=True)
        assert run_id != seeded.prior
        assert events[0]["payload"]["requested_profile"] == "fallback"
        assert events[0]["payload"]["resumes_run_id"] == seeded.prior
        manifest = json.loads(
            (root / ".local" / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["requested_profile"] == "fallback"
        assert manifest["reservation_owned"] is False
        assert seeded.worktree.is_dir()
        assert not any(c[:3] == ["git", "worktree", "remove"] for c in git_calls)


def test_build_resume_fallback_e2e_reuses_worktree_and_keeps_original_base(monkeypatch, tmp_path):
    import jaxflow_resume as jresume

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
            "CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd", "TMUX_PANE": "%3",
            "PATH": "/bin", "HOME": str(tmp_path),
        }
        first_id = jaxflow_build.dispatch_build(
            _build_args(
                plan=str(plan), branch="feat/x", whitelist="a.py",
                verify="printf v >> .local/verify-ran",
                build="printf b >> .local/build-ran",
            ),
            run=_run_with_tmux(fake, real_cwd=root),
            post=post, env=env, now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        first_manifest = json.loads(
            (root / ".local" / "runs" / first_id / "manifest.json").read_text(encoding="utf-8"))
        worktree = Path(first_manifest["worktree"])
        base_sha = first_manifest["base_sha"]
        code = jaxflow_worker.run_worker(
            str(root / ".local" / "runs" / first_id / "manifest.json"),
            run=_run_with_tmux(fake, real_cwd=worktree),
            post=post, popen=_E2EBuilderPopen, allowlist_root=allow_root, env=env,
        )
        assert code == 0
        finished = [e for e in events if e["type"] == "run-finished"]
        assert finished[-1]["payload"]["result"] == "failure"
        assert (worktree / "leftover.txt").is_file()
        head_after_fail = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        assert head_after_fail != base_sha
        first_report = (worktree / ".local" / "reports" / f"{first_id}.md").read_bytes()
        first_tests = (worktree / ".local" / "reports" / f"{first_id}.tests.txt").read_bytes()
        first_checkpoint = _checkpoint_path(root, first_id).read_bytes()
        checkpoint = jresume.read_checkpoint(_checkpoint_path(root, first_id))
        assert checkpoint["outcome"] == "failure"
        assert checkpoint["base"] == base_sha
        assert (worktree / ".local" / "verify-ran").read_text(encoding="utf-8") == "v"
        assert (worktree / ".local" / "build-ran").read_text(encoding="utf-8") == "b"

        second_id = jaxflow_build.dispatch_build(
            _build_args(resume=first_id, fallback=True),
            run=_run_with_tmux(fake, real_cwd=root),
            post=post, env=env, now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        assert second_id != first_id
        second_manifest = json.loads(
            (root / ".local" / "runs" / second_id / "manifest.json").read_text(encoding="utf-8"))
        assert second_manifest["base_sha"] == base_sha
        assert second_manifest["worktree"] == str(worktree)
        assert second_manifest["requested_profile"] == "fallback"
        assert second_manifest["resumes_run_id"] == first_id
        code = jaxflow_worker.run_worker(
            str(root / ".local" / "runs" / second_id / "manifest.json"),
            run=_run_with_tmux(fake, real_cwd=worktree),
            post=post, popen=_E2EBuilderPopen, allowlist_root=allow_root, env=env,
        )
        assert code == 0
        finished = [e for e in events if e["type"] == "run-finished"]
        assert finished[-1]["run_id"] == second_id
        assert finished[-1]["payload"]["result"] == "success"
        assert second_manifest["base_sha"] == base_sha
        assert (worktree / "leftover.txt").read_text(encoding="utf-8") == "untracked\n"
        assert (worktree / ".local" / "verify-ran").read_text(encoding="utf-8") == "vv"
        assert (worktree / ".local" / "build-ran").read_text(encoding="utf-8") == "bb"
        assert (worktree / ".local" / "reports" / f"{first_id}.md").read_bytes() == first_report
        assert (worktree / ".local" / "reports" / f"{first_id}.tests.txt").read_bytes() == first_tests
        assert _checkpoint_path(root, first_id).read_bytes() == first_checkpoint
        caller_session = second_manifest["caller_session"]
        session_dir = jaxflow_common.CALLBACKS_ROOT / caller_session
        first_line = (session_dir / f"{first_id}.line").read_text(encoding="utf-8")
        second_line = (session_dir / f"{second_id}.line").read_text(encoding="utf-8")
        assert first_id in first_line
        assert second_id in second_line
        assert first_id not in second_line


def test_build_resume_uses_current_saved_profile_after_settings_change(monkeypatch, tmp_path):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    _E2EBuilderPopen.launches = 0
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        config, _, settings_path = _stub_launch_paths(tmp_path, monkeypatch)
        db = Path(raw) / "jaxos.db"
        _fresh_db(db).close()
        monkeypatch.setattr(jr, "DB_PATH", db)
        fake = FakeTmux()
        events = []
        post = _ledger_post(db, events)
        env = {
            "CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd", "TMUX_PANE": "%3",
            "PATH": "/bin", "HOME": str(tmp_path),
        }
        first_id = jaxflow_build.dispatch_build(
            _build_args(
                plan=str(plan), branch="feat/x", whitelist="a.py",
                verify="true",
            ),
            run=_run_with_tmux(fake, real_cwd=root),
            post=post, env=env, now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        jaxflow_worker.run_worker(
            str(root / ".local" / "runs" / first_id / "manifest.json"),
            run=_run_with_tmux(fake, real_cwd=Path(allow_root / "demo-feat-x")),
            post=post, popen=_E2EBuilderPopen, allowlist_root=allow_root, env=env,
        )
        _rewrite_json(settings_path, lambda data: (
            data.__setitem__("revision", "22222222222222222222222222222222"),
            data["builders"]["default"].__setitem__("model", "changed-model"),
        ))
        config["provider"]["fixture"]["models"]["jaxflow-builder-default"]["id"] = "changed-model"
        captured = {}

        class CapturePopen(_E2EBuilderPopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None,
                         start_new_session=None, env=None):
                super().__init__(
                    argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr,
                    start_new_session=start_new_session, env=env,
                )
                captured["argv"] = list(argv)

        second_id = jaxflow_build.dispatch_build(
            _build_args(resume=first_id),
            run=_run_with_tmux(fake, real_cwd=root),
            post=post, env=env, now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        resume_manifest = json.loads(
            (root / ".local" / "runs" / second_id / "manifest.json").read_text(encoding="utf-8"))
        assert resume_manifest["requested_profile"] == "default"
        assert resume_manifest["model"] == "fixture/changed-model"
        jaxflow_worker.run_worker(
            str(root / ".local" / "runs" / second_id / "manifest.json"),
            run=_run_with_tmux(fake, real_cwd=Path(resume_manifest["worktree"])),
            post=post, popen=CapturePopen, allowlist_root=allow_root, env=env,
        )
        argv = captured["argv"]
        assert argv[argv.index("--model") + 1] == "fixture/jaxflow-builder-default"
        assert "changed-model" not in argv
        on_disk = json.loads(
            (root / ".local" / "runs" / second_id / "manifest.json").read_text(encoding="utf-8"))
        assert on_disk["launch_selection"]["model"] == "fixture/changed-model"
        assert on_disk["launch_selection"]["runtime_model"] == "fixture/jaxflow-builder-default"
        assert on_disk["launch_selection"]["settings_revision"] == "22222222222222222222222222222222"
        assert first_id != second_id


@pytest.mark.parametrize("case", [
    "success", "cancelled", "no-checkpoint", "old-runtime", "reviewer",
    "nonterminal", "missing-tree", "newer-attempt", "live-session",
])
def test_build_resume_refuses_ineligible(monkeypatch, tmp_path, case):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        seeded = _seed_resumable(monkeypatch, tmp_path, allow_root, root)
        fake = FakeTmux()
        if case == "success":
            con = sqlite3.connect(seeded.db)
            con.execute(
                "UPDATE workflow_events SET payload = ? WHERE type = 'run-finished'",
                (json.dumps({
                    "contract_status": "ok", "result": "success", "exit_code": 0,
                    "report_path": "/tmp/r.md", "summary": "ok", "phase": "P",
                    "head_sha": seeded.base_sha,
                }),),
            )
            con.commit()
            con.close()
        elif case == "cancelled":
            con = sqlite3.connect(seeded.db)
            con.execute(
                "UPDATE workflow_events SET payload = ? WHERE type = 'run-finished'",
                (json.dumps({
                    "contract_status": "cancelled", "result": None, "exit_code": None,
                    "report_path": None, "summary": "cancelled", "phase": "P",
                    "head_sha": None,
                }),),
            )
            con.commit()
            con.close()
        elif case == "no-checkpoint":
            _checkpoint_path(root, seeded.prior).unlink()
        elif case == "old-runtime":
            manifest_path = root / ".local" / "runs" / seeded.prior / "manifest.json"
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            prior_manifest["runtime"] = "opencode-grok"
            del prior_manifest["requested_profile"]
            manifest_path.write_text(json.dumps(prior_manifest), encoding="utf-8")
            con = sqlite3.connect(seeded.db)
            con.execute(
                "UPDATE workflow_events SET payload = ? WHERE type = 'run-started'",
                (json.dumps({
                    "phase": "P", "runtime": "opencode-grok", "kind": "build",
                    "target": "feat/x", "session": f"jax-demo-build-{seeded.prior}",
                    "repo": str(root),
                }),),
            )
            con.commit()
            con.close()
        elif case == "reviewer":
            con = sqlite3.connect(seeded.db)
            con.execute("UPDATE workflow_events SET role = 'reviewer'")
            con.commit()
            con.close()
        elif case == "nonterminal":
            con = sqlite3.connect(seeded.db)
            con.execute("DELETE FROM workflow_events WHERE type = 'run-finished'")
            con.commit()
            con.close()
        elif case == "missing-tree":
            shutil.rmtree(seeded.worktree)
        elif case == "newer-attempt":
            con = sqlite3.connect(seeded.db)
            _insert(con, "ffffffffffff", "demo", "builder", "run-started", {
                "phase": "P", "runtime": "opencode-builder", "kind": "build",
                "target": "feat/x", "session": "jax-demo-build-ffffffffffff",
                "repo": str(root), "requested_profile": "default",
                "root_build_run_id": seeded.prior, "resumes_run_id": seeded.prior,
            })
            con.close()
        elif case == "live-session":
            fake.existing.add(f"jax-demo-build-{seeded.prior}")
        with pytest.raises(ji.Refusal) as caught:
            jaxflow_build.dispatch_build(
                _build_args(resume=seeded.prior),
                run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True},
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root,
            )
        assert caught.value.code == "resume-ineligible"
        if case != "missing-tree":
            assert seeded.worktree.is_dir()
        assert fake.calls[:1] != [["tmux", "new-session"]]


def _rewrite_checkpoint(root, run_id, **fields):
    import jaxflow_resume as jresume

    path = _checkpoint_path(root, run_id)
    data = jresume.read_checkpoint(path)
    data.update(fields)
    path.unlink()
    jresume.write_checkpoint(path, data)


def _assert_resume_refused_without_new_attempt(seeded, fake, monkeypatch):
    with pytest.raises(ji.Refusal) as caught:
        jaxflow_build.dispatch_build(
            _build_args(resume=seeded.prior),
            run=_run_with_tmux(fake, real_cwd=seeded.root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=seeded.allow_root,
        )
    assert caught.value.code == "resume-ineligible"
    assert seeded.worktree.is_dir()
    assert _checkpoint_path(seeded.root, seeded.prior).is_file()
    names = {p.name for p in (seeded.root / ".local" / "runs").iterdir()}
    assert seeded.prior in names
    assert names <= {seeded.prior, "locks"}
    assert fake.calls[:1] != [["tmux", "new-session"]]


def test_build_resume_refuses_forged_checkpoint_base_sha(monkeypatch, tmp_path):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        (root / "README").write_text("y\n", encoding="utf-8")
        _git(root, "add", "README")
        _git(root, "commit", "-m", "second")
        seeded = _seed_resumable(monkeypatch, tmp_path, allow_root, root)
        parent = _run_real(["git", "rev-parse", "HEAD^"], seeded.worktree).stdout.strip()
        assert parent and parent != seeded.base_sha
        _rewrite_checkpoint(root, seeded.prior, base=parent)
        _assert_resume_refused_without_new_attempt(seeded, FakeTmux(), monkeypatch)


def test_build_resume_refuses_forged_checkpoint_root_id(monkeypatch, tmp_path):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        seeded = _seed_resumable(monkeypatch, tmp_path, allow_root, root)
        _rewrite_checkpoint(root, seeded.prior, root_build_run_id="ffffffffffff")
        _assert_resume_refused_without_new_attempt(seeded, FakeTmux(), monkeypatch)


def test_build_resume_refuses_checkpoint_replaced_while_waiting_for_claim(monkeypatch, tmp_path):
    import jaxflow_resume as jresume

    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        (root / "README").write_text("y\n", encoding="utf-8")
        _git(root, "add", "README")
        _git(root, "commit", "-m", "second")
        seeded = _seed_resumable(monkeypatch, tmp_path, allow_root, root)
        parent = _run_real(["git", "rev-parse", "HEAD^"], seeded.worktree).stdout.strip()
        real_claim = jresume.worktree_claim

        def wrapped(repo, worktree):
            _rewrite_checkpoint(root, seeded.prior, base=parent)
            return real_claim(repo, worktree)

        monkeypatch.setattr(jresume, "worktree_claim", wrapped)
        _assert_resume_refused_without_new_attempt(seeded, FakeTmux(), monkeypatch)


def _prior_manifest(root, run_id):
    return root / ".local" / "runs" / run_id / "manifest.json"


def test_build_resume_refuses_tampered_default_to_fallback_profile(monkeypatch, tmp_path):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        seeded = _seed_resumable(monkeypatch, tmp_path, allow_root, root)
        path = _prior_manifest(root, seeded.prior)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["requested_profile"] = "fallback"
        path.write_text(json.dumps(data), encoding="utf-8")
        os.chmod(path, 0o644)
        _assert_resume_refused_without_new_attempt(seeded, FakeTmux(), monkeypatch)


@pytest.mark.parametrize("case", ["symlink", "malformed", "oversized", "group-write"])
def test_build_resume_refuses_untrusted_prior_manifest(monkeypatch, tmp_path, case):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        seeded = _seed_resumable(monkeypatch, tmp_path, allow_root, root)
        path = _prior_manifest(root, seeded.prior)
        if case == "symlink":
            outside = tmp_path / "outside.json"
            outside.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(outside)
        elif case == "malformed":
            path.write_text("[]", encoding="utf-8")
            os.chmod(path, 0o644)
        elif case == "oversized":
            data = json.loads(path.read_text(encoding="utf-8"))
            data["pad"] = "a" * (1 << 20)
            path.write_text(json.dumps(data), encoding="utf-8")
            os.chmod(path, 0o644)
        else:
            path.chmod(0o664)
        _assert_resume_refused_without_new_attempt(seeded, FakeTmux(), monkeypatch)


def test_build_resume_refuses_manifest_replaced_while_waiting_for_claim(monkeypatch, tmp_path):
    import jaxflow_resume as jresume

    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        seeded = _seed_resumable(monkeypatch, tmp_path, allow_root, root)
        real_claim = jresume.worktree_claim

        def wrapped(repo, worktree):
            path = _prior_manifest(root, seeded.prior)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["requested_profile"] = "fallback"
            path.write_text(json.dumps(data), encoding="utf-8")
            os.chmod(path, 0o644)
            return real_claim(repo, worktree)

        monkeypatch.setattr(jresume, "worktree_claim", wrapped)
        _assert_resume_refused_without_new_attempt(seeded, FakeTmux(), monkeypatch)


_TL2_MUTATIONS = (
    ("git", "merge", "--no-ff"),
    ("git", "worktree", "remove"),
    ("git", "branch", "-d"),
    ("git", "branch", "-D"),
    ("git", "push"),
    ("git", "commit"),
)


def _tl2_bind(cfg):
    jr.DB_PATH = Path(cfg["db"])
    jset.SETTINGS_PATH = Path(cfg["settings"])
    # A spawned child re-imports jaxflow, so the autouse callback isolation does not reach it.
    jaxflow_common.CALLBACKS_ROOT = Path(cfg["callbacks"])
    jaxflow_common.CLAUDE_SETTINGS_PATH = Path(cfg["claude_settings"])
    os.chdir(cfg["root"])


def _tl2_write_result(path, data):
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def _tl2_ledger_post(db, published=None, proceed=None, hold=False):
    def post(event):
        run_id = event.get("run_id")
        if run_id:
            con = sqlite3.connect(db, timeout=30)
            try:
                con.execute("PRAGMA busy_timeout=30000")
                _insert(con, run_id, event["project"], event["role"], event["type"], event["payload"])
            finally:
                con.close()
        if hold and published is not None and event.get("type") == "run-started":
            published.set()
            if proceed is not None:
                proceed.wait(30)
        return {"ok": True}
    return post


def _tl2_resume_child(cfg, start, published, proceed):
    result = {"ok": False}
    try:
        _tl2_bind(cfg)
        if start is not None and not start.wait(30):
            _tl2_write_result(cfg["result"], {"ok": False, "error": "start-timeout"})
            return
        if cfg.get("role") == "waiter" and published is not None and not published.wait(30):
            _tl2_write_result(cfg["result"], {"ok": False, "error": "published-timeout"})
            return
        run_id = jaxflow_build.dispatch_build(
            _build_args(resume=cfg["prior"]),
            run=_run_with_tmux(FakeTmux(), real_cwd=Path(cfg["root"])),
            post=_tl2_ledger_post(
                cfg["db"], published, proceed, hold=cfg.get("role") == "holder",
            ),
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=Path(cfg["allow_root"]),
            db_path=Path(cfg["db"]),
        )
        result = {"ok": True, "run_id": run_id}
    except ji.Refusal as exc:
        result = {"ok": False, "code": exc.code}
    except Exception as exc:
        result = {"ok": False, "error": type(exc).__name__, "msg": str(exc)[:300]}
    _tl2_write_result(cfg["result"], result)


def _tl2_merge_child(cfg, start, published, proceed):
    mutations = []
    try:
        _tl2_bind(cfg)
        if start is not None and not start.wait(30):
            _tl2_write_result(cfg["result"], {"ok": False, "error": "start-timeout", "mutations": []})
            return
        if cfg.get("role") == "waiter" and published is not None and not published.wait(30):
            _tl2_write_result(cfg["result"], {"ok": False, "error": "published-timeout", "mutations": []})
            return
        fake_run, _calls = _switch_aware_runner(Path(cfg["root"]), cfg["sha"])

        def run(argv, cwd=None):
            if any(tuple(argv[: len(prefix)]) == prefix for prefix in _TL2_MUTATIONS):
                mutations.append(list(argv))
            if (
                cfg.get("role") == "holder"
                and published is not None
                and argv[:3] == ["git", "rev-parse", "--verify"]
                and str(argv[-1]).endswith("^2")
            ):
                published.set()
                if proceed is not None:
                    proceed.wait(30)
            return fake_run(argv, cwd)

        rc = jaxflow_merge.cmd_merge(
            _MergeArgs(sha=cfg["sha"], checks="true"),
            run=run, post=_tl2_ledger_post(cfg["db"]), env=_merge_env(),
            now=_fixed_now, allowlist_root=Path(cfg["allow_root"]),
        )
        _tl2_write_result(cfg["result"], {"ok": True, "rc": rc, "mutations": mutations})
    except ji.Refusal as exc:
        _tl2_write_result(cfg["result"], {"ok": False, "code": exc.code, "mutations": mutations})
    except Exception as exc:
        _tl2_write_result(
            cfg["result"],
            {"ok": False, "error": type(exc).__name__, "msg": str(exc)[:300], "mutations": mutations},
        )


def _tl2_cfg(seeded, tmp_path, name, role=None):
    cfg = {
        "root": str(seeded.root),
        "allow_root": str(seeded.allow_root),
        "db": str(seeded.db),
        "settings": str(tmp_path / "agent-settings.json"),
        "callbacks": str(tmp_path / "callbacks"),
        "claude_settings": str(tmp_path / "claude-settings.json"),
        "prior": seeded.prior,
        "result": str(tmp_path / name),
        "sha": "a" * 40,
    }
    if role is not None:
        cfg["role"] = role
    return cfg


def _tl2_result(cfg):
    return json.loads(Path(cfg["result"]).read_text(encoding="utf-8"))


def _tl2_join(procs, timeout=30):
    deadline = time.monotonic() + timeout
    hung = []
    for proc in procs:
        proc.join(max(0, deadline - time.monotonic()))
        if proc.is_alive():
            hung.append(proc)
    for proc in hung:
        proc.terminate()
    for proc in hung:
        proc.join(2)
        if proc.is_alive():
            proc.kill()
            proc.join(2)
    if hung:
        raise AssertionError("tl2 child timed out")


def _tl2_pair(holder_target, waiter_target, holder_cfg, waiter_cfg):
    ctx = multiprocessing.get_context("spawn")
    start, published, proceed = ctx.Event(), ctx.Event(), ctx.Event()
    holder = ctx.Process(target=holder_target, args=(holder_cfg, start, published, proceed))
    waiter = ctx.Process(target=waiter_target, args=(waiter_cfg, start, published, proceed))
    holder.start()
    waiter.start()
    start.set()
    if not published.wait(30):
        proceed.set()
        _tl2_join([holder, waiter])
        raise AssertionError("holder never reached barrier")
    waiter.join(30)
    proceed.set()
    _tl2_join([holder, waiter])
    return _tl2_result(holder_cfg), _tl2_result(waiter_cfg)


def _tl2_new_runs(root, prior):
    runs = Path(root) / ".local" / "runs"
    return {p.name for p in runs.iterdir() if p.name not in {prior, "locks"}}


def _tl2_seed(monkeypatch, tmp_path):
    allow_root = tmp_path / "repos"
    root = allow_root / "demo"
    _init_repo(root)
    seeded = _seed_resumable(monkeypatch, tmp_path, allow_root, root)
    con = sqlite3.connect(seeded.db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.close()
    return seeded


def _assert_retained(seeded):
    assert seeded.worktree.is_dir()
    assert _checkpoint_path(seeded.root, seeded.prior).is_file()
    assert (seeded.root / ".local" / "runs" / seeded.prior / "manifest.json").is_file()


def test_two_resumes_yield_at_most_one_new_run(monkeypatch, tmp_path):
    seeded = _tl2_seed(monkeypatch, tmp_path)
    holder = _tl2_cfg(seeded, tmp_path, "holder.json", "holder")
    waiter = _tl2_cfg(seeded, tmp_path, "waiter.json", "waiter")
    held, waited = _tl2_pair(_tl2_resume_child, _tl2_resume_child, holder, waiter)
    assert held.get("ok") is True
    assert waited.get("ok") is not True
    assert waited.get("code") in ("agent-settings-permissions", "resume-ineligible")
    assert len(_tl2_new_runs(seeded.root, seeded.prior)) == 1
    # The spawned child re-imports jaxflow, so its callback pointer must land in the
    # isolated root the child was bound to, never the real ~/.jax-os/callbacks.
    assert len(list((tmp_path / "callbacks").rglob("*.json"))) == 1
    _assert_retained(seeded)


def test_resume_vs_merge_one_mutation_boundary(monkeypatch, tmp_path):
    seeded = _tl2_seed(monkeypatch, tmp_path)
    resume_cfg = _tl2_cfg(seeded, tmp_path, "resume.json", "waiter")
    merge_cfg = _tl2_cfg(seeded, tmp_path, "merge.json", "holder")
    merged, resumed = _tl2_pair(_tl2_merge_child, _tl2_resume_child, merge_cfg, resume_cfg)
    assert merged.get("ok") is True
    assert resumed.get("ok") is not True
    assert resumed.get("code") in ("agent-settings-permissions", "resume-ineligible")
    assert merged.get("mutations")
    assert len(_tl2_new_runs(seeded.root, seeded.prior)) == 0
    _assert_retained(seeded)


def test_competing_resume_and_merge_refuse_after_started_before_worker(monkeypatch, tmp_path):
    seeded = _tl2_seed(monkeypatch, tmp_path)
    first = _tl2_cfg(seeded, tmp_path, "first.json")
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_tl2_resume_child, args=(first, None, None, None))
    proc.start()
    _tl2_join([proc])
    launched = _tl2_result(first)
    assert launched.get("ok") is True
    new = _tl2_new_runs(seeded.root, seeded.prior)
    assert len(new) == 1
    _assert_retained(seeded)
    resume_cfg = _tl2_cfg(seeded, tmp_path, "resume.json")
    merge_cfg = _tl2_cfg(seeded, tmp_path, "merge.json")
    resume_proc = ctx.Process(target=_tl2_resume_child, args=(resume_cfg, None, None, None))
    merge_proc = ctx.Process(target=_tl2_merge_child, args=(merge_cfg, None, None, None))
    resume_proc.start()
    merge_proc.start()
    _tl2_join([resume_proc, merge_proc])
    resumed = _tl2_result(resume_cfg)
    merged = _tl2_result(merge_cfg)
    # Either contender can lose the claim before checking the active run.
    assert resumed.get("ok") is not True
    assert resumed.get("code") in ("agent-settings-permissions", "resume-ineligible")
    assert merged.get("ok") is not True
    assert merged.get("code") in ("agent-settings-permissions", "resume-ineligible")
    assert not merged.get("mutations")
    assert _tl2_new_runs(seeded.root, seeded.prior) == new
    _assert_retained(seeded)


# ---- build: dispatch order + builder-role cancelled payload (spec §4.2/§2.4) ----

def test_build_dispatch_posts_run_started_before_tmux_with_builder_defaults(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        run_id = jaxflow_build.dispatch_build(
            _build_args(plan=str(plan)), run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"CODEX_THREAD_ID": "t1"}, now=_fixed_now, allowlist_root=allow_root,
        )
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["runtime"] == "opencode-grok"
        assert payload["model"] == "xai/grok-4.6"
        assert payload["effort"] == "n/a"
        assert payload["caller"] == "codex"
        tmux_cmds = [c[1] for c in fake.calls]
        assert tmux_cmds == ["has-session", "new-session"]


def test_build_tmux_failure_posts_cancelled_row_with_null_head_sha_and_cleans_worktree(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        fake.new_session_fails = True
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        try:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan), branch="feat/x"), run=_run_with_tmux(fake, real_cwd=root),
                post=post, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
                allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert exc.code == "tmux-failed"
        else:
            raise AssertionError("tmux-failed not raised")
        cancelled = events[1]["payload"]
        assert cancelled["contract_status"] == "cancelled"
        assert cancelled["head_sha"] is None
        assert not (allow_root / "demo-feat-x").exists()
        assert _run_real(["git", "rev-parse", "--verify", "feat/x"], root).returncode != 0


def test_build_tmux_new_session_raises_is_treated_as_tmux_failed(monkeypatch):
    # fixes cold review F7: an EXCEPTION from the new-session call (not just a non-zero
    # exit) must produce the same cancelled-row-then-refuse-then-cleanup sequence.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        def raising_run(argv, cwd=None):
            if argv[:2] == ["tmux", "new-session"]:
                raise OSError("tmux binary not found")
            return _run_with_tmux(FakeTmux(), real_cwd=root)(argv, cwd=cwd)

        try:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan), branch="feat/y"), run=raising_run, post=post,
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
                allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert exc.code == "tmux-failed"
        else:
            raise AssertionError("tmux-failed not raised")
        assert [e["type"] for e in events] == ["run-started", "run-finished"]
        assert events[1]["payload"]["contract_status"] == "cancelled"
        assert not (allow_root / "demo-feat-y").exists()
        assert _run_real(["git", "rev-parse", "--verify", "feat/y"], root).returncode != 0


def test_build_hub_unreachable_removes_manifest_worktree_and_branch_without_tmux_call(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()

        def post(event):
            raise RuntimeError("event post failed")

        try:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan), branch="feat/z"), run=_run_with_tmux(fake, real_cwd=root),
                post=post, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
                allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert exc.code == "hub-unreachable"
        else:
            raise AssertionError("hub-unreachable not raised")
        assert [c[1] for c in fake.calls] == ["has-session"]
        assert not (allow_root / "demo-feat-z").exists()
        assert not (root / ".local" / "runs").exists() or not any((root / ".local" / "runs").iterdir())
        assert _run_real(["git", "rev-parse", "--verify", "feat/z"], root).returncode != 0


def test_build_quotes_tmux_command_for_space_and_semicolon_in_repo_path(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo app; two"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()

        run_id = jaxflow_build.dispatch_build(
            _build_args(plan=str(plan)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        new_session_call = next(c for c in fake.calls if c[1] == "new-session")
        cmd = new_session_call[-1]
        manifest_path = root / ".local" / "runs" / run_id / "manifest.json"
        assert shlex.split(cmd) == [
            "env", "HONCHO_ENABLED=false", "JAXFLOW_ONESHOT=1",
            sys.executable, str(jaxflow_common.SCRIPT_PATH), "--run-worker", str(manifest_path),
        ]


# ---- build: --builder override + runtime-not-allowed end-to-end (spec §7.1's own note that
# slice a could not reach this refusal through the CLI -- `--builder` makes it reachable now) ----

def test_build_builder_override_and_runtime_not_allowed_end_to_end(capsys, monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        run_id = jaxflow_build.dispatch_build(
            _build_args(plan=str(plan), branch="feat/y", builder="opencode-deepseek"),
            run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
            allowlist_root=allow_root,
        )
        assert events[0]["payload"]["runtime"] == "opencode-deepseek"
        # An OpenCode runtime records its RESOLVED model id, not the "default" sentinel
        # (same rule the opencode-grok case above asserts).
        assert events[0]["payload"]["model"] == "openrouter/deepseek/deepseek-v4-flash-0731"

        code = jaxflow_cli.main(
            ["build", "--plan", str(plan), "--phase", "P", "--branch", "feat/z",
             "--whitelist", "a.py", "--verify", "true", "--builder", "bogus-runtime"],
            run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "runtime-not-allowed"
        assert not (allow_root / "demo-feat-z").exists()


def test_build_refuses_builder_on_default_branch_after_worktree_checkout(monkeypatch):
    # fixes cold review F1's own note that this refusal is only reachable when the fresh
    # branch NAME collides with the hardcoded {default, "main", "master"} set -- here the
    # repo's real default is "trunk" and "main" does not exist yet, so `git worktree add -b
    # main` succeeds and preflight (run against the new worktree) is what then refuses.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        root.mkdir(parents=True)
        _git(root, "init", "-b", "trunk")
        _git(root, "config", "user.email", "t@t.test")
        _git(root, "config", "user.name", "t")
        (root / "README").write_text("x\n", encoding="utf-8")
        (root / ".gitignore").write_text(".local/\n", encoding="utf-8")
        _git(root, "add", "README", ".gitignore")
        _git(root, "commit", "-m", "init")
        # fixture fix (execution rule): `jr._default_branch()` only resolves a real
        # non-main/master default via `refs/remotes/origin/HEAD` -- without this, it falls
        # back to the LITERAL string "main" even though "main" is not a real ref here,
        # which makes `git worktree add -b main <path> main` itself fail with "invalid
        # reference: main" (probed live) instead of reaching preflight at all. Setting this
        # symbolic ref (no real remote needed -- `_default_branch` only reads the ref's
        # target name, never dereferences it) makes `_default_branch()` correctly resolve
        # "trunk", matching what this test's own comment already claims.
        _git(root, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/trunk")
        plan = root / ".local" / "docs" / "plans" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Plan\n\n### Task 1: t\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        try:
            jaxflow_build.dispatch_build(
                _build_args(plan=str(plan), branch="main"),
                run=_run_with_tmux(fake, real_cwd=root), post=lambda e: {"ok": True},
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
                allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            assert exc.code == "builder-on-default-branch"
        else:
            raise AssertionError("builder-on-default-branch not raised")
        assert not (allow_root / "demo-main").exists()
        assert _run_real(["git", "rev-parse", "--verify", "main"], root).returncode != 0
