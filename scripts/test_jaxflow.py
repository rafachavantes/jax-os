#!/usr/bin/env python3
"""Stdlib-only tests for jaxflow.py (slice a)."""
import contextlib
import io
import json
import multiprocessing
import os
import re
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import time
import urllib.error
from subprocess import CompletedProcess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

import general_settings
import jaxflow
import jaxflow_workerkit
import jaxflow_common
import uuid
import shlex
import jaxflow_hook
import jaxflow_run as jr
import jaxflow_settings as jset
import jax_init as ji

_ALL_AGENTS_ON = {"claude": True, "codex": True, "opencode": True}


def _agents_setting(monkeypatch, **overrides):
    """Patch general_settings.read_settings with the given per-agent overrides (default all on)."""
    on = {**_ALL_AGENTS_ON, **overrides}
    monkeypatch.setattr(general_settings, "read_settings", lambda: {
        "ok": True, "data": {"integrations": {"classifier": True, "github": True, "agents": on}}})


@pytest.fixture(autouse=True)
def _integration_on_by_default(monkeypatch):
    # MOA-502 Decisions 1-2: build_review_tally reads integrations.classifier and the gh call
    # sites read integrations.github, both live through general_settings.read_settings().
    # Pre-existing tests exercise the real (non-injected) Jev/gh paths, so default both ON
    # here; the gate-specific tests re-patch read_settings with their own off fixture, which
    # wins (their setattr applies after this autouse one).
    # MOA-504: and integrations.agents (all on) for the dispatch gates.
    monkeypatch.setattr(general_settings, "read_settings",
                        lambda: {"ok": True, "data": {"integrations": {"classifier": True, "github": True, "agents": _ALL_AGENTS_ON}}})


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """run_worker installs real SIGTERM/SIGHUP handlers (fixes cold review F4/P8 --
    without this, a worker test leaves its handler installed for every test that runs
    after it in the same process)."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


@pytest.fixture(autouse=True)
def _isolate_agent_settings(monkeypatch):
    monkeypatch.setattr(jset, "SETTINGS_PATH", Path("/nonexistent/jaxflow-agent-settings.json"))


@pytest.fixture(autouse=True)
def _isolate_callbacks(tmp_path, monkeypatch):
    """Every callback pointer/.line file this suite writes lands under an isolated
    per-test tmp dir -- never the real ~/.jax-os/callbacks (mirrors
    _isolate_agent_settings above). The Claude hook-installed probe (D9) is pointed at a
    file that already contains a STRUCTURALLY valid claude-callback PostToolUse/Bash
    hook entry (cold review round 1 F6 -- D9 checks structure, not a substring), so that
    warning is silent by default across the whole suite; the tests that specifically
    exercise D9 override this locally."""
    monkeypatch.setattr(jaxflow_common, "CALLBACKS_ROOT", tmp_path / "callbacks")
    settings_path = tmp_path / "claude-settings.json"
    settings_path.write_text(json.dumps({
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python3 /home/rafa/repos/jax-os/scripts/jaxflow_hook.py claude-callback",
                            "asyncRewake": True,
                            "timeout": 14400,
                        }
                    ],
                }
            ]
        }
    }), encoding="utf-8")
    monkeypatch.setattr(jaxflow_common, "CLAUDE_SETTINGS_PATH", settings_path)


_SETTINGS_FIXTURE = Path(__file__).resolve().parents[1] / "workflow" / "fixtures" / "agent-settings-v1.json"

# A canonical (lowercase, hyphenated) placeholder Claude session id -- every manifest
# helper and inline env dict in this file used the old placeholders `s`/`sess-1` before
# this fixture sweep, which is not a shape `jaxflow.UUID_RE.fullmatch` accepts (spec
# §5.1); this value is.
_TEST_CLAUDE_SESSION_ID = "01234567-89ab-4cde-8f01-23456789abcd"


def _install_settings(tmp_path, monkeypatch):
    target = tmp_path / "agent-settings.json"
    target.write_bytes(_SETTINGS_FIXTURE.read_bytes())
    target.chmod(0o600)
    monkeypatch.setattr(jset, "SETTINGS_PATH", target)
    return target


def _native_config():
    def model_entry(allow_fallbacks):
        return {
            "id": "wire-real-model",
            "variants": {"high": {"reasoning": {"effort": "high"}}},
            "options": {"provider": {"sort": "price", "allow_fallbacks": allow_fallbacks}},
        }
    return {
        "provider": {
            "fixture": {
                "npm": "@openrouter/ai-sdk-provider",
                "models": {
                    "jaxflow-builder-default": model_entry(True),
                    "jaxflow-builder-fallback": model_entry(False),
                },
            },
        },
        "permission": {"read": "deny"},
        "agent": {"build": {}},
    }


def _stub_launch_paths(tmp_path, monkeypatch, *, native=None, settings=True):
    monkeypatch.setattr(jset, "LOCK_PATH", tmp_path / "agent-settings.lock")
    monkeypatch.setattr(jset, "SOURCE_PATHS", {
        "opencode.json": tmp_path / "opencode.json",
        "opencode.jsonc": tmp_path / "opencode.jsonc",
    })
    env_path = tmp_path / "hermes.env"
    env_path.write_text(
        "JAX_PROVIDER_FIXTURE_API_KEY=secret-value\nBWS_ACCESS_TOKEN=machine-token\n",
        encoding="utf-8",
    )
    env_path.chmod(0o600)
    monkeypatch.setattr(jset, "ENV_PATH", env_path)
    config = native if native is not None else _native_config()
    monkeypatch.setattr(jset, "read_effective_opencode_config", lambda **kw: config)
    settings_path = _install_settings(tmp_path, monkeypatch) if settings else None
    return config, env_path, settings_path


def _rewrite_json(path, mutate):
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    path.chmod(0o600)


def _managed_worker_repo(tmp_path, monkeypatch, **extra):
    allow_root = tmp_path / "repos"
    root = allow_root / "demo"
    _init_repo(root)
    worktree = _init_worktree(root)
    _write_plan(worktree)
    kwargs = {
        "runtime": "opencode-builder",
        "model": "fixture/wire-real-model",
        "effort": "high",
        "requested_profile": "default",
        "root_build_run_id": "bbbbccccdddd",
    }
    kwargs.update(extra)
    path, manifest = _write_manifest_for_builder_worker(root, worktree, **kwargs)
    db = tmp_path / "jaxos.db"
    con = sqlite3.connect(db) if db.exists() else _fresh_db(db)
    con.execute("DELETE FROM workflow_events WHERE run_id = ?", (manifest["run_id"],))
    _insert(con, manifest["run_id"], manifest["project"], "builder", "run-started", {
        k: manifest.get(k) for k in (
            "phase", "runtime", "kind", "target", "caller", "caller_session",
            "model", "effort", "session", "repo", "verify", "requested_profile",
            "root_build_run_id", "resumes_run_id",
        )
    })
    con.close()
    monkeypatch.setattr(jr, "DB_PATH", db)
    return allow_root, root, worktree, path, manifest


def _capture_builder_popen():
    captured = {}

    class CapturePopen(FakeBuilderPopen):
        def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None,
                     start_new_session=None, env=None):
            super().__init__(
                argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr,
                start_new_session=start_new_session, env=env,
            )
            captured["argv"] = list(argv)
            captured["env"] = None if env is None else dict(env)

    return captured, CapturePopen


def _run_builder_worker_test(worktree, manifest_path, allow_root, popen, env=None):
    events = []
    code = jaxflow.run_worker(
        str(manifest_path),
        run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
        post=lambda e: events.append(e) or {"ok": True},
        popen=popen,
        allowlist_root=allow_root,
        env=env,
    )
    return code, events


def _run_real(argv, cwd=None):
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)


def _completed(rc, out="", err=""):
    """The shape an injected `run` returns. The older fakes build this inline with
    SimpleNamespace; the merge fixtures name it once."""
    return SimpleNamespace(returncode=rc, stdout=out, stderr=err)


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "t@t.test")
    _git(root, "config", "user.name", "t")
    (root / "README").write_text("x\n", encoding="utf-8")
    (root / ".gitignore").write_text(".local/\n", encoding="utf-8")
    _git(root, "add", "README", ".gitignore")
    _git(root, "commit", "-m", "init")


def _spec_file(root: Path, name="2026-01-01-demo-spec.md", text="# Demo spec\n"):
    docs = root / ".local" / "docs" / "specs"
    docs.mkdir(parents=True, exist_ok=True)
    path = docs / name
    path.write_text(text, encoding="utf-8")
    return path


class FakeTmux:
    def __init__(self):
        self.calls = []
        self.existing = set()
        self.new_session_fails = False
        self.incarnation = "111:222"

    def handle(self, argv):
        self.calls.append(list(argv))
        cmd = argv[1]
        if cmd == "has-session":
            name = argv[argv.index("-t") + 1]
            return SimpleNamespace(returncode=0 if name in self.existing else 1, stdout="", stderr="")
        if cmd == "new-session":
            if self.new_session_fails:
                return SimpleNamespace(returncode=1, stdout="", stderr="fail")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd == "display-message":
            return SimpleNamespace(returncode=0, stdout=self.incarnation + "\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def _run_with_tmux(fake, real_cwd=None):
    def run(argv, cwd=None):
        if argv and argv[0] == "tmux":
            return fake.handle(argv)
        return _run_real(argv, cwd if cwd is not None else real_cwd)
    return run


def _run_with_tmux_and_log(fake, log, real_cwd=None):
    # Same as `_run_with_tmux`, but also records every non-tmux (git) argv -- used by
    # tests that need to prove NO git call happened before a given refusal (Part 1
    # diff-review F2).
    def run(argv, cwd=None):
        if argv and argv[0] == "tmux":
            return fake.handle(argv)
        log.append(list(argv))
        return _run_real(argv, cwd if cwd is not None else real_cwd)
    return run


def _fixed_now():
    # One injected, timezone-aware clock for every dispatch test in this file (fixes cold
    # review F5) -- `now=lambda: None` broke the moment Task 2 adds `_iso8601(now())` to
    # every manifest.
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def _build_args(**kw):
    resume = kw.get("resume")
    return SimpleNamespace(
        plan=kw.get("plan") if resume else kw["plan"],
        phase=kw.get("phase", None if resume else "P"),
        branch=kw.get("branch", None if resume else "feat/demo"),
        whitelist=kw.get("whitelist", None if resume else "a.py,b.py"),
        verify=kw.get("verify", None if resume else "true"),
        builder=kw.get("builder"), model=kw.get("model"), effort=kw.get("effort"),
        from_caller=kw.get("from_caller"), no_callback=kw.get("no_callback", False),
        base=kw.get("base"), build=kw.get("build"), fallback=kw.get("fallback", False),
        resume=resume,
    )


def _plan_file(root):
    plan = root / ".local" / "docs" / "plans" / "plan.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# Plan\n\n**Goal:** g\n\n### Task 1: t\n", encoding="utf-8")
    return plan


def test_split_spec_fragment_handles_no_hash_malformed_and_well_formed():
    assert jaxflow_common._split_spec_fragment("spec.md") == ("spec.md", None)
    assert jaxflow_common._split_spec_fragment("spec.md#Heading") == ("spec.md", "Heading")
    # malformed -- empty fragment: treated as no fragment, whole raw string as path
    assert jaxflow_common._split_spec_fragment("spec.md#") == ("spec.md#", None)
    # malformed -- fragment containing '..': same treatment
    assert jaxflow_common._split_spec_fragment("spec.md#../escape") == ("spec.md#../escape", None)
    # split at the LAST '#' when there is more than one
    assert jaxflow_common._split_spec_fragment("spec.md#one#two") == ("spec.md#one", "two")


def test_find_heading_matches_any_level_case_sensitively():
    text = "# T\n\n## Right Case\ntext\n\n###### Deep\nmore\n"
    assert jaxflow_common._find_heading(text, "Right Case") is True
    assert jaxflow_common._find_heading(text, "Deep") is True
    assert jaxflow_common._find_heading(text, "right case") is False
    assert jaxflow_common._find_heading(text, "Nope") is False


def test_iso8601_matches_stdlib_isoformat_seconds():
    # `_iso8601` used to hand-splice a colon into strftime's %z; this proves
    # `dt.isoformat(timespec="seconds")` gives byte-identical output for every
    # aware datetime this codebase actually feeds it (UTC via `_fixed_now`, a
    # negative offset, and one with microseconds -- both drop to whole seconds
    # the same way).
    for dt in (
        _fixed_now(),
        datetime(2026, 9, 6, 10, 0, 0, tzinfo=timezone(timedelta(hours=-3))),
        datetime(2026, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc),
    ):
        assert jaxflow_common._iso8601(dt) == dt.isoformat(timespec="seconds")


def _review_args(**kw):
    return SimpleNamespace(
        spec=kw.get("spec"), plan=kw.get("plan"), focus=kw.get("focus"),
        model=kw.get("model"), effort=kw.get("effort"), phase=kw.get("phase"),
        from_caller=kw.get("from_caller"), no_callback=kw.get("no_callback", False),
    )


# ---- caller detection ----

def test_resolve_caller_from_env_and_from_flag():
    assert jaxflow_common.resolve_caller({"CLAUDECODE": "1"}, None) == "claude"
    assert jaxflow_common.resolve_caller({"CODEX_THREAD_ID": "1"}, None) == "codex"
    for bad_env in ({}, {"CLAUDECODE": "1", "CODEX_THREAD_ID": "1"}):
        try:
            jaxflow_common.resolve_caller(bad_env, None)
        except ji.Refusal as exc:
            assert exc.code == "caller-unknown"
        else:
            raise AssertionError("caller-unknown not raised")
    assert jaxflow_common.resolve_caller({"CLAUDECODE": "1", "CODEX_THREAD_ID": "1"}, "codex") == "codex"


# ---- caller-session-missing (acceptance smoke 2026-09-06, fix 4) ----

def test_dispatch_refuses_caller_session_missing_before_manifest_and_post(capsys, monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        posted = []

        def post(event):
            posted.append(event)
            return {"ok": True}

        code = jaxflow.main(
            ["review", "--spec", str(target), "--from", "codex"],
            run=_run_with_tmux(fake, real_cwd=root), post=post, env={}, allowlist_root=allow_root,
        )
        err = capsys.readouterr().err.strip().splitlines()
        assert code == jaxflow_common.REFUSED
        assert err[0] == "caller-session-missing"
        assert "CODEX_THREAD_ID" in err[1]
        assert posted == []
        assert not (root / ".local" / "runs").exists()
        assert not (root / ".local" / "reports").exists() or not any((root / ".local" / "reports").iterdir())


def test_dispatch_proceeds_when_caller_session_present(monkeypatch):
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

        jaxflow.dispatch_review(
            _review_args(spec=str(target), from_caller="codex"),
            run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"CODEX_THREAD_ID": "thread-1"}, now=_fixed_now, allowlist_root=allow_root,
        )
        assert events[0]["payload"]["caller_session"] == "thread-1"


# ---- project slug ----

def test_slugify_project_bounds_and_fallback():
    assert jaxflow_common.slugify_project("Acme.AI") == "acme-ai"
    assert jaxflow_common.slugify_project("comenta.ia.br") == "comenta-ia-br"
    long_name = "a" * 70 + "---"
    slug = jaxflow_common.slugify_project(long_name)
    # 40, not 64 (fixes cold review F2): jax-<slug>-<kind>-<run_id> must stay under the
    # ledger's 80-char `session` bound -- 4 + 40 + 1 + 5("build") + 1 + 12 = 63 < 80.
    assert len(slug) == 40
    assert not slug.endswith("-")
    assert jaxflow_common.slugify_project("___") == "repo"


def test_slug_keeps_session_name_within_ledger_bound():
    slug = jaxflow_common.slugify_project("a" * 70)
    session = f"jax-{slug}-build-{'a' * 12}"
    assert len(session) <= 80


# ---- refusal mapping ----

# `runtime-not-allowed` has no end-to-end test through the slice-a CLI surface: the
# reviewer runtime is derived (never chosen by a flag) and `--builder` — the only slice-a
# input that could pick a disallowed runtime — arrives in slice b, so this unit test on
# `_map_refusal` is the only coverage this code path gets in this slice.
def test_map_refusal_translates_known_messages_and_passes_through_unknown():
    assert jaxflow_common._map_refusal("detached HEAD") == "detached-head"
    assert jaxflow_common._map_refusal("builder cannot run on default branch") == "builder-on-default-branch"
    assert jaxflow_common._map_refusal("runtime not allowed") == "runtime-not-allowed"
    assert jaxflow_common._map_refusal("dirty-tracked-tree") == "dirty-tracked-tree"
    assert jaxflow_common._map_refusal("malformed project") == "malformed project"


# ---- not-a-git-toplevel ----

def test_dispatch_refuses_not_a_git_toplevel(capsys, monkeypatch):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        root.mkdir(exist_ok=True)
        monkeypatch.chdir(root)
        code = jaxflow.main(
            ["review", "--spec", str(root / "x.md")], run=_run_real, post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1"},
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "not-a-git-toplevel"


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

        run_id = jaxflow.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now, allowlist_root=allow_root,
        )
        payload = events[0]["payload"]
        assert payload["repo"] == str(root)
        assert payload["session"] == f"jax-demo-spec-{run_id}"


# ---- path allowlist ----

def test_dispatch_refuses_path_outside_allowlist(capsys, monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        outside = Path(raw) / "outside"
        root = allow_root / "demo"
        _init_repo(root)
        outside.mkdir()
        target = outside / "spec.md"
        target.write_text("# spec\n", encoding="utf-8")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        code = jaxflow.main(
            ["review", "--spec", str(target)], run=_run_with_tmux(fake), post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1"}, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"


# ---- secret target refusal (fixes cold review F3) ----

def test_dispatch_refuses_secret_target_before_manifest_and_post(capsys, monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        posted = []

        def post(event):
            posted.append(event)
            return {"ok": True}

        for name in (".env.local", "id_rsa", "foo.pem"):
            target = root / name
            code = jaxflow.main(
                ["review", "--spec", str(target)], run=_run_with_tmux(fake, real_cwd=root), post=post,
                env={"CLAUDECODE": "1"}, allowlist_root=allow_root,
            )
            assert code == jaxflow_common.REFUSED
            assert capsys.readouterr().err.strip() == f"secret-detected: {target}"
        assert posted == []
        assert not (root / ".local" / "runs").exists()


def test_dispatch_refuses_secret_target_case_insensitive(capsys, monkeypatch):
    # fixes branch review G1(b): _is_secret_path used to be case-sensitive, so
    # `.ENV`/`.PEM`/`id_RSA` variants sailed past the dispatcher guard.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        target = root / ".ENV"
        target.write_text("SECRET=1\n", encoding="utf-8")
        code = jaxflow.main(
            ["review", "--spec", str(target)], run=_run_with_tmux(fake, real_cwd=root), post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1"}, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == f"secret-detected: {target}"


def test_worker_refuses_secret_manifest_target_without_reading_or_posting(capsys):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = root / ".env"
        target.write_text("SECRET=x\n", encoding="utf-8")
        manifest_path, manifest = _write_manifest_for_worker(root, target)
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()), post=post, popen=FakePopen, allowlist_root=root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == f"secret-detected: {target}"
        assert events == []
        assert not (root / ".local" / "reports").exists()


def test_worker_refuses_secret_target_via_symlink_using_resolved_path(capsys):
    # fixes branch review G1(a): the worker used to apply the secret guard to the
    # UNRESOLVED manifest target, so a `review.md -> .env` symlink alias read the
    # secret straight through. It must be judged (and reported) on the resolved path.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        secret = root / ".env"
        secret.write_text("SECRET=x\n", encoding="utf-8")
        alias = root / "review.md"
        alias.symlink_to(secret)
        manifest_path, manifest = _write_manifest_for_worker(root, alias)
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()), post=post, popen=FakePopen, allowlist_root=root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == f"secret-detected: {secret.resolve()}"
        assert events == []
        assert not (root / ".local" / "reports").exists()


def test_worker_refuses_target_outside_allowlist_root(capsys):
    # fixes branch review G1(a): the worker never re-checked containment on its own
    # manifest target, so a manifest pointing outside the allowlist root was read anyway.
    with TemporaryDirectory() as raw:
        raw_root = Path(raw).resolve()
        allow_root = raw_root / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        outside = raw_root / "outside.md"
        outside.write_text("# outside\n", encoding="utf-8")
        manifest_path, manifest = _write_manifest_for_worker(root, outside)
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()), post=post, popen=FakePopen,
            allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"
        assert events == []
        assert not (root / ".local" / "reports").exists()


# ---- dispatch order + run-started payload + hub-unreachable/tmux-failed ----

def test_dispatch_posts_run_started_before_tmux_and_prints_run_id(capsys, monkeypatch):
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

        code = jaxflow.main(
            ["review", "--spec", str(target)], run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, allowlist_root=allow_root,
        )
        out = capsys.readouterr().out.strip()
        assert code == jaxflow_common.OK
        run_id = out
        assert len(run_id) == 12
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["kind"] == "spec"
        assert payload["target"] == str(target)
        assert payload["caller"] == "claude"
        assert payload["caller_session"] == "01234567-89ab-4cde-8f01-23456789abcd"
        assert payload["model"] == "gpt-5.6-luna"
        assert payload["effort"] == "xhigh"
        assert payload["runtime"] == "codex"
        assert payload["repo"] == str(root)
        assert payload["session"] == f"jax-demo-spec-{run_id}"
        assert "caller_pane" not in payload
        assert "verify" not in payload
        tmux_cmds = [c[1] for c in fake.calls]
        assert tmux_cmds == ["has-session", "new-session"]


def test_dispatch_from_codex_caller_defaults_to_claude_sonnet_xhigh(capsys, monkeypatch):
    # The codex row of REVIEWER_DEFAULTS had no test at all until 2026-09-08 — only the
    # claude row above was pinned, so raising BOTH efforts to xhigh could have silently
    # regressed this direction. Both tech leads now get the same effort.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        events = []

        code = jaxflow.main(
            ["review", "--spec", str(target)], run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: events.append(e) or {"ok": True},
            env={"CODEX_THREAD_ID": "thr-1"}, allowlist_root=allow_root,
        )
        capsys.readouterr()
        assert code == jaxflow_common.OK
        payload = events[0]["payload"]
        assert payload["caller"] == "codex"
        assert payload["caller_session"] == "thr-1"
        assert payload["runtime"] == "claude"
        assert payload["model"] == "sonnet"
        assert payload["effort"] == "xhigh"


# ---- doc-review test-output evidence (acceptance smoke 2026-09-06, fix 2) ----

def test_dispatch_writes_doc_review_test_marker_before_dispatch(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()

        run_id = jaxflow.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        tests_file = root / ".local" / "reports" / f"{run_id}.tests.txt"
        assert tests_file.read_text(encoding="utf-8") == jaxflow.DOC_REVIEW_TEST_MARKER
        assert jaxflow.DOC_REVIEW_TEST_MARKER == "No test/build command was required by this handoff.\n"


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
            jaxflow.dispatch_review(
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
            jaxflow.dispatch_review(
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

        run_id = jaxflow.dispatch_review(
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

        jaxflow.dispatch_review(
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
        run_id = jaxflow.dispatch_review(
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
        run_id = jaxflow.dispatch_build(
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
        first_id = jaxflow.dispatch_build(
            _build_args(plan=str(plan), branch="feat/x", whitelist="a.py", verify="false"),
            run=_run_with_tmux(fake, real_cwd=root),
            post=post, env=env, now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        worktree = allow_root / "demo-feat-x"
        jaxflow.run_worker(
            str(root / ".local" / "runs" / first_id / "manifest.json"),
            run=_run_with_tmux(fake, real_cwd=worktree),
            post=post, popen=_E2EBuilderPopen, allowlist_root=allow_root, env=env,
        )
        second_id = jaxflow.dispatch_build(
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
        run_id = jaxflow.dispatch_diff_review(
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
        jaxflow.dispatch_review(
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
        jaxflow.dispatch_review(
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
        run_id = jaxflow.dispatch_review(
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
            jaxflow.dispatch_review(
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
            jaxflow.dispatch_build(
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
        code = jaxflow.main(
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
            jaxflow.dispatch_build(
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
        run_id = jaxflow.dispatch_review(
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
        run_id = jaxflow.dispatch_review(
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
        jaxflow.dispatch_review(
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
        run_id = jaxflow.dispatch_review(
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
        run_id = jaxflow.dispatch_review(
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
        run_id = jaxflow.dispatch_review(
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
        run_id = jaxflow.dispatch_review(
            _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": _TEST_CLAUDE_SESSION_ID},
            now=_fixed_now, allowlist_root=allow_root,
        )
        err = capsys.readouterr().err
        assert "no claude-callback hook installed" in err
        assert run_id in err


# ---- refusal translation end-to-end (detached HEAD) ----

def test_dispatch_detached_head_maps_to_kebab_code(capsys, monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        _git(root, "checkout", "--detach")
        monkeypatch.chdir(root)
        fake = FakeTmux()
        code = jaxflow.main(
            ["review", "--spec", str(target)], run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1"}, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "detached-head"


# ---- tmux command quoting (fixes cold review F7/G16) ----

def test_dispatch_quotes_tmux_command_for_space_and_semicolon_in_repo_path(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo app; two"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()

        run_id = jaxflow.dispatch_review(
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


# ---- worker: launches, stdin, process group, claude redirect ----

class FakePopen:
    """Stands in for the real LLM subprocess: writes the fixed report text either
    to its own stdout (claude path) or to the --output-last-message file codex was
    given in argv (codex path) -- mirroring what each real runtime does. MOA-470:
    also exposes `.stdout`/`.stderr` as readable in-memory streams when the caller
    requests `subprocess.PIPE`, so a reader loop has something to drain. Class-level
    `stdout`/`stderr` defaults of `None` are what let every OTHER fake in this file
    that never sets them (the report-fault fakes) work with `_capture_child_output`
    unchanged -- plain attribute inheritance, zero code in those classes."""
    stdout = None
    stderr = None
    CHILD_BYTES = b""  # a subclass overrides this to feed specific bytes to capture
    REPORT_TEXT = (
        "---\nrun_id: aaaabbbbcccc\nproject: demo\nrole: reviewer\nphase: PHASE\n"
        "verdict: approve\nsummary: looks fine\n---\nbody\n"
    )

    def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
        self.argv = argv
        self.cwd = cwd
        self.start_new_session = start_new_session
        self.env = env
        self.pid = 4242
        self.returncode = 0
        self.stdin_bytes = stdin.read() if stdin is not None else None
        # Codex writes its final report to --output-last-message REGARDLESS of what
        # stdout is wired to (spec §4.2) -- checked independently of the stdout branch
        # below, not chained as an elif of it.
        if "--output-last-message" in argv:
            out_path = Path(argv[argv.index("--output-last-message") + 1])
            out_path.write_text(type(self).REPORT_TEXT, encoding="utf-8")
        if stdout is subprocess.PIPE:
            self.stdout = io.BytesIO(self.CHILD_BYTES)
        elif stdout not in (None, subprocess.PIPE):
            stdout.write(type(self).REPORT_TEXT.encode("utf-8"))
        if stderr is subprocess.PIPE:
            self.stderr = io.BytesIO(self.CHILD_BYTES)

    def wait(self, timeout=None):
        return self.returncode


def _write_manifest_for_worker(root, target, *, run_id="aaaabbbbcccc", runtime="codex", **extra):
    manifest = {
        "kind": "spec", "role": "reviewer", "project": "demo", "phase": "PHASE",
        "repo": str(root), "target": str(target), "caller": "claude",
        "caller_session": "01234567-89ab-4cde-8f01-23456789abcd", "model": "m", "effort": "high",
        "runtime": runtime, "run_id": run_id, "session": f"jax-demo-spec-{run_id}",
        "no_callback": False, "focus": None,
    }
    manifest.update(extra)
    path = root / ".local" / "runs" / run_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_worker_pipes_prompt_on_stdin_for_both_runtimes_and_finalizes():
    for runtime in ("codex", "claude"):
        with TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            _init_repo(root)
            target = _spec_file(root, text="REVIEW ME\n")
            manifest_path, manifest = _write_manifest_for_worker(root, target, runtime=runtime)
            events = []

            def post(event):
                events.append(event)
                return {"ok": True}

            code = jaxflow.run_worker(
                str(manifest_path), run=_run_with_tmux(FakeTmux()), post=post, popen=FakePopen,
                allowlist_root=root,
            )
            assert code == 0
            assert len(events) == 1
            payload = events[0]["payload"]
            assert payload["contract_status"] == "ok", runtime
            assert payload["verdict"] == "approve", runtime
            report = root / ".local" / "reports" / "aaaabbbbcccc.md"
            assert report.is_file()
            assert "REVIEW ME" not in json.dumps(payload)


def test_worker_claude_runtime_redirects_stdout_to_reviewer_output_path():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        manifest_path, manifest = _write_manifest_for_worker(root, target, runtime="claude")
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()), post=post, popen=FakePopen, allowlist_root=root,
        )
        assert code == 0
        assert events[0]["payload"]["contract_status"] == "ok"


def test_worker_stdin_carries_prompt_text_no_shell_argv_leak():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root, text="SECRET-HANDOFF-TEXT\n")
        manifest_path, manifest = _write_manifest_for_worker(root, target, runtime="codex")
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["argv"] = argv
                captured["start_new_session"] = start_new_session
                captured["stdin_bytes"] = self.stdin_bytes
                captured["env"] = env

        snapshot = dict(os.environ)
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()), post=lambda e: {"ok": True}, popen=CapturePopen,
            allowlist_root=root,
        )
        assert captured["start_new_session"] is True
        assert b"SECRET-HANDOFF-TEXT" not in captured["stdin_bytes"]
        assert "SECRET-HANDOFF-TEXT" not in " ".join(captured["argv"])
        # sealed env reaches the child on a COPY; the real os.environ is never touched
        # (fixes cold review F4).
        assert captured["env"]["HONCHO_ENABLED"] == "false"
        assert captured["env"]["JAXFLOW_ONESHOT"] == "1"
        assert dict(os.environ) == snapshot


# ---- composed prompt: preamble + doc-review evidence line (acceptance smoke 2026-09-06,
# fixes 2 and 3) ----

def test_worker_prompt_starts_with_preamble_and_names_test_evidence_path():
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

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()), post=lambda e: {"ok": True}, popen=CapturePopen,
            allowlist_root=root,
        )
        prompt_text = captured["stdin_bytes"].decode("utf-8")
        assert prompt_text.startswith(jaxflow_workerkit.REVIEWER_PROMPT_PREAMBLE)
        tests_path = root / ".local" / "reports" / f"{manifest['run_id']}.tests.txt"
        assert f"Test-output evidence: {tests_path}" in prompt_text


def test_worker_prompt_frames_document_under_review_before_document_and_after_repo_block():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root, text="# Demo spec\n")
        manifest_path, manifest = _write_manifest_for_worker(root, target, runtime="codex")
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["stdin_bytes"] = self.stdin_bytes

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()), post=lambda e: {"ok": True}, popen=CapturePopen,
            allowlist_root=root,
        )
        prompt_text = captured["stdin_bytes"].decode("utf-8")
        document_marker = f"Document under review: {target}"
        review_kind_marker = "Review kind: spec — this is a document review."
        repo_line = f"repo: {root}\n"
        assert document_marker in prompt_text
        assert review_kind_marker in prompt_text
        assert prompt_text.index(repo_line) < prompt_text.index(document_marker)
        assert "# Demo spec" not in prompt_text
        assert prompt_text.index(document_marker) < prompt_text.index(
            "Read the named document and required evidence from disk before reviewing."
        )


@pytest.mark.parametrize("caller,runtime", [("codex", "claude"), ("claude", "codex")])
@pytest.mark.parametrize("kind", ["spec", "plan"])
@pytest.mark.parametrize("external", [False, True])
def test_doc_review_file_inputs_and_directory_grants(monkeypatch, caller, runtime, kind, external):
    with TemporaryDirectory() as raw:
        allowed = Path(raw).resolve()
        root = allowed / "demo repo"
        _init_repo(root)
        source = allowed / "other repo" if external else root
        if external:
            _init_repo(source)
        marker = "MOA460_BODY_MUST_BE_READ_FROM_DISK"
        target = _spec_file(source, text=f"# Document\nRafa gate: required\n\n{marker}\n")
        manifest_path, manifest = _write_manifest_for_worker(
            root, target, runtime=runtime, caller=caller, kind=kind,
        )
        assert jaxflow.REVIEWER_DEFAULTS[caller]["runtime"] == runtime
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured["child"] = self

        monkeypatch.setattr(
            jaxflow_common, "_update_status_md",
            lambda state, **kwargs: captured.update(gate=state.get("spec_gate_required")),
        )
        assert jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda event: {"ok": True}, popen=CapturePopen, allowlist_root=allowed,
        ) == 0
        child = captured["child"]
        prompt = child.stdin_bytes.decode("utf-8")
        assert marker not in prompt
        assert f"Document under review: {target}" in prompt
        assert "Read the named document" in prompt
        assert f"Test-output evidence: {root}/.local/reports/aaaabbbbcccc.tests.txt" in prompt
        assert captured["gate"] is True
        assert child.cwd == "/tmp"
        if runtime == "claude":
            granted = child.argv[child.argv.index("--add-dir") + 1:]
            assert granted == [str(root)] + ([str(target.parent)] if external else [])
            assert child.argv[child.argv.index("--tools") + 1] == "Read,Glob,Grep"
            assert child.argv[child.argv.index("--setting-sources") + 1] == ""
        else:
            assert "--add-dir" not in child.argv
            assert child.argv[child.argv.index("-C") + 1] == "/tmp"


def test_doc_review_refuses_outside_repo_before_directory_grant(capsys):
    with TemporaryDirectory() as raw:
        parent = Path(raw).resolve()
        allowed = parent / "allowed"
        root = allowed / "demo"
        _init_repo(root)
        target = _spec_file(root)
        manifest_path, manifest = _write_manifest_for_worker(root, target, runtime="claude")
        manifest["repo"] = str(parent / "outside")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        def no_child(*args, **kwargs):
            raise AssertionError("outside repo must not reach the runtime")

        assert jaxflow.run_worker(
            str(manifest_path), popen=no_child, post=lambda event: {"ok": True},
            allowlist_root=allowed,
        ) == jaxflow_common.REFUSED
        assert "path-outside-allowlist" in capsys.readouterr().err


# ---- MOA-467: minimal read-only directory grants for the Claude reviewer, derived from
# git metadata (never folder names), deduplicated, no sibling-repo grants, Codex argv
# unchanged ----

@pytest.mark.parametrize("kind", ["spec", "plan"])
def test_claude_doc_review_from_linked_worktree_grants_control_repo(kind):
    # A review STARTED in a linked worktree gets that worktree (its argv cwd) AND its
    # canonical control repo -- the repo names carry spaces, proving the grant is a plain
    # argv entry, never a shell-joined string.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        root = allow_root / "demo repo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        target = _spec_file(root, text="REVIEW ME\n")
        manifest_path, manifest = _write_manifest_for_worker(
            worktree, target, runtime="claude", kind=kind,
        )
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["argv"] = argv

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=allow_root,
        )
        assert code == 0
        granted = captured["argv"][captured["argv"].index("--add-dir") + 1:]
        assert granted == [str(worktree), str(root)]


def test_claude_doc_review_from_separate_git_dir_main_grants_its_linked_worktree():
    # MOA-467 acceptance fixes: a `--separate-git-dir` main whose common dir IS named
    # `.git` (the shape the old common-dir NAME check misread) must resolve to the
    # CHECKOUT -- the review grants the main plus the linked worktree holding the named
    # document, never the gitdir's parent.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        allow_root.mkdir()
        main = allow_root / "sep"
        gitdir = allow_root / "cache" / ".git"
        _init_separate_git_dir_repo(main, gitdir)
        worktree = _init_worktree(main, "feat/x")
        target = _spec_file(worktree, text="REVIEW ME\n")
        manifest_path, manifest = _write_manifest_for_worker(
            main, target, runtime="claude", kind="spec",
        )
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["argv"] = argv

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=main),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=allow_root,
        )
        assert code == 0
        granted = captured["argv"][captured["argv"].index("--add-dir") + 1:]
        assert granted == [str(main), str(worktree)]
        assert str(allow_root / "cache") not in granted


@pytest.mark.parametrize("kind", ["spec", "plan"])
def test_claude_doc_review_from_main_grants_the_linked_worktree_root(kind):
    # MOA-467 acceptance: a review launched in the CONTROL repo but targeting a document
    # inside one of its linked worktrees grants that worktree's ROOT -- not just the
    # nested document folder -- plus the control repo: the same inputs a review started
    # in the worktree would get.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        root = allow_root / "demo repo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        target = _spec_file(worktree, text="REVIEW ME\n")
        manifest_path, manifest = _write_manifest_for_worker(
            root, target, runtime="claude", kind=kind,
        )
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["argv"] = argv

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=allow_root,
        )
        assert code == 0
        granted = captured["argv"][captured["argv"].index("--add-dir") + 1:]
        assert granted == [str(root), str(worktree)]
        assert str(target.parent) not in granted


def test_codex_doc_review_from_main_into_a_worktree_argv_is_unchanged():
    # Codex has no directory-grant mechanism: the related-worktree grant the Claude path
    # receives changes nothing -- no --add-dir, -C stays /tmp.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        target = _spec_file(worktree, text="REVIEW ME\n")
        manifest_path, manifest = _write_manifest_for_worker(
            root, target, runtime="codex",
        )
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["argv"] = argv

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=allow_root,
        )
        assert code == 0
        argv = captured["argv"]
        assert "--add-dir" not in argv
        assert argv[argv.index("-C") + 1] == "/tmp"


def test_codex_doc_review_from_linked_worktree_argv_is_unchanged():
    # Codex has no directory-grant mechanism: no --add-dir ever appears, -C stays /tmp,
    # and the extra grants the Claude path would receive change nothing.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        target = _spec_file(root, text="REVIEW ME\n")
        manifest_path, manifest = _write_manifest_for_worker(
            worktree, target, runtime="codex",
        )
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["argv"] = argv

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=allow_root,
        )
        assert code == 0
        argv = captured["argv"]
        assert "--add-dir" not in argv
        assert argv[argv.index("-C") + 1] == "/tmp"
        # the worker itself still runs in /tmp for doc reviews (unchanged)
        assert argv[:2] == ["codex", "exec"]


def test_claude_diff_review_grants_worktree_control_and_external_spec_parent_deduped():
    # A diff review's Claude reviewer reads: the build worktree (argv cwd), the canonical
    # control repo (originals live there), and -- only for a declared spec that lives
    # outside every granted root -- that spec's OWN parent, never a sibling repo root.
    # The allowlist root carries a space so every granted path exercises space handling
    # (the slug-derived worktree identity check keeps working -- it compares paths, not
    # names).
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos dir"
        allow_root.mkdir()
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        (worktree / "changed.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "change")
        base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        spec_src = allow_root / "other repo" / "the-spec.md"
        spec_src.parent.mkdir(parents=True)
        spec_src.write_text("# The spec\n", encoding="utf-8")
        plan_src = root / ".local" / "docs" / "plans" / "plan.md"
        plan_src.parent.mkdir(parents=True)
        plan_src.write_text(
            f"# P\n\n**Goal:** g\n\n**Spec:** `{spec_src}`\n\n### Task 1: t\n",
            encoding="utf-8",
        )
        tests_path = worktree / ".local" / "reports" / "b1b1b1b1b1b1.tests.txt"
        tests_path.parent.mkdir(parents=True)
        tests_path.write_text("COMMAND: true\n\nEXIT: 0\n", encoding="utf-8")
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, runtime="claude", base_sha=base_sha, head_sha=head_sha,
            plan_path=str(plan_src), spec_path=str(spec_src), tests_path=str(tests_path),
        )
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["argv"] = argv

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=allow_root,
        )
        assert code == 0
        granted = captured["argv"][captured["argv"].index("--add-dir") + 1:]
        assert granted == [str(worktree), str(root), str(spec_src.parent)]


def test_claude_diff_review_grants_no_unrelated_sibling_dirs():
    # A sibling repo sitting right next to the control repo is never granted when no
    # named document lives in it -- and the control repo grant is not repeated for
    # documents inside it (deduplicated).
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos dir"
        allow_root.mkdir()
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        sibling = allow_root / "sibling"
        _init_repo(sibling)
        (worktree / "changed.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "change")
        base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        plan_src = root / ".local" / "docs" / "plans" / "plan.md"
        plan_src.parent.mkdir(parents=True)
        plan_src.write_text("# P\n\n**Goal:** g\n\n### Task 1: t\n", encoding="utf-8")
        tests_path = worktree / ".local" / "reports" / "b1b1b1b1b1b1.tests.txt"
        tests_path.parent.mkdir(parents=True)
        tests_path.write_text("COMMAND: true\n\nEXIT: 0\n", encoding="utf-8")
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, runtime="claude", base_sha=base_sha, head_sha=head_sha,
            plan_path=str(plan_src), spec_path=str(plan_src), tests_path=str(tests_path),
        )
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["argv"] = argv

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=allow_root,
        )
        assert code == 0
        granted = captured["argv"][captured["argv"].index("--add-dir") + 1:]
        assert granted == [str(worktree), str(root)]
        assert str(sibling) not in granted


def test_terminate_child_sends_term_then_kill_on_timeout():
    calls = []

    class FakeChild:
        def __init__(self):
            self.waited = 0

        def wait(self, timeout=None):
            self.waited += 1
            if self.waited == 1:
                raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
            return 0

    def fake_killpg(pgid, sig):
        calls.append((pgid, sig))

    child = FakeChild()
    jaxflow_workerkit._terminate_child(child, 999, killpg=fake_killpg, timeout=0.01)
    assert calls == [(999, signal.SIGTERM), (999, signal.SIGKILL)]
    assert child.waited == 2


def test_interrupted_payload_builder_with_no_worktree_has_null_head_sha():
    payload = jaxflow_workerkit._interrupted_payload(role="builder", phase="P1", signal_name="SIGTERM")
    assert payload == {
        "phase": "P1", "exit_code": None, "contract_status": "interrupted",
        "report_path": None, "stage": "worker", "diagnostic": "worker interrupted by SIGTERM",
        "summary": "worker interrupted by SIGTERM", "head_sha": None, "result": "failure",
    }


def test_interrupted_payload_builder_reads_real_head_when_descendant(tmp_path):
    worktree = tmp_path / "wt"
    _init_repo(worktree)
    base_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
    (worktree / "f.txt").write_text("x\n", encoding="utf-8")
    _git(worktree, "add", "f.txt")
    _git(worktree, "commit", "-m", "feat: work")
    head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
    payload = jaxflow_workerkit._interrupted_payload(
        role="builder", phase="P1", signal_name="SIGHUP",
        worktree=worktree, base_sha=base_sha, run=jr.run_command,
    )
    assert payload["head_sha"] == head_sha
    assert payload["result"] == "failure"


def test_interrupted_payload_reviewer_has_no_builder_only_fields():
    payload = jaxflow_workerkit._interrupted_payload(role="reviewer", phase="P1", signal_name="SIGTERM")
    assert "head_sha" not in payload
    assert "result" not in payload
    assert payload["stage"] == "worker"


def test_interrupted_payload_uses_a_terminal_error_last_line_when_given():
    # MOA-474 cold review F3: an existing diagnostic line upgrades stage to "runtime"
    # with the real cause, instead of the plain worker-interrupted text.
    last_line = json.dumps({  # real OpenCode shape: the cause nests under `error`
        "type": "error", "sessionID": "ses_x",
        "error": {"name": "APIError", "data": {"statusCode": 403, "message": "spending limit reached"}},
    })
    payload = jaxflow_workerkit._interrupted_payload(
        role="builder", phase="P1", signal_name="SIGTERM", last_line=last_line,
    )
    assert payload["stage"] == "runtime"
    assert payload["diagnostic"] == "APIError 403 spending limit reached"
    assert payload["summary"] == jr._bound(payload["diagnostic"], 200)


def test_interrupted_payload_falls_back_to_plain_text_on_a_non_error_last_line():
    payload = jaxflow_workerkit._interrupted_payload(
        role="builder", phase="P1", signal_name="SIGTERM",
        last_line=json.dumps({"type": "step_finish", "reason": "stop"}),
    )
    assert payload["stage"] == "worker"
    assert payload["diagnostic"] == "worker interrupted by SIGTERM"


def test_worker_signal_during_wait_kills_process_group_then_posts_interrupted(monkeypatch):
    """Fixes cold review F6: delivers a real SIGTERM while run_worker is inside
    child.wait(), via a fake child whose own .wait() raises the signal with
    signal.raise_signal (reliable synchronous delivery, unlike os.kill). The handler's
    real os._exit(0) is neutralized so the test process survives; the pre-existing
    `if terminated["flag"]: return 0` guard is what then makes run_worker return
    cleanly instead of falling through to finalize/post. MOA-474 §10.2: the handler
    itself now posts exactly one interrupted row."""
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        manifest_path, manifest = _write_manifest_for_worker(root, target, runtime="codex")
        killpg_calls = []

        def fake_killpg(pgid, sig):
            killpg_calls.append((pgid, sig))

        monkeypatch.setattr(os, "_exit", lambda code: None)

        class SignalingChild:
            def __init__(self):
                self.pid = os.getpid()  # a real, getpgid-able pid -- no fake needed
                self.returncode = 0
                self.wait_calls = 0
                self.stdout = None
                self.stderr = None

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    signal.raise_signal(signal.SIGTERM)
                    return 0
                if self.wait_calls == 2:
                    raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
                return 0

        def fake_popen(argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
            if stdin is not None:
                stdin.read()
            return SignalingChild()

        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()), post=post,
            popen=fake_popen, killpg=fake_killpg, allowlist_root=root,
        )
        assert code == 0
        assert [sig for _pgid, sig in killpg_calls] == [signal.SIGTERM, signal.SIGKILL]
        assert killpg_calls[0][0] == killpg_calls[1][0]
        # MOA-474 §10.2: the signal path now posts exactly ONE terminal row, built
        # directly (never through derive_outcome, D5) -- never the normal-completion
        # path's own post (which this fake child's .wait() never reaches: `wait_calls
        # == 2` raises TimeoutExpired, so run_worker's own child.wait() call is what
        # observes the signal, not a second real completion).
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["contract_status"] == "interrupted"
        assert payload["stage"] == "worker"
        assert payload["diagnostic"] == "worker interrupted by SIGTERM"
        assert payload["exit_code"] is None
        assert payload["report_path"] is None
        assert "head_sha" not in payload  # this manifest is a reviewer (doc) run


def test_worker_builder_signal_posts_interrupted_with_real_head_and_checkpoint(monkeypatch):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        base_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _write_plan(worktree)
        (worktree / "changed.txt").write_text("partial\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "feat: partial work before the crash")
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        # MOA-474 cold review 887b1d3994d4 F2: a NON-managed manifest (no requested_profile)
        # -- a managed one makes run_worker resolve the launch selection (agent settings,
        # DB row) and refuse before the child ever runs. The checkpoint half of this
        # behaviour is proven separately below over the managed-worker helpers.
        manifest_path, manifest = _write_manifest_for_builder_worker(
            root, worktree, base_sha=base_sha,
        )
        killpg_calls = []

        def fake_killpg(pgid, sig):
            killpg_calls.append((pgid, sig))

        monkeypatch.setattr(os, "_exit", lambda code: None)

        class SignalingChild:
            def __init__(self):
                self.pid = os.getpid()
                self.returncode = 0
                self.wait_calls = 0
                self.stdout = None
                self.stderr = None

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    signal.raise_signal(signal.SIGTERM)
                    return 0
                if self.wait_calls == 2:
                    raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
                return 0

        def fake_popen(argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
            if stdin is not None:
                stdin.read()
            return SignalingChild()

        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True},
            popen=fake_popen, killpg=fake_killpg, allowlist_root=root.parent,
        )
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["contract_status"] == "interrupted"
        assert payload["result"] == "failure"
        assert payload["stage"] == "worker"
        assert payload["diagnostic"] == "worker interrupted by SIGTERM"
        assert payload["head_sha"] == head_sha
        spool_path = jaxflow_common._spool_path(root, manifest["run_id"])
        assert not spool_path.exists()  # delivered (post returned {"ok": True}) -> deleted
        assert not (jaxflow_common._manifest_dir(root, manifest["run_id"]) / "resume-checkpoint.json").exists()


def test_worker_builder_signal_writes_checkpoint_for_managed_run(tmp_path, monkeypatch):
    # The checkpoint half: a MANAGED manifest, set up exactly like the existing managed
    # worker tests (`_stub_launch_paths` + `_managed_worker_repo`), so run_worker reaches
    # the child.
    _stub_launch_paths(tmp_path, monkeypatch)
    allow_root, root, worktree, manifest_path, manifest = _managed_worker_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(os, "_exit", lambda code: None)

    class SignalingChild:
        def __init__(self):
            self.pid = os.getpid()
            self.returncode = 0
            self.wait_calls = 0
            self.stdout = None
            self.stderr = None

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                signal.raise_signal(signal.SIGTERM)
            return 0

    def popen(argv, **kw):
        return SignalingChild()

    # Explicit no-op killpg: run_worker's own default binds os.killpg at import time,
    # so a monkeypatched os.killpg would NOT reach it -- and this child's pid is the
    # test process itself, so a real killpg would signal the whole test session.
    events = []
    jaxflow.run_worker(
        str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
        post=lambda e: events.append(e) or {"ok": True},
        popen=popen, killpg=lambda pgid, sig: None, allowlist_root=allow_root,
    )
    assert [e["payload"]["contract_status"] for e in events] == ["interrupted"]
    checkpoint_path = jaxflow_common._manifest_dir(root, manifest["run_id"]) / "resume-checkpoint.json"
    assert checkpoint_path.exists()
    assert json.loads(checkpoint_path.read_text(encoding="utf-8"))["outcome"] == "failure"


def test_worker_builder_signal_before_popen_still_posts_interrupted(monkeypatch):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        base_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree, base_sha=base_sha)
        monkeypatch.setattr(os, "_exit", lambda code: None)
        events = []

        def popen(argv, **kw):
            # The signal lands while Popen is "in progress": no child exists yet.
            signal.raise_signal(signal.SIGTERM)
            raise AssertionError("handler must have exited the worker before Popen returned")

        # With os._exit patched to a no-op the handler returns into `popen`, which then
        # raises -- so the AssertionError propagates out of run_worker; the events the
        # handler posted before that are what this test asserts.
        with pytest.raises(AssertionError):
            jaxflow.run_worker(
                str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
                post=lambda e: events.append(e) or {"ok": True}, popen=popen,
                killpg=lambda pgid, sig: None, allowlist_root=root.parent,
            )
        assert [e["payload"]["contract_status"] for e in events] == ["interrupted"]
        assert events[0]["payload"]["stage"] == "worker"
        assert events[0]["payload"]["diagnostic"] == "worker interrupted by SIGTERM"


def test_worker_diff_reviewer_signal_posts_interrupted_no_verdict(monkeypatch):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        # MOA-474 cold review 887b1d3994d4 F3: the diff worker runs a real
        # `git diff --name-only <base>..<head>` before Popen, so the helper's default
        # placeholder SHAs (a*40/b*40) would refuse before the handler is ever exercised.
        base_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        (worktree / "changed.txt").write_text("partial\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "feat: reviewed work")
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, base_sha=base_sha, head_sha=head_sha,
        )
        killpg_calls = []

        def fake_killpg(pgid, sig):
            killpg_calls.append((pgid, sig))

        monkeypatch.setattr(os, "_exit", lambda code: None)

        class SignalingChild:
            def __init__(self):
                self.pid = os.getpid()
                self.returncode = 0
                self.wait_calls = 0
                self.stdout = None
                self.stderr = None

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    signal.raise_signal(signal.SIGTERM)
                    return 0
                if self.wait_calls == 2:
                    raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
                return 0

        def fake_popen(argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
            if stdin is not None:
                stdin.read()
            return SignalingChild()

        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True},
            popen=fake_popen, killpg=fake_killpg, allowlist_root=root.parent,
        )
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["contract_status"] == "interrupted"
        assert payload["stage"] == "worker"
        assert payload["diagnostic"] == "worker interrupted by SIGTERM"
        assert "verdict" not in payload
        assert "head_sha" not in payload


def test_worker_doc_reviewer_signal_posts_interrupted_no_verdict(monkeypatch):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        manifest_path, manifest = _write_manifest_for_worker(root, target, runtime="codex")
        killpg_calls = []

        def fake_killpg(pgid, sig):
            killpg_calls.append((pgid, sig))

        monkeypatch.setattr(os, "_exit", lambda code: None)

        class SignalingChild:
            def __init__(self):
                self.pid = os.getpid()
                self.returncode = 0
                self.wait_calls = 0
                self.stdout = None
                self.stderr = None

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    signal.raise_signal(signal.SIGTERM)
                    return 0
                if self.wait_calls == 2:
                    raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
                return 0

        def fake_popen(argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
            if stdin is not None:
                stdin.read()
            return SignalingChild()

        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()),
            post=lambda e: events.append(e) or {"ok": True},
            popen=fake_popen, killpg=fake_killpg, allowlist_root=root,
        )
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["contract_status"] == "interrupted"
        assert payload["stage"] == "worker"
        assert payload["diagnostic"] == "worker interrupted by SIGTERM"
        assert "verdict" not in payload
        assert "head_sha" not in payload


def test_worker_post_failure_prints_delivery_failed_and_leaves_callback_line_exact(capsys):
    """Fixes cold review F6: the callback line format is fixed (`[JAXFLOW] <kind>
    <run_id> finished — <verdict|result>[ · <stage> — <diagnostic>] —
    <report path|no report>`) whether or not the finish POST succeeded; a failed POST
    is reported separately, to the worker's own stdout (the pane), never by mutating
    the callback line."""
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        fake = FakeTmux()
        manifest_path, manifest = _write_manifest_for_worker(root, target)

        def post(event):
            raise RuntimeError("event post failed")

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(fake), post=post, popen=FakePopen, allowlist_root=root,
        )
        assert code == 0
        line_path = jaxflow_common.CALLBACKS_ROOT / manifest["caller_session"] / "aaaabbbbcccc.line"
        line = line_path.read_text(encoding="utf-8").rstrip("\n")
        report = root / ".local" / "reports" / "aaaabbbbcccc.md"
        assert line == (
            f"[JAXFLOW] spec aaaabbbbcccc finished — approve — {report} · ledger pending"
        )
        assert "event delivery FAILED" not in line
        assert "event delivery FAILED: event post failed" in capsys.readouterr().out


@pytest.mark.parametrize("caller", ["claude", None, "unknown", "codex"])
def test_worker_no_callback_flag_sends_nothing(monkeypatch, caller):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        fake = FakeTmux()
        monkeypatch.setattr(
            subprocess, "run",
            lambda argv, **kwargs: pytest.fail(f"queue invoked under no_callback: {argv}"),
        )
        manifest_path, manifest = _write_manifest_for_worker(
            root, target, caller=caller, no_callback=True,
        )
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(fake), post=lambda e: {"ok": True}, popen=FakePopen,
            allowlist_root=root,
        )
        assert not any(c[1] == "send-keys" for c in fake.calls)
        assert not jaxflow_common.CALLBACKS_ROOT.exists() or not any(jaxflow_common.CALLBACKS_ROOT.rglob("*.line"))


_CAPTURED_THREAD = "11111111-2222-4333-8444-555555555555"
_WORKER_THREAD = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
_CALLBACK_SECRET = "sentinel-secret-not-for-logs"
_QUEUE_KW = dict(stdin=subprocess.DEVNULL, capture_output=True, timeout=10, check=False)


def _forbidden_tmux(argv, cwd=None):
    pytest.fail(f"Codex callback attempted tmux: {argv}")


def _intercept_codex_queue(monkeypatch, impl):
    original = subprocess.run

    def wrapped(argv, **kwargs):
        if list(argv[:2]) == ["codex", "queue"]:
            return impl(argv, **kwargs)
        return original(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", wrapped)


@pytest.mark.parametrize("kind", ["spec", "plan", "diff", "build"])
@pytest.mark.parametrize("pane", [None, "%3"])
def test_codex_callback_targets_captured_thread(monkeypatch, kind, pane):
    captured = _CAPTURED_THREAD
    monkeypatch.setenv("CODEX_THREAD_ID", _WORKER_THREAD)
    monkeypatch.setattr(time, "sleep", lambda seconds: pytest.fail("Codex callback slept"))
    calls = []

    def queue(argv, **kwargs):
        calls.append((argv, kwargs))
        return CompletedProcess(argv, 0, b"accepted", b"")

    monkeypatch.setattr(subprocess, "run", queue)
    manifest = dict(run_id="aaaabbbbcccc", caller="codex",
                    caller_session=captured, caller_pane=pane,
                    caller_incarnation="111:222")
    jaxflow_workerkit._send_callback(manifest, run=_forbidden_tmux, kind=kind,
                           outcome="success", summary="done", report_path="/report.md")
    assert calls == [([
        "codex", "queue", "--thread", captured, "--message",
        f"[JAXFLOW] {kind} aaaabbbbcccc finished — success — /report.md"
    ], dict(stdin=subprocess.DEVNULL, capture_output=True, timeout=10, check=False))]


@pytest.mark.parametrize("caller,session", [
    ("codex", _CAPTURED_THREAD),
    ("claude", "01234567-89ab-4cde-8f01-23456789abcd"),
])
def test_no_callback_skips_queue_and_tmux(monkeypatch, caller, session):
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kwargs: pytest.fail(f"queue invoked under no_callback: {argv}"),
    )
    monkeypatch.setattr(time, "sleep", lambda seconds: pytest.fail("sleep under no_callback"))
    manifest = dict(run_id="aaaabbbbcccc", caller=caller, caller_session=session,
                    caller_pane="%3", caller_incarnation="111:222", no_callback=True)
    jaxflow_workerkit._send_callback(
        manifest, run=_forbidden_tmux, kind="spec", outcome="success",
        summary="done", report_path="/report.md",
    )


@pytest.mark.parametrize("session", [
    None,
    "",
    "my-session-name",
    123,
    "AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE",
    "11111111222243338444555555555555",
])
def test_codex_callback_invalid_session_warns_without_delivery(monkeypatch, capsys, session):
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kwargs: pytest.fail(f"queue invoked for invalid session: {argv}"),
    )
    monkeypatch.setattr(time, "sleep", lambda seconds: pytest.fail("sleep on invalid session"))
    manifest = dict(run_id="aaaabbbbcccc", caller="codex", caller_session=session,
                    caller_pane="%3", caller_incarnation="111:222")
    jaxflow_workerkit._send_callback(
        manifest, run=_forbidden_tmux, kind="spec", outcome="success",
        summary="done", report_path="/report.md",
    )
    err = capsys.readouterr().err
    assert err.strip() == (
        "callback delivery failed: aaaabbbbcccc invalid-session; use jaxflow status/result"
    )


@pytest.mark.parametrize("mode,category", [
    ("returncode", "queue-failed"),
    ("missing", "cli-missing"),
    ("timeout", "queue-timeout"),
    ("oserror", "queue-os-error"),
])
def test_codex_callback_queue_failures_warn_once_without_tmux(monkeypatch, capsys, mode, category):
    monkeypatch.setenv("CODEX_THREAD_ID", _WORKER_THREAD)
    monkeypatch.setattr(time, "sleep", lambda seconds: pytest.fail("Codex callback slept"))
    calls = []

    def queue(argv, **kwargs):
        calls.append((argv, kwargs))
        secret = _CALLBACK_SECRET.encode()
        if mode == "returncode":
            return CompletedProcess(argv, 1, secret, secret)
        if mode == "missing":
            raise FileNotFoundError(_CALLBACK_SECRET)
        if mode == "timeout":
            raise subprocess.TimeoutExpired(argv, 10, output=secret, stderr=secret)
        raise OSError(_CALLBACK_SECRET)

    monkeypatch.setattr(subprocess, "run", queue)
    manifest = dict(run_id="aaaabbbbcccc", caller="codex",
                    caller_session=_CAPTURED_THREAD, caller_pane="%3",
                    caller_incarnation="111:222")
    jaxflow_workerkit._send_callback(
        manifest, run=_forbidden_tmux, kind="spec", outcome="success",
        summary="done", report_path="/report.md",
    )
    assert calls == [([
        "codex", "queue", "--thread", _CAPTURED_THREAD, "--message",
        "[JAXFLOW] spec aaaabbbbcccc finished — success — /report.md"
    ], dict(stdin=subprocess.DEVNULL, capture_output=True, timeout=10, check=False))]
    err = capsys.readouterr().err
    assert _CALLBACK_SECRET not in err
    assert err.strip() == (
        f"callback delivery failed: aaaabbbbcccc {category}; use jaxflow status/result"
    )


def test_post_event_treats_409_as_delivered_not_a_failure():
    # MOA-474 D3: a 409 is the DB's own terminal-claim race (src/server/db/workflows.ts:
    # 117-122) -- another writer (worker vs reaper, or a worker vs cmd_cancel) already
    # posted this run's terminal row first. That is a DELIVERED outcome, not an error.
    def opener(req, timeout):
        class Resp:
            status = 409
            def read(self_):
                return b'{"ok": false, "error": "already finished"}'
            def __enter__(self_):
                return self_
            def __exit__(self_, *a):
                return False
        return Resp()
    result = jaxflow_common._post_event({"x": 1}, opener=opener)
    # The plan's draft test asserted the decoded body here; the plan's own Step 1.4
    # implementation returns this explicit marker instead, so a caller can never
    # confuse a 409 with a 200 {ok:false} (which still raises and keeps the spool).
    assert result == {"ok": False, "claimed": True}


def test_post_event_still_raises_on_a_real_500():
    def opener(req, timeout):
        class Resp:
            status = 500
            def read(self_):
                return b'{"ok": false}'
            def __enter__(self_):
                return self_
            def __exit__(self_, *a):
                return False
        return Resp()
    with pytest.raises(RuntimeError):
        jaxflow_common._post_event({"x": 1}, opener=opener)


def test_spool_write_read_delete_roundtrip(tmp_path):
    repo = tmp_path / "demo"
    repo.mkdir()
    event = {"run_id": "aaaabbbbcccc", "payload": {"result": "failure"}}
    path = jaxflow_common._write_spool(repo, "aaaabbbbcccc", event)
    assert path == jaxflow_common._spool_path(repo, "aaaabbbbcccc")
    assert json.loads(path.read_text(encoding="utf-8")) == event
    jaxflow_common._delete_spool(repo, "aaaabbbbcccc")
    assert not path.exists()


@pytest.mark.parametrize("refuse_kind", ["builder", "unvalidated", "diff"])
def test_refusal_paths_spool_before_post_keep_on_failure_delete_on_delivery(tmp_path, monkeypatch, refuse_kind):
    # MOA-474 cold review F1: a failed refusal POST must leave a recoverable spool
    # behind; a delivered one must not.
    monkeypatch.setattr(jr, "DB_PATH", tmp_path / "unused-jaxos.db")
    root = tmp_path / "demo"
    _init_repo(root)
    run_id = "aaaabbbbcccc"
    run_dir = jaxflow_common._manifest_dir(root, run_id)
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    manifest = dict(run_id=run_id, project="demo", phase="P1", kind="build")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spool_path = run_dir / "finished.json"

    def failing_post(event):
        raise RuntimeError("network down")

    def ok_post(event):
        return {"ok": True}

    def call():
        if refuse_kind == "builder":
            worktree = _init_worktree(root)
            return jaxflow._refuse_builder_run(
                "boom", manifest=manifest, run=jr.run_command, post=post_fn,
                control_repo=root, worktree=worktree, branch="feat/x",
            )
        if refuse_kind == "unvalidated":
            return jaxflow._refuse_unvalidated_builder_run(
                "boom", manifest=manifest, manifest_path=manifest_path,
                run=jr.run_command, post=post_fn,
            )
        manifest["kind"] = "diff"
        return jaxflow_workerkit._refuse_diff_run(
            "boom", manifest=manifest, run=jr.run_command, post=post_fn,
            manifest_path=manifest_path,
        )

    post_fn = failing_post
    call()
    assert spool_path.exists()

    post_fn = ok_post
    call()
    assert not spool_path.exists()


@pytest.mark.parametrize("kind", ["build", "spec"])
def test_send_callback_omits_stage_segment_on_the_happy_path(kind):
    manifest = dict(run_id="aaaabbbbcccc", caller="claude", caller_session=_TEST_CLAUDE_SESSION_ID)
    jaxflow_workerkit._send_callback(
        manifest, run=_forbidden_tmux, kind=kind, outcome="success",
        summary="done", report_path="/report.md",
    )
    line_path = jaxflow_common.CALLBACKS_ROOT / _TEST_CLAUDE_SESSION_ID / "aaaabbbbcccc.line"
    assert line_path.read_text(encoding="utf-8") == (
        f"[JAXFLOW] {kind} aaaabbbbcccc finished — success — /report.md\n"
    )


def test_send_callback_renders_stage_diagnostic_and_report_flag():
    manifest = dict(run_id="aaaabbbbcccc", caller="claude", caller_session=_TEST_CLAUDE_SESSION_ID)
    jaxflow_workerkit._send_callback(
        manifest, run=_forbidden_tmux, kind="build", outcome="failure",
        summary="unused", report_path="/report.md", stage="runtime",
        diagnostic="APIError 403 budget exceeded", contract_status="invalid",
    )
    line_path = jaxflow_common.CALLBACKS_ROOT / _TEST_CLAUDE_SESSION_ID / "aaaabbbbcccc.line"
    assert line_path.read_text(encoding="utf-8") == (
        "[JAXFLOW] build aaaabbbbcccc finished — failure · runtime — "
        "APIError 403 budget exceeded [report invalid] — /report.md\n"
    )


def test_send_callback_renders_no_report_literal_and_ledger_pending():
    manifest = dict(run_id="aaaabbbbcccc", caller="claude", caller_session=_TEST_CLAUDE_SESSION_ID)
    jaxflow_workerkit._send_callback(
        manifest, run=_forbidden_tmux, kind="build", outcome="failure", summary="unused",
        report_path=None, stage="worker", diagnostic="worker interrupted by SIGTERM",
        ledger_pending=True,
    )
    line_path = jaxflow_common.CALLBACKS_ROOT / _TEST_CLAUDE_SESSION_ID / "aaaabbbbcccc.line"
    assert line_path.read_text(encoding="utf-8") == (
        "[JAXFLOW] build aaaabbbbcccc finished — failure · worker — "
        "worker interrupted by SIGTERM — no report · ledger pending\n"
    )


def test_send_callback_never_flags_report_status_on_ok_cancelled_or_interrupted():
    manifest = dict(run_id="aaaabbbbcccc", caller="claude", caller_session=_TEST_CLAUDE_SESSION_ID)
    for status in ("ok", "cancelled", "interrupted"):
        jaxflow_workerkit._send_callback(
            manifest, run=_forbidden_tmux, kind="build", outcome="x", summary="unused",
            report_path=None, contract_status=status,
        )
        line_path = jaxflow_common.CALLBACKS_ROOT / _TEST_CLAUDE_SESSION_ID / "aaaabbbbcccc.line"
        assert "[report" not in line_path.read_text(encoding="utf-8")


def test_send_callback_claude_skips_line_for_noncanonical_session(capsys):
    manifest = dict(
        run_id="aaaabbbbcccc", caller="claude", caller_session="not-a-uuid",
    )
    jaxflow_workerkit._send_callback(
        manifest, run=_forbidden_tmux, kind="spec", outcome="success",
        summary="done", report_path="/report.md",
    )
    assert not (jaxflow_common.CALLBACKS_ROOT / "not-a-uuid").exists()
    err = capsys.readouterr().err.strip()
    assert err == "callback delivery failed: 'aaaabbbbcccc' path-unsafe; use jaxflow status/result"


@pytest.mark.parametrize("bad_run_id", [
    "../x", "/tmp/jaxflow-escape", "aaaabbbbcccc\n",
])
def test_send_callback_claude_rejects_unsafe_run_id(capsys, bad_run_id):
    # Cold review round 2 F4: `ascii(bad_run_id)[:80]` is the production rendering (see `_write_callback_file`);
    # asserting against it here (rather than a hand-typed literal per case) is what proves
    # the "aaaabbbbcccc\n" case still prints as ONE line -- a raw newline would make `err`
    # (a real multi-line string) fail this equality against the escaped, single-line repr.
    manifest = dict(
        run_id=bad_run_id, caller="claude", caller_session=_TEST_CLAUDE_SESSION_ID,
    )
    jaxflow_workerkit._send_callback(
        manifest, run=_forbidden_tmux, kind="spec", outcome="success",
        summary="done", report_path="/report.md",
    )
    session_dir = jaxflow_common.CALLBACKS_ROOT / _TEST_CLAUDE_SESSION_ID
    assert not session_dir.exists() or not any(session_dir.iterdir())
    err = capsys.readouterr().err.strip()
    assert err == (
        f"callback delivery failed: {ascii(bad_run_id)[:80]} path-unsafe; "
        "use jaxflow status/result"
    )
    assert "\n" not in err


def test_send_callback_publish_failure_does_not_affect_result(monkeypatch, capsys):
    # A CALLBACKS_ROOT that is itself a FILE makes `session_dir.mkdir()` raise
    # NotADirectoryError (an OSError subclass) -- the write-error branch, not the
    # path-safety gate (caller_session/run_id are both well-formed here).
    blocked = jaxflow_common.CALLBACKS_ROOT.parent / "callbacks-is-a-file"
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("", encoding="utf-8")
    monkeypatch.setattr(jaxflow_common, "CALLBACKS_ROOT", blocked / "unreachable")
    manifest = dict(
        run_id="aaaabbbbcccc", caller="claude", caller_session=_TEST_CLAUDE_SESSION_ID,
    )
    # `_send_callback` itself never returns a result -- this proves it does not raise,
    # which is what a worker relies on (its own `worker_outcome` is set BEFORE this
    # call, per every one of the 6 real call sites read in Task 1's investigation).
    jaxflow_workerkit._send_callback(
        manifest, run=_forbidden_tmux, kind="spec", outcome="success",
        summary="done", report_path="/report.md",
    )
    err = capsys.readouterr().err.strip()
    assert err == "callback delivery failed: aaaabbbbcccc write-error; use jaxflow status/result"


def test_send_callback_codex_path_unchanged_with_no_pane_fields(monkeypatch):
    # F2: a manifest carrying NEITHER caller_pane NOR caller_incarnation (the shape
    # every Codex-caller manifest has from this task onward, since D8 stops writing
    # them for every caller) still delivers via `codex queue`, unaffected.
    monkeypatch.setenv("CODEX_THREAD_ID", _WORKER_THREAD)
    calls = []

    def queue(argv, **kwargs):
        calls.append((argv, kwargs))
        return CompletedProcess(argv, 0, b"accepted", b"")

    monkeypatch.setattr(subprocess, "run", queue)
    manifest = dict(run_id="aaaabbbbcccc", caller="codex", caller_session=_CAPTURED_THREAD)
    jaxflow_workerkit._send_callback(
        manifest, run=_forbidden_tmux, kind="build", outcome="success",
        summary="done", report_path="/report.md",
    )
    assert calls == [([
        "codex", "queue", "--thread", _CAPTURED_THREAD, "--message",
        "[JAXFLOW] build aaaabbbbcccc finished — success — /report.md",
    ], dict(stdin=subprocess.DEVNULL, capture_output=True, timeout=10, check=False))]
    assert not jaxflow_common.CALLBACKS_ROOT.exists()


def test_send_callback_claude_writes_line_file_on_builder_success():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=FakeBuilderPopen, allowlist_root=root.parent,
        )
        assert code == 0
        line_path = jaxflow_common.CALLBACKS_ROOT / manifest["caller_session"] / f"{manifest['run_id']}.line"
        assert line_path.read_text(encoding="utf-8").startswith(
            f"[JAXFLOW] build {manifest['run_id']} finished — success"
        )


def test_send_callback_claude_writes_line_file_on_diff_review_success():
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
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=FakePopen, allowlist_root=root.parent,
        )
        assert code == 0
        line_path = jaxflow_common.CALLBACKS_ROOT / manifest["caller_session"] / f"{manifest['run_id']}.line"
        assert line_path.read_text(encoding="utf-8").startswith(
            f"[JAXFLOW] diff {manifest['run_id']} finished — approve"
        )


def test_send_callback_claude_writes_line_file_on_doc_review_success():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        manifest_path, manifest = _write_manifest_for_worker(root, target, runtime="codex")
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()), post=lambda e: {"ok": True},
            popen=FakePopen, allowlist_root=root,
        )
        assert code == 0
        line_path = jaxflow_common.CALLBACKS_ROOT / manifest["caller_session"] / f"{manifest['run_id']}.line"
        assert line_path.read_text(encoding="utf-8").startswith(
            f"[JAXFLOW] spec {manifest['run_id']} finished — approve"
        )


def test_worker_codex_callback_queue_failure_still_finalizes(monkeypatch, capsys):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        fake = FakeTmux()
        monkeypatch.setenv("CODEX_THREAD_ID", _WORKER_THREAD)
        monkeypatch.setattr(time, "sleep", lambda seconds: pytest.fail("Codex callback slept"))
        queue_calls = []

        def queue(argv, **kwargs):
            queue_calls.append((argv, kwargs))
            return CompletedProcess(argv, 1, _CALLBACK_SECRET.encode(), _CALLBACK_SECRET.encode())

        _intercept_codex_queue(monkeypatch, queue)
        status_calls = []
        monkeypatch.setattr(
            jaxflow_common, "_update_status_md",
            lambda manifest, **kwargs: status_calls.append(dict(manifest)),
        )
        events = []
        manifest_path, manifest = _write_manifest_for_worker(
            root, target, caller="codex", caller_session=_CAPTURED_THREAD,
            caller_pane="%3", caller_incarnation=fake.incarnation,
        )
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(fake),
            post=lambda event: events.append(event) or {"ok": True},
            popen=FakePopen, allowlist_root=root,
        )
        assert code == 0
        assert len(queue_calls) == 1
        assert queue_calls[0][0][:4] == ["codex", "queue", "--thread", _CAPTURED_THREAD]
        assert queue_calls[0][1] == _QUEUE_KW
        assert not any(c[1] == "send-keys" for c in fake.calls)
        assert len(events) == 1
        assert events[0]["type"] == "run-finished"
        assert events[0]["payload"]["contract_status"] == "ok"
        assert events[0]["payload"]["verdict"] == "approve"
        assert len(status_calls) == 1
        err = capsys.readouterr().err
        assert _CALLBACK_SECRET not in err
        assert "callback delivery failed: aaaabbbbcccc queue-failed; use jaxflow status/result" in err


# ---- status/result/cancel ----

def _fresh_db(path):
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE workflow_events (id INTEGER PRIMARY KEY, ts TEXT, run_id TEXT, "
        "project TEXT, role TEXT, type TEXT, payload TEXT)"
    )
    con.commit()
    return con


def _insert(con, run_id, project, role, type_, payload, *, ts="t"):
    con.execute(
        "INSERT INTO workflow_events (ts, run_id, project, role, type, payload) VALUES (?, ?, ?, ?, ?, ?)",
        (ts, run_id, project, role, type_, json.dumps(payload)),
    )
    con.commit()


def _ledger_post(db, events):
    def post(event):
        events.append(event)
        con = sqlite3.connect(db)
        _insert(con, event.get("run_id"), event["project"], event["role"], event["type"], event["payload"])
        con.close()
        return {"ok": True}
    return post


# ---- MOA-495 2.2: review-round tally ----

def _all_pairs_noul(score):
    """Test helper: a `noul_batch` fake that scores every (new, prev) pair the same,
    matching `build_review_tally`'s batched call shape (new_ids, new_findings,
    prev_ids, prev_findings) -> {(new_id, prev_id): score}."""
    return lambda new_ids, _nf, prev_ids, _pf: {(n, p): score for n in new_ids for p in prev_ids}


def test_build_review_tally_reports_repeated_and_new_findings():
    new_text = (
        "F1. HIGH — cache race in the writer\n"
        "F2 — MEDIUM — misleading empty-file UI state\n"
    )
    prev_text = "F1 — HIGH — stale cache entry never evicted\n"

    def noul_batch(new_ids, new_findings, prev_ids, _prev_findings):
        return {
            (n, p): (0.89 if "misleading empty-file" in new_findings[n] else 0.1)
            for n in new_ids for p in prev_ids
        }
    tally = jaxflow_workerkit.build_review_tally(new_text, prev_text, noul_batch=noul_batch)
    assert tally == "tally: 2 findings — 1 repeated (F2≈prev F1 0.89), 1 new"


def test_build_review_tally_all_new_when_previous_report_has_no_findings():
    tally = jaxflow_workerkit.build_review_tally(
        "F1. HIGH — something bad\n", "approve, nothing to report\n",
        noul_batch=lambda *a: (_ for _ in ()).throw(AssertionError("no pairs to compare")),
    )
    assert tally == "tally: 1 findings — 0 repeated, 1 new"


def test_build_review_tally_none_when_new_report_has_no_findings():
    assert jaxflow_workerkit.build_review_tally("approve\n", "F1. HIGH — x\n", noul_batch=_all_pairs_noul(1.0)) is None


def test_build_review_tally_none_on_jev_error():
    def raising(*_a):
        raise RuntimeError("jev down")
    assert jaxflow_workerkit.build_review_tally("F1. HIGH — x\n", "F1. HIGH — y\n", noul_batch=raising) is None


def test_build_review_tally_threshold_is_inclusive_at_0_3():
    at_threshold = jaxflow_workerkit.build_review_tally("F1. HIGH — x\n", "F1. HIGH — y\n", noul_batch=_all_pairs_noul(0.3))
    assert at_threshold == "tally: 1 findings — 1 repeated (F1≈prev F1 0.30), 0 new"
    below_threshold = jaxflow_workerkit.build_review_tally("F1. HIGH — x\n", "F1. HIGH — y\n", noul_batch=_all_pairs_noul(0.29))
    assert below_threshold == "tally: 1 findings — 0 repeated, 1 new"


def test_build_review_tally_makes_exactly_one_batched_call_for_all_pairs():
    # Cold review F1: previously N*M sequential Noul requests. Now exactly one call,
    # carrying every pair's question, regardless of how many new/previous findings.
    new_text = "".join(f"F{i}. HIGH — bug {i}\n" for i in range(1, 4))
    prev_text = "".join(f"F{i}. HIGH — old bug {i}\n" for i in range(1, 3))
    calls = []

    def noul_batch(new_ids, new_findings, prev_ids, prev_findings):
        calls.append((tuple(new_ids), tuple(prev_ids)))
        return {(n, p): 0.0 for n in new_ids for p in prev_ids}
    jaxflow_workerkit.build_review_tally(new_text, prev_text, noul_batch=noul_batch)
    assert len(calls) == 1
    assert calls[0] == (("F1", "F2", "F3"), ("F1", "F2"))


def test_build_review_tally_skips_when_pairs_exceed_the_cap():
    # Cold review F1: N*M pairs above _TALLY_MAX_PAIRS never reach Jev at all -- skip
    # the tally rather than risk an oversized batch (worst case must stay 1 call).
    new_text = "".join(f"F{i}. HIGH — bug {i}\n" for i in range(1, 12))  # 11 new
    prev_text = "".join(f"F{i}. HIGH — old bug {i}\n" for i in range(1, 11))  # 10 previous
    assert 11 * 10 > jaxflow_workerkit._TALLY_MAX_PAIRS
    tally = jaxflow_workerkit.build_review_tally(
        new_text, prev_text, noul_batch=lambda *a: (_ for _ in ()).throw(AssertionError("must not call Jev")),
    )
    assert tally is None


def test_build_review_tally_caps_the_rendered_line_to_300_chars_dropping_trailing_matches():
    # Cold review F3: ~20 repeated findings render past the validator's 300-char limit.
    new_text = "".join(f"F{i}. HIGH — bug variant {i} in the writer\n" for i in range(1, 21))
    prev_text = "F1. HIGH — original bug in the writer\n"
    tally = jaxflow_workerkit.build_review_tally(new_text, prev_text, noul_batch=_all_pairs_noul(0.9))
    assert tally is not None
    assert len(tally) <= 300
    assert tally.startswith("tally: 20 findings — 20 repeated")
    assert tally.endswith(", 0 new")
    # Aggregate counts survive even though not every match example fits.
    assert tally.count("≈prev") < 20


def test_build_review_tally_omits_the_line_with_zero_jev_calls_when_classifier_is_off(monkeypatch):
    # `build_review_tally` swallows ANY exception from `noul_batch` into `None` (an injected
    # fake that only raises would let this test pass without the gate), so the fake records
    # its calls for the assertion; the text uses a real finding shape `extract_findings`
    # matches (the plan's `- [ ] F1 (MEDIUM):` shape is not matched by `_FINDING_ID_RE`).
    import general_settings
    monkeypatch.setattr(general_settings, "read_settings",
                         lambda: {"ok": True, "data": {"integrations": {"classifier": False}}})
    calls = []
    def counting_noul(*a):
        calls.append(a)
        return {("F1", "F1"): 0.9}
    new_text = "F1. HIGH — a\n"
    prev_text = "F1. HIGH — a\n"
    assert jaxflow_workerkit.build_review_tally(new_text, prev_text, noul_batch=counting_noul) is None
    assert calls == []


def test_previous_ok_review_finds_the_most_recent_matching_ok_round():
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        con.row_factory = sqlite3.Row
        _insert(con, "d1", "demo", "reviewer", "run-started", {"kind": "diff", "target": "feat/x"}, ts="1")
        _insert(con, "d1", "demo", "reviewer", "run-finished", {"contract_status": "ok"}, ts="1")
        _insert(con, "d2", "demo", "reviewer", "run-started", {"kind": "diff", "target": "feat/x"}, ts="2")
        _insert(con, "d2", "demo", "reviewer", "run-finished", {"contract_status": "invalid"}, ts="2")
        _insert(con, "d3", "demo", "reviewer", "run-started", {"kind": "diff", "target": "feat/x"}, ts="3")
        assert jaxflow_workerkit._previous_ok_review(con, "demo", "diff", "feat/x", "d3") == "d1"
        con.close()


def test_previous_ok_review_ignores_other_kind_target_project_or_unfinished():
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        con.row_factory = sqlite3.Row
        _insert(con, "s1", "demo", "reviewer", "run-started", {"kind": "spec", "target": "/a.md"}, ts="1")
        _insert(con, "s1", "demo", "reviewer", "run-finished", {"contract_status": "ok"}, ts="1")
        assert jaxflow_workerkit._previous_ok_review(con, "demo", "plan", "/a.md", "s2") is None
        assert jaxflow_workerkit._previous_ok_review(con, "demo", "spec", "/other.md", "s2") is None
        assert jaxflow_workerkit._previous_ok_review(con, "other-project", "spec", "/a.md", "s2") is None
        _insert(con, "s3", "demo", "reviewer", "run-started", {"kind": "spec", "target": "/a.md"}, ts="2")
        # no run-finished row for s3 at all -- an in-flight round is never "previous".
        assert jaxflow_workerkit._previous_ok_review(con, "demo", "spec", "/a.md", "s4") == "s1"
        con.close()


def test_review_round_tally_reads_the_previous_report_off_disk(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / ".local" / "reports").mkdir(parents=True)
    (repo / ".local" / "reports" / "aaaaaaaaaaaa.md").write_text("F1. HIGH — old bug\n", encoding="utf-8")
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    con.row_factory = sqlite3.Row
    _insert(con, "aaaaaaaaaaaa", "demo", "reviewer", "run-started", {"kind": "diff", "target": "feat/x"}, ts="1")
    _insert(con, "aaaaaaaaaaaa", "demo", "reviewer", "run-finished", {"contract_status": "ok"}, ts="1")
    con.close()
    monkeypatch.setattr(jaxflow_workerkit, "_same_problem_noul_batch", _all_pairs_noul(0.9))
    tally = jaxflow_workerkit._review_round_tally(
        repo, "demo", "diff", "feat/x", "bbbbbbbbbbbb", "F1. HIGH — old bug reworded\n", db_path=db,
    )
    assert tally == "tally: 1 findings — 1 repeated (F1≈prev F1 0.90), 0 new"


def test_review_round_tally_none_without_a_previous_round(tmp_path):
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    assert jaxflow_workerkit._review_round_tally(
        tmp_path, "demo", "diff", "feat/x", "bbbbbbbbbbbb", "F1. HIGH — x\n", db_path=db,
    ) is None


def test_review_round_tally_none_when_db_missing(tmp_path):
    assert jaxflow_workerkit._review_round_tally(
        tmp_path, "demo", "diff", "feat/x", "bbbbbbbbbbbb", "F1. HIGH — x\n",
        db_path=tmp_path / "does-not-exist.db",
    ) is None


def test_worker_doc_review_ok_row_gets_a_tally_against_the_previous_ok_round(monkeypatch):
    class FindingsPopen(FakePopen):
        REPORT_TEXT = (
            "---\nrun_id: aaaabbbbcccc\nproject: demo\nrole: reviewer\nphase: PHASE\n"
            "verdict: approve-with-changes\nsummary: 1 MEDIUM\n---\n"
            "F1. MEDIUM — misleading empty-file UI state\n"
        )

    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        manifest_path, manifest = _write_manifest_for_worker(root, target, runtime="codex")

        prev_reports = root / ".local" / "reports"
        prev_reports.mkdir(parents=True, exist_ok=True)
        (prev_reports / "1111aaaa2222.md").write_text("F1. MEDIUM — stale empty-file badge\n", encoding="utf-8")
        db = root / "jaxos.db"
        con = _fresh_db(db)
        con.row_factory = sqlite3.Row
        _insert(con, "1111aaaa2222", "demo", "reviewer", "run-started", {"kind": "spec", "target": str(target)}, ts="1")
        _insert(con, "1111aaaa2222", "demo", "reviewer", "run-finished", {"contract_status": "ok"}, ts="1")
        con.close()
        monkeypatch.setattr(jr, "DB_PATH", db)
        monkeypatch.setattr(jaxflow_workerkit, "_same_problem_noul_batch", _all_pairs_noul(0.9))

        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()), post=events.append, popen=FindingsPopen,
            allowlist_root=root,
        )
        assert code == 0
        payload = events[0]["payload"]
        assert payload["tally"] == "tally: 1 findings — 1 repeated (F1≈prev F1 0.90), 0 new"


def test_worker_doc_review_no_tally_without_a_previous_round(monkeypatch, tmp_path):
    root = tmp_path.resolve()
    _init_repo(root)
    target = _spec_file(root)
    manifest_path, manifest = _write_manifest_for_worker(root, target, runtime="codex")
    db = root / "jaxos.db"
    _fresh_db(db).close()
    monkeypatch.setattr(jr, "DB_PATH", db)

    events = []
    code = jaxflow.run_worker(
        str(manifest_path), run=_run_with_tmux(FakeTmux()), post=events.append, popen=FakePopen,
        allowlist_root=root,
    )
    assert code == 0
    assert "tally" not in events[0]["payload"]


def test_status_derivation_unknown_running_finished_dead():
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        fake = FakeTmux()
        assert jaxflow.cmd_status("nope", run=_run_with_tmux(fake), db_path=db) == "unknown"

        _insert(con, "r1", "demo", "reviewer", "run-started", {"session": "jax-demo-spec-r1"})
        fake.existing.add("jax-demo-spec-r1")
        assert jaxflow.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == "running"

        fake.existing.discard("jax-demo-spec-r1")
        assert jaxflow.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == "dead"

        _insert(con, "r1", "demo", "reviewer", "run-finished", {"contract_status": "ok"})
        # MOA-474 §12.2: a reviewer row with no verdict now renders the literal
        # "no verdict" instead of the old bare-ok fold.
        assert jaxflow.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == "finished(no verdict)"

        _insert(con, "r2", "demo", "reviewer", "run-started", {"session": "jax-demo-spec-r2"})
        _insert(con, "r2", "demo", "reviewer", "run-finished", {"contract_status": "invalid"})
        assert jaxflow.cmd_status("r2", run=_run_with_tmux(fake), db_path=db) == "finished(no verdict, report invalid)"

        _insert(con, "r3", "demo", "reviewer", "run-started", {"session": "jax-demo-spec-r3"})
        _insert(con, "r3", "demo", "reviewer", "run-finished", {"contract_status": "cancelled"})
        assert jaxflow.cmd_status("r3", run=_run_with_tmux(fake), db_path=db) == "finished(cancelled)"
        con.close()


def test_result_prints_path_and_content_or_refuses_unknown_and_mismatch():
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        root.mkdir(parents=True)
        (root / ".local" / "reports").mkdir(parents=True)
        report = root / ".local" / "reports" / "r1.md"
        report.write_text("REPORT BODY\n", encoding="utf-8")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        try:
            jaxflow.cmd_result("nope", allowlist_root=allow_root, db_path=db)
        except ji.Refusal as exc:
            assert exc.code == "unknown-run"
        else:
            raise AssertionError("unknown-run not raised")

        _insert(con, "r1", "demo", "reviewer", "run-started", {"repo": str(root)})
        _insert(con, "r1", "demo", "reviewer", "run-finished", {
            "contract_status": "ok", "report_path": str(report), "summary": "ok",
        })
        out = jaxflow.cmd_result("r1", allowlist_root=allow_root, db_path=db)
        assert out == f"{report}\nREPORT BODY\n"

        _insert(con, "r2", "demo", "reviewer", "run-started", {"repo": str(root)})
        _insert(con, "r2", "demo", "reviewer", "run-finished", {
            "contract_status": "ok", "report_path": "/tmp/not-the-real-path.md", "summary": "ok",
        })
        try:
            jaxflow.cmd_result("r2", allowlist_root=allow_root, db_path=db)
        except ji.Refusal as exc:
            assert exc.code == "report-path-mismatch"
        else:
            raise AssertionError("report-path-mismatch not raised")

        _insert(con, "r3", "demo", "reviewer", "run-started", {"repo": str(root)})
        _insert(con, "r3", "demo", "reviewer", "run-finished", {
            "contract_status": "cancelled", "report_path": None, "summary": "cancelled by claude at t",
        })
        assert jaxflow.cmd_result("r3", allowlist_root=allow_root, db_path=db) == "cancelled — cancelled by claude at t"
        con.close()


def test_result_refuses_report_path_mismatch_on_dot_dot_traversal():
    """Fixes cold review F8: the DB's report_path resolves to the right file but is not
    the string this run's own dispatch would have written, so the byte-for-byte compare
    in cmd_result must refuse it -- and must do so without ever opening the file."""
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        root.mkdir(parents=True)
        (root / ".local" / "reports").mkdir(parents=True)
        report = root / ".local" / "reports" / "r4.md"
        report.write_text("SHOULD NOT BE READ\n", encoding="utf-8")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _insert(con, "r4", "demo", "reviewer", "run-started", {"repo": str(root)})
        sneaky = str(root / ".local" / "reports" / ".." / "reports" / "r4.md")
        _insert(con, "r4", "demo", "reviewer", "run-finished", {
            "contract_status": "ok", "report_path": sneaky, "summary": "ok",
        })
        try:
            jaxflow.cmd_result("r4", allowlist_root=allow_root, db_path=db)
        except ji.Refusal as exc:
            assert exc.code == "report-path-mismatch"
        else:
            raise AssertionError("report-path-mismatch not raised")
        con.close()


def test_result_refuses_report_missing_when_the_recorded_path_has_no_file():
    """MOA-455's sibling, MOA-456. When the recorded path IS the one this run's dispatch
    would have written but the file itself is gone, `cmd_result` used to fall through to
    `read_text` and escape as a raw `FileNotFoundError` -- the only failure path in the
    CLI that was a traceback instead of a kebab-case code. It gets its OWN code rather
    than widening `report-path-mismatch`: the two have different remedies. A mismatch
    means the ledger disagrees with the dispatch convention; `report-missing` means the
    file is gone for good and re-running `result` is pointless. That second case is real
    here -- `git worktree remove` deletes the worktree's gitignored `.local/` with it."""
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        (root / ".local" / "reports").mkdir(parents=True)
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        expected = root / ".local" / "reports" / "r5.md"
        _insert(con, "r5", "demo", "reviewer", "run-started", {"repo": str(root)})
        _insert(con, "r5", "demo", "reviewer", "run-finished", {
            "contract_status": "ok", "report_path": str(expected), "summary": "ok",
        })
        assert not expected.exists()
        try:
            jaxflow.cmd_result("r5", allowlist_root=allow_root, db_path=db)
        except ji.Refusal as exc:
            assert exc.code == "report-missing"
        else:
            raise AssertionError("report-missing not raised")
        con.close()


@pytest.mark.parametrize("case,code", [
    ("success", None),
    ("forged", "report-path-mismatch"),
    ("missing", "report-missing"),
    ("escaping", "path-outside-allowlist"),
])
def test_result_builder_reads_worktree_report(case, code):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        control = allow_root / "demo"
        worktree = allow_root / "demo-feat-x"
        control.mkdir(parents=True)
        expected = worktree / ".local" / "reports" / "b1.md"
        if case == "success":
            expected.parent.mkdir(parents=True)
            expected.write_text("BUILDER BODY\n", encoding="utf-8")
            recorded = str(expected)
        elif case == "forged":
            decoy = control / ".local" / "reports" / "b1.md"
            decoy.parent.mkdir(parents=True)
            decoy.write_text("DECOY\n", encoding="utf-8")
            recorded = str(decoy)
        elif case == "missing":
            recorded = str(expected)
        else:
            expected.parent.mkdir(parents=True)
            outside = Path(raw) / "outside.md"
            outside.write_text("SECRET\n", encoding="utf-8")
            expected.symlink_to(outside)
            recorded = str(expected)
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _insert(con, "b1", "demo", "builder", "run-started", {
            "repo": str(control), "kind": "build", "target": "feat/x",
        })
        _insert(con, "b1", "demo", "builder", "run-finished", {
            "contract_status": "ok", "report_path": recorded, "summary": "ok",
        })
        if case == "success":
            out = jaxflow.cmd_result("b1", allowlist_root=allow_root, db_path=db)
            assert out == f"{expected.resolve()}\nBUILDER BODY\n"
            assert not (control / ".local" / "runs").exists()
        else:
            try:
                jaxflow.cmd_result("b1", allowlist_root=allow_root, db_path=db)
            except ji.Refusal as exc:
                assert exc.code == code
            else:
                raise AssertionError(f"{code} not raised")
        con.close()


def test_cancel_unknown_already_finished_and_happy_path(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        fake = FakeTmux()
        try:
            jaxflow.cmd_cancel(
                "nope", run=_run_with_tmux(fake), post=lambda url, body: (200, {"ok": True}),
                now=lambda: None, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "unknown-run"
        else:
            raise AssertionError("unknown-run not raised")

        _insert(con, "r1", "demo", "reviewer", "run-started", {
            "session": "jax-demo-spec-r1", "phase": "P", "caller": "claude",
        })
        _insert(con, "r1", "demo", "reviewer", "run-finished", {"contract_status": "ok"})
        try:
            jaxflow.cmd_cancel(
                "r1", run=_run_with_tmux(fake), post=lambda url, body: (200, {"ok": True}),
                now=lambda: None, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "already-finished"
        else:
            raise AssertionError("already-finished not raised")

        _insert(con, "r2", "demo", "reviewer", "run-started", {
            "session": "jax-demo-spec-r2", "phase": "P", "caller": "codex",
        })
        events = []

        def post(url, body):
            events.append(body)
            return 200, {"ok": True}

        import datetime as _dt
        out = jaxflow.cmd_cancel(
            "r2", run=_run_with_tmux(fake), post=post,
            now=lambda: _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc), db_path=db,
        )
        assert out == "cancelled"
        assert fake.calls[-1][:3] == ["tmux", "kill-session", "-t"] or fake.calls[0][:2] == ["tmux", "kill-session"]
        assert events[0]["payload"]["contract_status"] == "cancelled"
        assert events[0]["payload"]["exit_code"] is None
        assert events[0]["payload"]["report_path"] is None
        assert "cancelled by codex at" in events[0]["payload"]["summary"]
        con.close()


def test_cancel_post_status_outcomes(monkeypatch):
    """Fixes cold review F2: `post` is `_post`-shaped ((url, payload) -> (status, dict),
    raising only on a transport error), so cmd_cancel can tell a real 409 (someone else's
    terminal row won the race, §2.4) apart from a plain server error or a dropped
    connection -- the earlier draft mapped every one of those to `already-finished`."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _insert(con, "r1", "demo", "reviewer", "run-started", {
            "session": "jax-demo-spec-r1", "phase": "P", "caller": "claude",
        })
        con.close()
        fake = FakeTmux()
        import datetime as _dt
        now = lambda: _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)

        try:
            jaxflow.cmd_cancel("r1", run=_run_with_tmux(fake), post=lambda url, body: (409, {"ok": False}), now=now, db_path=db)
        except ji.Refusal as exc:
            assert exc.code == "already-finished"
        else:
            raise AssertionError("already-finished not raised on 409")

        try:
            jaxflow.cmd_cancel("r1", run=_run_with_tmux(fake), post=lambda url, body: (500, {"ok": False}), now=now, db_path=db)
        except ji.Refusal as exc:
            assert exc.code == "hub-rejected: 500"
        else:
            raise AssertionError("hub-rejected not raised on 500")

        def raising_post(url, body):
            raise OSError("connection refused")

        try:
            jaxflow.cmd_cancel("r1", run=_run_with_tmux(fake), post=raising_post, now=now, db_path=db)
        except ji.Refusal as exc:
            assert exc.code == "hub-unreachable"
        else:
            raise AssertionError("hub-unreachable not raised on transport error")

        # fixes cold review F5: a 200 that decodes to {ok: false} is the route's own
        # deliberate non-claim-failure shape (§ route.ts), not a success -- must not be
        # read as "cancelled".
        try:
            jaxflow.cmd_cancel(
                "r1", run=_run_with_tmux(fake),
                post=lambda url, body: (200, {"ok": False, "error": "x"}), now=now, db_path=db,
            )
        except ji.Refusal as exc:
            assert exc.code == "hub-rejected: 200 x"
        else:
            raise AssertionError("hub-rejected not raised on 200 ok:false")


def test_cancel_older_attempt_does_not_kill_resumed_session(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        repo = str(Path(raw) / "demo")
        _insert(con, "aaaaaaaaaaaa", "demo", "builder", "run-started", {
            "session": "jax-demo-build-aaaaaaaaaaaa", "phase": "P", "caller": "claude",
            "runtime": "opencode-builder", "kind": "build", "target": "feat/x",
            "repo": repo,
        })
        _insert(con, "ffffffffffff", "demo", "builder", "run-started", {
            "session": "jax-demo-build-ffffffffffff", "phase": "P", "caller": "claude",
            "runtime": "opencode-builder", "kind": "build", "target": "feat/x",
            "repo": repo, "resumes_run_id": "aaaaaaaaaaaa",
        })
        con.close()
        fake = FakeTmux()
        fake.existing.add("jax-demo-build-aaaaaaaaaaaa")
        fake.existing.add("jax-demo-build-ffffffffffff")
        import datetime as _dt
        with pytest.raises(ji.Refusal) as caught:
            jaxflow.cmd_cancel(
                "aaaaaaaaaaaa", run=_run_with_tmux(fake),
                post=lambda url, body: (200, {"ok": True}),
                now=lambda: _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc),
                db_path=db,
            )
        assert caught.value.code == "resume-ineligible"
        assert not any(c[:3] == ["tmux", "kill-session", "-t"] for c in fake.calls)


def test_cancel_succeeds_while_worktree_claim_held(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    import jaxflow_resume as jresume

    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, "bbbbbbbbbbbb", "demo", "builder", "run-started", {
        "session": "jax-demo-build-bbbbbbbbbbbb", "phase": "P", "caller": "claude",
        "runtime": "opencode-builder", "kind": "build", "target": "feat/x",
        "repo": str(tmp_path),
    })
    con.close()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    fake = FakeTmux()
    fake.existing.add("jax-demo-build-bbbbbbbbbbbb")
    import datetime as _dt
    with jresume.worktree_claim(tmp_path, worktree):
        out = jaxflow.cmd_cancel(
            "bbbbbbbbbbbb", run=_run_with_tmux(fake),
            post=lambda url, body: (200, {"ok": True}),
            now=lambda: _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc),
            db_path=db,
        )
    assert out == "cancelled"
    assert ["tmux", "kill-session", "-t", "jax-demo-build-bbbbbbbbbbbb"] in fake.calls


def test_cancel_reports_finalized_by_worker_when_the_row_lands_before_the_deadline(monkeypatch):
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _insert(con, "aaaabbbbcccc", "demo", "builder", "run-started", {
            "session": "jax-demo-build-aaaabbbbcccc", "phase": "P", "caller": "claude",
        })
        con.close()
        fake = FakeTmux()
        import datetime as _dt
        now = lambda: _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)

        # The worker "wins the race" on the SECOND poll: insert its own interrupted
        # row from a background-simulated call between polls.
        polls = {"n": 0}
        real_sleep_calls = []

        def fake_sleep(seconds):
            real_sleep_calls.append(seconds)
            polls["n"] += 1
            if polls["n"] == 1:
                con2 = sqlite3.connect(db)
                _insert(con2, "aaaabbbbcccc", "demo", "builder", "run-finished", {
                    "contract_status": "interrupted", "exit_code": None, "report_path": None,
                    "summary": "worker interrupted by SIGHUP", "stage": "worker",
                    "diagnostic": "worker interrupted by SIGHUP", "head_sha": None,
                    "result": "failure",
                })
                con2.close()

        monkeypatch.setattr(time, "sleep", fake_sleep)
        out = jaxflow.cmd_cancel(
            "aaaabbbbcccc", run=_run_with_tmux(fake),
            post=lambda url, body: pytest.fail("must not double-post once the worker won"),
            now=now, db_path=db,
        )
        assert out == "finalized by worker — failure · worker — worker interrupted by SIGHUP"
        assert real_sleep_calls  # confirms the poll loop actually ran


def test_cancel_reposts_the_spool_when_no_row_landed_but_a_spool_file_did(monkeypatch, tmp_path):
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        repo = Path(raw) / "demo"
        repo.mkdir()
        _insert(con, "aaaabbbbcccc", "demo", "builder", "run-started", {
            "session": "jax-demo-build-aaaabbbbcccc", "phase": "P", "caller": "claude", "repo": str(repo),
        })
        con.close()
        fake = FakeTmux()
        import datetime as _dt
        now = lambda: _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)
        spooled_event = {
            "run_id": "aaaabbbbcccc", "project": "demo", "role": "builder", "type": "run-finished",
            "source": "deterministic", "emitter": "wrapper",
            "payload": {
                "contract_status": "interrupted", "exit_code": None, "report_path": None,
                "summary": "worker interrupted by SIGHUP", "stage": "worker",
                "diagnostic": "worker interrupted by SIGHUP", "head_sha": None, "result": "failure",
            },
        }
        jaxflow_common._write_spool(repo, "aaaabbbbcccc", spooled_event)
        monkeypatch.setattr(time, "sleep", lambda s: None)
        posted = []

        def post(url, body):
            posted.append(body)
            return 200, {"ok": True}

        out = jaxflow.cmd_cancel("aaaabbbbcccc", run=_run_with_tmux(fake), post=post, now=now, db_path=db)
        assert out == "finalized from spool — failure · worker — worker interrupted by SIGHUP"
        assert posted == [spooled_event]
        assert not jaxflow_common._spool_path(repo, "aaaabbbbcccc").exists()


def test_cancel_refuses_hub_unreachable_instead_of_overwriting_a_pending_spool(monkeypatch):
    # Diff review e55f14eba5be F1: the worker's spool holds the real interrupted evidence;
    # when its redelivery is refused (200 {ok:false}) cancel must NOT post a plain
    # `cancelled` row -- that row would win the terminal claim and the reaper's later
    # redelivery would 409 and delete the evidence. Keep the spool, refuse.
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        repo = Path(raw) / "demo"
        repo.mkdir()
        _insert(con, "aaaabbbbcccc", "demo", "builder", "run-started", {
            "session": "jax-demo-build-aaaabbbbcccc", "phase": "P", "caller": "claude", "repo": str(repo),
        })
        con.close()
        import datetime as _dt
        now = lambda: _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)
        spooled_event = {
            "run_id": "aaaabbbbcccc", "project": "demo", "role": "builder", "type": "run-finished",
            "source": "deterministic", "emitter": "wrapper",
            "payload": {
                "contract_status": "interrupted", "exit_code": None, "report_path": None,
                "summary": "worker interrupted by SIGHUP", "stage": "worker",
                "diagnostic": "worker interrupted by SIGHUP", "head_sha": None, "result": "failure",
            },
        }
        jaxflow_common._write_spool(repo, "aaaabbbbcccc", spooled_event)
        monkeypatch.setattr(time, "sleep", lambda s: None)
        posted = []

        def post(url, body):
            posted.append(body)
            return 200, {"ok": False, "error": "invalid event"}

        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_cancel("aaaabbbbcccc", run=_run_with_tmux(FakeTmux()), post=post, now=now, db_path=db)
        assert exc.value.code == "hub-rejected: 200 invalid event"
        assert posted == [spooled_event]  # the spool was tried, the cancelled row never was
        assert jaxflow_common._spool_path(repo, "aaaabbbbcccc").exists()


def test_cancel_falls_back_to_a_plain_cancelled_row_when_neither_a_row_nor_a_spool_exists(monkeypatch):
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _insert(con, "aaaabbbbcccc", "demo", "reviewer", "run-started", {
            "session": "jax-demo-spec-aaaabbbbcccc", "phase": "P", "caller": "claude",
        })
        con.close()
        fake = FakeTmux()
        import datetime as _dt
        now = lambda: _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)
        monkeypatch.setattr(time, "sleep", lambda s: None)
        events = []

        def post(url, body):
            events.append(body)
            return 200, {"ok": True}

        out = jaxflow.cmd_cancel("aaaabbbbcccc", run=_run_with_tmux(fake), post=post, now=now, db_path=db)
        assert out == "cancelled"
        assert events[0]["payload"]["contract_status"] == "cancelled"


# ---- build: not-a-git-toplevel (fixes cold review round 2 F8 -- build-native and
# reachable via the CLI's very first check, but had no test of its own) ----

def test_build_refuses_not_a_git_toplevel(capsys, monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        outside = Path(raw) / "not-a-repo"
        outside.mkdir(parents=True)
        monkeypatch.chdir(outside)
        fake = FakeTmux()
        code = jaxflow.main(
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
        code = jaxflow.main(
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
        code = jaxflow.main(
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
        code = jaxflow.main(
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
            code = jaxflow.main(
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
        code = jaxflow.main(
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
            code = jaxflow.main(
                argv, run=_run_with_tmux(fake, real_cwd=root), post=lambda e: {"ok": True},
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, allowlist_root=allow_root,
            )
            expected_runtime = builder or jaxflow.BUILDER_DEFAULT
            assert code == jaxflow_common.REFUSED
            assert capsys.readouterr().err.strip() == f"effort-not-supported: {expected_runtime}"
        assert fake.calls == []
        assert not (allow_root / "demo-feat-x").exists()


def _probe_dispatch_build(monkeypatch, root, allow_root, plan, **flags):
    fake = FakeTmux()
    git_calls = []
    events = []
    mkdir_calls = []
    real_mkdir = os.mkdir

    def wrapped_mkdir(path, *a, **k):
        mkdir_calls.append(path)
        return real_mkdir(path, *a, **k)

    monkeypatch.setattr(os, "mkdir", wrapped_mkdir)

    def post(event):
        events.append(event)
        return {"ok": True}

    try:
        run_id = jaxflow.dispatch_build(
            _build_args(plan=str(plan), **flags),
            run=_run_with_tmux_and_log(fake, git_calls, real_cwd=root),
            post=post, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        return run_id, events, git_calls, mkdir_calls, None
    except ji.Refusal as exc:
        return None, events, git_calls, mkdir_calls, exc


def _assert_no_reservation(allow_root, git_calls, mkdir_calls, events):
    assert events == []
    assert mkdir_calls == []
    assert not any(c[:3] == ["git", "worktree", "add"] for c in git_calls)
    assert not (allow_root / "demo-feat-demo").exists()


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
        code = jaxflow.main(
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
        run_id = jaxflow.dispatch_build(
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
                jaxflow.dispatch_review(
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
        code = jaxflow.main(
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
            jaxflow.dispatch_build(
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
        run_id = jaxflow.dispatch_build(
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
                jaxflow.dispatch_build(
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
            jaxflow.dispatch_build(
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
        jaxflow.dispatch_build(
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
            jaxflow.dispatch_build(
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
            jaxflow.dispatch_build(
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
            jaxflow.dispatch_build(
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
            jaxflow.dispatch_build(
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
        run_id = jaxflow.dispatch_build(
            _build_args(plan=str(plan)), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        assert run_id
        assert (allow_root / "demo-feat-demo").is_dir()


def test_build_handoff_omits_goal_tasks_acceptance_and_names_the_plan_path():
    handoff = jaxflow._build_builder_handoff(
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
        run_id = jaxflow.dispatch_build(
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
            jaxflow.dispatch_build(
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
            jaxflow.dispatch_build(
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
            jaxflow.dispatch_build(
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
            jaxflow.dispatch_build(
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
            jaxflow.dispatch_build(
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

        run_id = jaxflow.dispatch_build(
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
        run_id = jaxflow.main(
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
        jaxflow.main(
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
        run_id = jaxflow.dispatch_build(
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
        run_id = jaxflow.dispatch_build(
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
            jaxflow.parse_args(base + ["--verify", blank])
        with pytest.raises(SystemExit):
            jaxflow.parse_args(base + ["--verify", "true", "--build", blank])
    # A command that merely LOOKS trivial is the caller's business, not the parser's.
    args = jaxflow.parse_args(base + ["--verify", "true", "--build", "true"])
    assert (args.verify, args.build) == ("true", "true")
    # --build is OPTIONAL: omitting it parses, and lands as None.
    assert jaxflow.parse_args(base + ["--verify", "true"]).build is None


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


def _seed_resumable(monkeypatch, tmp_path, allow_root, root, *, prior="aaaaaaaaaaaa"):
    import jaxflow_resume as jresume

    plan = _plan_file(root)
    worktree = _init_worktree(root)
    monkeypatch.chdir(root)
    _install_settings(tmp_path, monkeypatch)
    db = Path(allow_root).parent / "jaxos.db"
    monkeypatch.setattr(jr, "DB_PATH", db)
    base_sha = _run_real(["git", "rev-parse", "HEAD"], root).stdout.strip()
    _write_manifest_for_builder_worker(
        root, worktree, run_id=prior, runtime="opencode-builder",
        model="fixture/wire-real-model", effort="high",
        requested_profile="default", root_build_run_id=prior,
        reservation_owned=True, base_sha=base_sha, plan_path=str(plan),
        phase="P",
    )
    state = jresume.capture_work_state(worktree, plan, run=_run_real)
    jresume.write_checkpoint(_checkpoint_path(root, prior), {
        "version": 1,
        "run_id": prior,
        "root_build_run_id": prior,
        "repo": str(root),
        "worktree": str(worktree),
        "branch": "feat/x",
        "base": base_sha,
        "outcome": "failure",
        "plan_revision": state["plan_sha256"],
        "work_state": state,
    })
    con = _fresh_db(db)
    _insert(con, prior, "demo", "builder", "run-started", {
        "phase": "P", "runtime": "opencode-builder", "kind": "build",
        "target": "feat/x", "caller": "claude", "caller_session": "s",
        "model": "fixture/wire-real-model", "effort": "high",
        "session": f"jax-demo-build-{prior}", "repo": str(root),
        "verify": "true", "requested_profile": "default",
        "root_build_run_id": prior,
    })
    _insert(con, prior, "demo", "builder", "run-finished", {
        "contract_status": "ok", "result": "failure", "exit_code": 1,
        "report_path": "/tmp/r.md", "summary": "failed", "phase": "P",
        "head_sha": base_sha,
    })
    con.close()
    return SimpleNamespace(
        plan=plan, worktree=worktree, db=db, prior=prior,
        base_sha=base_sha, state=state, root=root, allow_root=allow_root,
    )


def _dispatch_resume(seeded, fake, **extra):
    events = []
    git_calls = []
    run_id = jaxflow.dispatch_build(
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
    code = jaxflow.main(
        argv, run=run, post=lambda e: pytest.fail("unexpected event"),
        env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
        now=_fixed_now, allowlist_root=tmp_path,
    )
    assert code == jaxflow_common.REFUSED
    captured = capsys.readouterr()
    assert captured.err.strip() == "build-missing-required-flags"
    assert captured.out == ""


def test_build_resume_parser_accepts_id_without_fresh_flags():
    args = jaxflow.parse_args(["build", "--resume", "aaaaaaaaaaaa"])
    assert args.resume == "aaaaaaaaaaaa"
    assert args.plan is None
    assert args.phase is None
    assert args.branch is None
    assert args.whitelist is None
    assert args.verify is None
    assert args.fallback is False
    args = jaxflow.parse_args([
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
                jaxflow.dispatch_build(_build_args(resume="aaaaaaaaaaaa", **extra), **kwargs)
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
            jaxflow.dispatch_build(
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
            jaxflow.dispatch_build(
                _build_args(resume=seeded.prior),
                run=_run_with_tmux(fake, real_cwd=root),
                post=lambda e: {"ok": True},
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root,
            )
        assert caught.value.code == "resume-ineligible"


def test_build_resume_has_no_force_flag():
    with pytest.raises(SystemExit):
        jaxflow.parse_args(["build", "--resume", "aaaaaaaaaaaa", "--force"])


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


class _E2EBuilderPopen:
    """Writes a report using the injected run_id; first launch leaves partial work."""
    launches = 0

    def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None,
                 start_new_session=None, env=None):
        self.argv = argv
        self.cwd = cwd
        self.env = env
        self.pid = 5252
        self.returncode = 0
        self.stdout = io.BytesIO() if stdout is subprocess.PIPE else None
        self.stderr = io.BytesIO() if stderr is subprocess.PIPE else None
        text = stdin.read().decode("utf-8") if stdin is not None else ""
        run_id = phase = report = None
        for line in text.splitlines():
            if line.startswith("run_id: "):
                run_id = line[8:]
            elif line.startswith("phase: "):
                phase = line[7:]
            elif line.startswith("report: "):
                report = line[8:]
        type(self).launches += 1
        if type(self).launches == 1:
            work = Path(cwd)
            (work / "a.py").write_text("partial\n", encoding="utf-8")
            _git(work, "add", "a.py")
            _git(work, "commit", "-m", "partial")
            (work / "leftover.txt").write_text("untracked\n", encoding="utf-8")
            result, summary = "failure", "stopped halfway"
        else:
            result, summary = "success", "finished"
        Path(report).write_text(
            f"---\nrun_id: {run_id}\nproject: demo\nrole: builder\nphase: {phase}\n"
            f"result: {result}\nsummary: {summary}\n---\nbody\n",
            encoding="utf-8",
        )

    def wait(self, timeout=None):
        return self.returncode


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
        first_id = jaxflow.dispatch_build(
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
        code = jaxflow.run_worker(
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

        second_id = jaxflow.dispatch_build(
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
        code = jaxflow.run_worker(
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
        first_id = jaxflow.dispatch_build(
            _build_args(
                plan=str(plan), branch="feat/x", whitelist="a.py",
                verify="true",
            ),
            run=_run_with_tmux(fake, real_cwd=root),
            post=post, env=env, now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        jaxflow.run_worker(
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

        second_id = jaxflow.dispatch_build(
            _build_args(resume=first_id),
            run=_run_with_tmux(fake, real_cwd=root),
            post=post, env=env, now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        resume_manifest = json.loads(
            (root / ".local" / "runs" / second_id / "manifest.json").read_text(encoding="utf-8"))
        assert resume_manifest["requested_profile"] == "default"
        assert resume_manifest["model"] == "fixture/changed-model"
        jaxflow.run_worker(
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
            jaxflow.dispatch_build(
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
        jaxflow.dispatch_build(
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
        run_id = jaxflow.dispatch_build(
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

        rc = jaxflow.cmd_merge(
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

        run_id = jaxflow.dispatch_build(
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
            jaxflow.dispatch_build(
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
            jaxflow.dispatch_build(
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
            jaxflow.dispatch_build(
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

        run_id = jaxflow.dispatch_build(
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

        run_id = jaxflow.dispatch_build(
            _build_args(plan=str(plan), branch="feat/y", builder="opencode-deepseek"),
            run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
            allowlist_root=allow_root,
        )
        assert events[0]["payload"]["runtime"] == "opencode-deepseek"
        # An OpenCode runtime records its RESOLVED model id, not the "default" sentinel
        # (same rule the opencode-grok case above asserts).
        assert events[0]["payload"]["model"] == "openrouter/deepseek/deepseek-v4-flash-0731"

        code = jaxflow.main(
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
            jaxflow.dispatch_build(
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


# ---- status.md by-product (spec §4.5) ----

_STATUS_TEMPLATE = (
    "---\n"
    "project: Demo Project\n"
    "stage: spec\n"
    "builder: claude-code\n"
    "branch: main\n"
    "tmux: demo-session\n"
    "flag: main client\n"
    "updated: 2020-01-01T00:00:00-03:00\n"
    "---\n\n"
    "## Now\n"
    "Old status text here.\n\n"
    "## Residuals\n"
    "- an accepted risk\n"
    "- another one\n"
)

# Every "a write should happen" test below dispatches its manifest AFTER the template's own
# `updated` value (2020) -- fixes cold review F5's own finding that the first draft's
# fixtures had dispatch_start (2000) BEFORE updated (2020), which made the (correct) stale
# guard skip every write the tests then asserted had happened.
_AFTER_TEMPLATE_UPDATED = "2026-01-01T00:00:00-03:00"


def _write_status_md(root, text=_STATUS_TEMPLATE):
    path = root / ".jax-os" / "status.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_status_md_missing_file_is_a_silent_noop():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        manifest = {"repo": str(root), "kind": "build", "runtime": "opencode-grok"}
        jaxflow_common._update_status_md(manifest, run=_run_real, allowlist_root=root)
        assert not (root / ".jax-os").exists()


def test_status_md_forged_repo_outside_allowlist_is_skipped_before_any_read(capsys):
    # fixes F2 (HIGH, NEW): a forged manifest["repo"] pointing outside the allowlist root
    # must never be canonicalized-then-trusted -- the existing status.md at that path is
    # left completely untouched, and the skip fires before the file is ever opened.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        outside = Path(raw).resolve() / "outside"
        status_path = _write_status_md(outside)
        before = status_path.read_text(encoding="utf-8")
        manifest = {
            "repo": str(outside), "kind": "build", "runtime": "opencode-grok",
            "branch": "feat/demo", "dispatch_start": _AFTER_TEMPLATE_UPDATED,
            "worker_summary": "built the thing", "worker_contract_status": "ok",
            "worker_outcome": "success",
        }
        jaxflow_common._update_status_md(manifest, run=_run_real, allowlist_root=allow_root)
        assert status_path.read_text(encoding="utf-8") == before
        assert "status.md: skipped (repo outside allowlist)" in capsys.readouterr().out


def test_status_md_build_run_preserves_unmanaged_fields_and_residuals():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        status_path = _write_status_md(root)
        manifest = {
            "repo": str(root), "kind": "build", "runtime": "opencode-grok",
            "branch": "feat/demo", "dispatch_start": _AFTER_TEMPLATE_UPDATED,
            "worker_summary": "built the thing", "worker_contract_status": "ok",
            "worker_outcome": "success",
        }
        jaxflow_common._update_status_md(manifest, run=_run_real, allowlist_root=root)
        text = status_path.read_text(encoding="utf-8")
        assert "project: Demo Project" in text
        assert "tmux: demo-session" in text
        assert "flag: main client" in text
        assert "- an accepted risk\n- another one" in text
        assert "stage: build" in text
        assert "builder: opencode-grok" in text
        assert "branch: feat/demo" in text
        assert "## Now\nbuilt the thing" in text
        assert "Old status text here." not in text


def test_status_md_stale_write_guard_skips_when_file_is_newer(capsys):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        status_path = _write_status_md(root, _STATUS_TEMPLATE.replace(
            "updated: 2020-01-01T00:00:00-03:00", "updated: 2099-01-01T00:00:00-03:00",
        ))
        before = status_path.read_text(encoding="utf-8")
        manifest = {
            "repo": str(root), "kind": "build", "runtime": "opencode-grok",
            "branch": "feat/demo", "dispatch_start": _AFTER_TEMPLATE_UPDATED,
            "worker_summary": "built the thing", "worker_contract_status": "ok",
            "worker_outcome": "success",
        }
        jaxflow_common._update_status_md(manifest, run=_run_real, allowlist_root=root)
        assert status_path.read_text(encoding="utf-8") == before
        assert "status.md: skipped stale write" in capsys.readouterr().out


def test_status_md_gate_rules_for_diff_verdict_and_no_gate_otherwise():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        status_path = _write_status_md(root)
        # A real `diff` manifest (Part 2, Task 5) has no "branch" key -- `target` IS the
        # branch under review. Omitting "branch" here is deliberate: it is what catches a
        # `_update_status_md` regression that falls back to the CONTROL repo's own current
        # branch instead of the reviewed one.
        manifest = {
            "repo": str(root), "kind": "diff", "runtime": "codex",
            "target": "feat/demo", "dispatch_start": _AFTER_TEMPLATE_UPDATED,
            "worker_summary": "reject — 1 HIGH", "worker_contract_status": "ok",
            "worker_outcome": "reject",
        }
        jaxflow_common._update_status_md(manifest, run=_run_real, allowlist_root=root)
        assert "gate:" not in status_path.read_text(encoding="utf-8")

        status_path2 = _write_status_md(root)
        manifest["worker_outcome"] = "approve"
        manifest["worker_summary"] = "approve — clean"
        jaxflow_common._update_status_md(manifest, run=_run_real, allowlist_root=root)
        text = status_path2.read_text(encoding="utf-8")
        assert "gate: awaiting-approval" in text
        assert "stage: review" in text
        assert "branch: feat/demo" in text


def test_status_md_spec_review_gate_requires_both_no_high_and_header_opt_in():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        base_manifest = {
            "repo": str(root), "kind": "spec", "runtime": "codex",
            "branch": "main", "dispatch_start": _AFTER_TEMPLATE_UPDATED,
            "worker_summary": "approve — clean", "worker_contract_status": "ok",
            "worker_outcome": "approve",
        }

        # verdict approve, but the reviewed doc did NOT opt in -> no gate.
        status_path = _write_status_md(root)
        manifest = dict(base_manifest, spec_gate_required=False)
        jaxflow_common._update_status_md(manifest, run=_run_real, allowlist_root=root)
        assert "gate:" not in status_path.read_text(encoding="utf-8")

        # verdict reject, doc DID opt in -> still no gate (HIGH findings exist).
        status_path2 = _write_status_md(root)
        manifest2 = dict(base_manifest, worker_outcome="reject", spec_gate_required=True)
        jaxflow_common._update_status_md(manifest2, run=_run_real, allowlist_root=root)
        assert "gate:" not in status_path2.read_text(encoding="utf-8")

        # both true -> gate.
        status_path3 = _write_status_md(root)
        manifest3 = dict(base_manifest, worker_outcome="approve-with-changes", spec_gate_required=True)
        jaxflow_common._update_status_md(manifest3, run=_run_real, allowlist_root=root)
        assert "gate: awaiting-approval" in status_path3.read_text(encoding="utf-8")
        assert "stage: spec" in status_path3.read_text(encoding="utf-8")


# ---- status.md: every failure mode is a silent skip, never an exception (fixes cold review F8) ----

def test_status_md_unparseable_frontmatter_is_a_silent_skip(capsys):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        status_path = _write_status_md(root, "not frontmatter at all\n")
        manifest = {
            "repo": str(root), "kind": "build", "runtime": "opencode-grok",
            "branch": "feat/demo", "dispatch_start": _AFTER_TEMPLATE_UPDATED,
            "worker_summary": "x", "worker_contract_status": "ok", "worker_outcome": "success",
        }
        jaxflow_common._update_status_md(manifest, run=_run_real, allowlist_root=root)
        assert status_path.read_text(encoding="utf-8") == "not frontmatter at all\n"
        assert "status.md: unparseable frontmatter" in capsys.readouterr().out


def test_status_md_malformed_timestamp_does_not_crash_and_proceeds():
    # fixes cold review F8: `existing_updated >= dispatch_start` as a plain STRING compare
    # is wrong across offsets, AND a parse failure must never raise -- it is treated as "not
    # provably stale", so the write proceeds rather than being silently and wrongly skipped.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        status_path = _write_status_md(root, _STATUS_TEMPLATE.replace(
            "updated: 2020-01-01T00:00:00-03:00", "updated: not-a-real-timestamp",
        ))
        manifest = {
            "repo": str(root), "kind": "build", "runtime": "opencode-grok",
            "branch": "feat/demo", "dispatch_start": _AFTER_TEMPLATE_UPDATED,
            "worker_summary": "built the thing", "worker_contract_status": "ok",
            "worker_outcome": "success",
        }
        jaxflow_common._update_status_md(manifest, run=_run_real, allowlist_root=root)
        text = status_path.read_text(encoding="utf-8")
        assert "## Now\nbuilt the thing" in text
        assert "stage: build" in text


def test_status_md_current_branch_run_error_is_a_silent_skip(capsys):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        status_path = _write_status_md(root)
        before = status_path.read_text(encoding="utf-8")
        manifest = {
            # kind "spec" (not "build"/"diff") is what reaches `_current_branch`, the only
            # call site in `_update_status_md` that invokes the injected `run()`.
            "repo": str(root), "kind": "spec", "runtime": "codex",
            "dispatch_start": _AFTER_TEMPLATE_UPDATED, "worker_summary": "approve — clean",
            "worker_contract_status": "ok", "worker_outcome": "approve",
        }

        def raising_run(argv, cwd=None):
            raise OSError("git not found")

        jaxflow_common._update_status_md(manifest, run=raising_run, allowlist_root=root)
        assert status_path.read_text(encoding="utf-8") == before
        assert "status.md: skipped" in capsys.readouterr().out


def test_status_md_write_oserror_is_a_silent_skip(capsys):
    # fixes cold review round 2 F4: chmod on the DIRECTORY (the first draft's approach)
    # never blocks Path.write_text() truncating an EXISTING, already-writable file -- only
    # the FILE's own permission bit does. Removing the owner's write bit on `status.md`
    # itself is what actually makes the final `path.write_text(...)` call raise.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        status_path = _write_status_md(root)
        manifest = {
            "repo": str(root), "kind": "build", "runtime": "opencode-grok",
            "branch": "feat/demo", "dispatch_start": _AFTER_TEMPLATE_UPDATED,
            "worker_summary": "x", "worker_contract_status": "ok", "worker_outcome": "success",
        }
        os.chmod(status_path, 0o400)  # read-only
        try:
            jaxflow_common._update_status_md(manifest, run=_run_real, allowlist_root=root)
        finally:
            os.chmod(status_path, 0o600)  # restore so TemporaryDirectory can clean up
        assert status_path.read_text(encoding="utf-8") == _STATUS_TEMPLATE
        assert "status.md: skipped (" in capsys.readouterr().out


def test_status_md_byte_preserves_unmanaged_frontmatter_with_crlf_and_noncanonical_order():
    # fixes cold review round 2 F7: the OWNED keys (stage/builder/branch/updated/gate) are
    # the only lines this function may ever rewrite -- every other line, INCLUDING its own
    # EOL style and the file's own field order, must survive byte-for-byte. Deliberately
    # uses CRLF and an order that does not match the canonical template.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        original = (
            "---\r\n"
            "flag: main client\r\n"
            "project: Demo Project\r\n"
            "stage: spec\r\n"
            "builder: claude-code\r\n"
            "tmux: demo-session\r\n"
            "branch: main\r\n"
            "updated: 2020-01-01T00:00:00-03:00\r\n"
            "---\r\n\r\n"
            "## Now\r\n"
            "Old status text here.\r\n"
        )
        status_path = _write_status_md(root, original)
        manifest = {
            "repo": str(root), "kind": "build", "runtime": "opencode-grok",
            "branch": "feat/demo", "dispatch_start": _AFTER_TEMPLATE_UPDATED,
            "worker_summary": "built the thing", "worker_contract_status": "ok",
            "worker_outcome": "success",
        }
        jaxflow_common._update_status_md(manifest, run=_run_real, allowlist_root=root)
        # fixture fix (execution rule): `Path.read_text` without `newline=""` applies
        # universal-newline translation, silently converting the CRLF this assertion is
        # trying to observe back to LF -- the same gap the production read needed fixing
        # for (jaxflow.py's `_update_status_md`).
        lines = status_path.read_text(encoding="utf-8", newline="").splitlines(keepends=True)
        assert lines[0] == "---\r\n"
        assert lines[1] == "flag: main client\r\n"  # unmanaged: untouched, same position
        assert lines[2] == "project: Demo Project\r\n"  # unmanaged: untouched, same position
        assert lines[3] == "stage: build\r\n"  # owned: rewritten IN PLACE, CRLF preserved
        assert lines[4] == "builder: opencode-grok\r\n"  # owned: rewritten in place
        assert lines[5] == "tmux: demo-session\r\n"  # unmanaged: untouched, same position
        assert lines[6] == "branch: feat/demo\r\n"  # owned: rewritten in place
        assert lines[7].startswith("updated: ") and lines[7].endswith("\r\n")  # owned
        assert lines[8] == "---\r\n"  # closing delimiter: untouched
        assert "gate:" not in "".join(lines)


# ---- worker: builder run (spec §4.2 step 3-4) ----

class FakeBuilderPopen:
    """Stands in for the builder LLM subprocess: writes report.md DIRECTLY at the path named
    in its own prompt (mirroring a real builder's own Write-tool behavior), then exits.
    MOA-470: also exposes `.stdout`/`.stderr` as readable in-memory streams when piped
    (opencode's stdout was never the report channel -- report delivery, above, is
    untouched by this)."""
    stdout = None
    stderr = None
    CHILD_BYTES = b""

    REPORT_TEXT = (
        "---\nrun_id: bbbbccccdddd\nproject: demo\nrole: builder\nphase: PHASE\n"
        "result: success\nsummary: built the thing\n---\nbody\n"
    )

    def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
        self.argv = argv
        self.cwd = cwd
        self.pid = 5252
        self.returncode = 0
        self.stdin_bytes = stdin.read() if stdin is not None else None
        text = self.stdin_bytes.decode("utf-8")
        for line in text.splitlines():
            if line.startswith("report: "):
                Path(line[len("report: "):]).write_text(type(self).REPORT_TEXT, encoding="utf-8")
                break
        if stdout is subprocess.PIPE:
            self.stdout = io.BytesIO(self.CHILD_BYTES)
        if stderr is subprocess.PIPE:
            self.stderr = io.BytesIO(self.CHILD_BYTES)

    def wait(self, timeout=None):
        return self.returncode


def _write_manifest_for_builder_worker(root, worktree, *, run_id="bbbbccccdddd", verify="true",
                                        runtime="opencode-grok", model="xai/grok-4.6",
                                        effort="n/a", build=None, **extra):
    manifest = {
        "kind": "build", "role": "builder", "project": "demo", "phase": "PHASE",
        "repo": str(root), "worktree": str(worktree), "branch": "feat/x",
        "target": "feat/x", "caller": "claude", "caller_session": "01234567-89ab-4cde-8f01-23456789abcd",
        "model": model, "effort": effort, "runtime": runtime,
        "run_id": run_id, "session": f"jax-demo-build-{run_id}", "no_callback": False,
        "whitelist": ["a.py"], "verify": verify,
        "plan_path": str(worktree / ".local" / "docs" / "plans" / "plan.md"),
        "dispatch_start": "2026-01-01T00:00:00-03:00",
    }
    if build:
        manifest["build"] = build
    manifest.update(extra)
    path = root / ".local" / "runs" / run_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    os.chmod(path, 0o644)
    return path, manifest


def _init_worktree(root, branch="feat/x"):
    """A real worktree of `root`, checked out on `branch` with one commit -- close enough to
    what `git worktree add -b` produces for the worker's own git calls (rev-parse HEAD,
    the --verify shell command) to run against."""
    worktree = root.parent / f"{root.name}-{branch.replace('/', '-')}"
    default = _run_real(["git", "branch", "--show-current"], root).stdout.strip()
    _git(root, "worktree", "add", "-b", branch, str(worktree), default)
    return worktree


def _init_separate_git_dir_repo(root: Path, gitdir: Path):
    """A main checkout whose git metadata lives elsewhere (`--separate-git-dir`): its
    `.git` is a FILE, git reports gitdir == common dir, and its linked worktrees' admin
    dirs live under `<gitdir>/worktrees/` -- the layout whose common dir is NOT named
    `.git` (MOA-467 acceptance fixes)."""
    root.mkdir(parents=True, exist_ok=True)
    gitdir.parent.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-b", "main", f"--separate-git-dir={gitdir}")
    _git(root, "config", "user.email", "t@t.test")
    _git(root, "config", "user.name", "t")
    (root / "README").write_text("x\n", encoding="utf-8")
    (root / ".gitignore").write_text(".local/\n", encoding="utf-8")
    _git(root, "add", "README", ".gitignore")
    _git(root, "commit", "-m", "init")


def _write_plan(worktree, text="# Plan\n\n**Goal:** ship the thing.\n\n### Task 1: do it\n"):
    plan_dir = worktree / ".local" / "docs" / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "plan.md").write_text(text, encoding="utf-8")


# ---- worker: manifest identity is validated before ANY path construction (fixes Part 3
# diff-review F1) ----

def test_worker_builder_refuses_forged_run_id_before_any_path_construction(capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        manifest["run_id"] = "/tmp/jaxflow-escape"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        def popen(*a, **kw):
            raise AssertionError("popen must never be called for a forged run_id")

        fake = FakeTmux()
        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(fake, real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True}, popen=popen, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        err_lines = capsys.readouterr().err.strip().splitlines()
        assert err_lines[0] == "path-outside-allowlist"
        assert err_lines[1] == (
            "callback delivery failed: '/tmp/jaxflow-escape' path-unsafe; use jaxflow status/result"
        )
        assert fake.calls == []
        assert not Path("/tmp/jaxflow-escape.md").exists()
        assert not Path("/tmp/jaxflow-escape").exists()
        # fixes F1 (RECURRENCE part3-F3): a _validate_run_paths failure now posts the
        # cancelled terminal row for this run and removes only the manifest dir the
        # worker was actually invoked with -- the real worktree/branch it never
        # validated stay untouched.
        assert len(events) == 1
        assert events[0]["payload"]["contract_status"] == "cancelled"
        assert events[0]["role"] == "builder"
        assert manifest_path.parent.exists()
        assert worktree.is_dir()


def test_worker_builder_refuses_worktree_mismatched_from_manifest_derivation(capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        other = allow_root / "some-other-dir"
        other.mkdir()
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        manifest["worktree"] = str(other)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        def popen(*a, **kw):
            raise AssertionError("popen must never be called for a mismatched worktree")

        fake = FakeTmux()
        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(fake, real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True}, popen=popen, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"
        assert fake.calls == []
        assert len(events) == 1
        assert events[0]["payload"]["contract_status"] == "cancelled"
        assert manifest_path.parent.exists()
        # neither the real worktree nor the forged "other" path is ever touched.
        assert worktree.is_dir()
        assert other.is_dir()
        line_path = jaxflow_common.CALLBACKS_ROOT / manifest["caller_session"] / f"{manifest['run_id']}.line"
        assert line_path.read_text(encoding="utf-8").startswith(
            f"[JAXFLOW] build {manifest['run_id']} finished — cancelled — no report"
        )


def test_worker_builder_refuses_worktree_outside_allowlist_root(capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        outside = Path(raw).resolve() / "outside-escape"
        outside.mkdir()
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        manifest["worktree"] = str(outside)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        def popen(*a, **kw):
            raise AssertionError("popen must never be called for an escaping worktree")

        fake = FakeTmux()
        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(fake, real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True}, popen=popen, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"
        assert fake.calls == []
        assert len(events) == 1
        assert events[0]["payload"]["contract_status"] == "cancelled"
        assert manifest_path.parent.exists()
        assert worktree.is_dir()
        assert outside.is_dir()


def test_worker_builder_refuses_branch_target_mismatch_before_validation_leaves_victim_branch(capsys):
    # fixes F4 (REGRESSION): a manifest whose "branch" disagrees with its own "target"
    # now fails _validate_run_paths itself, taking the SAME lightweight, no-cleanup
    # refusal path a forged run_id/worktree does (F1) -- an unrelated real branch that
    # happens to share the forged branch name is therefore never touched.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)  # branch feat/x, matches target
        _git(root, "branch", "victim")
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        manifest["branch"] = "victim"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        def popen(*a, **kw):
            raise AssertionError("popen must never be called for a branch/target mismatch")

        fake = FakeTmux()
        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(fake, real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True}, popen=popen, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"
        assert len(events) == 1
        assert events[0]["payload"]["contract_status"] == "cancelled"
        assert manifest_path.parent.exists()
        # the run's own worktree/branch (feat/x) AND the unrelated "victim" branch both
        # survive -- cleanup never runs for an unvalidated manifest.
        assert worktree.is_dir()
        branches = _run_real(["git", "branch", "--list", "victim"], root).stdout
        assert "victim" in branches


def test_worker_builder_refusal_after_validation_cleans_up_the_validated_target_branch():
    # fixes F4: once validation succeeds, _refuse_builder_run cleans up manifest["target"]
    # -- the validated identity -- never a raw manifest["branch"] value, even when
    # "branch" is absent from the manifest entirely.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)  # branch feat/x
        _write_plan(worktree, "# Plan\n\n**Goal:** g.\n\n**Spec:** /outside/spec.md\n\n### Task 1: t\n")
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        del manifest["branch"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        def popen(*a, **kw):
            raise AssertionError("popen must never be called for an unresolvable spec reference")

        fake = FakeTmux()
        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(fake, real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True}, popen=popen, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert len(events) == 1
        assert events[0]["payload"]["contract_status"] == "cancelled"
        # the real worktree/branch this run owns (feat/x, derived from "target") is
        # cleaned up -- not left dangling for lack of a "branch" key to pass along.
        assert not worktree.exists()
        branches = _run_real(["git", "branch", "--list", "feat/x"], root).stdout
        assert "feat/x" not in branches


def test_worker_builder_manifest_missing_plan_path_is_a_cleaned_refusal(capsys):
    # fixes F1 (RECURRENCE): a manifest that passes `_validate_run_paths` but has no
    # "plan_path" key used to raise a bare KeyError out of the pre-launch `try` block --
    # `except (Refusal, OSError, ValueError)` never caught it, so the worker crashed with
    # no cancelled run-finished row, no callback, and the reservation (worktree, branch,
    # manifest dir) left behind. Now every pre-launch failure, KeyError included, routes
    # through `_refuse_builder_run`.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)  # branch feat/x, matches target
        _write_plan(worktree)
        fake = FakeTmux()
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        del manifest["plan_path"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        def popen(*a, **kw):
            raise AssertionError("popen must never be called for a manifest missing plan_path")

        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(fake, real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True}, popen=popen, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        capsys.readouterr()
        assert len(events) == 1
        assert events[0]["payload"]["contract_status"] == "cancelled"
        assert events[0]["payload"]["summary"] == "manifest missing plan_path"

        line_path = jaxflow_common.CALLBACKS_ROOT / manifest["caller_session"] / f"{manifest['run_id']}.line"
        assert line_path.read_text(encoding="utf-8").rstrip("\n") == (
            f"[JAXFLOW] build {manifest['run_id']} finished — cancelled — no report"
        )

        assert not manifest_path.parent.exists()
        assert not worktree.exists()
        branches = _run_real(["git", "branch", "--list", "feat/x"], root).stdout
        assert "feat/x" not in branches
        admin = _run_real(["git", "worktree", "list", "--porcelain"], root).stdout
        assert worktree.name not in admin


def test_worker_builder_run_writes_report_records_head_sha_and_verify_passes():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree), post=post,
            popen=FakeBuilderPopen, allowlist_root=root.parent,
        )
        assert code == 0
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["contract_status"] == "ok"
        assert payload["result"] == "success"
        assert payload["head_sha"] is not None
        assert len(payload["head_sha"]) == 40
        tests_file = worktree / ".local" / "reports" / "bbbbccccdddd.tests.txt"
        assert "COMMAND: true" in tests_file.read_text(encoding="utf-8")
        assert "EXIT: 0" in tests_file.read_text(encoding="utf-8")
        report = worktree / ".local" / "reports" / "bbbbccccdddd.md"
        assert report.is_file()
        assert oct(report.stat().st_mode)[-3:] == "444"


def test_worker_builder_verify_failure_overrides_result_but_not_contract_status():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree, verify="false")
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree), post=post,
            popen=FakeBuilderPopen, allowlist_root=root.parent,
        )
        payload = events[0]["payload"]
        # The report itself still says "success" (the builder's own claim, untouched) --
        # jaxflow's own post-build verify overrides only the LEDGER's `result` field.
        assert payload["contract_status"] == "ok"
        assert payload["result"] == "failure"
        report_text = (worktree / ".local" / "reports" / "bbbbccccdddd.md").read_text(encoding="utf-8")
        assert "result: success" in report_text


def test_worker_builder_runs_both_commands_writes_both_frames_and_wires_the_handoff():
    """Spec §7.1 #35, plus the WIRING (cold-review F4). Asserting only on the evidence file
    would let the whole change pass while the worker still called
    `_build_builder_handoff(..., build_cmd=None)` -- the handoff would keep saying
    `commands.build: none`, which is the exact defect MOA-454 is named for. So this test
    reads the prompt the builder actually received."""
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, _ = _write_manifest_for_builder_worker(
            root, worktree, verify="true", build="true")
        events = []
        prompts = []

        class CapturingBuilderPopen(FakeBuilderPopen):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                prompts.append(self.stdin_bytes.decode("utf-8"))

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True},
            popen=CapturingBuilderPopen, allowlist_root=root.parent,
        )
        assert events[0]["payload"]["result"] == "success"
        text = (worktree / ".local" / "reports" / "bbbbccccdddd.tests.txt").read_text(encoding="utf-8")
        assert text.count("COMMAND: ") == 2
        assert text.count("EXIT: 0") == 2
        # the wiring: the handoff the builder was actually handed
        assert len(prompts) == 1
        assert "  test: true\n" in prompts[0]
        assert "  build: true\n" in prompts[0]
        assert "build: none" not in prompts[0]


def test_worker_builder_a_failing_build_command_fails_the_run_and_still_records_the_test_frame():
    """EITHER command failing makes the run a failure -- and the passing test's frame is
    still there, so the tech lead can see WHICH half broke."""
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, _ = _write_manifest_for_builder_worker(
            root, worktree, verify="true", build="false")
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True},
            popen=FakeBuilderPopen, allowlist_root=root.parent,
        )
        assert events[0]["payload"]["result"] == "failure"
        text = (worktree / ".local" / "reports" / "bbbbccccdddd.tests.txt").read_text(encoding="utf-8")
        assert "EXIT: 0\n" in text and "EXIT: 1\n" in text


def test_worker_builder_still_runs_the_build_when_the_test_command_failed():
    """The no-short-circuit property, end to end (spec §7.1 #35). Removing `--build`'s
    unconditional run must fail THIS test and nothing else."""
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, _ = _write_manifest_for_builder_worker(
            root, worktree, verify="false", build="true")
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True},
            popen=FakeBuilderPopen, allowlist_root=root.parent,
        )
        text = (worktree / ".local" / "reports" / "bbbbccccdddd.tests.txt").read_text(encoding="utf-8")
        assert text.count("COMMAND: ") == 2
        assert events[0]["payload"]["result"] == "failure"


class FailingReportPopen(FakeBuilderPopen):
    """A builder that honestly reports `result: failure` -- e.g. it stopped halfway, or
    `.githooks/commit-msg` rejected every commit -- while the code it did write still
    passes the verify chain."""

    REPORT_TEXT = (
        "---\nrun_id: bbbbccccdddd\nproject: demo\nrole: builder\nphase: PHASE\n"
        "result: failure\nsummary: stopped on task 3\n---\nbody\n"
    )

    def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
        self.argv = argv
        self.cwd = cwd
        self.pid = 5252
        self.returncode = 0
        self.stdin_bytes = stdin.read() if stdin is not None else None
        text = self.stdin_bytes.decode("utf-8")
        for line in text.splitlines():
            if line.startswith("report: "):
                Path(line[len("report: "):]).write_text(
                    FailingReportPopen.REPORT_TEXT, encoding="utf-8")
                break


class NoLabelReportPopen(FakeBuilderPopen):
    """Writes a report with real content but no parseable `result:`/`verdict:` label
    anywhere -- the b756335468ba shape (spec §2.1, parser table row 6, decision 3
    branch (c))."""
    REPORT_TEXT = "# Builder report\n\nDid five tasks. All green.\n"


def _run_failing_builder(root, worktree, fake, *, verify):
    manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree, verify=verify)
    events = []
    jaxflow.run_worker(
        str(manifest_path), run=_run_with_tmux(fake, real_cwd=worktree),
        post=lambda event: events.append(event) or {"ok": True},
        popen=FailingReportPopen, allowlist_root=root.parent,
    )
    line_path = jaxflow_common.CALLBACKS_ROOT / manifest["caller_session"] / f"{manifest['run_id']}.line"
    return events[0]["payload"], line_path.read_text(encoding="utf-8").rstrip("\n")


def test_worker_builder_failure_with_passing_verify_flags_the_contradiction(monkeypatch):
    """MOA-455. The verify chain is a veto on SUCCESS only -- it never overturns a
    builder's claimed failure, because the builder can know something the verify cannot
    see. That asymmetry stays; what changes is that the disagreement stops being
    invisible. The ledger `result` is untouched: only the callback line says so."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        payload, line = _run_failing_builder(root, worktree, FakeTmux(), verify="true")

        assert payload["result"] == "failure"
        assert "(jaxflow verify passed — read the evidence)" in line
        assert line.startswith(
            "[JAXFLOW] build bbbbccccdddd finished — "
            "failure (jaxflow verify passed — read the evidence) · report — "
        )


def test_worker_builder_failure_with_failing_verify_stays_a_plain_failure(monkeypatch):
    """The other half of MOA-455: when both agree, the note must NOT appear. Without
    this the fix passes by appending the note unconditionally."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        payload, line = _run_failing_builder(root, worktree, FakeTmux(), verify="false")

        assert payload["result"] == "failure"
        assert "jaxflow verify passed" not in line
        assert line.startswith("[JAXFLOW] build bbbbccccdddd finished — failure · report — ")


def test_worker_builder_normal_path_reports_runtime_stage_on_a_terminal_stream_error():
    # child.log's LAST line is a top-level error event; report is missing (no report.md
    # ever written -- the worker's own __init__ never runs the write branch).
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)

        class CrashingPopen(FakeBuilderPopen):
            CHILD_BYTES = (
                b'{"type": "step_start"}\n'
                b'{"type": "error", "error": {"name": "APIError", '
                b'"data": {"statusCode": 403, "message": "spending limit reached"}}}\n'
            )

            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None,
                         start_new_session=None, env=None):
                self.argv = argv
                self.cwd = cwd
                self.pid = 5254
                self.returncode = 0
                self.stdin_bytes = stdin.read() if stdin is not None else None
                if stdout is subprocess.PIPE:
                    self.stdout = io.BytesIO(self.CHILD_BYTES)
                if stderr is subprocess.PIPE:
                    self.stderr = io.BytesIO(self.CHILD_BYTES)

            def wait(self, timeout=None):
                return self.returncode

        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True},
            popen=CrashingPopen, allowlist_root=root.parent,
        )
        payload = events[0]["payload"]
        assert payload["result"] == "failure"
        assert payload["stage"] == "runtime"
        assert payload["diagnostic"] == "APIError 403 spending limit reached"
        assert payload["contract_status"] == "missing"


def test_worker_builder_normal_path_omits_stage_on_the_happy_path():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True},
            popen=FakeBuilderPopen, allowlist_root=root.parent,
        )
        payload = events[0]["payload"]
        assert "stage" not in payload or payload["stage"] is None
        assert "diagnostic" not in payload or payload["diagnostic"] is None


def test_worker_builder_missing_report_gets_a_verify_derived_result():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)

        class NoReportPopen(FakeBuilderPopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None,
                         start_new_session=None, env=None):
                self.stdin_bytes = stdin.read() if stdin is not None else None
                self.pid = 5253
                self.returncode = 0

            def wait(self, timeout=None):
                return 0

        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree), post=post,
            popen=NoReportPopen, allowlist_root=root.parent,
        )
        payload = events[0]["payload"]
        assert payload["contract_status"] == "missing"
        # AC 4: green verify, but no base_sha in the manifest -> no commit-beyond-base
        # evidence -> the fallback records failure, never a bare absence any more.
        assert payload["result"] == "failure"


def test_worker_diff_reviewer_normal_path_reports_no_verdict_on_terminal_stream_error():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        # The diff worker runs a real `git diff --name-only <base>..<head>` BEFORE Popen,
        # so placeholder SHAs would refuse before this fixture is ever reached.
        (worktree / "changed.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "change")
        base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()

        class CrashingDiffPopen(FakePopen):
            CHILD_BYTES = (
                b'{"type": "error", "name": "E", "data": {"statusCode": 403, "message": "m"}}\n'
            )

            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None,
                         start_new_session=None, env=None):
                self.argv = argv
                self.cwd = cwd
                self.pid = 4243
                self.returncode = 0
                self.stdin_bytes = stdin.read() if stdin is not None else None
                # No report written (--output-last-message intentionally skipped) --
                # this fixture proves contract_status: missing, not invalid.
                if stdout is subprocess.PIPE:
                    self.stdout = io.BytesIO(self.CHILD_BYTES)
                if stderr is subprocess.PIPE:
                    self.stderr = io.BytesIO(self.CHILD_BYTES)

            def wait(self, timeout=None):
                return self.returncode

        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, base_sha=base_sha, head_sha=head_sha,
        )
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True},
            popen=CrashingDiffPopen, allowlist_root=root.parent,
        )
        payload = events[0]["payload"]
        assert "verdict" not in payload
        assert payload["stage"] == "runtime"
        assert payload["diagnostic"] == "E 403 m"


def test_worker_doc_reviewer_normal_path_reports_report_stage_on_clean_missing_report():
    class NoReportPopen(FakePopen):
        def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None,
                     start_new_session=None, env=None):
            self.argv = argv
            self.cwd = cwd
            self.pid = 4244
            self.returncode = 0
            self.stdin_bytes = stdin.read() if stdin is not None else None
            # No report written; a clean (empty) stream -- proves the report-stage
            # branch fires from contract_status alone, no runtime/verify noise.
            if stdout is subprocess.PIPE:
                self.stdout = io.BytesIO(self.CHILD_BYTES)
            if stderr is subprocess.PIPE:
                self.stderr = io.BytesIO(self.CHILD_BYTES)

        def wait(self, timeout=None):
            return self.returncode

    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        manifest_path, manifest = _write_manifest_for_worker(root, target)
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()),
            post=lambda e: events.append(e) or {"ok": True},
            popen=NoReportPopen, allowlist_root=root,
        )
        payload = events[0]["payload"]
        assert "verdict" not in payload
        assert payload["stage"] == "report"
        assert payload["diagnostic"] == "report missing"


def test_reviewer_unreadable_verdict_stays_invalid_with_no_fallback():
    # AC 7 / plan self-review gap (5e2ef5708068 F3): a reviewer report with real
    # prose but no parseable `verdict:` label anywhere must stay `invalid` with no
    # outcome at all -- reviewers never get a fallback, unlike a builder.
    class NoVerdictPopen(FakePopen):
        REPORT_TEXT = "# Review\n\nEverything looks fine. No issues found.\n"

    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        manifest_path, manifest = _write_manifest_for_worker(root, target)
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()),
            post=lambda e: events.append(e) or {"ok": True},
            popen=NoVerdictPopen, allowlist_root=root,
        )
        payload = events[0]["payload"]
        assert payload["contract_status"] == "invalid"
        assert "verdict" not in payload


def test_worker_builder_stage_two_only_report_posts_run_finished():
    # AC 15 / plan self-review gap (5e2ef5708068 F3): a stage-2-only report (no
    # `---`-bounded frontmatter block at all) must still reach run-finished with the
    # recovered outcome threaded through -- the only worker-level proof of the
    # crash path that made `report_outcome` get deleted; a parser-only test can't
    # see it.
    class LooseResultPopen(FakeBuilderPopen):
        REPORT_TEXT = "# Builder report\n\nresult: success\nsummary: all tasks done\n"

    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True},
            popen=LooseResultPopen, allowlist_root=root.parent,
        )
        payload = events[0]["payload"]
        assert payload["contract_status"] == "ok"
        assert payload["result"] == "success"


def test_worker_reviewer_stage_two_only_report_posts_run_finished():
    # Same gap, reviewer side -- representative of both reviewer worker paths (doc
    # review and diff review both call the same `finalize_reviewer_report`).
    class LooseVerdictPopen(FakePopen):
        REPORT_TEXT = "# Review\n\nverdict: approve\nsummary: no issues at all\n"

    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        manifest_path, manifest = _write_manifest_for_worker(root, target)
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()),
            post=lambda e: events.append(e) or {"ok": True},
            popen=LooseVerdictPopen, allowlist_root=root,
        )
        payload = events[0]["payload"]
        assert payload["contract_status"] == "ok"
        assert payload["verdict"] == "approve"


@pytest.mark.parametrize("commit,verify,result", [
    pytest.param(True, None, "success", id="green-verify-and-a-commit-beyond-base"),
    pytest.param(False, None, "failure", id="green-verify-with-no-commit-beyond-base"),
    pytest.param(True, "false", "failure", id="red-verify-regardless-of-commits"),
])
def test_builder_fallback_records_the_result_from_verify_and_commits(commit, verify, result):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        base_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _write_plan(worktree)
        if commit:
            (worktree / "changed.txt").write_text("done\n", encoding="utf-8")
            _git(worktree, "add", "changed.txt")
            _git(worktree, "commit", "-m", "feat: task work")
        manifest_path, manifest = _write_manifest_for_builder_worker(
            root, worktree, base_sha=base_sha, **({} if verify is None else {"verify": verify}))
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True},
            popen=NoLabelReportPopen, allowlist_root=root.parent,
        )
        payload = events[0]["payload"]
        assert payload["contract_status"] == "invalid"
        assert payload["result"] == result


def test_builder_fallback_applies_when_the_only_result_label_is_out_of_enum():
    # MOA-474 spec review: an out-of-enum result:/verdict: label is not a label at all
    # for the decision table -- it folds into Part 3 (no-label) exactly like a report
    # with no label anywhere, so the MOA-472 fallback DOES apply here (green verify +
    # commit beyond base -> success). Before this spec the row shipped with no `result`
    # key at all (the "neither branch" gap, §2.3 item 5).
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        base_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _write_plan(worktree)
        (worktree / "changed.txt").write_text("done\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "feat: task work")

        class GarbageResultPopen(FakeBuilderPopen):
            REPORT_TEXT = "result: banana\nsummary: built it\n"

        manifest_path, manifest = _write_manifest_for_builder_worker(
            root, worktree, base_sha=base_sha)
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True},
            popen=GarbageResultPopen, allowlist_root=root.parent,
        )
        payload = events[0]["payload"]
        assert payload["contract_status"] == "invalid"
        assert payload["result"] == "success"
        assert payload["stage"] == "report"
        assert payload["diagnostic"] == "report invalid"


def test_builder_fallback_never_applies_to_a_green_verify_on_an_unrelated_head():
    # 77f30f829248 F1: inequality alone is not descent. An unrelated head_sha (not a
    # descendant of base_sha at all) must resolve failure, never success.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        tree = _run_real(["git", "write-tree"], worktree).stdout.strip()
        unrelated_base_sha = _run_real(
            ["git", "commit-tree", tree, "-m", "unrelated root"], worktree
        ).stdout.strip()
        manifest_path, manifest = _write_manifest_for_builder_worker(
            root, worktree, base_sha=unrelated_base_sha)
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True},
            popen=NoLabelReportPopen, allowlist_root=root.parent,
        )
        payload = events[0]["payload"]
        assert payload["result"] == "failure"


def test_builder_fallback_survives_a_non_string_base_sha_in_a_hand_edited_manifest():
    # F4: manifest.json can carry ANY JSON value at "base_sha" once hand-edited (or
    # corrupted) -- a non-string reaching `re.match` after the child has already exited
    # crashed the whole worker with no terminal event posted, exactly the failure
    # MOA-470 shipped to prevent.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        (worktree / "changed.txt").write_text("done\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "feat: task work")
        manifest_path, manifest = _write_manifest_for_builder_worker(
            root, worktree, base_sha=12345)
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True},
            popen=NoLabelReportPopen, allowlist_root=root.parent,
        )
        assert events, "a terminal event must still post despite the malformed manifest"
        payload = events[0]["payload"]
        assert payload["result"] == "failure"


def test_worker_builder_blocked_with_failing_verify_stays_blocked():
    # decision 10, AC 14: the verify veto is a veto on a CLAIMED SUCCESS only.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)

        class BlockedReportPopen(FakeBuilderPopen):
            REPORT_TEXT = (
                "---\nrun_id: bbbbccccdddd\nproject: demo\nrole: builder\nphase: PHASE\n"
                "result: blocked\nsummary: stuck on task 3\n---\nbody\n"
            )

        manifest_path, manifest = _write_manifest_for_builder_worker(
            root, worktree, verify="false")
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True},
            popen=BlockedReportPopen, allowlist_root=root.parent,
        )
        payload = events[0]["payload"]
        assert payload["contract_status"] == "ok"
        assert payload["result"] == "blocked"


def test_builder_manifest_records_base_sha_at_dispatch(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        expected_sha = _run_real(["git", "rev-parse", "HEAD"], root).stdout.strip()
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        run_id = jaxflow.dispatch_build(
            _build_args(plan=str(plan)), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root,
        )
        manifest = json.loads(
            (root / ".local" / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["base_sha"] == expected_sha


def test_cmd_status_prints_finished_result_for_a_missing_row_carrying_result():
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        fake = FakeTmux()
        _insert(con, "r1", "demo", "builder", "run-started", {"session": "jax-demo-build-r1"})
        _insert(con, "r1", "demo", "builder", "run-finished",
                {"contract_status": "invalid", "result": "success", "exit_code": 0})
        assert jaxflow.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == (
            "finished(success, report invalid)"
        )
        con.close()


def test_cmd_status_renders_a_garbage_value_row_unchanged():
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        fake = FakeTmux()
        _insert(con, "r1", "demo", "builder", "run-started", {"session": "jax-demo-build-r1"})
        _insert(con, "r1", "demo", "builder", "run-finished",
                {"contract_status": "invalid", "exit_code": 0})
        # MOA-474 §12.2 branch 4 (§Backward compatibility): a legacy row with NEITHER
        # result nor reason changes from finished(failed) to finished(no result) --
        # this is the one pinned test §Backward compatibility names explicitly.
        assert jaxflow.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == "finished(no result)"
        con.close()


def test_cmd_status_renders_interrupted_row_with_stage_and_diagnostic():
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        fake = FakeTmux()
        _insert(con, "r1", "demo", "builder", "run-started", {"session": "jax-demo-build-r1"})
        _insert(con, "r1", "demo", "builder", "run-finished", {
            "contract_status": "interrupted", "exit_code": None, "report_path": None,
            "summary": "worker interrupted by SIGTERM", "stage": "worker",
            "diagnostic": "worker interrupted by SIGTERM", "head_sha": None, "result": "failure",
        })
        assert jaxflow.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == (
            "finished(interrupted) · worker — worker interrupted by SIGTERM"
        )
        con.close()


def test_cmd_status_renders_builder_result_with_stage_and_report_flag():
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        fake = FakeTmux()
        _insert(con, "r1", "demo", "builder", "run-started", {"session": "jax-demo-build-r1"})
        _insert(con, "r1", "demo", "builder", "run-finished", {
            "contract_status": "missing", "exit_code": 1, "result": "failure",
            "stage": "runtime", "diagnostic": "APIError 403 budget exceeded",
        })
        assert jaxflow.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == (
            "finished(failure, report missing) · runtime — APIError 403 budget exceeded"
        )
        con.close()


def test_cmd_status_renders_reviewer_verdict_with_stage_omitted_on_happy_path():
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        fake = FakeTmux()
        _insert(con, "r1", "demo", "reviewer", "run-started", {"session": "jax-demo-spec-r1"})
        _insert(con, "r1", "demo", "reviewer", "run-finished", {
            "contract_status": "ok", "exit_code": 0, "verdict": "approve-with-changes",
        })
        assert jaxflow.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == (
            "finished(approve-with-changes)"
        )
        con.close()


def test_cmd_result_renders_interrupted_row():
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _insert(con, "r1", "demo", "reviewer", "run-started", {"repo": "/tmp/demo"})
        _insert(con, "r1", "demo", "reviewer", "run-finished", {
            "contract_status": "interrupted", "exit_code": None, "report_path": None,
            "summary": "worker interrupted by SIGHUP", "stage": "worker",
            "diagnostic": "worker interrupted by SIGHUP",
        })
        out = jaxflow.cmd_result("r1", db_path=db)
        assert out == "interrupted — no verdict · worker — worker interrupted by SIGHUP"
        con.close()


def test_cmd_result_reason_block_gains_outcome_and_stage_lines(capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        root.mkdir(parents=True)
        report = root / ".local" / "reports" / "r5.md"
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        # `target` is required by cmd_result's builder branch (the plan's draft fixture
        # omitted it, which raised KeyError before the Refusal this test expects).
        _insert(con, "r5", "demo", "builder", "run-started", {"repo": str(root), "target": "feat/x"})
        _insert(con, "r5", "demo", "builder", "run-finished", {
            "contract_status": "missing", "exit_code": 1, "report_path": str(report),
            "summary": "no report written", "reason": "crash", "tail": "line one\nline two",
            "result": "failure", "stage": "runtime", "diagnostic": "exit 1 — no cause emitted",
        })
        try:
            jaxflow.cmd_result("r5", allowlist_root=allow_root, db_path=db)
        except ji.Refusal:
            pass
        out = capsys.readouterr().out
        assert "outcome: failure\n" in out
        assert "stage: runtime\n" in out
        assert "diagnostic: exit 1 — no cause emitted\n" in out
        assert out.index("outcome:") < out.index("reason:") < out.index("tail:")
        con.close()


# ---- worker: symlinked report/tests destinations refuse instead of following (fixes Part
# 3 diff-review F2), and a chmod failure at lock time is a non-success outcome (F5) ----

class SymlinkReportPopen(FakeBuilderPopen):
    """Plants report.md as a symlink to an outside file instead of writing it directly --
    the outside file already has valid, well-formed report content, so if the worker ever
    followed the symlink it would read/chmod/succeed on it."""

    def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
        self.argv = argv
        self.cwd = cwd
        self.pid = 5260
        self.returncode = 0
        self.stdin_bytes = stdin.read() if stdin is not None else None
        text = self.stdin_bytes.decode("utf-8")
        for line in text.splitlines():
            if line.startswith("report: "):
                report_path = Path(line[len("report: "):])
                outside = report_path.parent.parent / "outside-report.md"
                outside.write_text(FakeBuilderPopen.REPORT_TEXT, encoding="utf-8")
                report_path.symlink_to(outside)
                break

    def wait(self, timeout=None):
        return self.returncode


def test_worker_builder_symlinked_report_is_invalid_and_never_followed():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree), post=post,
            popen=SymlinkReportPopen, allowlist_root=root.parent,
        )
        payload = events[0]["payload"]
        assert payload["contract_status"] == "invalid"
        assert payload["result"] == "failure"
        report_path = worktree / ".local" / "reports" / "bbbbccccdddd.md"
        # the symlink itself is left untouched -- never followed, never chmoded.
        assert report_path.is_symlink()
        outside = worktree / ".local" / "outside-report.md"
        assert oct(outside.stat().st_mode)[-3:] != "444"


class SymlinkTestsPopen(FakeBuilderPopen):
    """Writes report.md normally but plants the tests-evidence path as a symlink to an
    outside file -- proving the post-build verify write skips it rather than following it."""

    def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
        self.argv = argv
        self.cwd = cwd
        self.pid = 5261
        self.returncode = 0
        self.stdin_bytes = stdin.read() if stdin is not None else None
        text = self.stdin_bytes.decode("utf-8")
        for line in text.splitlines():
            if line.startswith("report: "):
                Path(line[len("report: "):]).write_text(FakeBuilderPopen.REPORT_TEXT, encoding="utf-8")
            if line.startswith("tests: "):
                tests_path = Path(line[len("tests: "):])
                outside = tests_path.parent.parent / "outside-tests.txt"
                outside.write_text("pre-existing\n", encoding="utf-8")
                tests_path.symlink_to(outside)

    def wait(self, timeout=None):
        return self.returncode


def test_worker_builder_symlinked_tests_path_skips_write_and_reports_failure():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree), post=post,
            popen=SymlinkTestsPopen, allowlist_root=root.parent,
        )
        payload = events[0]["payload"]
        assert payload["contract_status"] == "ok"
        # the write was skipped -- treated exactly like a verify failure.
        assert payload["result"] == "failure"
        outside = worktree / ".local" / "outside-tests.txt"
        assert outside.read_text(encoding="utf-8") == "pre-existing\n"
        tests_path = worktree / ".local" / "reports" / "bbbbccccdddd.tests.txt"
        assert tests_path.is_symlink()


def test_worker_builder_report_chmod_failure_is_invalid_not_ok(monkeypatch):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        # fixes F5: the lock now goes through os.fchmod(fd, ...) on a no-follow file
        # descriptor, not os.chmod(path, ...) -- fchmod is the call to intercept.
        def failing_fchmod(fd, mode, *a, **kw):
            raise OSError("lock failed")

        monkeypatch.setattr(os, "fchmod", failing_fchmod)

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree), post=post,
            popen=FakeBuilderPopen, allowlist_root=root.parent,
        )
        payload = events[0]["payload"]
        assert payload["contract_status"] == "invalid"
        # MOA-474 §5.1 item 4: the report CONTENT parsed a claimed success, so this is
        # Part 2's labeled-success branch -- the run's execution outcome is `success`
        # even though the report is invalid (the lock, not the content, failed).
        assert payload["result"] == "success"
        # fixes Part 3 diff-review F5: the report content itself was valid -- only the
        # lock failed -- so the file must be left as-is, still readable, just unlocked.
        report = worktree / ".local" / "reports" / "bbbbccccdddd.md"
        assert oct(report.stat().st_mode)[-3:] != "444"


def test_worker_builder_report_symlink_planted_between_read_and_lock_never_chmoded(monkeypatch):
    # fixes F5 (RECURRENCE part3-F2): the read and the lock each re-resolve and re-open
    # no-follow immediately before acting. A symlink planted in the GAP between them --
    # after content validation already passed, before the chmod -- must be caught by the
    # lock's own re-check, not silently followed because the earlier read's check passed.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        report_path = worktree / ".local" / "reports" / "bbbbccccdddd.md"
        outside = worktree / ".local" / "outside-report.md"
        real_validate_report = jr.validate_report

        def swap_after_validating(text, *a, **kw):
            status, parsed, outcome = real_validate_report(text, *a, **kw)
            if status == "ok" and not outside.exists():
                outside.write_text(text, encoding="utf-8")
                report_path.unlink()
                report_path.symlink_to(outside)
            return status, parsed, outcome

        monkeypatch.setattr(jr, "validate_report", swap_after_validating)

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree), post=post,
            popen=FakeBuilderPopen, allowlist_root=root.parent,
        )
        payload = events[0]["payload"]
        assert payload["contract_status"] == "invalid"
        # Same as the chmod-failure sibling above: content parsed success, so Part 2
        # labeled-success decides the execution outcome.
        assert payload["result"] == "success"
        assert report_path.is_symlink()
        assert oct(outside.stat().st_mode)[-3:] != "444"


def test_worker_builder_stdin_prompt_and_process_group():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        captured = {}

        class CapturePopen(FakeBuilderPopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["argv"] = argv
                captured["cwd"] = cwd
                captured["start_new_session"] = start_new_session

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
        )
        assert captured["cwd"] == str(worktree)
        assert captured["start_new_session"] is True
        assert captured["argv"][0] == "opencode"


def test_worker_uses_current_saved_profile_after_dispatch_edit(tmp_path, monkeypatch):
    config, _, settings_path = _stub_launch_paths(tmp_path, monkeypatch)
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(tmp_path, monkeypatch)
    _rewrite_json(settings_path, lambda data: (
        data.__setitem__("revision", "22222222222222222222222222222222"),
        data["builders"]["default"].__setitem__("model", "changed-model"),
    ))
    config["provider"]["fixture"]["models"]["jaxflow-builder-default"]["id"] = "changed-model"
    captured, popen = _capture_builder_popen()
    inherited = {"PATH": "/bin", "HOME": "/h", "BWS_ACCESS_TOKEN": "machine-token"}
    code, events = _run_builder_worker_test(
        worktree, manifest_path, allow_root, popen, env=inherited,
    )
    assert code == 0
    argv = captured["argv"]
    assert argv[argv.index("--model") + 1] == "fixture/jaxflow-builder-default"
    assert argv[argv.index("--variant") + 1] == "high"
    assert "--agent" in argv and argv[argv.index("--agent") + 1] == "build"
    assert "wire-real-model" not in argv
    assert "changed-model" not in argv
    assert f"jaxflow-builder-bbbbccccdddd" not in argv
    assert captured["env"]["PATH"] == "/bin"
    assert "BWS_ACCESS_TOKEN" not in captured["env"]
    assert captured["env"]["JAX_PROVIDER_FIXTURE_API_KEY"] == "secret-value"
    assert "OPENCODE_CONFIG" not in captured["env"]
    assert "OPENCODE_CONFIG_CONTENT" not in captured["env"]
    assert "OPENCODE_CONFIG_DIR" not in captured["env"]
    assert config["permission"]["read"] == "deny"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["model"] == "fixture/wire-real-model"
    assert data["effort"] == "high"
    assert data["requested_profile"] == "default"
    assert data["root_build_run_id"] == "bbbbccccdddd"
    assert data["launch_selection"] == {
        "profile_name": "default",
        "settings_revision": "22222222222222222222222222222222",
        "model": "fixture/changed-model",
        "runtime_model": "fixture/jaxflow-builder-default",
        "effort": "high",
        "credential": {"kind": "env", "env": "JAX_PROVIDER_FIXTURE_API_KEY"},
    }
    blob = manifest_path.read_text(encoding="utf-8") + json.dumps(events) + str(argv)
    assert "secret-value" not in blob
    assert inherited["BWS_ACCESS_TOKEN"] == "machine-token"


def test_worker_managed_refuses_deleted_or_malformed_settings(tmp_path, monkeypatch):
    def popen(*a, **kw):
        raise AssertionError("popen must never be called")

    _stub_launch_paths(tmp_path, monkeypatch, settings=False)
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(tmp_path, monkeypatch)
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert events[0]["payload"]["contract_status"] == "cancelled"
    assert events[0]["payload"]["summary"] == "agent-settings-uninitialized"
    assert not worktree.exists()

    _stub_launch_paths(tmp_path, monkeypatch, settings=False)
    broken = tmp_path / "agent-settings.json"
    broken.write_text("{broken", encoding="utf-8")
    broken.chmod(0o600)
    monkeypatch.setattr(jset, "SETTINGS_PATH", broken)
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(tmp_path / "malformed", monkeypatch)
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert events[0]["payload"]["summary"] == "agent-settings-malformed"
    assert not worktree.exists()


def test_worker_managed_refuses_missing_alias_and_routing_conflict(tmp_path, monkeypatch):
    def popen(*a, **kw):
        raise AssertionError("popen must never be called")

    config, _, _ = _stub_launch_paths(tmp_path, monkeypatch)
    del config["provider"]["fixture"]["models"]["jaxflow-builder-default"]
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(tmp_path, monkeypatch)
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert events[0]["payload"]["summary"] == "agent-profile-conflict"
    assert not worktree.exists()

    config = _native_config()
    config["provider"]["fixture"]["models"]["jaxflow-builder-default"]["options"]["provider"]["allow_fallbacks"] = False
    monkeypatch.setattr(jset, "read_effective_opencode_config", lambda **kw: config)
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(tmp_path / "routing", monkeypatch)
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert events[0]["payload"]["summary"] == "agent-profile-conflict"
    assert not worktree.exists()


@pytest.mark.parametrize("profile,alias,real_id,other_id", [
    ("default", "jaxflow-builder-default", "default-wire-id", "fallback-wire-id"),
    ("fallback", "jaxflow-builder-fallback", "fallback-wire-id", "default-wire-id"),
])
def test_worker_argv_uses_selected_profile_alias(
    tmp_path, monkeypatch, profile, alias, real_id, other_id,
):
    config, _, settings_path = _stub_launch_paths(tmp_path, monkeypatch)
    other = "fallback" if profile == "default" else "default"
    _rewrite_json(settings_path, lambda data: (
        data["builders"][profile].__setitem__("model", real_id),
        data["builders"][other].__setitem__("model", other_id),
    ))
    config["provider"]["fixture"]["models"][alias]["id"] = real_id
    config["provider"]["fixture"]["models"][f"jaxflow-builder-{other}"]["id"] = other_id
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(
        tmp_path, monkeypatch, requested_profile=profile,
    )
    captured, popen = _capture_builder_popen()
    code, _ = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == 0
    argv = captured["argv"]
    assert argv[argv.index("--model") + 1] == f"fixture/{alias}"
    blob = " ".join(argv)
    assert real_id not in blob
    assert other_id not in blob
    assert f"jaxflow-builder-{other}" not in blob


def test_worker_non_reasoning_omits_variant(tmp_path, monkeypatch):
    config, _, settings_path = _stub_launch_paths(tmp_path, monkeypatch)
    _rewrite_json(settings_path, lambda data: (
        data["builders"]["default"].__setitem__("effort", None),
        data["builders"]["default"].__setitem__("routing", None),
    ))
    config["provider"]["fixture"]["npm"] = "@ai-sdk/openai"
    del config["provider"]["fixture"]["models"]["jaxflow-builder-default"]["options"]
    del config["provider"]["fixture"]["models"]["jaxflow-builder-default"]["variants"]
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(tmp_path, monkeypatch)
    captured, popen = _capture_builder_popen()
    code, _ = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == 0
    argv = captured["argv"]
    assert argv[argv.index("--model") + 1] == "fixture/jaxflow-builder-default"
    assert "--variant" not in argv
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["launch_selection"]["effort"] is None


def test_worker_env_credential_injects_selected_key(tmp_path, monkeypatch):
    _, _, settings_path = _stub_launch_paths(tmp_path, monkeypatch)
    _rewrite_json(settings_path, lambda data: data["builders"]["default"].__setitem__("credential", {
        "kind": "env",
        "env": "JAX_PROVIDER_FIXTURE_API_KEY",
    }))
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(tmp_path, monkeypatch)
    captured, popen = _capture_builder_popen()
    inherited = {"PATH": "/bin", "BWS_ACCESS_TOKEN": "machine-token", "OTHER": "keep"}
    code, events = _run_builder_worker_test(
        worktree, manifest_path, allow_root, popen, env=inherited,
    )
    assert code == 0
    assert captured["env"]["JAX_PROVIDER_FIXTURE_API_KEY"] == "secret-value"
    assert captured["env"]["OTHER"] == "keep"
    assert "BWS_ACCESS_TOKEN" not in captured["env"]
    blob = manifest_path.read_text(encoding="utf-8") + json.dumps(events) + str(captured["argv"])
    assert "secret-value" not in blob
    assert inherited["BWS_ACCESS_TOKEN"] == "machine-token"


def test_worker_history_write_failure_refuses_before_launch(tmp_path, monkeypatch):
    _stub_launch_paths(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(jaxflow, "_persist_launch_selection", boom)

    def popen(*a, **kw):
        raise AssertionError("popen must never be called")

    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(tmp_path, monkeypatch)
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert events[0]["payload"]["contract_status"] == "cancelled"
    assert not worktree.exists()
    assert not manifest_path.exists()


def test_worker_forged_reservation_owned_cannot_delete_resumed_worktree(tmp_path, monkeypatch):
    _stub_launch_paths(tmp_path, monkeypatch)
    prior = "aaaaaaaaaaaa"
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(
        tmp_path, monkeypatch, reservation_owned=True, root_build_run_id=prior,
    )
    marker = worktree / "kept.txt"
    marker.write_text("keep\n", encoding="utf-8")
    con = sqlite3.connect(jr.DB_PATH)
    con.execute("DELETE FROM workflow_events")
    con.commit()
    _insert(con, "bbbbccccdddd", "demo", "builder", "run-started", {
        "phase": "PHASE", "runtime": "opencode-builder", "kind": "build",
        "target": "feat/x", "caller": "claude", "caller_session": "01234567-89ab-4cde-8f01-23456789abcd",
        "model": "fixture/wire-real-model", "effort": "high",
        "session": "jax-demo-build-bbbbccccdddd", "repo": str(root),
        "verify": "true", "requested_profile": "default",
        "root_build_run_id": prior, "resumes_run_id": prior,
    })
    con.close()
    captured, popen = _capture_builder_popen()
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert "argv" not in captured
    assert events[0]["payload"]["contract_status"] == "cancelled"
    assert marker.is_file()
    assert worktree.is_dir()


def test_worker_resume_prelaunch_refusal_keeps_inherited_worktree(tmp_path, monkeypatch):
    _stub_launch_paths(tmp_path, monkeypatch)
    prior = "aaaaaaaaaaaa"

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(jaxflow, "_persist_launch_selection", boom)
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(
        tmp_path, monkeypatch, reservation_owned=True, root_build_run_id=prior,
        resumes_run_id=prior,
    )
    marker = worktree / "kept.txt"
    marker.write_text("keep\n", encoding="utf-8")

    def popen(*a, **kw):
        raise AssertionError("popen must never be called")

    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert events[0]["payload"]["contract_status"] == "cancelled"
    assert marker.is_file()
    assert worktree.is_dir()


def test_worker_resume_state_changed_refuses_before_popen(tmp_path, monkeypatch):
    import jaxflow_resume as jresume

    _stub_launch_paths(tmp_path, monkeypatch)
    prior = "aaaaaaaaaaaa"
    allow_root, root, worktree, manifest_path, manifest = _managed_worker_repo(
        tmp_path, monkeypatch, reservation_owned=False, root_build_run_id=prior,
        resumes_run_id=prior,
    )
    plan = Path(manifest["plan_path"])
    state = jresume.capture_work_state(worktree, plan, run=_run_real)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["resume_start"] = state
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    (worktree / "moved.py").write_text("x\n", encoding="utf-8")
    captured, popen = _capture_builder_popen()
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert "argv" not in captured
    assert events[0]["payload"]["contract_status"] == "cancelled"
    assert worktree.is_dir()
    assert (worktree / "moved.py").is_file()


def test_worker_newer_attempt_refuses_before_popen(tmp_path, monkeypatch):
    _stub_launch_paths(tmp_path, monkeypatch)
    prior = "aaaaaaaaaaaa"
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(
        tmp_path, monkeypatch, reservation_owned=False, root_build_run_id=prior,
        resumes_run_id=prior,
    )
    con = sqlite3.connect(jr.DB_PATH)
    _insert(con, "ffffffffffff", "demo", "builder", "run-started", {
        "phase": "PHASE", "runtime": "opencode-builder", "kind": "build",
        "target": "feat/x", "session": "jax-demo-build-ffffffffffff",
        "repo": str(root), "requested_profile": "default",
        "root_build_run_id": prior, "resumes_run_id": prior,
    })
    con.close()
    captured, popen = _capture_builder_popen()
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert "argv" not in captured
    assert events[0]["payload"]["contract_status"] == "cancelled"
    assert worktree.is_dir()


def test_worker_worktree_claim_refuses_when_held(tmp_path, monkeypatch):
    import jaxflow_resume as jresume

    _stub_launch_paths(tmp_path, monkeypatch)
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(tmp_path, monkeypatch)
    captured, popen = _capture_builder_popen()
    with jresume.worktree_claim(root, worktree):
        code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert "argv" not in captured
    assert events[0]["payload"]["contract_status"] == "cancelled"
    assert worktree.is_dir()


def _spy_cleanup(monkeypatch):
    cleaned = {"worktree": [], "rmtree": []}

    def fake_cleanup(worktree, branch, *, run, repo):
        cleaned["worktree"].append(Path(worktree).resolve())

    def fake_rmtree(path, ignore_errors=False):
        cleaned["rmtree"].append(Path(path).resolve())

    monkeypatch.setattr(jaxflow_common, "_cleanup_worktree", fake_cleanup)
    monkeypatch.setattr(shutil, "rmtree", fake_rmtree)
    return cleaned


def _assert_no_reservation_cleanup(cleaned, *paths):
    forbidden = {Path(p).resolve() for p in paths}
    assert cleaned["worktree"] == []
    assert forbidden.isdisjoint(set(cleaned["rmtree"]))


def test_unvalidated_and_uncertain_refusals_do_not_request_cleanup(tmp_path, monkeypatch):
    _stub_launch_paths(tmp_path, monkeypatch)
    allow_root, root, worktree, manifest_path, manifest = _managed_worker_repo(tmp_path, monkeypatch)
    marker = worktree / "kept.txt"
    marker.write_text("keep\n", encoding="utf-8")
    prior_cp = _checkpoint_path(root, manifest["run_id"])
    prior_cp.write_text("retained\n", encoding="utf-8")

    def popen(*a, **kw):
        raise AssertionError("popen must never be called")

    cleaned = _spy_cleanup(monkeypatch)
    con = sqlite3.connect(jr.DB_PATH)
    con.execute("DELETE FROM workflow_events")
    con.commit()
    con.close()
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert events[0]["payload"]["contract_status"] == "cancelled"
    _assert_no_reservation_cleanup(cleaned, worktree, manifest_path.parent)
    assert marker.is_file()
    assert prior_cp.is_file()
    assert worktree.is_dir()


def test_malformed_and_unavailable_started_do_not_request_cleanup(tmp_path, monkeypatch):
    _stub_launch_paths(tmp_path, monkeypatch)

    def popen(*a, **kw):
        raise AssertionError("popen must never be called")

    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(tmp_path, monkeypatch)
    marker = worktree / "kept.txt"
    marker.write_text("keep\n", encoding="utf-8")
    con = sqlite3.connect(jr.DB_PATH)
    con.execute("UPDATE workflow_events SET payload = '{' WHERE type = 'run-started'")
    con.commit()
    con.close()
    cleaned = _spy_cleanup(monkeypatch)
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert events[0]["payload"]["contract_status"] == "cancelled"
    _assert_no_reservation_cleanup(cleaned, worktree, manifest_path.parent)
    assert marker.is_file()

    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(
        tmp_path / "unavailable", monkeypatch,
    )
    marker = worktree / "kept.txt"
    marker.write_text("keep\n", encoding="utf-8")

    def boom(*a, **k):
        raise sqlite3.Error("unavailable")

    monkeypatch.setattr(jaxflow_common, "_open_ro", boom)
    cleaned = _spy_cleanup(monkeypatch)
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    _assert_no_reservation_cleanup(cleaned, worktree, manifest_path.parent)
    assert marker.is_file()


def test_claim_refusal_and_replayed_checkpoint_do_not_request_cleanup(tmp_path, monkeypatch):
    import jaxflow_resume as jresume

    _stub_launch_paths(tmp_path, monkeypatch)
    allow_root, root, worktree, manifest_path, manifest = _managed_worker_repo(tmp_path, monkeypatch)
    marker = worktree / "kept.txt"
    marker.write_text("keep\n", encoding="utf-8")
    captured, popen = _capture_builder_popen()
    cleaned = _spy_cleanup(monkeypatch)
    with jresume.worktree_claim(root, worktree):
        code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert "argv" not in captured
    _assert_no_reservation_cleanup(cleaned, worktree, manifest_path.parent)
    assert marker.is_file()

    allow_root, root, worktree, manifest_path, manifest = _managed_worker_repo(
        tmp_path / "replay", monkeypatch,
    )
    marker = worktree / "kept.txt"
    marker.write_text("keep\n", encoding="utf-8")
    jresume.write_checkpoint(_checkpoint_path(root, manifest["run_id"]), {
        "version": 1,
        "run_id": manifest["run_id"],
        "root_build_run_id": manifest["run_id"],
        "repo": str(root),
        "worktree": str(worktree),
        "branch": "feat/x",
        "base": "0" * 40,
        "outcome": "failure",
        "plan_revision": "b" * 64,
        "work_state": {"head": "c" * 40, "plan_sha256": "d" * 64, "fingerprint": "e" * 64},
    })

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(jaxflow, "_persist_launch_selection", boom)
    cleaned = _spy_cleanup(monkeypatch)
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    _assert_no_reservation_cleanup(cleaned, worktree, manifest_path.parent)
    assert marker.is_file()
    assert _checkpoint_path(root, manifest["run_id"]).is_file()


def test_refuse_builder_run_missing_started_does_not_own_resume(tmp_path, monkeypatch):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    marker = worktree / "kept.txt"
    marker.write_text("keep\n", encoding="utf-8")
    run_dir = tmp_path / ".local" / "runs" / "bbbbccccdddd"
    run_dir.mkdir(parents=True)
    (run_dir / "resume-checkpoint.json").write_text("retained\n", encoding="utf-8")
    manifest = {
        "run_id": "bbbbccccdddd", "project": "demo", "phase": "P", "kind": "build",
        "requested_profile": "default", "root_build_run_id": "aaaaaaaaaaaa",
        "resumes_run_id": "aaaaaaaaaaaa", "reservation_owned": True,
        "no_callback": True,
    }
    monkeypatch.setattr(jaxflow, "_builder_started_payload", lambda *a, **k: None)
    cleaned = _spy_cleanup(monkeypatch)

    def forbidden(*a, **k):
        raise AssertionError("git must not run")

    jaxflow._refuse_builder_run(
        "disk full", manifest=manifest, run=forbidden, post=lambda e: {"ok": True},
        control_repo=tmp_path, worktree=worktree, branch="feat/x",
    )
    _assert_no_reservation_cleanup(cleaned, worktree, run_dir)
    assert marker.is_file()
    assert (run_dir / "resume-checkpoint.json").is_file()


@pytest.mark.parametrize("profile", [None, "nope", "default"])
@pytest.mark.parametrize("evidence", ["missing", "malformed", "unavailable"])
def test_managed_runtime_uncertain_evidence_does_not_request_cleanup(
    tmp_path, monkeypatch, profile, evidence,
):
    _stub_launch_paths(tmp_path, monkeypatch)
    extra = {"runtime": "opencode-builder"}
    if profile is not None:
        extra["requested_profile"] = profile
    allow_root, root, worktree, manifest_path, manifest = _managed_worker_repo(
        tmp_path, monkeypatch, **extra,
    )
    if profile is None:
        _rewrite_json(manifest_path, lambda data: data.pop("requested_profile", None))
    marker = worktree / "kept.txt"
    marker.write_text("keep\n", encoding="utf-8")
    prior_cp = _checkpoint_path(root, manifest["run_id"])
    prior_cp.write_text("retained\n", encoding="utf-8")

    def popen(*a, **kw):
        raise AssertionError("popen must never be called")

    con = sqlite3.connect(jr.DB_PATH)
    if evidence == "missing":
        con.execute("DELETE FROM workflow_events")
        con.commit()
    elif evidence == "malformed":
        con.execute("UPDATE workflow_events SET payload = '{' WHERE type = 'run-started'")
        con.commit()
    con.close()
    if evidence == "unavailable":
        def boom(*a, **k):
            raise sqlite3.Error("unavailable")
        monkeypatch.setattr(jaxflow_common, "_open_ro", boom)
    cleaned = _spy_cleanup(monkeypatch)
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert events[0]["payload"]["contract_status"] == "cancelled"
    _assert_no_reservation_cleanup(cleaned, worktree, manifest_path.parent)
    assert marker.is_file()
    assert prior_cp.is_file()
    assert worktree.is_dir()


def test_fresh_owner_prelaunch_refusal_requests_cleanup(tmp_path, monkeypatch):
    _stub_launch_paths(tmp_path, monkeypatch)
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(jaxflow, "_persist_launch_selection", boom)

    def popen(*a, **kw):
        raise AssertionError("popen must never be called")

    cleaned = _spy_cleanup(monkeypatch)
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert events[0]["payload"]["contract_status"] == "cancelled"
    assert worktree.resolve() in cleaned["worktree"]
    assert manifest_path.parent.resolve() in cleaned["rmtree"]


def test_status_and_result_label_preview_and_launch_selection():
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        control = allow_root / "demo"
        worktree = allow_root / "demo-feat-x"
        run_id = "bbbbccccdddd"
        worktree.mkdir(parents=True)
        report = worktree / ".local" / "reports" / f"{run_id}.md"
        report.parent.mkdir(parents=True)
        report.write_text("REPORT BODY\n", encoding="utf-8")
        manifest_path = control / ".local" / "runs" / run_id / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({
            "model": "fixture/wire-real-model",
            "effort": "high",
            "launch_selection": {
                "profile_name": "default",
                "settings_revision": "11111111111111111111111111111111",
                "model": "fixture/changed-model",
                "runtime_model": "fixture/jaxflow-builder-default",
                "effort": "high",
                "credential": {"kind": "native"},
            },
        }), encoding="utf-8")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        started = {
            "session": f"jax-demo-build-{run_id}",
            "repo": str(control),
            "runtime": "opencode-builder",
            "model": "fixture/wire-real-model",
            "effort": "high",
            "target": "feat/x",
        }
        _insert(con, run_id, "demo", "builder", "run-started", started)
        fake = FakeTmux()
        fake.existing.add(started["session"])
        text = jaxflow.cmd_status(run_id, run=_run_with_tmux(fake), db_path=db)
        assert "running" in text
        assert "preview model: fixture/wire-real-model" in text
        assert "preview effort: high" in text
        assert "resolved selection model: fixture/changed-model" in text
        assert "resolved selection effort: high" in text
        assert "launched model:" not in text
        _insert(con, run_id, "demo", "builder", "run-finished", {
            "contract_status": "ok",
            "report_path": str(report),
            "summary": "ok",
        })
        out = jaxflow.cmd_result(run_id, allowlist_root=allow_root, db_path=db)
        assert "preview model: fixture/wire-real-model" in out
        assert "resolved selection model: fixture/changed-model" in out
        assert "launched model:" not in out
        assert "REPORT BODY" in out
        con.close()


def test_status_omits_invalid_or_missing_launch_selection():
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        control = allow_root / "demo"
        run_id = "bbbbccccdddd"
        manifest_path = control / ".local" / "runs" / run_id / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({
            "model": "fixture/wire-real-model",
            "effort": "high",
        }), encoding="utf-8")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        started = {
            "session": f"jax-demo-build-{run_id}",
            "repo": str(control),
            "runtime": "opencode-builder",
            "model": "fixture/wire-real-model",
            "effort": "high",
            "target": "feat/x",
        }
        _insert(con, run_id, "demo", "builder", "run-started", started)
        fake = FakeTmux()
        fake.existing.add(started["session"])
        text = jaxflow.cmd_status(run_id, run=_run_with_tmux(fake), db_path=db)
        assert "preview model: fixture/wire-real-model" in text
        assert "resolved selection" not in text
        assert "launched " not in text
        _rewrite_json(manifest_path, lambda data: data.update({
            "launch_selection": {
                "profile_name": "default",
                "settings_revision": "11111111111111111111111111111111",
                "model": "fixture/changed-model",
                "runtime_model": "fixture/jaxflow-builder-default",
                "effort": "high",
                "credential": {"kind": "native"},
                "extra": True,
            },
        }))
        text = jaxflow.cmd_status(run_id, run=_run_with_tmux(fake), db_path=db)
        assert "resolved selection" not in text
        assert "launched " not in text
        con.close()


def test_status_history_tolerates_non_object_manifests():
    with TemporaryDirectory() as raw:
        control = Path(raw) / "repos" / "demo"
        run_id = "bbbbccccdddd"
        manifest_path = control / ".local" / "runs" / run_id / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        started = {
            "session": f"jax-demo-build-{run_id}",
            "repo": str(control),
            "runtime": "opencode-builder",
            "model": "fixture/wire-real-model",
            "effort": "high",
            "target": "feat/x",
        }
        _insert(con, run_id, "demo", "builder", "run-started", started)
        fake = FakeTmux()
        fake.existing.add(started["session"])
        for body in (
            "null", "[]", '"not-an-object"', "1", "{",
            "[" * 60000 + "]" * 60000,
        ):
            manifest_path.write_text(body, encoding="utf-8")
            text = jaxflow.cmd_status(run_id, run=_run_with_tmux(fake), db_path=db)
            assert "preview model: fixture/wire-real-model" in text
            assert "resolved selection" not in text
            assert "launched " not in text
        manifest_path.write_text(json.dumps({
            "nested": {"deep": {"x": 1}},
            "launch_selection": "not-an-object",
        }), encoding="utf-8")
        text = jaxflow.cmd_status(run_id, run=_run_with_tmux(fake), db_path=db)
        assert "preview model: fixture/wire-real-model" in text
        assert "resolved selection" not in text
        manifest_path.write_text(json.dumps({
            "launch_selection": {
                "profile_name": "default",
                "settings_revision": "11111111111111111111111111111111",
                "model": "fixture/changed-model",
                "runtime_model": "fixture/jaxflow-builder-default",
                "effort": None,
                "credential": {"kind": "native"},
            },
        }), encoding="utf-8")
        text = jaxflow.cmd_status(run_id, run=_run_with_tmux(fake), db_path=db)
        assert "resolved selection model: fixture/changed-model" in text
        assert "resolved selection effort: n/a" in text
        con.close()


_LAUNCH_SELECTION = {
    "profile_name": "default",
    "settings_revision": "11111111111111111111111111111111",
    "model": "fixture/wire-real-model",
    "runtime_model": "fixture/jaxflow-builder-default",
    "effort": "high",
    "credential": {"kind": "native"},
}


def test_persist_launch_selection_binds_canonical_manifest(tmp_path):
    repo = tmp_path / "demo"
    run_id = "bbbbccccdddd"
    canonical = repo / ".local" / "runs" / run_id / "manifest.json"
    canonical.parent.mkdir(parents=True)
    manifest = {"run_id": run_id}
    canonical.write_text(json.dumps(manifest), encoding="utf-8")
    canonical.chmod(0o600)
    copied = tmp_path / "other" / "manifest.json"
    copied.parent.mkdir()
    copied.write_bytes(canonical.read_bytes())
    copied.chmod(0o600)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow._persist_launch_selection(
            copied, dict(manifest), _LAUNCH_SELECTION, control_repo=repo,
        )
    assert exc.value.code in ("path-outside-allowlist", "agent-settings-permissions")
    assert "launch_selection" not in json.loads(copied.read_text(encoding="utf-8"))
    assert "launch_selection" not in json.loads(canonical.read_text(encoding="utf-8"))
    linked_parent = tmp_path / "via-runs"
    linked_parent.symlink_to(canonical.parent.parent)
    linked = linked_parent / run_id / "manifest.json"
    with pytest.raises(ji.Refusal):
        jaxflow._persist_launch_selection(
            linked, dict(manifest), _LAUNCH_SELECTION, control_repo=repo,
        )
    assert "launch_selection" not in json.loads(canonical.read_text(encoding="utf-8"))
    jaxflow._persist_launch_selection(
        canonical, dict(manifest), _LAUNCH_SELECTION, control_repo=repo,
    )
    stored = json.loads(canonical.read_text(encoding="utf-8"))["launch_selection"]
    assert stored == _LAUNCH_SELECTION


def test_persist_launch_selection_refuses_parent_switch(tmp_path, monkeypatch):
    repo = tmp_path / "demo"
    run_id = "bbbbccccdddd"
    canonical = repo / ".local" / "runs" / run_id / "manifest.json"
    canonical.parent.mkdir(parents=True)
    prior = json.dumps({"run_id": run_id}).encode("utf-8")
    canonical.write_bytes(prior)
    canonical.chmod(0o600)
    outside = tmp_path / "outside" / run_id
    outside.parent.mkdir()
    real_open = os.open
    swapped = {"n": False}

    def gated(path, flags, mode=0o777, *args, dir_fd=None, **kwargs):
        name = os.fspath(path)
        if str(name).endswith(".tmp") and not swapped["n"]:
            swapped["n"] = True
            if canonical.parent.exists() and not outside.exists():
                canonical.parent.rename(outside)
                canonical.parent.mkdir(parents=True)
                decoy = canonical.parent / "manifest.json"
                decoy.write_bytes(prior)
                decoy.chmod(0o600)
        if dir_fd is None:
            return real_open(path, flags, mode, *args, **kwargs)
        return real_open(path, flags, mode, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(os, "open", gated)
    with pytest.raises(ji.Refusal):
        jaxflow._persist_launch_selection(
            canonical, {"run_id": run_id}, _LAUNCH_SELECTION, control_repo=repo,
        )
    assert (outside / "manifest.json").read_bytes() == prior
    decoy = canonical.parent / "manifest.json"
    if decoy.exists():
        assert b"launch_selection" not in decoy.read_bytes()
    assert _tmps_here(canonical.parent) == []
    assert _tmps_here(outside) == []


def _tmps_here(folder):
    return [p for p in folder.iterdir() if p.name.endswith(".tmp")]


def test_worker_copied_manifest_is_not_a_write_destination(tmp_path, monkeypatch):
    _stub_launch_paths(tmp_path, monkeypatch)
    allow_root, root, worktree, manifest_path, manifest = _managed_worker_repo(tmp_path, monkeypatch)
    original = manifest_path.read_text(encoding="utf-8")
    copied = tmp_path / "copied" / "manifest.json"
    copied.parent.mkdir()
    copied.write_bytes(manifest_path.read_bytes())
    copied.chmod(0o600)
    cleaned = []
    monkeypatch.setattr(jaxflow_common, "_cleanup_worktree", lambda *a, **k: cleaned.append("worktree"))
    real_rmtree = shutil.rmtree

    def fake_rmtree(path, ignore_errors=False):
        resolved = Path(path).resolve()
        if resolved in {worktree.resolve(), manifest_path.parent.resolve()}:
            raise AssertionError(f"cleanup of real run {path}")
        return real_rmtree(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(shutil, "rmtree", fake_rmtree)
    captured, popen = _capture_builder_popen()
    code, events = _run_builder_worker_test(worktree, copied, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert "argv" not in captured
    assert cleaned == []
    assert worktree.is_dir()
    assert manifest_path.read_text(encoding="utf-8") == original
    if copied.exists():
        assert "launch_selection" not in json.loads(copied.read_text(encoding="utf-8"))


def test_worker_symlinked_manifest_is_not_a_write_destination(tmp_path, monkeypatch):
    _stub_launch_paths(tmp_path, monkeypatch)
    allow_root, root, worktree, manifest_path, manifest = _managed_worker_repo(tmp_path, monkeypatch)
    original = manifest_path.read_text(encoding="utf-8")
    via = tmp_path / "via-runs"
    via.symlink_to(root / ".local" / "runs")
    linked = via / manifest["run_id"] / "manifest.json"
    cleaned = []
    monkeypatch.setattr(jaxflow_common, "_cleanup_worktree", lambda *a, **k: cleaned.append("worktree"))
    real_rmtree = shutil.rmtree

    def fake_rmtree(path, ignore_errors=False):
        resolved = Path(path).resolve()
        if resolved in {worktree.resolve(), manifest_path.parent.resolve()}:
            raise AssertionError(f"cleanup of real run {path}")
        return real_rmtree(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(shutil, "rmtree", fake_rmtree)
    captured, popen = _capture_builder_popen()
    code, events = _run_builder_worker_test(worktree, linked, allow_root, popen)
    assert code == jaxflow_common.REFUSED
    assert "argv" not in captured
    assert cleaned == []
    assert worktree.is_dir()
    assert manifest_path.read_text(encoding="utf-8") == original


def test_pure_config_run_bounds_stdout_and_reaps(tmp_path):
    exe = sys.executable
    under = jaxflow._pure_config_run(
        [exe, "-c", "import sys; sys.stdout.buffer.write(b'{}')"], cap=64,
    )
    assert under.returncode == 0
    assert under.stdout == b"{}"
    exact = jaxflow._pure_config_run(
        [exe, "-c", "import sys; sys.stdout.buffer.write(b'x' * 64)"], cap=64,
    )
    assert exact.returncode == 0
    assert exact.stdout == b"x" * 64

    pid_file = tmp_path / "over.pid"
    over_script = (
        "import os, sys, time\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "sys.stdout.buffer.write(b'x' * 80)\n"
        "sys.stdout.flush()\n"
        "time.sleep(30)\n"
    )
    with pytest.raises(ji.Refusal) as exc:
        jaxflow._pure_config_run([exe, "-c", over_script], cap=64)
    assert exc.value.code == "agent-profile-conflict"
    pid = int(pid_file.read_text(encoding="utf-8"))
    with pytest.raises(OSError):
        os.kill(pid, 0)

    err = jaxflow._pure_config_run(
        [exe, "-c", "import sys; sys.stderr.write('secret-stderr\\n')"], cap=64,
    )
    assert err.stdout == b""
    assert err.stderr in (b"", None, "")

    stall_pid = tmp_path / "stall.pid"
    stall_script = (
        "import os, sys, time\n"
        f"open({str(stall_pid)!r}, 'w').write(str(os.getpid()))\n"
        "sys.stdout.buffer.write(b'{')\n"
        "sys.stdout.flush()\n"
        "time.sleep(30)\n"
    )
    with pytest.raises(ji.Refusal) as exc:
        jaxflow._pure_config_run([exe, "-c", stall_script], deadline=0.2, cap=64)
    assert exc.value.code == "agent-profile-conflict"
    pid = int(stall_pid.read_text(encoding="utf-8"))
    with pytest.raises(OSError):
        os.kill(pid, 0)


def test_read_effective_opencode_config_refuses_empty_invalid_and_nonzero():
    repo = Path("/repo")
    for result in (
        SimpleNamespace(returncode=1, stdout="{}", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="{", stderr=""),
        SimpleNamespace(returncode=0, stdout="[]", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr="only-stderr"),
    ):
        with pytest.raises(ji.Refusal) as exc:
            jset.read_effective_opencode_config(
                run=lambda *a, **k: result, env={}, repo=repo,
            )
        assert exc.value.code == "agent-profile-conflict"


def test_persisted_selection_survives_failed_popen_as_resolved(tmp_path, monkeypatch):
    _stub_launch_paths(tmp_path, monkeypatch)
    allow_root, root, worktree, manifest_path, manifest = _managed_worker_repo(
        tmp_path, monkeypatch,
    )

    def popen(*a, **kw):
        raise OSError("exec failed")

    with pytest.raises(OSError):
        _run_builder_worker_test(worktree, manifest_path, allow_root, popen)
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))["launch_selection"]
    assert stored["runtime_model"] == "fixture/jaxflow-builder-default"
    assert stored["model"] == "fixture/wire-real-model"
    fake = FakeTmux()
    fake.existing.add(manifest["session"])
    text = jaxflow.cmd_status(
        manifest["run_id"], run=_run_with_tmux(fake), db_path=jr.DB_PATH,
    )
    assert "resolved selection model: fixture/wire-real-model" in text
    assert "launched model:" not in text


def _checkpoint_path(root, run_id="bbbbccccdddd"):
    return root / ".local" / "runs" / run_id / "resume-checkpoint.json"


def test_worker_checkpoint_captures_post_verify_state_and_outcomes(tmp_path, monkeypatch):
    import jaxflow_resume as jresume

    _stub_launch_paths(tmp_path, monkeypatch)
    verify = (
        "git -c user.email=t@t.test -c user.name=t commit --allow-empty -m v"
        " && printf 'x\\n' > from-verify.py"
    )
    allow_root, root, worktree, manifest_path, manifest = _managed_worker_repo(
        tmp_path, monkeypatch, verify=verify, base_sha="0" * 40,
    )
    head_before = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, FakeBuilderPopen)
    assert code == 0
    assert events[0]["payload"]["result"] == "success"
    data = jresume.read_checkpoint(_checkpoint_path(root))
    assert data["outcome"] == "success"
    assert data["run_id"] == manifest["run_id"]
    assert data["root_build_run_id"] == "bbbbccccdddd"
    assert data["base"] == "0" * 40
    assert data["work_state"]["head"] != head_before
    assert (worktree / "from-verify.py").read_text(encoding="utf-8") == "x\n"
    jresume.require_same_work_state(
        data["work_state"],
        jresume.capture_work_state(worktree, Path(manifest["plan_path"]), run=_run_real),
    )
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["requested_profile"] == "default"
    assert "worker_outcome" not in on_disk

    class BlockedPopen(FakeBuilderPopen):
        REPORT_TEXT = FakeBuilderPopen.REPORT_TEXT.replace("result: success", "result: blocked")

    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(
        tmp_path / "blocked", monkeypatch, base_sha="0" * 40,
    )
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, BlockedPopen)
    assert code == 0
    assert events[0]["payload"]["result"] == "blocked"
    assert jresume.read_checkpoint(_checkpoint_path(root))["outcome"] == "blocked"

    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(
        tmp_path / "fail", monkeypatch, base_sha="0" * 40,
    )
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, FailingReportPopen)
    assert code == 0
    assert events[0]["payload"]["result"] == "failure"
    assert jresume.read_checkpoint(_checkpoint_path(root))["outcome"] == "failure"


def test_worker_checkpoint_write_failure_or_secret_does_not_change_result(tmp_path, monkeypatch):
    import jaxflow_resume as jresume

    _stub_launch_paths(tmp_path, monkeypatch)
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(tmp_path, monkeypatch)

    original = jresume.write_checkpoint

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(jresume, "write_checkpoint", boom)
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, FakeBuilderPopen)
    assert code == 0
    assert events[0]["payload"]["result"] == "success"
    assert not _checkpoint_path(root).exists()

    monkeypatch.setattr(jresume, "write_checkpoint", original)
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(tmp_path / "secret", monkeypatch)
    (worktree / ".env").write_text("nope\n", encoding="utf-8")
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, FakeBuilderPopen)
    assert code == 0
    assert events[0]["payload"]["result"] == "success"
    assert not _checkpoint_path(root).exists()


def test_worker_legacy_and_missing_report_checkpoint_rules(tmp_path, monkeypatch):
    import jaxflow_resume as jresume

    class SilentPopen(FakeBuilderPopen):
        def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None,
                     start_new_session=None, env=None):
            self.argv = argv
            self.cwd = cwd
            self.pid = 5252
            self.returncode = 0
            self.stdin_bytes = stdin.read() if stdin is not None else None
            if stdout is subprocess.PIPE:
                self.stdout = io.BytesIO(self.CHILD_BYTES)
            if stderr is subprocess.PIPE:
                self.stderr = io.BytesIO(self.CHILD_BYTES)

    _stub_launch_paths(tmp_path, monkeypatch)
    allow_root, root, worktree, manifest_path, _ = _managed_worker_repo(
        tmp_path, monkeypatch, base_sha="0" * 40,
    )
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, SilentPopen)
    assert code == 0
    assert events[0]["payload"]["result"] == "failure"
    assert jresume.read_checkpoint(_checkpoint_path(root))["outcome"] == "failure"

    allow_root = tmp_path / "legacy-repos"
    root = allow_root / "demo"
    _init_repo(root)
    worktree = _init_worktree(root)
    _write_plan(worktree)
    manifest_path, _ = _write_manifest_for_builder_worker(root, worktree)
    code, events = _run_builder_worker_test(worktree, manifest_path, allow_root, FakeBuilderPopen)
    assert code == 0
    assert events[0]["payload"]["result"] == "success"
    assert not _checkpoint_path(root).exists()


def test_status_and_result_checkpoint_hint_from_fixed_path():
    import jaxflow_resume as jresume

    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        control = allow_root / "demo"
        worktree = allow_root / "demo-feat-x"
        run_id = "bbbbccccdddd"
        worktree.mkdir(parents=True)
        report = worktree / ".local" / "reports" / f"{run_id}.md"
        report.parent.mkdir(parents=True)
        report.write_text("REPORT BODY\n", encoding="utf-8")
        dest = _checkpoint_path(control, run_id)
        dest.parent.mkdir(parents=True)
        jresume.write_checkpoint(dest, {
            "version": 1,
            "run_id": run_id,
            "root_build_run_id": run_id,
            "repo": str(control),
            "worktree": str(worktree),
            "branch": "feat/x",
            "base": "a" * 40,
            "outcome": "failure",
            "plan_revision": "b" * 64,
            "work_state": {"head": "c" * 40, "plan_sha256": "d" * 64, "fingerprint": "e" * 64},
        })
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        started = {
            "session": f"jax-demo-build-{run_id}",
            "repo": str(control),
            "runtime": "opencode-builder",
            "model": "m",
            "effort": "high",
            "target": "feat/x",
        }
        _insert(con, run_id, "demo", "builder", "run-started", started)
        _insert(con, run_id, "demo", "builder", "run-finished", {
            "contract_status": "ok",
            "result": "failure",
            "report_path": str(report),
            "summary": "stopped",
        })
        text = jaxflow.cmd_status(run_id, run=_run_with_tmux(FakeTmux()), db_path=db)
        assert "checkpoint: present" in text
        out = jaxflow.cmd_result(run_id, allowlist_root=allow_root, db_path=db)
        assert "checkpoint: present" in out
        dest.unlink()
        text = jaxflow.cmd_status(run_id, run=_run_with_tmux(FakeTmux()), db_path=db)
        assert "checkpoint: absent" in text
        con.close()


# ---- worker: builder handoff is contract-valid (fixes cold review F2) ----

def test_builder_handoff_carries_both_commands_and_never_hardcodes_none():
    """MOA-454, the defect this issue is named for: `commands.build` was the literal
    `none` no matter what, so a tech lead who chained a build into --verify got a handoff
    that told the builder it had no build command."""
    handoff = jaxflow._build_builder_handoff(
        plan_dest=Path("/p/plan.md"),
        spec_dest=Path("/p/spec.md"), agents_path=Path("/p/AGENTS.md"),
        branch="feat/x", head_sha="a" * 40, whitelist=["src"],
        verify_cmd="pnpm test", build_cmd="pnpm build",
    )
    assert "  test: pnpm test\n" in handoff
    assert "  build: pnpm build\n" in handoff
    assert "build: none" not in handoff


def test_builder_handoff_says_none_only_when_there_is_no_build_command():
    handoff = jaxflow._build_builder_handoff(
        plan_dest=Path("/p/plan.md"),
        spec_dest=Path("/p/spec.md"), agents_path=Path("/p/AGENTS.md"),
        branch="feat/x", head_sha="a" * 40, whitelist=["src"],
        verify_cmd="pnpm test", build_cmd=None,
    )
    assert "  build: none\n" in handoff


def test_worker_builder_handoff_is_contract_valid_with_plan_fallback_spec():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree, text=(
            "# Ship the thing\n\n**Goal:** make the widget spin.\n\n"
            "### Task 1: add the spinner\n\n- [ ] Step 1: write it\n\n"
            "### Task 2: wire it up\n\n- [ ] Step 1: wire it\n"
        ))
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree, verify="true")
        captured = {}

        class CapturePopen(FakeBuilderPopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["stdin_bytes"] = self.stdin_bytes

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
        )
        prompt = captured["stdin_bytes"].decode("utf-8")
        plan_dest = str(worktree / ".local" / "docs" / "plans" / "plan.md")
        # every field orchestrator-handoff.md's BUILDER HANDOFF PACKAGE requires is present.
        assert "goal:" not in prompt
        assert "tasks:" not in prompt
        assert "acceptance:" not in prompt
        assert "paths:" in prompt
        # no **Spec:** line in the plan -> paths.spec falls back to the plan copy itself
        # (orchestrator-handoff.md requires paths.spec non-empty -- see this plan's header).
        assert f"spec: {plan_dest}" in prompt
        assert f"plan: {plan_dest}" in prompt
        assert "assumptions: The plan is the sole authority" in prompt
        assert "commands:" in prompt
        assert "test: true" in prompt
        assert "verification: rafa-verifies" in prompt
        assert "out-of-scope: Anything the plan does not name" in prompt
        # §2.7's fixed inputs. No AGENTS.md exists in this fixture's control repo -- the
        # path is still named explicitly (kept as-is, fixes Part 3 diff-review F6), it just
        # does not exist on disk.
        assert f"Project AGENTS.md: {worktree / 'AGENTS.md'}" in prompt
        assert not (worktree / "AGENTS.md").exists()
        assert "Branch: feat/x" in prompt
        assert "target: feat/x" in prompt
        assert "Never print secrets" in prompt
        assert "whitelist: a.py" in prompt


def test_worker_builder_handoff_names_target_and_existing_copied_agents_md():
    # fixes Part 3 diff-review F6: with an ACTUAL AGENTS.md copied into the worktree
    # (simulating dispatch_build's own copy step), the handoff must name that exact,
    # existing path -- not just a path that happens to be well-formed.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        (worktree / "AGENTS.md").write_text("# Project rules\n", encoding="utf-8")
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        captured = {}

        class CapturePopen(FakeBuilderPopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["stdin_bytes"] = self.stdin_bytes

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
        )
        prompt = captured["stdin_bytes"].decode("utf-8")
        agents_path = worktree / "AGENTS.md"
        assert agents_path.is_file()
        assert f"Project AGENTS.md: {agents_path}\n" in prompt
        assert "Branch: feat/x\n" in prompt
        assert "target: feat/x\n" in prompt


def test_worker_builder_handoff_names_referenced_spec_original_without_copy():
    # MOA-467: the declared **Spec:** file is resolved to its ORIGINAL validated
    # location and named in the handoff -- no copy is made into the worktree.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        spec_src = root / ".local" / "docs" / "specs" / "2026-01-01-demo-spec.md"
        spec_src.parent.mkdir(parents=True)
        spec_src.write_text("# The Spec\n", encoding="utf-8")
        _write_plan(worktree, text=(
            f"# Ship the thing\n\n**Goal:** make it spin.\n\n**Spec:** `{spec_src}`\n\n"
            "### Task 1: do it\n"
        ))
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        captured = {}

        class CapturePopen(FakeBuilderPopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["stdin_bytes"] = self.stdin_bytes

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
        )
        assert not (worktree / ".local" / "docs" / "specs").exists()
        prompt = captured["stdin_bytes"].decode("utf-8")
        assert f"  spec: {spec_src}" in prompt


def test_worker_builder_handoff_names_spec_fragment_when_the_plan_declares_a_heading():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        spec_src = root / ".local" / "docs" / "specs" / "2026-01-01-demo-spec.md"
        spec_src.parent.mkdir(parents=True)
        spec_src.write_text(
            "# The Spec\n\n## Section One\ntext\n\n## Section Two\nmore\n", encoding="utf-8",
        )
        _write_plan(worktree, text=(
            f"# Ship the thing\n\n**Goal:** make it spin.\n\n"
            f"**Spec:** `{spec_src}#Section Two`\n\n### Task 1: do it\n"
        ))
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        captured = {}

        class CapturePopen(FakeBuilderPopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["stdin_bytes"] = self.stdin_bytes

        assert jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
        ) == 0
        prompt = captured["stdin_bytes"].decode("utf-8")
        assert f"  spec: {spec_src}#Section Two" in prompt


# ---- MOA-467: original-path manifests and the validated read identity ----

def test_worker_builder_handoff_names_original_plan_and_declared_spec_without_copies():
    # A NEW-style manifest: plan_path points at the ORIGINAL plan in the control repo,
    # whose **Spec:** names the ORIGINAL spec. The handoff names both originals and
    # emits the validated read scope; nothing is copied into the worktree.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        spec_src = root / ".local" / "docs" / "specs" / "2026-01-01-demo-spec.md"
        spec_src.parent.mkdir(parents=True)
        spec_src.write_text("# The Spec\n", encoding="utf-8")
        plan_src = root / ".local" / "docs" / "plans" / "plan.md"
        plan_src.parent.mkdir(parents=True)
        plan_src.write_text(
            f"# Ship the thing\n\n**Goal:** make it spin.\n\n**Spec:** `{spec_src}`\n\n"
            "### Task 1: do it\n", encoding="utf-8",
        )
        manifest_path, manifest = _write_manifest_for_builder_worker(
            root, worktree, plan_path=str(plan_src),
        )
        captured = {}

        class CapturePopen(FakeBuilderPopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["stdin_bytes"] = self.stdin_bytes

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
        )
        prompt = captured["stdin_bytes"].decode("utf-8")
        assert f"  plan: {plan_src}" in prompt
        assert f"  spec: {spec_src}" in prompt
        # the validated read scope is emitted (control repo + assigned worktree)
        assert f"read-scope:\n  - {root}\n  - {worktree}\n" in prompt
        assert not (worktree / ".local" / "docs" / "specs").exists()
        assert not (worktree / ".local" / "docs" / "plans").exists()


def test_worker_builder_refuses_plan_outside_the_allowlist(capsys):
    # An explicitly named allowed external file is a read input (see the external-plan
    # regression below); a path OUTSIDE the allowlist is still refused -- the same
    # canonical containment `dispatch_build` applies.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        outside = Path(raw).resolve() / "outside" / "forged.md"
        outside.parent.mkdir(parents=True)
        outside.write_text("# Forged\n", encoding="utf-8")
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(
            root, worktree, plan_path=str(outside),
        )
        events = []

        def popen(*a, **kw):
            raise AssertionError("popen must never run for a plan outside the allowlist")

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True}, popen=popen,
            allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"
        assert len(events) == 1
        assert events[0]["payload"]["contract_status"] == "cancelled"
        # the refusal cleaned up the reservation exactly like any pre-launch refusal
        assert not worktree.exists()
        assert not manifest_path.parent.exists()


def test_external_plan_survives_dispatch_builder_and_diff_launch(monkeypatch):
    # The MOA-467 acceptance regression: a plan in ANOTHER allowlist repo, explicitly
    # named by the caller and validated at dispatch, must survive the builder launch and
    # the later diff launch -- no asynchronous refusal, no permission-only copy.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        allow_root.mkdir()
        root = allow_root / "demo"
        _init_repo(root)
        external = allow_root / "docs-repo"
        _init_repo(external)
        plan_src = external / "plans" / "external-plan.md"
        plan_src.parent.mkdir(parents=True)
        plan_src.write_text(
            "# External plan\n\n**Goal:** g\n\n### Task 1: t\n", encoding="utf-8",
        )
        monkeypatch.chdir(root)
        run_id = jaxflow.dispatch_build(
            _build_args(plan=str(plan_src), branch="feat/x"),
            run=_run_with_tmux(FakeTmux(), real_cwd=root), post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"}, now=_fixed_now,
            allowlist_root=allow_root,
        )
        worktree = allow_root / "demo-feat-x"
        manifest_path = root / ".local" / "runs" / run_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["plan_path"] == str(plan_src)
        assert not (worktree / ".local" / "docs").exists()  # no permission-only copy

        # Builder launch: the worker accepts the external original, names it (and the
        # spec it declares -- none here, so the plan is both inputs) and grants its
        # parent only, never the sibling repo root.
        captured = {}

        class CaptureBuilderPopen(FakeBuilderPopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["stdin_bytes"] = self.stdin_bytes

        assert jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CaptureBuilderPopen, allowlist_root=allow_root,
        ) == 0
        prompt = captured["stdin_bytes"].decode("utf-8")
        assert f"  plan: {plan_src}" in prompt
        assert f"  spec: {plan_src}" in prompt
        assert f"  - {plan_src.parent}\n" in prompt
        assert f"  - {external}\n" not in prompt

        # Diff launch: the same external original flows into the diff manifest, still
        # with no copy anywhere in the worktree.
        head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _seed_finished_build(con, run_id, root, worktree, verify="true", head_sha=head,
                             plan_path=plan_src)
        con.close()
        diff_run_id = jaxflow.dispatch_diff_review(
            _diff_args(run_id), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        diff_manifest = json.loads(
            (root / ".local" / "runs" / diff_run_id / "manifest.json").read_text(encoding="utf-8"))
        assert diff_manifest["plan_path"] == str(plan_src)
        assert diff_manifest["spec_path"] == str(plan_src)
        assert not (worktree / ".local" / "docs").exists()

        # Diff worker launch: the reviewer handoff names the same external original.
        captured_diff = {}

        class CaptureDiffPopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured_diff["stdin_bytes"] = self.stdin_bytes

        assert jaxflow.run_worker(
            str(root / ".local" / "runs" / diff_run_id / "manifest.json"),
            run=_run_with_tmux(FakeTmux(), real_cwd=worktree), post=lambda e: {"ok": True},
            popen=CaptureDiffPopen, allowlist_root=allow_root,
        ) == 0
        diff_prompt = captured_diff["stdin_bytes"].decode("utf-8")
        assert f"  plan: {plan_src}" in diff_prompt
        assert f"  spec: {plan_src}" in diff_prompt


def test_worker_builder_refuses_a_missing_plan_file(capsys):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        missing = worktree / ".local" / "docs" / "plans" / "missing.md"
        manifest_path, manifest = _write_manifest_for_builder_worker(
            root, worktree, plan_path=str(missing),
        )

        def popen(*a, **kw):
            raise AssertionError("popen must never run for a missing plan")

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=popen, allowlist_root=root.parent,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"


def test_worker_builder_refuses_symlinked_secret_plan_path(capsys):
    # A symlink planted in the worktree that resolves to a control-repo `.env` refuses
    # with the secret code -- resolve() FIRST, then the shape checks, exactly like the
    # reviewer target guard.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
        plan_link = worktree / "fake-plan.md"
        plan_link.symlink_to(root / ".env")
        manifest_path, manifest = _write_manifest_for_builder_worker(
            root, worktree, plan_path=str(plan_link),
        )

        def popen(*a, **kw):
            raise AssertionError("popen must never run for a symlinked secret plan")

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=popen, allowlist_root=root.parent,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == f"secret-detected: {root / '.env'}"


# ---- worker: model/effort override reaches runtime_argv (fixes cold review F4) ----

def test_worker_builder_model_effort_override_reaches_runtime_argv():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(
            root, worktree, runtime="opencode-deepseek", model="custom-id", effort="n/a",
        )
        captured = {}

        class CapturePopen(FakeBuilderPopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["argv"] = argv

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
        )
        argv = captured["argv"]
        assert argv[argv.index("--model") + 1] == "custom-id"
        # The runtime's OWN variant survives a model override -- it is part of the runtime
        # definition, not something the caller supplied (MOA-451).
        assert argv[argv.index("--variant") + 1] == "high"

        # No override -- the "default"/"n/a" ledger sentinels convert back to None, so the
        # spawned argv falls back to the runtime's own model id, with no --model token
        # carrying a caller value.
        manifest_path2, _ = _write_manifest_for_builder_worker(
            root, worktree, run_id="ccccddddeeee", runtime="opencode-grok",
            model="default", effort="n/a",
        )
        captured2 = {}

        class CapturePopen2(FakeBuilderPopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured2["argv"] = argv

        jaxflow.run_worker(
            str(manifest_path2), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen2, allowlist_root=root.parent,
        )
        argv2 = captured2["argv"]
        assert argv2[argv2.index("--model") + 1] == "xai/grok-4.6"
        assert argv2[argv2.index("--variant") + 1] == "xhigh"


# ---- worker: post-build verify evidence is redacted (fixes cold review F6) ----

def test_write_verify_tests_file_redacts_secrets_in_command_and_output():
    with TemporaryDirectory() as raw:
        worktree = Path(raw).resolve()
        path = worktree / "run.tests.txt"
        verify_cmd = 'echo "Authorization: Bearer sekret-tok-999"'
        result = _run_real(["/bin/sh", "-c", verify_cmd], raw)
        jaxflow_workerkit._write_verify_tests_file(path, [(verify_cmd, result)], worktree=worktree)
        text = path.read_text(encoding="utf-8")
        assert "sekret-tok-999" not in text
        assert "[REDACTED]" in text
        assert "EXIT: 0" in text


def test_write_verify_tests_file_writes_one_frame_per_command_test_then_build():
    """Spec §4.2 step 4 / §7.1 #34. One COMMAND:/output/EXIT: frame per non-none command,
    in order. status-report-contract.md already specifies this framing; only the writer
    was single-command."""
    with TemporaryDirectory() as raw:
        worktree = Path(raw).resolve()
        path = worktree / "out.tests.txt"
        frames = [
            ("pnpm test", CompletedProcess(args=[], returncode=1, stdout="test out\n", stderr="")),
            ("pnpm build", CompletedProcess(args=[], returncode=0, stdout="build out\n", stderr="")),
        ]
        assert jaxflow_workerkit._write_verify_tests_file(path, frames, worktree=worktree) is True
        assert path.read_text(encoding="utf-8") == (
            "COMMAND: pnpm test\ntest out\nEXIT: 1\n"
            "COMMAND: pnpm build\nbuild out\nEXIT: 0\n"
        )


def test_write_verify_tests_file_single_frame_is_byte_identical_to_the_old_format():
    """Spec §4.2 step 4: with no --build the file must be byte-identical to what the
    single-command implementation wrote. This is the backward-compatibility guard --
    `review --diff` on an old build run must produce the same evidence it always did."""
    with TemporaryDirectory() as raw:
        worktree = Path(raw).resolve()
        path = worktree / "out.tests.txt"
        result = CompletedProcess(args=[], returncode=0, stdout="hello\n", stderr="")
        assert jaxflow_workerkit._write_verify_tests_file(path, [("true", result)], worktree=worktree) is True
        assert path.read_text(encoding="utf-8") == "COMMAND: true\nhello\nEXIT: 0\n"


def test_run_verify_commands_always_runs_the_build_even_when_the_test_fails():
    """Spec §4.2 step 4 / §7.1 #35. Short-circuiting would restore the `&&` semantics this
    change removes: when the tests fail you would still not learn whether the tree builds."""
    calls = []

    def fake_run(argv, cwd=None):
        calls.append(argv[-1])
        return CompletedProcess(args=argv, returncode=1 if argv[-1] == "T" else 0, stdout="", stderr="")

    frames = jaxflow_workerkit._run_verify_commands(fake_run, Path("/tmp"), "T", "B")
    assert calls == ["T", "B"]
    assert [c for c, _ in frames] == ["T", "B"]
    assert [r.returncode for _, r in frames] == [1, 0]


def test_run_verify_commands_runs_only_the_test_when_there_is_no_build_command():
    calls = []

    def fake_run(argv, cwd=None):
        calls.append(argv[-1])
        return CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    frames = jaxflow_workerkit._run_verify_commands(fake_run, Path("/tmp"), "T", None)
    assert calls == ["T"]
    assert len(frames) == 1


# ---- worker: a DECLARED but bad **Spec:** reference refuses loudly, never silently
# substitutes the plan copy (fixes cold review round 2 F5) ----

def test_worker_builder_refuses_declared_spec_outside_allowlist_posts_cancelled_and_cleans_up(
    capsys, monkeypatch,
):
    # fixes Part 3 diff-review F3 (every pre-launch worker refusal posts a terminal
    # cancelled row and undoes the entire dispatched reservation -- worktree, branch, Git's
    # own worktree admin entry, the manifest dir, and the scratch dir) and F4 (this test
    # used to pass even if the builder-role branch never actually reached
    # _resolve_handoff_spec_path, since an unrelated code path could coincidentally print
    # the same string -- wrapping the real resolver proves the builder branch's own spec
    # check actually ran).
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        outside = Path(raw) / "outside"
        outside.mkdir()
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        outside_spec = outside / "spec.md"
        outside_spec.write_text("# Outside\n", encoding="utf-8")
        _write_plan(worktree, text=(
            f"# Ship the thing\n\n**Goal:** make it spin.\n\n**Spec:** `{outside_spec}`\n\n"
            "### Task 1: do it\n"
        ))
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)

        real_resolve = jaxflow_common._resolve_handoff_spec_path
        calls = []

        def spy(*a, **kw):
            calls.append((a, kw))
            return real_resolve(*a, **kw)

        monkeypatch.setattr(jaxflow_common, "_resolve_handoff_spec_path", spy)

        def popen(*a, **kw):
            raise AssertionError("popen must never be called once the declared spec refuses")

        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=post, popen=popen, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"
        assert len(calls) == 1  # the builder-role spec resolver actually ran

        assert len(events) == 1
        assert events[0]["payload"]["contract_status"] == "cancelled"

        assert not worktree.exists()
        assert _run_real(["git", "rev-parse", "--verify", "feat/x"], root).returncode != 0
        admin = _run_real(["git", "worktree", "list", "--porcelain"], root).stdout
        assert "demo-feat-x" not in admin
        assert not (root / ".local" / "runs" / manifest["run_id"]).exists()
        assert not (worktree / ".local" / "scratch" / manifest["run_id"]).exists()


def test_worker_builder_refuses_declared_secret_spec(capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        secret_spec = root / ".local" / "docs" / "specs" / ".env"
        secret_spec.parent.mkdir(parents=True)
        secret_spec.write_text("SECRET=1\n", encoding="utf-8")
        _write_plan(worktree, text=(
            f"# Ship the thing\n\n**Goal:** make it spin.\n\n**Spec:** `{secret_spec}`\n\n"
            "### Task 1: do it\n"
        ))
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=FakeBuilderPopen, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == f"secret-detected: {secret_spec}"


# ---- worker: a pre-launch refusal is a FINISHED run for the caller too (fixes S1, real
# smoke 6912107edb4f: a refused worker used to leave the caller with no [JAXFLOW] line and
# an empty status.md `## Now`) ----

def test_worker_builder_refusal_sends_jaxflow_callback_and_sets_worker_fields(capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        outside = Path(raw) / "outside"
        outside.mkdir()
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        outside_spec = outside / "spec.md"
        outside_spec.write_text("# Outside\n", encoding="utf-8")
        _write_plan(worktree, text=(
            f"# Ship the thing\n\n**Goal:** make it spin.\n\n**Spec:** `{outside_spec}`\n\n"
            "### Task 1: do it\n"
        ))
        fake = FakeTmux()
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)

        def popen(*a, **kw):
            raise AssertionError("popen must never be called once the declared spec refuses")

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(fake, real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=popen, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        capsys.readouterr()
        # `_refuse_builder_run` already deleted this run's own manifest dir (it "owns"
        # the reservation, no started row for it -- see its own docstring) BEFORE calling
        # `_send_callback` -- proving delivery survives that cleanup (D6).
        assert not manifest_path.parent.exists()
        line_path = jaxflow_common.CALLBACKS_ROOT / manifest["caller_session"] / f"{manifest['run_id']}.line"
        assert line_path.read_text(encoding="utf-8").rstrip("\n") == (
            f"[JAXFLOW] build {manifest['run_id']} finished — cancelled — no report"
        )


def test_worker_builder_refusal_codex_callback_attempts_queue_once(monkeypatch, capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        outside = Path(raw) / "outside"
        outside.mkdir()
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        outside_spec = outside / "spec.md"
        outside_spec.write_text("# Outside\n", encoding="utf-8")
        _write_plan(worktree, text=(
            f"# Ship the thing\n\n**Goal:** make it spin.\n\n**Spec:** `{outside_spec}`\n\n"
            "### Task 1: do it\n"
        ))
        fake = FakeTmux()
        queue_calls = []

        def queue(argv, **kwargs):
            queue_calls.append((argv, kwargs))
            return CompletedProcess(argv, 1, _CALLBACK_SECRET.encode(), _CALLBACK_SECRET.encode())

        _intercept_codex_queue(monkeypatch, queue)
        monkeypatch.setattr(time, "sleep", lambda seconds: pytest.fail("Codex callback slept"))
        manifest_path, manifest = _write_manifest_for_builder_worker(
            root, worktree, caller="codex", caller_session=_CAPTURED_THREAD,
            caller_pane="%3", caller_incarnation=fake.incarnation,
        )

        def popen(*a, **kw):
            raise AssertionError("popen must never be called once the declared spec refuses")

        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(fake, real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True}, popen=popen, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        err = capsys.readouterr().err
        assert "path-outside-allowlist" in err
        assert _CALLBACK_SECRET not in err
        assert f"callback delivery failed: {manifest['run_id']} queue-failed; use jaxflow status/result" in err
        assert len(queue_calls) == 1
        assert queue_calls[0][0][:4] == ["codex", "queue", "--thread", _CAPTURED_THREAD]
        assert queue_calls[0][1] == _QUEUE_KW
        assert not any(c[1] == "send-keys" for c in fake.calls)
        assert len(events) == 1
        assert events[0]["payload"]["contract_status"] == "cancelled"
        assert not worktree.exists()
        assert _run_real(["git", "rev-parse", "--verify", "feat/x"], root).returncode != 0
        assert not (root / ".local" / "runs" / manifest["run_id"]).exists()


def test_worker_diff_review_refusal_sends_jaxflow_callback_line(capsys):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        (worktree / "id_rsa").write_text("SECRET=1\n", encoding="utf-8")
        _git(worktree, "add", "id_rsa")
        _git(worktree, "commit", "-m", "add secret")
        base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        fake = FakeTmux()
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, base_sha=base_sha, head_sha=head_sha,
        )

        def popen(*a, **kw):
            raise AssertionError("popen must never be called for a secret path")

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(fake, real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=popen, allowlist_root=root.parent,
        )
        assert code == jaxflow_common.REFUSED
        capsys.readouterr()
        line_path = jaxflow_common.CALLBACKS_ROOT / manifest["caller_session"] / f"{manifest['run_id']}.line"
        assert line_path.read_text(encoding="utf-8").rstrip("\n") == (
            f"[JAXFLOW] diff {manifest['run_id']} finished — cancelled — no report"
        )


def test_worker_diff_review_refusal_codex_callback_attempts_queue_once(monkeypatch, capsys):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        (worktree / "id_rsa").write_text("SECRET=1\n", encoding="utf-8")
        _git(worktree, "add", "id_rsa")
        _git(worktree, "commit", "-m", "add secret")
        base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        fake = FakeTmux()
        queue_calls = []

        def queue(argv, **kwargs):
            queue_calls.append((argv, kwargs))
            return CompletedProcess(argv, 1, _CALLBACK_SECRET.encode(), _CALLBACK_SECRET.encode())

        _intercept_codex_queue(monkeypatch, queue)
        monkeypatch.setattr(time, "sleep", lambda seconds: pytest.fail("Codex callback slept"))
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, base_sha=base_sha, head_sha=head_sha,
            caller="codex", caller_session=_CAPTURED_THREAD,
            caller_pane="%3", caller_incarnation=fake.incarnation,
        )

        def popen(*a, **kw):
            raise AssertionError("popen must never be called for a secret path")

        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(fake, real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True}, popen=popen, allowlist_root=root.parent,
        )
        assert code == jaxflow_common.REFUSED
        err = capsys.readouterr().err
        assert "secret-detected: id_rsa" in err
        assert _CALLBACK_SECRET not in err
        assert f"callback delivery failed: {manifest['run_id']} queue-failed; use jaxflow status/result" in err
        assert len(queue_calls) == 1
        assert queue_calls[0][0][:4] == ["codex", "queue", "--thread", _CAPTURED_THREAD]
        assert queue_calls[0][1] == _QUEUE_KW
        assert not any(c[1] == "send-keys" for c in fake.calls)
        assert len(events) == 1
        assert events[0]["payload"]["contract_status"] == "cancelled"
        assert worktree.exists()


def test_worker_builder_refusal_no_callback_flag_sends_nothing():
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        outside = Path(raw) / "outside"
        outside.mkdir()
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        outside_spec = outside / "spec.md"
        outside_spec.write_text("# Outside\n", encoding="utf-8")
        _write_plan(worktree, text=(
            f"# Ship the thing\n\n**Goal:** make it spin.\n\n**Spec:** `{outside_spec}`\n\n"
            "### Task 1: do it\n"
        ))
        fake = FakeTmux()
        manifest_path, manifest = _write_manifest_for_builder_worker(
            root, worktree, no_callback=True,
        )

        def popen(*a, **kw):
            raise AssertionError("popen must never be called once the declared spec refuses")

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(fake, real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=popen, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert not any(c[1] == "send-keys" for c in fake.calls)
        assert not jaxflow_common.CALLBACKS_ROOT.exists() or not any(jaxflow_common.CALLBACKS_ROOT.rglob("*.line"))


def test_worker_builder_refusal_updates_status_md_now_and_stage_with_no_gate():
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        outside = Path(raw) / "outside"
        outside.mkdir()
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        outside_spec = outside / "spec.md"
        outside_spec.write_text("# Outside\n", encoding="utf-8")
        _write_plan(worktree, text=(
            f"# Ship the thing\n\n**Goal:** make it spin.\n\n**Spec:** `{outside_spec}`\n\n"
            "### Task 1: do it\n"
        ))
        status_path = _write_status_md(root)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)

        def popen(*a, **kw):
            raise AssertionError("popen must never be called once the declared spec refuses")

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=popen, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        text = status_path.read_text(encoding="utf-8")
        assert "stage: build" in text
        assert "## Now\npath-outside-allowlist" in text
        assert "gate:" not in text


# ---- worker: assertions the plan's own header claimed but never wrote (fixes cold review
# round 2 F6) ----

def test_worker_builder_invalid_report_is_invalid_not_missing():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)

        class InvalidReportPopen(FakeBuilderPopen):
            # An out-of-enum result fails both parse stages -- "invalid", never
            # "missing" (the file DOES exist, it just doesn't validate).
            REPORT_TEXT = "result: banana\nsummary: built the thing\n"

            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                self.stdin_bytes = stdin.read() if stdin is not None else None
                self.pid = 5254
                self.returncode = 0
                text = self.stdin_bytes.decode("utf-8")
                for line in text.splitlines():
                    if line.startswith("report: "):
                        Path(line[len("report: "):]).write_text(
                            InvalidReportPopen.REPORT_TEXT, encoding="utf-8",
                        )
                        break

            def wait(self, timeout=None):
                return 0

        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree), post=post,
            popen=InvalidReportPopen, allowlist_root=root.parent,
        )
        payload = events[0]["payload"]
        assert payload["contract_status"] == "invalid"
        # MOA-474: an out-of-enum label is "no label" for the decision table -> Part 3,
        # where no base_sha in this manifest means no commit beyond base -> failure.
        assert payload["result"] == "failure"
        assert payload["stage"] == "report"
        assert payload["diagnostic"] == "no commit beyond base"


def test_worker_builder_seals_env_for_child_process():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        captured = {}

        class CapturePopen(FakeBuilderPopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["env"] = env

        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
        )
        assert captured["env"]["HONCHO_ENABLED"] == "false"
        assert captured["env"]["JAXFLOW_ONESHOT"] == "1"


def test_worker_builder_updates_control_repo_status_md():
    # fixes cold review round 2 F6: no test proved the builder branch actually calls
    # _update_status_md against the CONTROL repo (Step 8's wiring, this file) -- the
    # existing signal tests only exercise the reviewer branch.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        status_path = _write_status_md(root)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=FakeBuilderPopen, allowlist_root=root.parent,
        )
        text = status_path.read_text(encoding="utf-8")
        assert "stage: build" in text
        assert "branch: feat/x" in text
        assert "## Now\nbuilt the thing" in text


# ---- MOA-467 Task 4: the CONTROL repo owns status for every run kind, even when the run
# originates in a linked worktree; a worktree status sentinel is never touched ----

@pytest.mark.parametrize("kind,target_kw", [
    ("build", {"target": "feat/x"}),
    ("spec", {}),
    ("diff", {"target": "feat/x"}),
    ("merge", {"target": "feat/x"}),
])
def test_status_md_linked_worktree_run_updates_only_the_control_repo(kind, target_kw):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        control_status = _write_status_md(root)
        sentinel = _write_status_md(worktree)
        before_sentinel = sentinel.read_text(encoding="utf-8")
        manifest = {
            "repo": str(worktree), "kind": kind, "runtime": "codex",
            "dispatch_start": _AFTER_TEMPLATE_UPDATED, "worker_summary": "s",
            "worker_contract_status": "ok", "worker_outcome": "approve",
            **target_kw,
        }
        jaxflow_common._update_status_md(manifest, run=_run_real, allowlist_root=root.parent)
        control_text = control_status.read_text(encoding="utf-8")
        assert "stage:" in control_text, kind
        assert "## Now\ns" in control_text, kind
        # the linked worktree's own status file stays byte-for-byte untouched
        assert sentinel.read_text(encoding="utf-8") == before_sentinel, kind


@pytest.mark.parametrize("kind,target_kw", [
    ("build", {"target": "feat/x"}),
    ("spec", {}),
    ("diff", {"target": "feat/x"}),
    ("merge", {"target": "feat/x"}),
])
def test_status_md_skips_write_when_the_owner_lookup_fails(kind, target_kw):
    # MOA-467 acceptance: an unresolved owner lookup for a `.git`-FILE checkout must
    # never redirect the write to the worktree itself -- skip best-effort instead, on a
    # non-zero probe AND on a raising one; the worktree sentinel stays byte-for-byte.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        control_status = _write_status_md(root)
        control_before = control_status.read_text(encoding="utf-8")
        sentinel = _write_status_md(worktree)
        sentinel_before = sentinel.read_text(encoding="utf-8")
        manifest = {
            "repo": str(worktree), "kind": kind, "runtime": "codex",
            "dispatch_start": _AFTER_TEMPLATE_UPDATED, "worker_summary": "s",
            "worker_contract_status": "ok", "worker_outcome": "approve",
            **target_kw,
        }

        def failing_probe(argv, cwd=None):
            if argv[:2] == ["git", "rev-parse"] and "--git-common-dir" in argv:
                return SimpleNamespace(returncode=128, stdout="", stderr="fatal: injected")
            return _run_real(argv, cwd)

        def raising_probe(argv, cwd=None):
            if argv[:2] == ["git", "rev-parse"] and "--git-common-dir" in argv:
                raise OSError("git exploded")
            return _run_real(argv, cwd)

        for probe in (failing_probe, raising_probe):
            jaxflow_common._update_status_md(manifest, run=probe, allowlist_root=root.parent)
            assert sentinel.read_text(encoding="utf-8") == sentinel_before, kind
            assert control_status.read_text(encoding="utf-8") == control_before, kind


@pytest.mark.parametrize("kind,target_kw", [
    ("build", {"target": "feat/x"}),
    ("spec", {}),
    ("diff", {"target": "feat/x"}),
    ("merge", {"target": "feat/x"}),
])
def test_status_md_separate_git_dir_main_repo_owns_its_card(kind, target_kw):
    # A `--separate-git-dir` main repo also has a `.git` FILE, but git reports its
    # gitdir and common dir as the SAME path -- it is no one's linked worktree, so it
    # still owns its own card, for build/review/merge alike.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "repos" / "sep"
        root.mkdir(parents=True)
        gitdir = Path(raw).resolve() / "sep-gitdir"
        _git(root, "init", "-b", "main", f"--separate-git-dir={gitdir}")
        status_path = _write_status_md(root)
        manifest = {
            "repo": str(root), "kind": kind, "runtime": "opencode-grok",
            "dispatch_start": _AFTER_TEMPLATE_UPDATED,
            "worker_summary": "built the thing", "worker_contract_status": "ok",
            "worker_outcome": "success",
            **target_kw,
        }
        jaxflow_common._update_status_md(manifest, run=_run_real, allowlist_root=Path(raw).resolve())
        text = status_path.read_text(encoding="utf-8")
        assert "stage:" in text, kind
        assert "## Now\nbuilt the thing" in text, kind


def test_control_owner_separate_git_dir_main_and_linked_worktree():
    # MOA-467 acceptance fixes: the shared relationship resolves a `--separate-git-dir`
    # MAIN to itself (gitdir == common dir); git's worktree list reports the COMMON GITDIR
    # as the main entry for its linked worktree, which fails checkout validation -- the
    # lookup is unresolved, so reads fall back to the worktree and status skips, never
    # the gitdir and never the worktree.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        allow_root.mkdir()
        main = allow_root / "sep"
        gitdir = Path(raw).resolve() / "sep-gitdir"
        _init_separate_git_dir_repo(main, gitdir)
        worktree = _init_worktree(main, "feat/x")
        assert jaxflow_common._control_owner(main, _run_real, allow_root) == (main.resolve(), True)
        assert jaxflow_common._control_owner(worktree, _run_real, allow_root) == (worktree.resolve(), False)
        assert jaxflow_workerkit._control_repo_of(worktree, _run_real, allow_root) == worktree.resolve()


def test_builder_read_roots_separate_git_dir_main_resolves_to_the_checkout():
    # MOA-467 acceptance fixes: a `--separate-git-dir` main whose common dir IS named
    # `.git` used to resolve its control repo to the gitdir's parent via the old name
    # check -- the shared relationship now names the checkout itself.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        allow_root.mkdir()
        main = allow_root / "sep"
        gitdir = allow_root / "cache" / ".git"
        _init_separate_git_dir_repo(main, gitdir)
        worktree = _init_worktree(main, "feat/x")
        roots = jaxflow._builder_read_roots(
            main, worktree, run=_run_real, allowlist_root=allow_root,
        )
        assert roots == (main.resolve(), worktree.resolve())
        assert (allow_root / "cache").resolve() not in roots


@pytest.mark.parametrize("kind,target_kw", [
    ("build", {"target": "feat/x"}),
    ("spec", {}),
    ("diff", {"target": "feat/x"}),
    ("merge", {"target": "feat/x"}),
])
def test_status_md_separate_git_dir_worktree_run_never_redirects_to_the_worktree(kind, target_kw):
    # A linked worktree of a `--separate-git-dir` main has no git-derivable main checkout
    # (git's worktree list main entry is the common gitdir), so the owner lookup is
    # unresolved: the write is skipped -- neither the worktree sentinel nor the main card
    # changes, for build/review/merge alike.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        allow_root.mkdir()
        main = allow_root / "sep"
        gitdir = Path(raw).resolve() / "sep-gitdir"
        _init_separate_git_dir_repo(main, gitdir)
        worktree = _init_worktree(main, "feat/x")
        control_status = _write_status_md(main)
        control_before = control_status.read_text(encoding="utf-8")
        sentinel = _write_status_md(worktree)
        sentinel_before = sentinel.read_text(encoding="utf-8")
        manifest = {
            "repo": str(worktree), "kind": kind, "runtime": "codex",
            "dispatch_start": _AFTER_TEMPLATE_UPDATED, "worker_summary": "s",
            "worker_contract_status": "ok", "worker_outcome": "approve",
            **target_kw,
        }
        jaxflow_common._update_status_md(manifest, run=_run_real, allowlist_root=allow_root)
        assert sentinel.read_text(encoding="utf-8") == sentinel_before, kind
        assert control_status.read_text(encoding="utf-8") == control_before, kind


def test_worker_doc_review_from_linked_worktree_writes_control_status_not_the_worktree():
    # End to end through run_worker (the review branch): a doc review whose manifest repo
    # is a real linked worktree updates the CONTROL repo's card -- the worktree sentinel
    # is untouched.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        control_status = _write_status_md(root)
        sentinel = _write_status_md(worktree)
        before_sentinel = sentinel.read_text(encoding="utf-8")
        target = _spec_file(root, text="REVIEW ME\n")
        manifest_path, manifest = _write_manifest_for_worker(
            worktree, target, runtime="claude",
        )
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=FakePopen, allowlist_root=allow_root,
        )
        assert code == 0
        control_text = control_status.read_text(encoding="utf-8")
        assert "stage: spec" in control_text
        assert "builder: claude" in control_text
        assert sentinel.read_text(encoding="utf-8") == before_sentinel



# ---- review --diff: dispatch (spec §4.3 steps 1-4) ----

def _diff_args(builder_run_id, **kw):
    return SimpleNamespace(
        diff=builder_run_id, focus=kw.get("focus"), model=kw.get("model"),
        effort=kw.get("effort"), phase=kw.get("phase"), from_caller=kw.get("from_caller"),
        no_callback=kw.get("no_callback", False), reverify=kw.get("reverify", False),
        since=kw.get("since"), full=kw.get("full"),
    )


def _seed_finished_build(con, run_id, root, worktree, *, verify="true", build=None, head_sha=None, plan_path=None, contract_status="ok", base_sha=None):
    """Seeds a finished `build` run-started/run-finished pair in the given sqlite
    connection, matching the shape `dispatch_build`/the builder worker actually write.
    Also writes the build's own `.local/runs/<run_id>/manifest.json` with a `plan_path`
    (fixes S2, real smoke b12a3db3da4b: `review --diff` now reads THIS file, not the DB
    rows, to resolve the diff handoff's own spec/plan) -- skipped when `worktree` was
    never actually created (the "missing worktree" dispatch test deliberately seeds a
    build for a worktree that is never created, and refuses before this file would ever
    be read; writing into it here would create it as a side effect and break that test).
    `plan_path` lets a caller point at a plan it already wrote itself (with its own
    **Spec:** line, e.g.) instead of getting the default no-Spec-line plan."""
    started_payload = {
        "phase": "PHASE", "runtime": "opencode-grok", "kind": "build", "target": "feat/x",
        "caller": "claude", "caller_session": "s", "model": "xai/grok-4.6", "effort": "n/a",
        "session": f"jax-demo-build-{run_id}", "repo": str(root), "verify": verify,
    }
    if build:
        started_payload["build"] = build
    _insert(con, run_id, "demo", "builder", "run-started", started_payload)
    _insert(con, run_id, "demo", "builder", "run-finished", {
        "phase": "PHASE", "exit_code": 0, "contract_status": contract_status,
        "report_path": str(worktree / ".local" / "reports" / f"{run_id}.md"),
        "summary": "built the thing", "head_sha": head_sha or "a" * 40,
        **({"result": "success"} if contract_status == "ok" else {}),
    })
    if worktree.is_dir():
        if plan_path is None:
            _write_plan(worktree)
            plan_path = worktree / ".local" / "docs" / "plans" / "plan.md"
        manifest_dir = root / ".local" / "runs" / run_id
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "manifest.json").write_text(
            json.dumps({"plan_path": str(plan_path), **({"base_sha": base_sha} if base_sha is not None else {})}),
            encoding="utf-8",
        )


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
        jaxflow.dispatch_diff_review(
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
        jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
        jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
                jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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

        run_id = jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
            _diff_args("b1"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
            now=_fixed_now, allowlist_root=allow_root, db_path=db,
        )
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text())
        assert manifest["guard"]["override"] == "none"
        assert manifest["guard"]["prior_verdict"] == "reject"

        run_id2 = jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
        jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
                jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
                jaxflow.dispatch_diff_review(
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
        assert jaxflow.cmd_status("aaaaaaaaaaa1", run=_run_with_tmux(FakeTmux()), db_path=db) == (
            f"chain:\n  aaaaaaaaaaa1 {merge_base[:12]}..{root_head[:12]} approve\nfinished(approve)"
        )

        _seed_diff_review(con, "aaaaaaaaaaa2", root, worktree, verdict="approve-with-changes", ts="2026-02-01T00:00:00+00:00")
        _write_diff_manifest(root, "aaaaaaaaaaa2", base_sha=root_head, head_sha=delta_head, since_review_run_id="aaaaaaaaaaa1")
        _seed_diff_review(con, "aaaaaaaaaaa3", root, worktree, verdict="reject", ts="2026-03-01T00:00:00+00:00")
        _write_diff_manifest(root, "aaaaaaaaaaa3", base_sha=delta_head, head_sha=delta2_head, since_review_run_id="aaaaaaaaaaa2")
        con.close()
        status = jaxflow.cmd_status("aaaaaaaaaaa3", run=_run_with_tmux(FakeTmux()), db_path=db)
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
        status4 = jaxflow.cmd_status("aaaaaaaaaaa4", run=_run_with_tmux(FakeTmux()), db_path=db)
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
        status = jaxflow.cmd_status("aaaaaaaaaaa1", run=_run_with_tmux(FakeTmux()), db_path=db)
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
        status = jaxflow.cmd_status("aaaaaaaaaaa1", run=_run_with_tmux(FakeTmux()), db_path=db)
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
        assert jaxflow._chain_block("aaaaaaaaaaa1", db_path=db) == "chain: broken (malformed)"

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
        assert jaxflow._chain_block("aaaaaaaaaaa2", db_path=db) == "chain: broken (malformed)"

        # (c) non-string head_sha in an otherwise well-formed root manifest.
        _seed_diff_review(con, "aaaaaaaaaaa3", root, None, verdict="approve")
        (root / ".local" / "runs" / "aaaaaaaaaaa3" / "manifest.json").write_text(
            json.dumps({"kind": "diff", "base_sha": "f" * 40, "head_sha": 12345}), encoding="utf-8",
        )
        con.close()
        assert jaxflow._chain_block("aaaaaaaaaaa3", db_path=db) == "chain: broken (malformed)"


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
        assert jaxflow.cmd_status("s1", run=_run_with_tmux(FakeTmux()), db_path=db) == "finished(no verdict)"


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
        out = jaxflow.cmd_result("aaaaaaaaaaa1", allowlist_root=allow_root, db_path=db)
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
    # Regression for the Path-vs-str comparison hazard at scripts/jaxflow.py:1731 --
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

        code = jaxflow.run_worker(
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
            jaxflow.dispatch_diff_review(
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
                jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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

        run_id = jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
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

        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
            jaxflow.dispatch_diff_review(
                _diff_args("b2"), run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: {"ok": True}, env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root, db_path=db2,
            )
        except ji.Refusal as exc:
            assert exc.code == "spec-reference-invalid"
        else:
            raise AssertionError("spec-reference-invalid not raised")


# ---- worker: diff review (spec §4.3 step 4, §6 I2's Claude resolution) ----

def _write_manifest_for_diff_worker(root, worktree, *, run_id="aaaabbbbcccc", runtime="codex", **extra):
    # fixes cold review round 2 F2: the default must match FakePopen.REPORT_TEXT's own
    # hardcoded `run_id: aaaabbbbcccc` frontmatter, or the finalize-report test below
    # (which relies on the default) looks for a report filename the worker never wrote.
    # fixes S2: every diff-worker manifest now carries `plan_path`/`spec_path` (the build's
    # own handoff inputs, re-validated worker-side) -- a real plan file inside the
    # worktree by default, so pre-existing tests here (which never set these) still pass
    # the worker's own re-validation; a test proving the refusal path overrides them.
    if "plan_path" not in extra:
        _write_plan(worktree)
        extra["plan_path"] = str(worktree / ".local" / "docs" / "plans" / "plan.md")
    extra.setdefault("spec_path", extra["plan_path"])
    manifest = {
        "kind": "diff", "role": "reviewer", "project": "demo", "phase": "PHASE",
        "repo": str(root), "worktree": str(worktree), "target": "feat/x",
        # fixes F6: builder_run_id must be 12-hex (it is embedded into a filesystem path
        # by the worker itself, spec §4.3 step 1's convention) -- "b1b1b1b1b1b1" keeps the
        # old "b1" shorthand recognizable while satisfying _RUN_ID_RE.
        "builder_run_id": "b1b1b1b1b1b1", "base_sha": "a" * 40, "head_sha": "b" * 40,
        "tests_path": str(worktree / ".local" / "reports" / "b1b1b1b1b1b1.tests.txt"),
        "caller": "claude", "caller_session": "01234567-89ab-4cde-8f01-23456789abcd", "model": "m", "effort": "high",
        "runtime": runtime, "run_id": run_id, "session": f"jax-demo-diff-{run_id}",
        "no_callback": False, "focus": None,
    }
    manifest.update(extra)
    path = root / ".local" / "runs" / run_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


# ---- worker: manifest identity is validated before ANY path construction (fixes Part 2
# diff-review F1, via the SAME `_validate_run_paths` helper the builder worker uses) ----

def test_worker_diff_review_refuses_forged_run_id_before_any_path_construction(capsys):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        manifest_path, manifest = _write_manifest_for_diff_worker(root, worktree)
        manifest["run_id"] = "/tmp/jaxflow-escape"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        def popen(*a, **kw):
            raise AssertionError("popen must never be called for a forged run_id")

        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True}, popen=popen, allowlist_root=root.parent,
        )
        assert code == jaxflow_common.REFUSED
        err_lines = capsys.readouterr().err.strip().splitlines()
        assert err_lines[0] == "path-outside-allowlist"
        assert err_lines[1] == (
            "callback delivery failed: '/tmp/jaxflow-escape' path-unsafe; use jaxflow status/result"
        )
        # fixes Part 2 diff-review F1: unlike the builder's own top-of-function identity
        # check (which posts nothing), a diff-worker mismatch DOES post the cancelled
        # terminal row -- but never touches the worktree, which it never owns.
        assert len(events) == 1
        assert events[0]["payload"]["contract_status"] == "cancelled"
        assert not Path("/tmp/jaxflow-escape.md").exists()
        assert not Path("/tmp/jaxflow-escape").exists()


def test_worker_diff_review_refuses_repo_outside_allowlist_root(capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        outside = Path(raw).resolve() / "outside-escape"
        outside.mkdir()
        manifest_path, manifest = _write_manifest_for_diff_worker(root, worktree)
        manifest["repo"] = str(outside)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        def popen(*a, **kw):
            raise AssertionError("popen must never be called for an escaping repo")

        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True}, popen=popen, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"
        assert len(events) == 1
        assert events[0]["payload"]["contract_status"] == "cancelled"


def test_worker_diff_review_refuses_worktree_mismatched_from_manifest_derivation(capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        other = allow_root / "some-other-dir"
        other.mkdir()
        manifest_path, manifest = _write_manifest_for_diff_worker(root, worktree)
        manifest["worktree"] = str(other)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        def popen(*a, **kw):
            raise AssertionError("popen must never be called for a mismatched worktree")

        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True}, popen=popen, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"
        assert len(events) == 1
        # never touches the worktree -- a diff review never owns it (unlike the builder's
        # own `_refuse_builder_run`, which runs `_cleanup_worktree` on this same mismatch).
        assert worktree.is_dir()


# ---- worker: plan_path/spec_path re-validation, mirroring the builder worker's own
# (fixes S2) ----

def test_worker_diff_review_refuses_forged_plan_path_outside_allowlist(capsys):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw).resolve() / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        outside_plan = Path(raw).resolve() / "outside-plan.md"
        outside_plan.write_text("# Forged\n", encoding="utf-8")
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, plan_path=str(outside_plan), spec_path=str(outside_plan),
        )

        def popen(*a, **kw):
            raise AssertionError("popen must never be called for a forged plan_path")

        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True}, popen=popen, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"
        assert len(events) == 1
        assert events[0]["payload"]["contract_status"] == "cancelled"


def test_worker_diff_review_refuses_forged_tests_path_never_reaches_prompt(capsys):
    # fixes F6 (HIGH, NEW): the evidence path is DERIVED from worktree+builder_run_id,
    # never trusted verbatim from the manifest -- a forged tests_path that disagrees with
    # the derived path refuses before the reviewer is ever launched, so it can never be
    # echoed into the prompt.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        outside = Path(raw).resolve() / "outside-evidence.txt"
        outside.write_text("attacker-controlled\n", encoding="utf-8")
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, tests_path=str(outside),
        )

        def popen(*a, **kw):
            raise AssertionError("popen must never be called for a forged tests_path")

        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True}, popen=popen, allowlist_root=root.parent,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"
        assert len(events) == 1
        assert events[0]["payload"]["contract_status"] == "cancelled"


def test_worker_diff_review_accepts_original_control_repo_plan_and_spec():
    # A NEW-style diff manifest: plan and derived spec live in the CONTROL repo, outside
    # the worktree -- the worker accepts them (no-Spec plan = plan as both inputs), and
    # the handoff names the originals.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        (worktree / "changed.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "change")
        base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        plan_src = root / ".local" / "docs" / "plans" / "plan.md"
        plan_src.parent.mkdir(parents=True)
        plan_src.write_text("# Plan\n\n**Goal:** g\n\n### Task 1: t\n", encoding="utf-8")
        tests_path = worktree / ".local" / "reports" / "b1b1b1b1b1b1.tests.txt"
        tests_path.parent.mkdir(parents=True)
        tests_path.write_text("COMMAND: true\n\nEXIT: 0\n", encoding="utf-8")
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, base_sha=base_sha, head_sha=head_sha,
            plan_path=str(plan_src), spec_path=str(plan_src), tests_path=str(tests_path),
        )
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["stdin_bytes"] = self.stdin_bytes

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
        )
        assert code == 0
        prompt = captured["stdin_bytes"].decode("utf-8")
        assert f"  spec: {plan_src}" in prompt
        assert f"  plan: {plan_src}" in prompt


def test_worker_diff_review_accepts_a_legacy_spec_copy_when_the_original_is_gone():
    # A legacy diff manifest (created by the old dispatch) named the spec COPY it placed
    # inside the worktree. The plan text may declare an original that no longer exists --
    # the copy must keep working, exactly as it did before MOA-467 ("existing legacy
    # copied inputs still work").
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        (worktree / "changed.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "change")
        base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        _write_plan(worktree, text=(
            "# P\n\n**Goal:** g\n\n"
            f"**Spec:** `{root / '.local' / 'docs' / 'specs' / 'gone-spec.md'}`\n\n"
            "### Task 1: t\n"
        ))
        legacy_copy = worktree / ".local" / "docs" / "specs" / "gone-spec.md"
        legacy_copy.parent.mkdir(parents=True)
        legacy_copy.write_text("# Copied spec\n", encoding="utf-8")
        tests_path = worktree / ".local" / "reports" / "b1b1b1b1b1b1.tests.txt"
        tests_path.parent.mkdir(parents=True)
        tests_path.write_text("COMMAND: true\n\nEXIT: 0\n", encoding="utf-8")
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, base_sha=base_sha, head_sha=head_sha, spec_path=str(legacy_copy),
            tests_path=str(tests_path),
        )
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["stdin_bytes"] = self.stdin_bytes

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
        )
        assert code == 0
        assert f"  spec: {legacy_copy}" in captured["stdin_bytes"].decode("utf-8")


def test_worker_diff_review_refuses_a_spec_not_named_by_the_plan(capsys):
    # The manifest's spec_path must equal the spec the plan itself declares (new style)
    # or a legacy worktree copy -- an arbitrary sibling file is a forged value.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        forged = Path(raw).resolve() / "forged-spec.md"
        forged.write_text("# Forged\n", encoding="utf-8")
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, spec_path=str(forged),
        )

        def popen(*a, **kw):
            raise AssertionError("popen must never run for a forged spec_path")

        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True}, popen=popen,
            allowlist_root=root.parent,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"
        assert len(events) == 1
        assert events[0]["payload"]["contract_status"] == "cancelled"


def test_worker_diff_review_refuses_a_missing_plan_file(capsys):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        missing = worktree / ".local" / "docs" / "plans" / "missing.md"
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, plan_path=str(missing), spec_path=str(missing),
        )

        def popen(*a, **kw):
            raise AssertionError("popen must never run for a missing plan")

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=popen, allowlist_root=root.parent,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"


def test_worker_diff_review_refuses_symlinked_secret_plan(capsys):
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
        plan_link = worktree / "fake-plan.md"
        plan_link.symlink_to(root / ".env")
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, plan_path=str(plan_link), spec_path=str(plan_link),
        )

        def popen(*a, **kw):
            raise AssertionError("popen must never run for a symlinked secret plan")

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=popen, allowlist_root=root.parent,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == f"secret-detected: {root / '.env'}"


def test_worker_diff_review_prompt_renders_spec_and_plan_before_test_output():
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
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["stdin_bytes"] = self.stdin_bytes

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
        )
        assert code == 0
        prompt = captured["stdin_bytes"].decode("utf-8")
        spec_line = f"  spec: {manifest['spec_path']}"
        plan_line = f"  plan: {manifest['plan_path']}"
        test_line = f"  test-output: {manifest['tests_path']}"
        assert spec_line in prompt
        assert plan_line in prompt
        assert prompt.index(spec_line) < prompt.index(plan_line) < prompt.index(test_line)


def _run_diff_worker_and_capture_prompt(manifest_path, worktree, root):
    captured = {}

    class CapturePopen(FakePopen):
        def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
            super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
            captured["stdin_bytes"] = self.stdin_bytes

    code = jaxflow.run_worker(
        str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
        post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
    )
    assert code == 0
    return captured["stdin_bytes"].decode("utf-8")


def test_worker_diff_review_renders_correction_header_and_prior_review_path():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        (worktree / "changed.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "change")
        base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        since_run_id = "cccccccccccc"
        manifest_path, _ = _write_manifest_for_diff_worker(
            root, worktree, base_sha=base_sha, head_sha=head_sha,
            since_review_run_id=since_run_id, since_verdict="approve-with-changes",
        )
        prompt = _run_diff_worker_and_capture_prompt(manifest_path, worktree, root)
        expected_report = root / ".local" / "reports" / f"{since_run_id}.md"
        assert f"Correction review of {since_run_id} (verdict approve-with-changes)\n" in prompt
        assert f"  prior-review: {expected_report}\n" in prompt

        manifest_path2, _ = _write_manifest_for_diff_worker(
            root, worktree, run_id="eeeeeeeeeeee", base_sha=base_sha, head_sha=head_sha,
        )
        prompt2 = _run_diff_worker_and_capture_prompt(manifest_path2, worktree, root)
        # The reviewer contract itself mentions the header phrase, so match the rendered
        # line, not the bare words.
        assert not re.search(r"Correction review of [0-9a-f]{12} \(verdict", prompt2)
        assert not re.search(r"^  prior-review: ", prompt2, re.M)


# ---- worker: secret-named changed paths refuse before the full diff is ever read (fixes
# Part 2 diff-review F2) ----

def test_worker_diff_review_refuses_secret_named_changed_paths(capsys):
    for name in (".env.local", "id_rsa", "foo.pem", "aws.key", "credentials.json"):
        with TemporaryDirectory() as raw:
            root = Path(raw).resolve() / "demo"
            _init_repo(root)
            worktree = _init_worktree(root, "feat/x")
            (worktree / name).write_text("SECRET=1\n", encoding="utf-8")
            _git(worktree, "add", name)
            _git(worktree, "commit", "-m", "add secret")
            base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
            head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
            manifest_path, manifest = _write_manifest_for_diff_worker(
                root, worktree, base_sha=base_sha, head_sha=head_sha,
            )

            def popen(*a, **kw):
                raise AssertionError(f"popen must never be called for a secret path ({name})")

            events = []
            code = jaxflow.run_worker(
                str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
                post=lambda e: events.append(e) or {"ok": True}, popen=popen, allowlist_root=root.parent,
            )
            assert code == jaxflow_common.REFUSED, name
            assert capsys.readouterr().err.strip() == f"secret-detected: {name}", name
            assert len(events) == 1, name
            assert events[0]["payload"]["contract_status"] == "cancelled", name
            # no prompt (and thus no diff/secret content) was ever written to disk -- the
            # run-specific scratch dir is never even created, since the FULL diff is never
            # read for a secret-named change.
            assert not (root / ".local" / "scratch" / manifest["run_id"]).exists(), name


def test_worker_diff_review_prompt_embeds_diff_text_and_grammar_for_both_runtimes():
    for runtime in ("codex", "claude"):
        with TemporaryDirectory() as raw:
            root = Path(raw).resolve() / "demo"
            _init_repo(root)
            worktree = _init_worktree(root, "feat/x")
            (worktree / "changed.txt").write_text("CHANGED-MARKER-TEXT\n", encoding="utf-8")
            _git(worktree, "add", "changed.txt")
            _git(worktree, "commit", "-m", "the change under review")
            base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
            head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
            tests_dir = worktree / ".local" / "reports"
            tests_dir.mkdir(parents=True, exist_ok=True)
            # fixes F6: must match the default builder_run_id ("b1b1b1b1b1b1") this
            # manifest carries -- the worker now DERIVES the evidence path from
            # worktree+builder_run_id and refuses a manifest tests_path that disagrees.
            tests_path = tests_dir / "b1b1b1b1b1b1.tests.txt"
            tests_path.write_text("COMMAND: true\n\nEXIT: 0\n", encoding="utf-8")

            manifest_path, manifest = _write_manifest_for_diff_worker(
                root, worktree, runtime=runtime, base_sha=base_sha, head_sha=head_sha,
                tests_path=str(tests_path),
                caller="codex" if runtime == "claude" else "claude",
            )
            captured = {}

            class CapturePopen(FakePopen):
                def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                    super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                    captured["argv"] = argv
                    captured["cwd"] = cwd
                    captured["stdin_bytes"] = self.stdin_bytes

            code = jaxflow.run_worker(
                str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
                post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
            )
            assert code == 0, runtime
            prompt = captured["stdin_bytes"].decode("utf-8")
            assert prompt.startswith(jaxflow_workerkit.REVIEWER_PROMPT_PREAMBLE), runtime
            assert f"diff: {base_sha}..{head_sha}" in prompt, runtime
            assert f"  test-output: {tests_path}" in prompt, runtime
            assert "CHANGED-MARKER-TEXT" in prompt, runtime  # the embedded diff text
            assert captured["cwd"] == str(worktree), runtime
            if runtime == "codex":
                assert captured["argv"][captured["argv"].index("-C") + 1] == str(worktree), runtime
            if runtime == "claude":
                granted = captured["argv"][captured["argv"].index("--add-dir") + 1:]
                # MOA-467: the Claude reviewer is granted the build worktree AND its
                # canonical control repo (the default-manifest plan/spec live in the
                # worktree here, so no extra document parent is added).
                assert granted == [str(worktree), str(root)]
                for key in ("spec_path", "plan_path", "tests_path"):
                    assert Path(manifest[key]).is_relative_to(worktree)
            else:
                assert "--add-dir" not in captured["argv"]

            # fixes cold review F7: prove the composed handoff actually round-trips
            # through the real grammar parser, not just through ad-hoc substring checks.
            # The full prompt also embeds reviewer-contract.md (via assemble_prompt),
            # which itself contains a literal "test-output:" substring on an unrelated
            # line, AND the embedded diff text that follows the handoff block could
            # itself legitimately contain the substring "test-output:" on some changed
            # line (fixes cold review round 2 F9 -- slicing to the END of the prompt, as
            # a first draft of this test did, would then fail the parser's own "exactly
            # one test-output: occurrence" rule for a reason that has nothing to do with
            # the handoff block itself). Isolate ONLY the fixed 3-line grammar block --
            # from its own unique `diff:` line through the end of the `test-output:` line
            # -- exactly like a real handoff consumer would receive just that block, not
            # the surrounding contract text or the diff that follows it.
            handoff_start = prompt.index(f"diff: {base_sha}..{head_sha}")
            test_output_line = f"  test-output: {tests_path}"
            handoff_end = prompt.index(test_output_line, handoff_start) + len(test_output_line)
            isolated = prompt[handoff_start:handoff_end]
            parsed = jr.parse_reviewer_handoff(isolated)
            assert parsed["base_sha"] == base_sha, runtime
            assert parsed["head_sha"] == head_sha, runtime
            assert parsed["test_output"] == str(tests_path), runtime
            assert parsed["builder_run_id"] == "b1b1b1b1b1b1", runtime


def test_worker_diff_review_prompt_survives_diff_containing_test_output_literal():
    # fixes cold review round 2 F9: a coincidental "test-output:" line INSIDE the embedded
    # diff text (e.g. a doc change describing this very handoff grammar) must not confuse
    # the fixed-block isolation technique the test above relies on -- proving the slicing
    # fix, not a worker code path (the worker never re-parses its own composed prompt;
    # only the dispatch-time stub is parsed, by preflight()).
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        (worktree / "changed.txt").write_text("  test-output: /some/unrelated/path\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "the change under review")
        base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        tests_dir = worktree / ".local" / "reports"
        tests_dir.mkdir(parents=True, exist_ok=True)
        # fixes F6: must match the default builder_run_id ("b1b1b1b1b1b1") this manifest
        # carries -- see the sibling test above for why.
        tests_path = tests_dir / "b1b1b1b1b1b1.tests.txt"
        tests_path.write_text("COMMAND: true\n\nEXIT: 0\n", encoding="utf-8")

        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, runtime="codex", base_sha=base_sha, head_sha=head_sha,
            tests_path=str(tests_path),
        )
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)
                captured["stdin_bytes"] = self.stdin_bytes

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
        )
        assert code == 0
        prompt = captured["stdin_bytes"].decode("utf-8")
        # the embedded diff really does carry the coincidental literal.
        assert "test-output: /some/unrelated/path" in prompt
        handoff_start = prompt.index(f"diff: {base_sha}..{head_sha}")
        test_output_line = f"  test-output: {tests_path}"
        handoff_end = prompt.index(test_output_line, handoff_start) + len(test_output_line)
        isolated = prompt[handoff_start:handoff_end]
        parsed = jr.parse_reviewer_handoff(isolated)
        assert parsed["head_sha"] == head_sha
        assert parsed["test_output"] == str(tests_path)


def test_worker_diff_review_finalizes_report_via_control_repo_paths():
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
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree), post=post,
            popen=FakePopen, allowlist_root=root.parent,
        )
        assert code == 0
        assert events[0]["payload"]["contract_status"] == "ok"
        assert events[0]["payload"]["verdict"] == "approve"
        report = root / ".local" / "reports" / "aaaabbbbcccc.md"  # FakePopen's fixed run_id in REPORT_TEXT
        assert report.is_file()  # lives under the CONTROL repo, not the worktree
        assert not (worktree / ".local" / "reports" / "aaaabbbbcccc.md").exists()


def test_worker_diff_review_updates_control_repo_status_md_with_approval_gate():
    # fixes Part 2 diff-review F6: no test proved the diff-worker branch's wiring into
    # `_update_status_md` actually sets `stage: review`/`gate: awaiting-approval` on an
    # approving diff review, preserves an existing status.md's unmanaged content, or
    # leaves the worktree's OWN `.jax-os/` (there is none -- the control repo's is the
    # only one ever written, spec §4.5) untouched.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        (worktree / "changed.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "change")
        base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        status_path = _write_status_md(root)
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, base_sha=base_sha, head_sha=head_sha,
        )

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=FakePopen, allowlist_root=root.parent,
        )
        assert code == 0
        text = status_path.read_text(encoding="utf-8")
        assert "stage: review" in text
        assert "gate: awaiting-approval" in text
        # unmanaged fields and the "## Residuals" section are preserved byte-for-byte.
        assert "project: Demo Project" in text
        assert "tmux: demo-session" in text
        assert "flag: main client" in text
        assert "## Residuals\n- an accepted risk\n- another one\n" in text
        # nothing is ever written inside the WORKTREE's own `.jax-os/` -- the by-product
        # always targets the control repo (spec §4.5).
        assert not (worktree / ".jax-os").exists()


def test_worker_diff_review_git_diff_failure_is_a_hard_failure_not_placeholder_text():
    # fixes cold review F6: a failed `git diff` must not become "(git diff failed)" prose
    # dispatched to the reviewer anyway -- it is an explicit worker failure instead, with
    # no run-finished event posted (surfaces as `dead` via `jaxflow status <run_id>`).
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, base_sha="f" * 40, head_sha="e" * 40,  # neither sha exists
        )
        events = []
        try:
            jaxflow.run_worker(
                str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
                post=lambda e: events.append(e) or {"ok": True}, popen=FakePopen,
                allowlist_root=root.parent,
            )
        except RuntimeError as exc:
            assert "git diff" in str(exc)
        else:
            raise AssertionError("git diff failure did not raise")
        assert events == []

# ---------------------------------------------------------------- slice c: preset parser

def _agents(tmp_path, text):
    """Writes an AGENTS.md into tmp_path and returns the repo path."""
    (tmp_path / "AGENTS.md").write_text(text, encoding="utf-8")
    return tmp_path


_TEMPLATE_DUAL_BRANCH_BLOCK = """## Deploy policy

**Preset: `dual-branch`** — small/internal project, low ceremony.

- Base branch: `staging` (default). Delivery target: `staging`, merged and pushed per
  the merge contract.

## Something else
"""


def test_delivery_target_falls_back_to_default_branch_with_no_preset_block(tmp_path):
    repo = _agents(tmp_path, "# AGENTS.md\n\nNo deploy policy here.\n")
    calls = []

    def fake_run(argv, cwd=None):
        calls.append(argv)
        if argv[:2] == ["git", "symbolic-ref"]:
            return _completed(0, "refs/remotes/origin/trunk\n")
        return _completed(1, "")

    target, no_preset, preset = jaxflow._resolve_delivery_target(repo, run=fake_run)
    assert (target, no_preset, preset) == ("trunk", True, None)
    assert calls, "the fallback must actually consult _default_branch"


def test_delivery_target_absent_agents_file_is_the_same_fallback(tmp_path):
    target, no_preset, preset = jaxflow._resolve_delivery_target(
        tmp_path, run=lambda argv, cwd=None: _completed(0, "refs/remotes/origin/main\n")
    )
    assert (target, no_preset, preset) == ("main", True, None)


def test_delivery_target_reads_the_dual_branch_preset_block(tmp_path):
    repo = _agents(tmp_path, _TEMPLATE_DUAL_BRANCH_BLOCK)

    def fake_run(argv, cwd=None):
        raise AssertionError("a resolved preset must never call _default_branch")

    assert jaxflow._resolve_delivery_target(repo, run=fake_run) == ("staging", False, "dual-branch")


@pytest.mark.parametrize("preset", ["single-branch", "single-branch-pr", "bubble-buildprint"])
def test_delivery_target_reads_the_other_presets(tmp_path, preset):
    repo = _agents(
        tmp_path,
        f"## Deploy policy\n\n**Preset: `{preset}`** — notes.\n\n"
        "- Base branch: `main`. Delivery target: `main`.\n",
    )
    target, no_preset, resolved = jaxflow._resolve_delivery_target(
        repo, run=lambda argv, cwd=None: _completed(1, "")
    )
    assert (target, no_preset, resolved) == ("main", False, preset)


def test_delivery_target_reads_the_dual_branch_pr_preset_block(tmp_path):
    repo = _agents(
        tmp_path,
        "**Preset: `dual-branch-pr`** — production project.\n\n"
        "- Base branch: `staging` (default). Delivery target: feature-branch PR against "
        "`staging`; executor and gate per `merge-contract.md`.\n"
        "- Production target: `main`.\n",
    )
    assert jaxflow._resolve_delivery_target(
        repo, run=lambda argv, cwd=None: _completed(1, ""),
    ) == ("staging", False, "dual-branch-pr")


@pytest.mark.parametrize("agents_text", [
    pytest.param(
        "**Preset: `dual-branch`** — a.\n\nDelivery target: `staging`.\n\n"
        "**Preset: `single-branch`** — b.\n\nDelivery target: `main`.\n",
        id="two-preset-lines"),
    pytest.param(
        "**Preset: `dual-branch`** — a.\n\n- Base branch: `staging`.\n",
        id="block-without-a-delivery-target"),
    # A `Delivery target:` that belongs to a LATER section must not be borrowed.
    pytest.param(
        "**Preset: `dual-branch`** — a.\n\n- Base branch: `staging`.\n\n"
        "## Another section\n\nDelivery target: `production`.\n",
        id="block-ends-at-the-next-heading"),
    pytest.param(
        "**Preset: `experimental`** — a.\n\nDelivery target: `main`.\n",
        id="unknown-preset-name"),
])
def test_delivery_target_refuses_with_preset_unknown(tmp_path, agents_text):
    repo = _agents(tmp_path, agents_text)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow._resolve_delivery_target(repo, run=lambda argv, cwd=None: _completed(1, ""))
    assert exc.value.code == "preset-unknown"


_FIVE_PRESETS = ["single-branch", "single-branch-pr", "dual-branch", "dual-branch-pr", "bubble-buildprint"]


@pytest.mark.parametrize("preset,target", [
    ("single-branch", "main"), ("single-branch-pr", "main"), ("dual-branch", "staging"),
    ("dual-branch-pr", "staging"), ("bubble-buildprint", "main"),
])
def test_delivery_target_resolves_each_of_the_five_presets(tmp_path, preset, target):
    repo = _agents(tmp_path, f"**Preset: `{preset}`** — a.\n\n- Delivery target: `{target}`.\n")
    assert jaxflow._resolve_delivery_target(
        repo, run=lambda argv, cwd=None: _completed(1, ""),
    ) == (target, False, preset)


@pytest.mark.parametrize("old", ["strict", "simple", "greenfield"])  # old-name-ok
def test_delivery_target_refuses_an_old_preset_name_and_lists_the_five(tmp_path, old):
    repo = _agents(tmp_path, f"**Preset: `{old}`** — a.\n\nDelivery target: `main`.\n")
    with pytest.raises(ji.Refusal) as exc:
        jaxflow._resolve_delivery_target(repo, run=lambda argv, cwd=None: _completed(1, ""))
    assert exc.value.code == "preset-unknown"
    assert all(name in exc.value.hint for name in _FIVE_PRESETS)


def test_preset_sets_partition_the_five_names():
    local, pr = set(jaxflow._LOCAL_PRESETS), set(jaxflow._PR_PRESETS)
    assert local | pr == set(_FIVE_PRESETS) and not local & pr
    assert set(jaxflow._RELEASE_PRESETS) == {"dual-branch-pr"}


# cold review e24e7fb33bc8 F1: the production target resolves for RELEASE presets only, so a stray
# `Production target:` line in a non-release block must still refuse.
_PRODUCTION_REFUSALS = [
    *(pytest.param(
        f"**Preset: `{preset}`** — a.\n\nDelivery target: `main`.\n- Production target: `main`.\n",
        id=f"{preset}-with-a-stray-production-line")
      for preset in ("single-branch", "single-branch-pr", "dual-branch", "bubble-buildprint")),
    *(pytest.param(
        f"**Preset: `{preset}`** — a.\n\nDelivery target: `main`.\n",
        id=f"{preset}-without-the-line")
      for preset in ("dual-branch", "single-branch", "single-branch-pr", "bubble-buildprint")),
    pytest.param(
        "**Preset: `dual-branch-pr`** — a.\n\nDelivery target: `staging`.\n",
        id="dual-branch-pr-missing-the-line"),
    pytest.param("# AGENTS.md\n\nNo deploy policy here.\n", id="no-preset-block-at-all"),
]


@pytest.mark.parametrize("agents_text", _PRODUCTION_REFUSALS)
def test_production_target_refuses(tmp_path, agents_text):
    repo = _agents(tmp_path, agents_text)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow._resolve_production_target(repo)
    assert exc.value.code == "production-target-unconfigured"


def test_production_target_reads_the_dual_branch_pr_block(tmp_path):
    repo = _agents(tmp_path, "**Preset: `dual-branch-pr`** — a.\n\nDelivery target: `staging`.\nProduction target: `main`.\n")
    assert jaxflow._resolve_production_target(repo) == "main"


def test_required_target_for_branch_is_delivery_for_a_feature_head(tmp_path):
    repo = _agents(tmp_path, "**Preset: `dual-branch-pr`** — a.\n\nDelivery target: `staging`.\nProduction target: `main`.\n")
    assert jaxflow._required_target_for_branch(
        repo, "feat/x", run=lambda argv, cwd=None: _completed(1, ""),
    ) == ("staging", "dual-branch-pr")


def test_required_target_for_branch_is_production_for_a_release_head(tmp_path):
    repo = _agents(tmp_path, "**Preset: `dual-branch-pr`** — a.\n\nDelivery target: `staging`.\nProduction target: `main`.\n")
    assert jaxflow._required_target_for_branch(
        repo, "release/2026-09-27-staging-promotion", run=lambda argv, cwd=None: _completed(1, ""),
    ) == ("main", "dual-branch-pr")


def test_required_target_for_branch_on_a_local_preset_refuses_a_release_head(tmp_path):
    repo = _agents(tmp_path, "**Preset: `dual-branch`** — a.\n\nDelivery target: `staging`.\n")
    # A release/* branch on a local preset has no Production target configured at all —
    # this only matters once `merge`/`pr open` actually route a release/* head through here
    # (Task 6/7); this test pins the helper's own behavior in isolation.
    with pytest.raises(ji.Refusal) as exc:
        jaxflow._required_target_for_branch(
            repo, "release/x", run=lambda argv, cwd=None: _completed(1, ""),
        )
    assert exc.value.code == "production-target-unconfigured"


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
    assert jaxflow.parse_threat_model(_THREAT_MODEL_BLOCK) == "internal-single-user"
    public_text = _THREAT_MODEL_BLOCK.replace("internal-single-user", "public-app")
    assert jaxflow.parse_threat_model(public_text) == "public-app"


def test_parse_threat_model_unknown_mode_is_none():
    assert jaxflow.parse_threat_model("## Threat model\nmode: something-else\n") is None


def test_parse_threat_model_missing_section_is_none():
    assert jaxflow.parse_threat_model("# AGENTS.md\n\nNo threat model here.\n") is None
    assert jaxflow.parse_threat_model("") is None


def test_parse_threat_model_heading_with_trailing_comment_case_insensitive():
    text = "## THREAT MODEL  <!-- CHOOSE ONE MODE -->\nmode: public-app\n"
    assert jaxflow.parse_threat_model(text) == "public-app"


def test_parse_threat_model_mode_line_surrounding_spaces():
    text = "## Threat model\n   mode:    internal-single-user   \n"
    assert jaxflow.parse_threat_model(text) == "internal-single-user"


def test_parse_threat_model_stops_at_the_next_heading():
    # A `mode:` line belonging to a LATER section must not be borrowed (same idea as
    # `_resolve_delivery_target`'s Deploy policy block boundary).
    text = "## Threat model\n\n## Another section\nmode: public-app\n"
    assert jaxflow.parse_threat_model(text) is None


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
    assert jaxflow._threat_model_for(repo) == "internal-single-user"


def test_threat_model_for_missing_agents_md_prints_note_and_returns_none(tmp_path, capsys):
    assert jaxflow._threat_model_for(tmp_path) is None
    assert capsys.readouterr().err.strip() == _NO_THREAT_MODEL_NOTE


def test_threat_model_for_no_valid_mode_prints_note_and_returns_none(tmp_path, capsys):
    repo = _agents(tmp_path, "## Threat model\nmode: something-else\n")
    assert jaxflow._threat_model_for(repo) is None
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
        run_id = jaxflow.dispatch_review(
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
        run_id = jaxflow.dispatch_review(
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
        run_id = jaxflow.dispatch_diff_review(
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

        code = jaxflow.run_worker(
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

        jaxflow.run_worker(
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


# ------------------------------------------------------------------ slice d: PR delivery (github helpers)

# MOA-502 Decision 2: every gh call site refuses `github-integration-disabled` before any
# git/GitHub mutation when integrations.github is off. The pre-gate probes (each command's
# own toplevel/branch reads) already ran — the assertion is that no gh/push call followed.

def test_pr_open_refuses_github_integration_disabled_before_any_gh_call(tmp_path, monkeypatch):
    import general_settings
    monkeypatch.setattr(general_settings, "read_settings",
                         lambda: {"ok": True, "data": {"integrations": {"github": False}}})
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_pr_open(_pr_open_args(), run=fake_run, env=_merge_env())
    assert exc.value.code == "github-integration-disabled"
    assert not any(c[:2] == ["gh", "pr"] or c[:2] == ["git", "push"] for c in calls), calls


def test_merge_pr_refuses_github_integration_disabled_before_any_gh_call(tmp_path, monkeypatch):
    import general_settings
    monkeypatch.setattr(general_settings, "read_settings",
                         lambda: {"ok": True, "data": {"integrations": {"github": False}}})
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(target="staging"), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "github-integration-disabled"
    assert not any(c[:2] == ["gh", "pr"] or c[:2] == ["git", "push"] for c in calls), calls


def test_release_refuses_github_integration_disabled_before_any_gh_call(tmp_path, monkeypatch):
    import general_settings
    monkeypatch.setattr(general_settings, "read_settings",
                         lambda: {"ok": True, "data": {"integrations": {"github": False}}})
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_release(SimpleNamespace(from_caller="claude"), run=fake_run, env=_merge_env())
    assert exc.value.code == "github-integration-disabled"
    assert not any(c[:2] == ["gh", "pr"] or c[:2] == ["git", "push"] for c in calls), calls


@pytest.mark.parametrize("preset,target,pr_refused,release_refused", [
    ("single-branch", "main", True, True),
    ("single-branch-pr", "main", False, True),
    ("dual-branch", "staging", True, True),
    ("dual-branch-pr", "staging", False, False),
    ("bubble-buildprint", "main", True, True),
])
def test_preset_matrix_pr_open_and_release_refusals(
        tmp_path, monkeypatch, preset, target, pr_refused, release_refused):
    # D2/D4: single-branch refuses both; single-branch-pr refuses only `release`;
    # dual-branch-pr refuses neither (its release path is exercised by the release tests below).
    monkeypatch.chdir(tmp_path)
    agents = f"**Preset: `{preset}`** — a.\n\nDelivery target: `{target}`.\n"
    if preset == "dual-branch-pr":
        agents += "Production target: `main`.\n"
    fake_run, calls = _merge_runner(tmp_path, agents=agents, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "check-ref-format",): _completed(0, ""),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_pr_open(
            _pr_open_args(target="not-the-target"), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == ("preset-not-pr" if pr_refused else "target-mismatch")
    if release_refused:
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_release(
                SimpleNamespace(from_caller="claude"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
        assert exc.value.code == "preset-no-release"
    assert not any(c[0] == "gh" or c[:2] == ["git", "push"] for c in calls)
    assert not any(c[:2] in (["git", "fetch"], ["git", "update-ref"]) for c in calls)


def test_github_repo_slug_parses_ssh_and_https_remotes(tmp_path):
    for url, expected in [
        ("git@github.com:acme/route-converter-se.git", "acme/route-converter-se"),
        ("https://github.com/acme/route-converter-se.git", "acme/route-converter-se"),
        ("https://github.com/acme/route-converter-se", "acme/route-converter-se"),
    ]:
        def run(argv, cwd=None, url=url):
            if argv[:3] == ["git", "remote", "get-url"]:
                return _completed(0, f"{url}\n")
            return _completed(1, "")
        assert jaxflow._github_repo_slug(run, tmp_path) == expected


def test_github_repo_slug_refuses_when_origin_is_unreadable_or_not_github(tmp_path):
    for result in (_completed(1, ""), _completed(0, "https://gitlab.com/acme/x.git\n"),
                   # cold review 75e934eacdca F5: a lookalike host must not match on the
                   # "github.com" substring alone -- _GH_REMOTE_RE is anchored to the real host.
                   _completed(0, "https://evilgithub.com/acme/x.git\n")):
        run = lambda argv, cwd=None, result=result: result
        with pytest.raises(ji.Refusal) as exc:
            jaxflow._github_repo_slug(run, tmp_path)
        assert exc.value.code == "github-unreachable"


_GH_PR_VIEW_BODY = {
    "number": 7, "url": "https://github.com/acme/x/pull/7", "state": "OPEN",
    "headRefOid": "a" * 40, "headRefName": "feat/x", "baseRefName": "staging",
    "mergeable": "MERGEABLE", "mergeCommit": None,
}


def test_gh_pr_view_parses_json_and_passes_the_exact_argv():
    calls = []
    def run(argv, cwd=None):
        calls.append(argv)
        return _completed(0, json.dumps(_GH_PR_VIEW_BODY))
    view = jaxflow._gh_pr_view(run, Path("/repo"), "acme/x", 7)
    assert view["state"] == "OPEN"
    assert calls == [["gh", "pr", "view", "7", "--repo", "acme/x", "--json",
                       "number,url,state,headRefOid,headRefName,baseRefName,mergeable,mergeCommit"]]


def test_gh_pr_view_refuses_github_unreachable_on_failure_or_bad_json(tmp_path):
    for result in (_completed(1, ""), _completed(0, "not json")):
        run = lambda argv, cwd=None, result=result: result
        with pytest.raises(ji.Refusal) as exc:
            jaxflow._gh_pr_view(run, tmp_path, "acme/x", 7)
        assert exc.value.code == "github-unreachable"


def test_resolve_pr_number_zero_one_and_ambiguous_matches(tmp_path):
    def run_for(matches):
        return lambda argv, cwd=None: _completed(0, json.dumps(matches))
    assert jaxflow._resolve_pr_number(run_for([]), tmp_path, "acme/x", "feat/x", "staging") is None
    assert jaxflow._resolve_pr_number(run_for([{"number": 9}]), tmp_path, "acme/x", "feat/x", "staging") == 9
    with pytest.raises(ji.Refusal) as exc:
        jaxflow._resolve_pr_number(run_for([{"number": 9}, {"number": 10}]), tmp_path, "acme/x", "feat/x", "staging")
    assert exc.value.code == "pr-ambiguous"


def test_resolve_pr_number_refuses_github_unreachable_on_failure_or_bad_json(tmp_path):
    for result in (_completed(1, ""), _completed(0, "not json")):
        run = lambda argv, cwd=None, result=result: result
        with pytest.raises(ji.Refusal) as exc:
            jaxflow._resolve_pr_number(run, tmp_path, "acme/x", "feat/x", "staging")
        assert exc.value.code == "github-unreachable"


def test_find_recorded_pr_returns_the_latest_matching_branch_row(tmp_path):
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, None, "demo", "lead", "pr-opened",
            {"branch": "feat/x", "sha": "a" * 40, "pr_number": 7}, ts="2026-09-27T10:00:00")
    _insert(con, None, "demo", "lead", "pr-opened",
            {"branch": "feat/x", "sha": "b" * 40, "pr_number": 7}, ts="2026-09-27T11:00:00")
    _insert(con, None, "demo", "lead", "pr-opened",
            {"branch": "feat/y", "sha": "c" * 40, "pr_number": 9}, ts="2026-09-27T12:00:00")
    con.close()
    found = jaxflow._find_recorded_pr("demo", "feat/x", db_path=db)
    assert found == {"branch": "feat/x", "sha": "b" * 40, "pr_number": 7}


def test_find_recorded_pr_returns_none_with_no_matching_row(tmp_path):
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    assert jaxflow._find_recorded_pr("demo", "feat/x", db_path=db) is None


def test_find_recorded_pr_returns_none_with_no_db_at_all(tmp_path):
    assert jaxflow._find_recorded_pr("demo", "feat/x", db_path=tmp_path / "nope.db") is None


def _pr_open_args(branch="feat/x", sha="a" * 40, target="staging", title="Ship it",
                   body_file=None, from_caller="claude"):
    return SimpleNamespace(branch=branch, sha=sha, target=target, title=title,
                            body_file=body_file, from_caller=from_caller)


_DUAL_PR_AGENTS = (
    "**Preset: `dual-branch-pr`** — production project.\n\n"
    "- Base branch: `staging` (default). Delivery target: feature-branch PR against "
    "`staging`; executor and gate per `merge-contract.md`.\n"
    "- Production target: `main`.\n"
)

_SINGLE_PR_AGENTS = (
    "**Preset: `single-branch-pr`** — GitHub Flow.\n\n"
    "- Base branch: `main`. Delivery target: `main`.\n"
)


def test_pr_open_happy_path_pushes_creates_and_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    events = []
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("git", "push", "origin", f"{'a' * 40}:refs/heads/feat/x"): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, "[]"),
        ("gh", "pr", "create"): _completed(0, "https://github.com/acme/x/pull/7\n"),
    })
    with monkeypatch.context() as m:
        # `jr.DB_PATH` alone is enough: `_find_recorded_pr`'s own `db_path = db_path or
        # jr.DB_PATH` fallback already resolves to this test's db (no need to also patch
        # `_open_ro` -- a stray earlier attempt to do that as `jaxflow._open_ro = lambda path:
        # jaxflow._open_ro(db)` was self-referential and would recurse forever the moment it
        # actually ran).
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow.cmd_pr_open(
            _pr_open_args(), run=fake_run, post=_ledger_post(db, events),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result == {"number": 7, "url": "https://github.com/acme/x/pull/7", "repo_slug": "acme/x"}
    assert events[0]["type"] == "pr-opened"
    assert events[0]["payload"] == {
        "repo": "acme/x", "branch": "feat/x", "sha": "a" * 40, "base": "staging",
        "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature",
    }
    assert ["git", "push", "origin", f"{'a' * 40}:refs/heads/feat/x"] in calls
    assert ["gh", "pr", "create", "--repo", "acme/x", "--head", "feat/x", "--base", "staging",
            "--title", "Ship it"] in calls


def test_pr_open_resumes_at_the_same_sha_with_no_mutation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    # project MUST be the real slug cmd_pr_open computes (slugify_project(repo.name), where
    # repo.name is tmp_path's own pytest-generated basename) — a hardcoded "demo" would never
    # match and _find_recorded_pr would silently see nothing (same convention as
    # scripts/test_jaxflow.py:13158's existing resume-ineligible fixture).
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened",
            {"repo": "acme/x", "branch": "feat/x", "sha": "a" * 40, "base": "staging",
             "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature"})
    con.close()
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({
            **_GH_PR_VIEW_BODY, "headRefOid": "a" * 40, "baseRefName": "staging",
            "url": "https://github.com/acme/x/pull/7",
        })),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow.cmd_pr_open(
            _pr_open_args(), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["number"] == 7
    for banned in ("git push", "gh pr create"):
        assert not any(" ".join(c).startswith(banned) for c in calls), f"{banned} ran on a no-op reuse"


def test_pr_open_fast_forward_fix_pushes_and_refreshes_the_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened",
            {"repo": "acme/x", "branch": "feat/x", "sha": "a" * 40, "base": "staging",
             "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature"})
    con.close()
    new_sha = "b" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", f"feat/x^{{commit}}"): _completed(0, f"{new_sha}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({
            **_GH_PR_VIEW_BODY, "headRefOid": "a" * 40, "baseRefName": "staging",
            "url": "https://github.com/acme/x/pull/7",
        })),
        ("git", "merge-base", "--is-ancestor", "a" * 40, new_sha): _completed(0, ""),
        ("git", "push", "origin", f"{new_sha}:refs/heads/feat/x"): _completed(0, ""),
    })
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        jaxflow.cmd_pr_open(
            _pr_open_args(sha=new_sha), run=fake_run, post=_ledger_post(db, events),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert ["git", "push", "origin", f"{new_sha}:refs/heads/feat/x"] in calls
    assert events[0]["payload"]["sha"] == new_sha


def test_pr_open_refuses_a_diverged_remote_head(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened",
            {"repo": "acme/x", "branch": "feat/x", "sha": "a" * 40, "base": "staging",
             "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature"})
    con.close()
    other_sha = "c" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", f"feat/x^{{commit}}"): _completed(0, f"{other_sha}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({
            **_GH_PR_VIEW_BODY, "headRefOid": "a" * 40, "baseRefName": "staging",
        })),
        ("git", "merge-base", "--is-ancestor", "a" * 40, other_sha): _completed(1, ""),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_pr_open(
                _pr_open_args(sha=other_sha), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-remote-diverged"
    assert not any(c[:2] == ["git", "push"] for c in calls)


def test_pr_open_reports_a_closed_unmerged_pr_never_reopening_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened",
            {"repo": "acme/x", "branch": "feat/x", "sha": "a" * 40, "base": "staging",
             "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature"})
    con.close()
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({**_GH_PR_VIEW_BODY, "state": "CLOSED"})),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_pr_open(
                _pr_open_args(), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-closed"
    assert not any(c[:3] == ["gh", "pr", "create"] for c in calls)


def test_pr_open_refuses_a_base_mismatch_on_a_recorded_pr_before_any_push_or_event(tmp_path, monkeypatch):
    # cold review 75e934eacdca F4: reusing a recorded PR whose ACTUAL GitHub base no longer
    # matches the approved --target must refuse before any push/event, even though --target
    # itself already matched the configured Delivery target (a separate, earlier check).
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened",
            {"repo": "acme/x", "branch": "feat/x", "sha": "a" * 40, "base": "staging",
             "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature"})
    con.close()
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({**_GH_PR_VIEW_BODY, "baseRefName": "wrong-base"})),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_pr_open(
                _pr_open_args(), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-identity-mismatch"
    assert not any(c[:2] == ["git", "push"] or c[:3] == ["gh", "pr", "create"] for c in calls)


def test_pr_open_refuses_a_head_branch_mismatch_on_a_recorded_pr_before_any_push_or_event(tmp_path, monkeypatch):
    # cold review 24597072c8ac F2: same guard as the base-mismatch test above, for the head
    # branch -- a recorded/resolved PR number whose actual GitHub head branch isn't `branch`
    # must refuse before any push/event too.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened",
            {"repo": "acme/x", "branch": "feat/x", "sha": "a" * 40, "base": "staging",
             "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature"})
    con.close()
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({**_GH_PR_VIEW_BODY, "headRefName": "feat/other"})),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_pr_open(
                _pr_open_args(), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-identity-mismatch"
    assert not any(c[:2] == ["git", "push"] or c[:3] == ["gh", "pr", "create"] for c in calls)


@pytest.mark.parametrize("agents,target,pushed_or_opened", [
    pytest.param(
        _DUAL_PR_AGENTS, "main",
        lambda c: c[:2] == ["git", "push"] or c[:3] == ["gh", "pr", "create"],
        id="feature-head-against-the-wrong-base"),
    pytest.param(
        _SINGLE_PR_AGENTS, "staging",
        lambda c: c[0] == "gh" or c[:2] == ["git", "push"],
        id="single-branch-pr-refuses-a-staging-target"),
])
def test_pr_open_refuses_target_mismatch(tmp_path, monkeypatch, agents, target, pushed_or_opened):
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(tmp_path, agents=agents, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "check-ref-format",): _completed(0, ""),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_pr_open(
            _pr_open_args(target=target), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "target-mismatch"
    assert not any(pushed_or_opened(c) for c in calls)


def test_pr_open_argparse_wiring_end_to_end(monkeypatch, tmp_path):
    seen = {}
    def fake_cmd_pr_open(args, **kwargs):
        seen["branch"] = args.branch
        seen["title"] = args.title
        return {"number": 1, "url": "https://github.com/acme/x/pull/1", "repo_slug": "acme/x"}
    monkeypatch.setattr(jaxflow, "cmd_pr_open", fake_cmd_pr_open)
    argv = ["pr", "open", "feat/x", "--sha", "a" * 40, "--target", "staging", "--title", "Ship it"]
    assert jaxflow.main(argv) == jaxflow_common.OK
    assert seen == {"branch": "feat/x", "title": "Ship it"}


def test_pr_open_rejects_a_blank_title(tmp_path):
    with pytest.raises(SystemExit):
        jaxflow.parse_args(["pr", "open", "feat/x", "--sha", "a" * 40, "--target", "staging", "--title", "   "])


def test_pr_open_refuses_a_local_preset_before_any_push_or_gh_call(tmp_path, monkeypatch):
    # cold review e441aa1770e3 F1: pr-open is PR-preset-only -- a single-branch/dual-branch/bubble-buildprint/no-preset repo must refuse before any push or gh call, not silently deliver.
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(
        tmp_path, agents="**Preset: `dual-branch`** — a.\n\nDelivery target: `staging`.\n", script={
            ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
            ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
            ("git", "check-ref-format",): _completed(0, ""),
        })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_pr_open(
            _pr_open_args(), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "preset-not-pr"
    assert not any(c[0] == "gh" for c in calls)
    assert not any(c[:2] == ["git", "push"] for c in calls)


def test_pr_open_refuses_a_no_preset_repo_before_any_push_or_gh_call(tmp_path, monkeypatch):
    # Same refusal for a repo with no Deploy policy block at all -- the common case (most repos
    # on this machine predate the preset system) must never be treated as implicitly a PR preset.
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(
        tmp_path, agents="# AGENTS.md\n\nNo deploy policy here.\n", script={
            ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
            ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
            ("git", "check-ref-format",): _completed(0, ""),
        })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_pr_open(
            _pr_open_args(), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "preset-not-pr"
    assert not any(c[0] == "gh" for c in calls)
    assert not any(c[:2] == ["git", "push"] for c in calls)


@pytest.mark.parametrize("agents", [
    "**Preset: `single-branch`** — a.\n\nDelivery target: `main`.\n",
    "**Preset: `dual-branch`** — a.\n\nDelivery target: `staging`.\n",
    "**Preset: `bubble-buildprint`** — a.\n\nDelivery target: `main`.\n",
    "# AGENTS.md\n\nNo deploy policy here.\n",
])
def test_pr_open_refuses_preset_not_pr_for_a_release_head_before_target_resolution(
        tmp_path, monkeypatch, agents):
    # cold review 9f7f7290c510 F2: the preset check precedes `_required_target_for_branch`, which
    # would otherwise raise production-target-unconfigured for a release/* head first.
    monkeypatch.chdir(tmp_path)
    head = "release/2026-10-07-staging-promotion"
    fake_run, calls = _merge_runner(tmp_path, agents=agents, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", f"{head}^{{commit}}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "check-ref-format",): _completed(0, ""),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_pr_open(
            _pr_open_args(branch=head, target="main"), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "preset-not-pr"
    assert not any(c[0] == "gh" or c[:2] == ["git", "push"] for c in calls)


def test_pr_open_single_branch_pr_targets_main_and_records_the_pr(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    events = []
    fake_run, calls = _merge_runner(tmp_path, agents=_SINGLE_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("git", "push", "origin", f"{'a' * 40}:refs/heads/feat/x"): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, "[]"),
        ("gh", "pr", "create"): _completed(0, "https://github.com/acme/x/pull/7\n"),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow.cmd_pr_open(
            _pr_open_args(target="main"), run=fake_run, post=_ledger_post(db, events),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["number"] == 7
    assert events[0]["type"] == "pr-opened" and events[0]["payload"]["base"] == "main"
    assert ["gh", "pr", "create", "--repo", "acme/x", "--head", "feat/x", "--base", "main",
            "--title", "Ship it"] in calls


def test_pr_open_single_branch_pr_refuses_a_release_head_even_with_a_production_target(
        tmp_path, monkeypatch):
    # cold review e24e7fb33bc8 F1. Picked REFUSE over "treat it as a feature targeting main":
    # spec D3 says a release/* head on single-branch-pr refuses production-target-unconfigured, and
    # the preset has no release step, so the branch name must never be read as a feature head.
    monkeypatch.chdir(tmp_path)
    head = "release/2026-10-07-staging-promotion"
    fake_run, calls = _merge_runner(
        tmp_path, agents=_SINGLE_PR_AGENTS + "- Production target: `main`.\n", script={
            ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
            ("git", "rev-parse", "--verify", f"{head}^{{commit}}"): _completed(0, f"{'a' * 40}\n"),
            ("git", "check-ref-format",): _completed(0, ""),
        })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_pr_open(
            _pr_open_args(branch=head, target="main"), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "production-target-unconfigured"
    assert not any(c[0] == "gh" or c[:2] == ["git", "push"] for c in calls)


_GH_PR_VIEW_MERGED = {
    "number": 7, "url": "https://github.com/acme/x/pull/7", "state": "MERGED",
    "headRefOid": "a" * 40, "headRefName": "feat/x", "baseRefName": "staging",
    "mergeable": "MERGEABLE", "mergeCommit": {"oid": "d" * 40},
}


def _pr_merge_setup(tmp_path, db, *, pr_state="OPEN", mergeable="MERGEABLE", target="staging",
                         branch="feat/x", sha="a" * 40, worktree_registered=True,
                         merge_result=None):
    # project MUST be the real slug cmd_merge computes (slugify_project(repo.name)) — reuses
    # the existing `_worktree_path` helper (scripts/test_jaxflow.py:12971-12973) for the exact
    # same reason every OTHER merge test does: a literal like "demo" would never match.
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened",
            {"repo": "acme/x", "branch": branch, "sha": sha, "base": target,
             "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature"})
    con.close()
    worktree = _worktree_path(tmp_path, branch)
    if worktree_registered:
        # A feature branch's PR-path merge runs --checks IN this worktree (decision 7), and
        # `_cmd_merge_pr` gates on `checks_dir.is_dir()` being a REAL directory before it
        # ever calls `_is_registered_worktree` -- unlike the fully-mocked release path (which
        # never reaches this branch's `worktree`), that check is not fakeable through `run`.
        # Every existing local-preset merge test that needs this same gate creates the directory
        # too (e.g. `_worktree_path(tmp_path).mkdir(parents=True)` at :13066/:13154/:13178/...).
        worktree.mkdir(parents=True, exist_ok=True)
    view = {"number": 7, "url": "https://github.com/acme/x/pull/7", "state": pr_state,
            "headRefOid": sha, "headRefName": branch, "baseRefName": target,
            "mergeable": mergeable, "mergeCommit": None}
    script = {
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", f"{branch}^{{commit}}"): _completed(0, f"{sha}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps(view)),
        ("git", "rev-parse", "HEAD"): _completed(0, f"{sha}\n"),
        ("git", "status", "--porcelain", "--untracked-files=no"): _completed(0, ""),
        ("/bin/bash", "-lc"): _completed(0, ""),
        ("gh", "pr", "merge",): merge_result or _completed(0, ""),
        ("git", "worktree", "list", "--porcelain"): _completed(
            0, f"worktree {worktree}\nbranch refs/heads/{branch}\n" if worktree_registered else ""),
        ("git", "fetch", "origin", target): _completed(0, ""),
        ("git", "switch", target): _completed(0, ""),
        ("git", "merge", "--ff-only"): _completed(0, ""),
        ("git", "worktree", "remove", str(worktree)): _completed(0, ""),
        ("git", "branch", "-d", branch): _completed(0, ""),
    }
    return script, worktree


def test_merge_pr_happy_path_verifies_checks_merges_and_syncs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, worktree = _pr_merge_setup(tmp_path, db)
    view_calls = {"n": 0}
    def gh_view_then_merged(argv, cwd=None):
        view_calls["n"] += 1
        if view_calls["n"] == 1:
            return _completed(0, json.dumps({
                "number": 7, "url": "https://github.com/acme/x/pull/7", "state": "OPEN",
                "headRefOid": "a" * 40, "headRefName": "feat/x", "baseRefName": "staging",
                "mergeable": "MERGEABLE", "mergeCommit": None,
            }))
        return _completed(0, json.dumps(_GH_PR_VIEW_MERGED))
    # `run()` intercepts every "gh pr view" call itself (stateful: OPEN first, MERGED on the
    # post-merge re-read) — `_pr_merge_setup`'s own static "gh pr view" script entry is
    # never reached for THIS test, only the entries after it (checks, merge, cleanup).
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    def run(argv, cwd=None):
        if tuple(argv[:3]) == ("gh", "pr", "view"):
            calls.append(list(argv))
            return gh_view_then_merged(argv, cwd)
        return fake_run(argv, cwd)
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow.cmd_merge(
            _MergeArgs(target="staging"), run=run, post=_ledger_post(db, events),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result == jaxflow_common.OK
    merge_call = next(c for c in calls if c[:3] == ["gh", "pr", "merge"])
    assert merge_call == ["gh", "pr", "merge", "7", "--repo", "acme/x", "--merge",
                           "--match-head-commit", "a" * 40, "--subject", "feat: Phase X (merge feat/x)"]
    assert events[-1]["type"] == "merge-approved"
    assert events[-1]["payload"]["merge_sha"] == "d" * 40
    assert events[-1]["payload"]["pr_number"] == 7


def test_merge_pr_github_merge_refused_leaves_nothing_recorded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(
        tmp_path, db, merge_result=_completed(1, "", "required status check \"ci\" is expected"))
    script[("gh", "pr", "checks",)] = _completed(0, "ci\tpending\thttps://x\n")
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_ledger_post(db, events),
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "github-merge-refused"
    assert "required status check" in exc.value.hint
    assert events == []


def test_merge_pr_merge_queue_required(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(
        tmp_path, db, merge_result=_completed(1, "", "Pull request is in a merge queue"))
    script[("gh", "pr", "checks",)] = _completed(0, "")
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "merge-queue-required"


def test_merge_pr_reports_a_success_that_did_not_actually_merge(tmp_path, monkeypatch):
    # cold review 24597072c8ac F1 (downgraded to LOW -- no project uses a merge queue --
    # relabelled): `gh pr merge` can exit 0 without the PR actually merging (a merge queue or
    # auto-merge accepted the request asynchronously). The post-merge re-read still showing
    # OPEN must refuse the specific `merge-not-completed`, not the generic `github-unreachable`,
    # and nothing may be recorded as merged.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    still_open = {"number": 7, "url": "https://github.com/acme/x/pull/7", "state": "OPEN",
                  "headRefOid": "a" * 40, "headRefName": "feat/x", "baseRefName": "staging",
                  "mergeable": "MERGEABLE", "mergeCommit": None}
    view_calls = {"n": 0}
    def gh_view(argv, cwd=None):
        view_calls["n"] += 1
        return _completed(0, json.dumps(still_open))
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    def run(argv, cwd=None):
        if tuple(argv[:3]) == ("gh", "pr", "view"):
            calls.append(list(argv))
            return gh_view(argv, cwd)
        return fake_run(argv, cwd)
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_merge(
                _MergeArgs(target="staging"), run=run, post=_ledger_post(db, events),
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "merge-not-completed"
    assert events == []


def test_merge_pr_identity_mismatch_on_base(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    view = {"number": 7, "url": "https://github.com/acme/x/pull/7", "state": "OPEN",
            "headRefOid": "a" * 40, "headRefName": "feat/x", "baseRefName": "wrong-base",
            "mergeable": "MERGEABLE", "mergeCommit": None}
    script[("gh", "pr", "view")] = _completed(0, json.dumps(view))
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-identity-mismatch"
    _assert_no_mutation(calls)


def test_merge_pr_refuses_a_head_branch_mismatch(tmp_path, monkeypatch):
    # cold review 24597072c8ac F2: the resolved PR's actual head branch must be verified
    # against args.branch, not just its head sha -- a same-sha PR whose recorded/resolved
    # number actually points at a different branch must refuse before any GitHub mutation.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    view = {"number": 7, "url": "https://github.com/acme/x/pull/7", "state": "OPEN",
            "headRefOid": "a" * 40, "headRefName": "feat/other", "baseRefName": "staging",
            "mergeable": "MERGEABLE", "mergeCommit": None}
    script[("gh", "pr", "view")] = _completed(0, json.dumps(view))
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-identity-mismatch"
    _assert_no_mutation(calls)
    assert not any(c[:3] == ["gh", "pr", "merge"] for c in calls)


def test_merge_pr_head_moved_since_approval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    view = {"number": 7, "url": "https://github.com/acme/x/pull/7", "state": "OPEN",
            "headRefOid": "e" * 40, "headRefName": "feat/x", "baseRefName": "staging",
            "mergeable": "MERGEABLE", "mergeCommit": None}
    script[("gh", "pr", "view")] = _completed(0, json.dumps(view))
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-head-moved"


def test_merge_pr_mergeability_unknown_after_bounded_retries_mutates_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db, mergeable="UNKNOWN")
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "mergeability-unknown"
    assert not any(c[:3] == ["gh", "pr", "merge"] for c in calls)


def test_merge_pr_rerun_after_a_real_github_merge_is_recording_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, worktree = _pr_merge_setup(tmp_path, db)
    script[("gh", "pr", "view")] = _completed(0, json.dumps(_GH_PR_VIEW_MERGED))
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow.cmd_merge(
            _MergeArgs(target="staging"), run=fake_run, post=_ledger_post(db, events),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result == jaxflow_common.OK
    assert not any(c[:3] == ["gh", "pr", "merge"] for c in calls)
    assert not any(c[:2] == ["/bin/bash", "-lc"] for c in calls)
    assert events[-1]["payload"]["merge_sha"] == "d" * 40


def test_merge_pr_already_merged_head_mismatch_refuses_pr_head_moved(tmp_path, monkeypatch):
    # cold review e441aa1770e3 F2: the already-merged recovery path must verify GitHub's
    # headRefOid against the approved sha before recording -- a PR merged with extra commits
    # (pushed and merged outside the approval flow) must never be recorded as an approval of a
    # DIFFERENT sha. Base is covered separately by
    # test_merge_pr_identity_mismatch_on_base (that check runs unconditionally, before
    # state is even inspected).
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    script[("gh", "pr", "view")] = _completed(0, json.dumps({**_GH_PR_VIEW_MERGED, "headRefOid": "e" * 40}))
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-head-moved"
    _assert_no_mutation(calls)


@pytest.mark.parametrize("branch,target,expected", [
    ("release/2026-09-27-staging-promotion", "main", True),
    ("release/2026-09-27-staging-promotion", "staging", False),
    ("feat/x", "main", False),
])
def test_merge_pr_target_mutual_exclusion_by_head_shape(tmp_path, monkeypatch, branch, target, expected):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    sha = "a" * 40
    if expected:
        script, checks_dir = _pr_merge_setup(
            tmp_path, db, branch=branch, sha=sha, target=target, worktree_registered=False)
        script[("git", "worktree", "add", "--detach")] = _completed(0, "")
        # cold review 75e934eacdca F2: cleanup no longer passes --force (plain removal).
        script[("git", "worktree", "remove", str(checks_dir))] = _completed(0, "")
    else:
        script, _ = _pr_merge_setup(tmp_path, db, branch=branch, sha=sha, target="staging")
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    # Plan deviation: a successful release merge needs the SAME stateful post-merge re-read
    # `test_merge_pr_happy_path...` already overrides for the feature case -- a static
    # OPEN view would make the implementation's own post-merge verification refuse
    # github-unreachable (the plan's fixture does not wire this; see report).
    if expected:
        view_calls = {"n": 0}
        def run(argv, cwd=None):
            if tuple(argv[:3]) == ("gh", "pr", "view"):
                calls.append(list(argv))
                view_calls["n"] += 1
                if view_calls["n"] == 1:
                    return _completed(0, json.dumps({
                        "number": 7, "url": "https://github.com/acme/x/pull/7", "state": "OPEN",
                        "headRefOid": sha, "headRefName": branch, "baseRefName": target,
                        "mergeable": "MERGEABLE", "mergeCommit": None,
                    }))
                return _completed(0, json.dumps({
                    **_GH_PR_VIEW_MERGED, "baseRefName": target, "headRefOid": sha,
                    "headRefName": branch,
                }))
            return fake_run(argv, cwd)
    else:
        run = fake_run
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        if expected:
            result = jaxflow.cmd_merge(
                _MergeArgs(branch=branch, sha=sha, target=target), run=run,
                post=_ledger_post(db, events), env=_merge_env(), now=_fixed_now,
                allowlist_root=tmp_path.parent,
            )
            assert result == jaxflow_common.OK
        else:
            with pytest.raises(ji.Refusal) as exc:
                jaxflow.cmd_merge(
                    _MergeArgs(branch=branch, sha=sha, target=target), run=run,
                    post=_never_post, env=_merge_env(), now=_fixed_now,
                    allowlist_root=tmp_path.parent,
                )
            assert exc.value.code == "target-mismatch"


def test_merge_single_branch_pr_feature_head_takes_the_pr_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    sha = "a" * 40
    script, _ = _pr_merge_setup(tmp_path, db, branch="feat/x", sha=sha, target="main")
    fake_run, calls = _merge_runner(tmp_path, agents=_SINGLE_PR_AGENTS, script=script)
    # Same post-merge re-read the dual-branch-pr happy path overrides: a static OPEN view
    # makes the implementation refuse merge-not-completed. The plan fixture does not wire this.
    view_calls = {"n": 0}
    def run(argv, cwd=None):
        if tuple(argv[:3]) == ("gh", "pr", "view"):
            calls.append(list(argv))
            view_calls["n"] += 1
            body = {
                "number": 7, "url": "https://github.com/acme/x/pull/7",
                "headRefOid": sha, "headRefName": "feat/x", "baseRefName": "main",
                "mergeable": "MERGEABLE",
            }
            if view_calls["n"] == 1:
                return _completed(0, json.dumps({**body, "state": "OPEN", "mergeCommit": None}))
            return _completed(0, json.dumps({**body, "state": "MERGED", "mergeCommit": {"oid": "d" * 40}}))
        return fake_run(argv, cwd)
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow.cmd_merge(
            _MergeArgs(branch="feat/x", sha=sha, target="main"), run=run,
            post=_ledger_post(db, events), env=_merge_env(), now=_fixed_now,
            allowlist_root=tmp_path.parent,
        )
    assert result == jaxflow_common.OK
    assert any(c[:3] == ["gh", "pr", "merge"] for c in calls)
    # Local delivery is `git merge --no-ff`. The PR path's best-effort sync (decision 10)
    # does call `git merge --ff-only origin/<target>`, so that prefix is not the local path.
    assert not any(c[:4] == ["git", "merge", "--no-ff", "--no-commit"] for c in calls)


def test_merge_single_branch_pr_refuses_a_release_head(tmp_path, monkeypatch):
    # spec D3: single-branch-pr has no release path, even with a Production target line.
    monkeypatch.chdir(tmp_path)
    agents = _SINGLE_PR_AGENTS + "Production target: `main`.\n"
    fake_run, calls = _merge_runner(tmp_path, agents=agents, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(
            _MergeArgs(branch="release/2026-10-07", sha="a" * 40, target="main"),
            run=fake_run, post=_never_post, env=_merge_env(), now=_fixed_now,
            allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "production-target-unconfigured"
    _assert_no_mutation(calls)


def test_merge_pr_no_recorded_pr_and_zero_gh_list_matches_refuses_pr_not_found(tmp_path, monkeypatch):
    # Ledger & card reconciliation: merge ALSO falls back to `gh pr list` when the hub has no
    # record (same lookup `pr open` uses) — pr-not-found only fires once THAT also finds
    # nothing, never before.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()  # no pr-opened row recorded at all
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        # cold review 75e934eacdca F1: without a real GitHub origin, _github_repo_slug refuses
        # github-unreachable before ever reaching gh pr list -- this fixture must reach the
        # intended pr-not-found refusal instead.
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "list"): _completed(0, "[]"),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-not-found"
    _assert_no_mutation(calls)


def test_merge_pr_no_recorded_pr_and_ambiguous_gh_list_refuses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()  # no pr-opened row recorded at all
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        # cold review 75e934eacdca F1: same reasoning as the pr-not-found fixture above -- reach
        # gh pr list (and its ambiguity) instead of refusing github-unreachable first.
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "list"): _completed(0, json.dumps([{"number": 7}, {"number": 8}])),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-ambiguous"
    _assert_no_mutation(calls)


def test_merge_pr_checks_dirtied_tree_refuses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    dirty_calls = {"n": 0}
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    def run(argv, cwd=None):
        if tuple(argv[:3]) == ("git", "status", "--porcelain"):
            dirty_calls["n"] += 1
            return _completed(0, "" if dirty_calls["n"] == 1 else " M dirty.txt\n")
        return fake_run(argv, cwd)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_merge(
                _MergeArgs(target="staging"), run=run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "checks-dirtied-tree"


# ------------------------------------------------------------------ slice e: release

# The plan's own fixtures name 2026-09-27, but `_fixed_now` -- the injected clock every
# dispatch test in this file uses -- returns 2026-01-01, so the computed release date (and
# therefore every computed branch literal below) is 2026-01-01.

_DUAL_PR_AGENTS = (
    "**Preset: `dual-branch-pr`** — production project.\n\n"
    "- Base branch: `staging` (default). Delivery target: feature-branch PR against "
    "`staging`; executor and gate per `merge-contract.md`.\n"
    "- Production target: `main`.\n"
)


def test_release_happy_path_fetches_snapshots_and_opens_the_pr(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    snapshot = "e" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("git", "fetch", "origin", "staging"): _completed(0, ""),
        ("git", "rev-parse", "--verify", "origin/staging^{commit}"): _completed(0, f"{snapshot}\n"),
        ("git", "rev-parse", "--verify", "refs/remotes/origin/release/2026-01-01-staging-promotion"): _completed(1, ""),
        ("git", "update-ref",): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, "[]"),
        ("gh", "pr", "create"): _completed(0, "https://github.com/acme/x/pull/11\n"),
    })
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_ledger_post(db, events),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["branch"] == "release/2026-01-01-staging-promotion"
    assert result["snapshot_sha"] == snapshot
    assert result["number"] == 11
    assert events[0]["payload"] == {
        "repo": "acme/x", "branch": "release/2026-01-01-staging-promotion", "sha": snapshot,
        "base": "main", "pr_number": 11, "pr_url": "https://github.com/acme/x/pull/11",
        "kind": "release", "snapshot_sha": snapshot,
    }
    assert ["git", "update-ref", "refs/heads/release/2026-01-01-staging-promotion", snapshot] in calls


def test_release_name_collision_same_day_gets_a_numeric_suffix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    snapshot = "e" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("git", "fetch", "origin", "staging"): _completed(0, ""),
        ("git", "rev-parse", "--verify", "origin/staging^{commit}"): _completed(0, f"{snapshot}\n"),
        ("git", "rev-parse", "--verify", "refs/remotes/origin/release/2026-01-01-staging-promotion"): _completed(0, f"{'f' * 40}\n"),
        ("git", "rev-parse", "--verify", "refs/remotes/origin/release/2026-01-01-staging-promotion-2"): _completed(1, ""),
        ("git", "update-ref",): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, "[]"),
        ("gh", "pr", "create"): _completed(0, "https://github.com/acme/x/pull/12\n"),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_ledger_post(db, []),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["branch"] == "release/2026-01-01-staging-promotion-2"


def test_release_name_collision_local_ref_gets_a_numeric_suffix(tmp_path, monkeypatch):
    # F2: the collision an ORIGIN-only check misses -- an interrupted run's local
    # `refs/heads/<branch>` with no remote counterpart yet must still trigger the suffix.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    snapshot = "e" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("git", "fetch", "origin", "staging"): _completed(0, ""),
        ("git", "rev-parse", "--verify", "origin/staging^{commit}"): _completed(0, f"{snapshot}\n"),
        ("git", "rev-parse", "--verify", "refs/heads/release/2026-01-01-staging-promotion"): _completed(0, f"{'f' * 40}\n"),
        ("git", "rev-parse", "--verify", "refs/remotes/origin/release/2026-01-01-staging-promotion-2"): _completed(1, ""),
        ("git", "update-ref",): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, "[]"),
        ("gh", "pr", "create"): _completed(0, "https://github.com/acme/x/pull/14\n"),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_ledger_post(db, []),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["branch"] == "release/2026-01-01-staging-promotion-2"


def test_release_reuses_an_open_release_pr_without_advancing_the_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    old_snapshot = "b" * 40
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened", {
        "repo": "acme/x", "branch": "release/2026-09-20-staging-promotion", "sha": old_snapshot,
        "base": "main", "pr_number": 9, "pr_url": "https://github.com/acme/x/pull/9",
        "kind": "release", "snapshot_sha": old_snapshot,
    })
    con.close()
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({
            "number": 9, "url": "https://github.com/acme/x/pull/9", "state": "OPEN",
            "headRefOid": old_snapshot, "baseRefName": "main", "mergeable": "MERGEABLE",
            "mergeCommit": None,
        })),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["snapshot_sha"] == old_snapshot
    assert result["number"] == 9
    assert not any(c[:2] == ["git", "fetch"] for c in calls)
    assert not any(c[:3] == ["gh", "pr", "create"] for c in calls)


def test_release_reconciles_an_interrupted_attempt_instead_of_duplicating(tmp_path, monkeypatch):
    # decision 9: a PARTIALLY-completed release (PR created on GitHub, ledger post never
    # landed) is found via _open_pr's own gh-list reconciliation, never re-created.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    snapshot = "e" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("git", "fetch", "origin", "staging"): _completed(0, ""),
        ("git", "rev-parse", "--verify", "origin/staging^{commit}"): _completed(0, f"{snapshot}\n"),
        ("git", "rev-parse", "--verify", "refs/remotes/origin/release/2026-01-01-staging-promotion"): _completed(0, f"{snapshot}\n"),
        ("git", "update-ref",): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, json.dumps([{"number": 13}])),
        ("gh", "pr", "view"): _completed(0, json.dumps({
            "number": 13, "url": "https://github.com/acme/x/pull/13", "state": "OPEN",
            "headRefOid": snapshot, "headRefName": "release/2026-01-01-staging-promotion",
            "baseRefName": "main", "mergeable": "MERGEABLE", "mergeCommit": None,
        })),
    })
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_ledger_post(db, events),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["number"] == 13
    assert not any(c[:3] == ["gh", "pr", "create"] for c in calls)


def test_release_refuses_a_repo_outside_the_allowlist_before_any_mutation(tmp_path, monkeypatch):
    # cold review e83ed7ea5eeb F1: refused right after the toplevel resolves -- before the
    # ledger read, the fetch, or the local ref write. No DB or ledger involved: the check
    # runs before either is ever reached.
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path / "elsewhere",
        )
    assert exc.value.code == "path-outside-allowlist"
    assert not any(c[:2] == ["git", "fetch"] for c in calls)
    assert not any(c[:2] == ["git", "update-ref"] for c in calls)


def test_release_does_not_reuse_a_same_numbered_pr_recorded_for_another_repo(tmp_path, monkeypatch):
    # cold review e83ed7ea5eeb F2: `payload["repo"]` must match the CURRENT repo_slug -- a
    # same-named repo under a different owner must never donate its PR number.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened", {
        "repo": "acme/other", "branch": "release/2026-09-20-staging-promotion", "sha": "b" * 40,
        "base": "main", "pr_number": 9, "pr_url": "https://github.com/acme/other/pull/9",
        "kind": "release", "snapshot_sha": "b" * 40,
    })
    con.close()
    snapshot = "e" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("git", "fetch", "origin", "staging"): _completed(0, ""),
        ("git", "rev-parse", "--verify", "origin/staging^{commit}"): _completed(0, f"{snapshot}\n"),
        ("git", "update-ref",): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, "[]"),
        ("gh", "pr", "create"): _completed(0, "https://github.com/acme/x/pull/20\n"),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_ledger_post(db, []),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["number"] == 20
    assert ["git", "fetch", "origin", "staging"] in calls
    assert not any(c[:3] == ["gh", "pr", "view"] for c in calls)


def test_release_does_not_reuse_when_the_live_pr_head_no_longer_matches_the_recorded_snapshot(
    tmp_path, monkeypatch,
):
    # cold review e83ed7ea5eeb F2: a recorded snapshot_sha the live PR's head has moved past
    # (something pushed to the release branch outside jaxflow) must not be handed back as
    # current -- falls through to a fresh fetch/reconcile instead.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    old_snapshot = "b" * 40
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened", {
        "repo": "acme/x", "branch": "release/2026-09-20-staging-promotion", "sha": old_snapshot,
        "base": "main", "pr_number": 9, "pr_url": "https://github.com/acme/x/pull/9",
        "kind": "release", "snapshot_sha": old_snapshot,
    })
    con.close()
    snapshot = "e" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({
            "number": 9, "url": "https://github.com/acme/x/pull/9", "state": "OPEN",
            "headRefOid": "c" * 40, "baseRefName": "main", "mergeable": "MERGEABLE",
            "mergeCommit": None,
        })),
        ("git", "fetch", "origin", "staging"): _completed(0, ""),
        ("git", "rev-parse", "--verify", "origin/staging^{commit}"): _completed(0, f"{snapshot}\n"),
        ("git", "update-ref",): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, "[]"),
        ("gh", "pr", "create"): _completed(0, "https://github.com/acme/x/pull/21\n"),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_ledger_post(db, []),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["number"] == 21
    assert ["git", "fetch", "origin", "staging"] in calls


def test_release_argparse_wiring_end_to_end(monkeypatch):
    seen = {}
    def fake_cmd_release(args, **kwargs):
        seen["from_caller"] = args.from_caller
        return {"number": 1, "url": "https://github.com/acme/x/pull/1",
                "snapshot_sha": "a" * 40, "branch": "release/x"}
    monkeypatch.setattr(jaxflow, "cmd_release", fake_cmd_release)
    assert jaxflow.main(["release", "--from", "claude"]) == jaxflow_common.OK
    assert seen == {"from_caller": "claude"}


# ------------------------------------------------------------------ slice c: merge verb

class _MergeArgs:
    def __init__(self, branch="feat/x", sha="a" * 40, phase="Phase X", checks="true",
                 from_caller="claude", target="main", recheck=False):
        self.branch, self.sha, self.phase = branch, sha, phase
        self.checks, self.from_caller = checks, from_caller
        self.target = target
        self.recheck = recheck


def _merge_runner(tmp_path, *, script=None, agents=None):
    """Returns (fake_run, calls). `script` maps an argv prefix tuple to a
    CompletedProcess; anything unmatched succeeds with empty output."""
    script = script or {}
    calls = []

    def fake_run(argv, cwd=None):
        calls.append(list(argv))
        for prefix, result in script.items():
            if tuple(argv[: len(prefix)]) == prefix:
                return result
        return _completed(0, "")

    if agents is not None:
        (tmp_path / "AGENTS.md").write_text(agents, encoding="utf-8")
    return fake_run, calls


def _merge_env():
    return {"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "s1", "TMUX_PANE": "%1"}


def _never_post(event):
    raise AssertionError(f"no event may be posted here, got {event.get('type')}")


def _assert_no_mutation(calls):
    """Spec §7.1 #19/#22: a refusal stops after the reads that detected it. Nothing that
    changes git state, the filesystem or the ledger may follow — the full set, not a
    sample of it (cold review F4)."""
    flat = [" ".join(c) for c in calls]
    for banned in ("git merge --no-ff", "git commit", "git push", "git worktree",
                    "git branch -d", "/bin/bash -lc"):
        assert not any(f.startswith(banned) for f in flat), f"{banned} ran after a refusal"


def _merge_args_without_target(**kw):
    args = _MergeArgs(**kw)
    delattr(args, "target")
    return args


def _merge_run_real_ref_format(tmp_path, *, script=None, agents=None):
    """Fake merge runner except `git check-ref-format`, which is the real binary."""
    fake_run, calls = _merge_runner(tmp_path, script=script, agents=agents)

    def run(argv, cwd=None):
        if tuple(argv[:2]) == ("git", "check-ref-format"):
            calls.append(list(argv))
            return _run_real(argv, cwd=cwd)
        return fake_run(argv, cwd=cwd)

    return run, calls


def _assert_no_delivery(calls, *, policy_target="release"):
    """MOA-458: mismatch/invalid target must not reach resume, switch, or delivery."""
    _assert_no_mutation(calls)
    flat = [" ".join(c) for c in calls]
    for banned in (f"git rev-parse --verify {policy_target}^2", "git switch",
                   "git write-tree"):
        assert not any(f.startswith(banned) for f in flat), f"{banned} ran after a refusal"


def test_merge_refuses_a_sha_that_is_not_full_40_hex(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(
        tmp_path, script={("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")}
    )
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha="abc1234"), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "sha-mismatch"
    _assert_no_mutation(calls)


def test_merge_refuses_when_head_is_not_the_approved_sha(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha, other = "a" * 40, "b" * 40
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        # the branch RESOLVES -- to a commit that is not the approved one. A missing ref is
        # a different case and would pass this test without proving anything (part-1 cold
        # review), so the specific entry comes first and the broad one only answers the
        # resume probe.
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{other}\n"),
        ("git", "rev-parse", "--verify"): _completed(128, ""),      # not a resume
        ("git", "branch", "--show-current"): _completed(0, "feat/x\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "sha-mismatch"
    _assert_no_mutation(calls)  # §7.1 #19: zero mutating calls past the detecting reads


def test_merge_refuses_a_dirty_tracked_tree_untracked_files_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    base = {
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "--verify"): _completed(128, ""),      # not a resume
        ("git", "branch", "--show-current"): _completed(0, "feat/x\n"),
    }
    dirty = {**base, ("git", "status", "--porcelain"): _completed(0, " M src/app.py\n")}
    fake_run, calls = _merge_runner(tmp_path, script=dirty)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "dirty-tracked-tree"
    _assert_no_mutation(calls)

    # a FAILED probe is not a clean tree either: empty stdout from a git that errored
    # would otherwise read as "nothing dirty" and let the merge proceed on a tree whose
    # state was never established (round-4 F4)
    broken = {**base, ("git", "status", "--porcelain"):
              _completed(128, "", "fatal: not a git repository\n")}
    fake_run3, calls3 = _merge_runner(tmp_path, script=broken)
    with pytest.raises(ji.Refusal) as exc3:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run3, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc3.value.code == "dirty-tracked-tree"
    assert "git status failed" in exc3.value.hint
    _assert_no_mutation(calls3)

    # untracked-only output must NOT refuse: the status probe asks git to omit untracked
    # files, so an untracked artifact produces empty output and the flow continues past
    # this gate (it will fail later for an unrelated reason, which this test ignores).
    clean = {**base, ("git", "status", "--porcelain"): _completed(0, "")}
    fake_run2, calls2 = _merge_runner(tmp_path, script=clean)
    with contextlib.suppress(Exception):
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run2, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    # The whole argv, not a 3-element prefix (diff review 2026-09-07): what actually
    # makes an untracked artifact pass this gate is `--untracked-files=no` on the probe,
    # and a prefix assertion stays green if that flag is ever dropped -- which would turn
    # every untracked build artifact into a refused delivery.
    assert ["git", "status", "--porcelain", "--untracked-files=no"] in calls2
    assert any(c[:2] == ["git", "switch"] for c in calls2), \
        "a clean tracked tree must reach the target switch — the only switch there is"


def test_merge_refuses_when_the_target_switch_lands_elsewhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "--verify"): _completed(128, ""),      # not a resume
        ("git", "status", "--porcelain"): _completed(0, ""),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        # the switch to `main` returned 0 but silently did not take effect — exactly what
        # target-mismatch guards against
        ("git", "branch", "--show-current"): _completed(0, "feat/x\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "target-mismatch"
    _assert_no_mutation(calls)


def _worktree_path(tmp_path, branch="feat/x"):
    """The path `build` derives for a branch — and the only one `merge` may ever remove."""
    return tmp_path.parent / f"{jaxflow_common.slugify_project(tmp_path.name)}-{branch.replace('/', '-')}"


def _merge_happy_script(tmp_path, sha, *, remote=True, target="main"):
    return {
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, f"refs/remotes/origin/{target}\n"),
        # SPECIFIC before broad: the branch probe and the resume probe are both
        # `rev-parse --verify`, and the runner returns the first matching prefix.
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "--verify", f"{target}^2"): _completed(128, ""),  # not a resume
        ("git", "branch", "--show-current"): _completed(0, "feat/x\n"),
        ("git", "status", "--porcelain"): _completed(0, ""),
        ("git", "rev-parse", "HEAD"): _completed(0, "c" * 40 + "\n"),
        # New (MOA-472): every existing happy-path test must keep exercising the FULL
        # checks sequence -- the safe default is "not an ancestor", i.e. not a
        # fast-forward. A dedicated fast-forward test overrides this explicitly.
        ("git", "merge-base", "--is-ancestor"): _completed(1, ""),
        # Exit 2, not 1: "No such remote" is exit 2 on real git, and round-5 F4 made that
        # the ONLY value that means "no origin" -- exit 1 would now refuse `push-failed`.
        ("git", "remote", "get-url"): _completed(0 if remote else 2, "git@github:x/y.git\n"),
        # git's own registry: the derived path IS the checkout of feat/x. Without this
        # entry `_is_registered_worktree` returns False, the cleanup block is unreachable
        # and every test that asserts on it fails (cold review round 2, F3).
        ("git", "worktree", "list"): _completed(
            0,
            f"worktree {tmp_path}\nHEAD {'b' * 40}\nbranch refs/heads/main\n\n"
            f"worktree {_worktree_path(tmp_path)}\nHEAD {sha}\nbranch refs/heads/feat/x\n\n",
        ),
    }


def _switch_aware_runner(tmp_path, sha, *, overrides=None, dirty_after_abort=None,
                         trees=None, switch_fails=None, **kw):
    """Like _merge_runner, but stateful in the three ways the delivery path needs. Every
    delivery test uses THIS runner: the same fake was hand-copied into six tests and each
    copy silently lacked one of these behaviours (cold review round 2).

    * `git branch --show-current` answers with whatever branch the last `git switch` chose.
    * `git status --porcelain` keeps answering from `script` until `git merge --abort`
      runs, then answers `dirty_after_abort`. A test that simply scripted a dirty status
      would be refused by the PREFLIGHT dirty-tracked-tree check and never reach the abort.
    * `git write-tree` walks `trees`, so the two index fingerprints can differ; by default
      both calls return the same value, i.e. the checks changed nothing. An entry that is
      not a string is returned as the result itself, which is how a FAILING `write-tree`
      is scripted.
    * `git switch <switch_fails>` returns non-zero AND leaves the previous branch checked
      out — real git's behaviour, and the reason a post-switch branch read is not by itself
      proof that the switch happened (round-3 F8). Only the target switch exists now
      (round-4 F12), so this is always the target's name."""
    script = _merge_happy_script(tmp_path, sha, **kw)
    script.update(overrides or {})
    calls, state = [], {"branch": "feat/x", "aborted": False}
    trees = iter(trees or [])

    def fake_run(argv, cwd=None):
        calls.append(list(argv))
        if argv[:2] == ["git", "switch"]:
            if argv[2] == switch_fails:
                return _completed(1, "", "fatal: invalid reference")   # branch unchanged
            state["branch"] = argv[2]
            return _completed(0, "")
        if argv[:3] == ["git", "branch", "--show-current"]:
            # An EXPLICIT override wins over the stateful answer (diff review
            # 2026-09-07). `overrides`, not `script` -- `script` always carries the happy
            # default, so consulting it would destroy the state tracking for every test.
            # Without this, a test that needs this read to FAIL cannot express it: the
            # override is swallowed here and the merge sails past the gate under test.
            forced = (overrides or {}).get(("git", "branch", "--show-current"))
            return forced if forced is not None else _completed(0, state["branch"] + "\n")
        if argv[:2] == ["git", "write-tree"]:
            # fixture fix (execution rule): the default tree id must be a real 40-hex
            # shape, or `_git_read`'s `shape=r"[0-9a-f]{40}"` check (round-5 F3) rejects it
            # on every test that does not override `trees=`, misreporting a fingerprinting
            # failure instead of exercising the delivery path under test.
            nxt = next(trees, "1" * 40)  # a str is a tree id; anything else is a result
            return _completed(0, nxt + "\n") if isinstance(nxt, str) else nxt
        if argv[:3] == ["git", "status", "--porcelain"] and state["aborted"]:
            return _completed(0, dirty_after_abort or "")
        if argv[:3] == ["git", "merge", "--abort"]:
            state["aborted"] = True      # and fall through to the script for its exit code
        for prefix, result in script.items():
            if tuple(argv[: len(prefix)]) == prefix:
                return result
        return _completed(0, "")

    return fake_run, calls


def test_merge_happy_path_call_order_and_printed_outcome_with_remote(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    posted = []
    _worktree_path(tmp_path).mkdir(parents=True)   # the checkout the cleanup block removes
    fake_run, calls = _switch_aware_runner(tmp_path, sha)

    def record_post(event):
        # The POST goes into the SAME list as the git calls, so its position is assertable
        # against the push instead of merely "something was posted" (cold review F6).
        calls.append(["POST", event["type"]])
        posted.append(event)

    rc = jaxflow.cmd_merge(_MergeArgs(sha=sha, checks="pytest -q"), run=fake_run,
                           post=record_post, env=_merge_env(), now=_fixed_now,
                           allowlist_root=tmp_path.parent)
    assert rc == jaxflow_common.OK
    flat = [" ".join(c) for c in calls]

    def idx(pred):
        return next(i for i, f in enumerate(flat) if pred(f))

    assert not any(f.startswith("git switch feat/x") for f in flat), \
        "the source branch is held by the builder worktree and is never checked out (F12)"
    # Every required READ, not only the mutations (diff review 2026-09-07): asserting
    # the mutation chain alone stays green if the source-ref, status, branch or remote
    # probe drifts across a mutation -- and a gate that runs after what it gates is not
    # a gate. §7.1 #21 asks for the FULL order, so pin the reads to it.
    i_srcref = idx(lambda f: f == "git rev-parse --verify feat/x^{commit}")
    i_status = idx(lambda f: f.startswith("git status --porcelain --untracked-files=no"))
    i_switch_target = idx(lambda f: f == "git switch main")
    i_oncheck = idx(lambda f: f == "git branch --show-current")
    i_merge = idx(lambda f: f.startswith("git merge --no-ff --no-commit"))
    i_checks = idx(lambda f: "pytest -q" in f)
    i_diff = idx(lambda f: f.startswith("git diff --quiet"))
    i_commit = idx(lambda f: f.startswith("git commit"))
    i_post = idx(lambda f: f.startswith("POST"))
    i_push = idx(lambda f: f.startswith("git push"))
    assert (i_srcref < i_status < i_switch_target < i_oncheck
            < i_merge < i_checks < i_diff < i_commit < i_post < i_push)
    i_remote = idx(lambda f: f.startswith("git remote get-url origin"))
    assert i_commit < i_remote < i_push, "the remote is probed only after a durable commit"
    # the index is fingerprinted on both sides of the checks (cold review F5)
    i_tree_before = idx(lambda f: f == "git write-tree")
    i_tree_after = max(i for i, f in enumerate(flat) if f == "git write-tree")
    assert i_merge < i_tree_before < i_checks < i_tree_after < i_commit
    assert flat[i_checks].startswith("/bin/bash -lc")
    assert flat[i_push] == "git push origin main"
    # the commit subject is the contract's own, never a caller-supplied message
    assert "feat: Phase X (merge feat/x)" in flat[i_commit]

    assert len(posted) == 1
    ev = posted[0]
    assert ev["type"] == "merge-approved" and ev["role"] == "lead" and ev["emitter"] == "wrapper"
    assert ev["payload"] == {
        "phase": "Phase X", "branch": "feat/x", "sha": sha, "target": "main",
        "approved_by": "rafa", "merge_sha": "c" * 40,
        "checks": {"mode": "run"},
    }
    assert ev["pane"] == "%1"
    assert "run_id" not in ev, "merge-approved is not run-scoped"
    # worktree removal strictly before branch deletion, both after the push
    # (§7.1 #21 full order, cold review C2/F6).
    i_worktree = idx(lambda f: f.startswith("git worktree remove"))
    i_branch_del = idx(lambda f: f == "git branch -d feat/x")
    assert i_push < i_worktree < i_branch_del

    out = capsys.readouterr().out.strip().splitlines()
    assert out[-2:] == [f"merged {'c' * 40} pushed origin/main", "checks: pytest -q exit 0"]


def test_merge_refuses_when_worktree_claim_held(tmp_path, monkeypatch):
    import jaxflow_resume as jresume

    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    worktree = _worktree_path(tmp_path)
    worktree.mkdir(parents=True)
    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    with jresume.worktree_claim(tmp_path, worktree):
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_merge(
                _MergeArgs(sha=sha, checks="true"), run=fake_run,
                post=lambda e: {"ok": True}, env=_merge_env(), now=_fixed_now,
                allowlist_root=tmp_path.parent,
            )
    assert exc.value.code in ("resume-ineligible", "agent-settings-permissions")
    _assert_no_mutation(calls)


def test_merge_refuses_nonterminal_newer_build(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    _worktree_path(tmp_path).mkdir(parents=True)
    db = tmp_path / "jaxos.db"
    monkeypatch.setattr(jr, "DB_PATH", db)
    con = _fresh_db(db)
    _insert(con, "bbbbbbbbbbbb", jaxflow_common.slugify_project(tmp_path.name), "builder", "run-started", {
        "phase": "P", "runtime": "opencode-builder", "kind": "build",
        "target": "feat/x", "session": "jax-demo-build-bbbbbbbbbbbb",
        "repo": str(tmp_path.resolve()),
    })
    con.close()
    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(
            _MergeArgs(sha=sha, checks="true"), run=fake_run,
            post=lambda e: {"ok": True}, env=_merge_env(), now=_fixed_now,
            allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "resume-ineligible"
    _assert_no_mutation(calls)


def test_merge_runs_checks_on_a_non_fast_forward_merge(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    _worktree_path(tmp_path).mkdir(parents=True)
    fake_run, calls = _switch_aware_runner(tmp_path, sha)  # default: not an ancestor
    rc = jaxflow.cmd_merge(_MergeArgs(sha=sha, checks="pytest -q"), run=fake_run,
                           post=lambda e: {"ok": True}, env=_merge_env(), now=_fixed_now,
                           allowlist_root=tmp_path.parent)
    assert rc == jaxflow_common.OK
    flat = [" ".join(c) for c in calls]
    assert any("pytest -q" in f for f in flat), "a non-fast-forward merge must still run checks_cmd"
    assert any(f.startswith("git diff --quiet") for f in flat)
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "checks: pytest -q exit 0"


def test_merge_still_refuses_checks_failed_on_a_non_fast_forward(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha, overrides={("/bin/bash", "-lc"): _completed(1, "boom")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "checks-failed"
    flat = [" ".join(c) for c in calls]
    assert any(f.startswith("git merge --no-ff") for f in flat), "must still attempt the merge"


def test_merge_refuses_an_equal_tip_merge_as_merge_failed(tmp_path, monkeypatch):
    # F3, AC 10: an equal tip is excluded from the skip, runs the full sequence, and
    # fails at `git commit` with nothing staged -- identical to today.
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha,
        overrides={
            ("git", "rev-parse", "HEAD"): _completed(0, sha + "\n"),  # target tip == approved sha
            # Scripted to say "yes" on purpose: this proves the equal-tip short-circuit
            # inside `_is_strict_descendant` -- not an unlucky default -- is what excludes
            # this merge from the skip.
            ("git", "merge-base", "--is-ancestor"): _completed(0, ""),
            ("git", "commit"): _completed(1, "nothing to commit, working tree clean"),
        },
    )
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha, checks="true"), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "merge-failed"
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("git merge-base --is-ancestor") for f in flat), \
        "equal tip must be excluded before ever consulting ancestry"
    assert any(f.startswith("/bin/bash -lc") for f in flat), "checks_cmd must still run"


def test_merge_fast_forward_uses_the_captured_pre_merge_tip_not_a_later_head_read(
        tmp_path, monkeypatch, capsys):
    # AC 8: the fast-forward decision is pinned to the tip captured BEFORE `git merge`
    # runs -- never a later HEAD re-read.
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    _worktree_path(tmp_path).mkdir(parents=True)
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha, overrides={("git", "merge-base", "--is-ancestor"): _completed(0, "")})
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: {"ok": True},
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    flat = [" ".join(c) for c in calls]
    i_tip = flat.index("git rev-parse HEAD")
    i_merge = next(i for i, f in enumerate(flat) if f.startswith("git merge --no-ff --no-commit"))
    i_ancestor = next(i for i, f in enumerate(flat) if f.startswith("git merge-base --is-ancestor"))
    assert i_tip < i_merge < i_ancestor
    assert "git rev-parse HEAD" not in flat[i_merge:i_ancestor], \
        "the fast-forward check must reuse the tip captured before the merge, never a fresh HEAD read"


def test_merge_a_non_fast_forward_target_does_not_skip_checks(tmp_path, monkeypatch, capsys):
    # F1, AC 9: pins the DIRECTION -- a reversed base/head would report a fast-forward
    # here (args.sha IS an ancestor of target_tip -- the WRONG relationship), and this
    # test fails if that ever recurs.
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    target_tip = "c" * 40
    _worktree_path(tmp_path).mkdir(parents=True)

    def is_ancestor(argv):
        if argv[3:] == [sha, target_tip]:
            return _completed(0, "")   # the WRONG direction: would wrongly say "yes"
        if argv[3:] == [target_tip, sha]:
            return _completed(1, "")   # the correct direction: no, not an ancestor
        raise AssertionError(f"unexpected merge-base call: {argv}")

    base_fake, calls = _switch_aware_runner(
        tmp_path, sha, overrides={("git", "rev-parse", "HEAD"): _completed(0, target_tip + "\n")})

    def wrapped(argv, cwd=None):
        if argv[:3] == ["git", "merge-base", "--is-ancestor"]:
            calls.append(list(argv))
            return is_ancestor(argv)
        return base_fake(argv, cwd=cwd)

    rc = jaxflow.cmd_merge(_MergeArgs(sha=sha, checks="pytest -q"), run=wrapped,
                           post=lambda e: {"ok": True}, env=_merge_env(), now=_fixed_now,
                           allowlist_root=tmp_path.parent)
    assert rc == jaxflow_common.OK
    flat = [" ".join(c) for c in calls]
    assert any("pytest -q" in f for f in flat), \
        "the correct direction finds no fast-forward, so checks_cmd must still run"


def test_merge_without_a_remote_skips_push_and_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha, remote=False)
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert not any(c[:2] == ["git", "push"] for c in calls)
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-2] == f"merged {'c' * 40} — no remote delivery configured"
    assert out[-1] == "checks: true exit 0"


def test_merge_prints_the_no_preset_block_line(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert "target: main (no preset block)" in capsys.readouterr().out


def test_merge_omits_pane_when_tmux_pane_is_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha, posted = "a" * 40, []
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    env = _merge_env()
    del env["TMUX_PANE"]
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=posted.append,
                      env=env, now=_fixed_now, allowlist_root=tmp_path.parent)
    assert "pane" not in posted[0], "absent pane is omitted, never sent as null"


@pytest.mark.parametrize("failing,code", [
    (("git", "merge", "--no-ff"), "merge-failed"),
    (("/bin/bash", "-lc"), "checks-failed"),
    (("git", "diff", "--quiet"), "checks-dirtied-tree"),
])
def test_merge_aborts_cleanly_with_zero_further_calls(tmp_path, monkeypatch, failing, code):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha,
                                           overrides={failing: _completed(1, "boom")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == code
    flat = [" ".join(c) for c in calls]
    assert "git merge --abort" in flat
    # everything after the abort is READ-ONLY state reporting (_abort_merge's two probes);
    # nothing that changes git state, the filesystem or the ledger may follow.
    tail = flat[flat.index("git merge --abort") + 1:]
    assert all(f.startswith("git rev-parse -q --verify MERGE_HEAD") or
               f.startswith("git status --porcelain") for f in tail), tail
    assert not any(f.startswith("git commit") or f.startswith("git push") or
                   f.startswith("git worktree") or f.startswith("git branch -d") or
                   f.startswith("git reset") for f in flat)


def test_merge_post_failure_keeps_the_commit_and_prints_the_resume_line(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha)

    def failing_post(event):
        raise RuntimeError("hub down")

    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=failing_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "hub-unreachable"
    flat = [" ".join(c) for c in calls]
    assert any(f.startswith("git commit") for f in flat), "the commit is kept"
    assert not any(f.startswith("git push") or f.startswith("git worktree") for f in flat)
    assert f"merge commit {'c' * 40} kept; re-run the same jaxflow merge to resume" in capsys.readouterr().out


def test_merge_push_failure_keeps_everything_and_cleans_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    posted = []
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha, overrides={("git", "push"): _completed(1, "rejected")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=posted.append,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "push-failed"
    assert len(posted) == 1, "the audit row was already posted and is not retried"
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("git worktree") or f.startswith("git branch -d") for f in flat)


def test_merge_copies_worktree_reports_before_removing_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    # the worktree `build` would have created for this branch, with a builder report in it
    worktree = tmp_path.parent / f"{jaxflow_common.slugify_project(tmp_path.name)}-feat-x"
    (worktree / ".local" / "reports").mkdir(parents=True)
    (worktree / ".local" / "reports" / "abc123abc123.md").write_text("report", encoding="utf-8")
    # the spec requires the verification evidence to survive the removal too, not just the
    # report markdown (part-1 cold review round 2)
    (worktree / ".local" / "reports" / "abc123abc123.tests.txt").write_text(
        "42 passed", encoding="utf-8")
    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    dest = tmp_path / ".local" / "reports"
    assert dest.joinpath("abc123abc123.md").read_text(encoding="utf-8") == "report"
    assert dest.joinpath("abc123abc123.tests.txt").read_text(encoding="utf-8") == "42 passed", \
        "the verification evidence survives too (spec §5.3)"
    flat = [" ".join(c) for c in calls]
    assert any(f.startswith("git worktree remove") for f in flat)
    assert any(f == "git branch -d feat/x" for f in flat)


def test_merge_writes_status_md_stage_ship_and_drops_the_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    (tmp_path / ".jax-os").mkdir()
    (tmp_path / ".jax-os" / "status.md").write_text(
        # fixture fix (execution rule): `_fixed_now()` (already defined at the top of this
        # file) returns 2026-01-01, not the 2026-09-07 the plan's own reference definition
        # assumed -- an `updated` stamp of 2026-09-01 would read as NEWER than
        # `dispatch_start` and trip the staleness guard, skipping the write this test
        # means to exercise. Any timestamp before 2026-01-01 keeps the same intent.
        "---\nproject: Demo\nstage: review\nbuilder: codex\nbranch: feat/x\n"
        "gate: awaiting-approval\nupdated: 2020-01-01T00:00:00-03:00\n---\n\n"
        "## Now\nOld text.\n\n## Residuals\nKeep me.\n",
        encoding="utf-8",
    )
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    text = (tmp_path / ".jax-os" / "status.md").read_text(encoding="utf-8")
    assert "stage: ship" in text
    assert "gate:" not in text, "a completed delivery has no pending decision"
    assert "project: Demo" in text and "## Residuals\nKeep me." in text
    assert "merged" in text.split("## Now")[1]


# ---- resume path (moved here from Task 2 so every commit gate stays green) -------------

def test_merge_detects_a_resume_and_skips_merge_checks_and_commit(tmp_path, monkeypatch):
    """§7.1 #26: target HEAD already carries the merge of --sha."""
    monkeypatch.chdir(tmp_path)
    sha, posted = "a" * 40, []
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        # the merge commit is read from TARGET, never from HEAD (round-2 F4)
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        # ...and it records the branch the approval named (the round-2 F4 binding)
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        # ...and the source branch still resolves to the approved sha (the F2 gate)
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{sha}\n"),
        ("git", "remote", "get-url"): _completed(2, ""),   # 2 == no origin
    })
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=posted.append,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    flat = [" ".join(c) for c in calls]
    for banned in ("git merge --no-ff", "git commit", "git diff --quiet", "/bin/bash -lc",
                   "git write-tree"):
        assert not any(f.startswith(banned) for f in flat), f"{banned} ran on a resume"
    assert posted and posted[0]["type"] == "merge-approved"
    assert posted[0]["payload"]["merge_sha"] == "c" * 40, "the target's merge, not HEAD"
    assert posted[0]["payload"]["checks"] == {"mode": "resumed"}


def test_merge_does_not_mistake_an_unrelated_merge_commit_for_a_resume(tmp_path, monkeypatch):
    """§7.1 #26, negative half: the normal Step 1 sequence runs in full."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    # main's HEAD IS a merge commit, but of a DIFFERENT sha. The override has to name the
    # SPECIFIC key `_merge_happy_script` already defines -- a broad prefix is inserted after
    # it and never reached, which made this test pass without testing anything (part-1 cold
    # review).
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha,
        overrides={("git", "rev-parse", "--verify", "main^2"): _completed(0, "d" * 40 + "\n")})
    _worktree_path(tmp_path).mkdir(parents=True)
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run,
                      post=lambda e: calls.append(["POST", e["type"]]),
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    flat = [" ".join(c) for c in calls]
    # the FULL normal sequence in order, not merely "some switch ran" (cold review C2/F6)
    at = -1
    for expected in ("git switch main", "git merge --no-ff",
                     "/bin/bash -lc", "git write-tree", "git diff --quiet", "git commit",
                     "POST", "git push", "git worktree remove", "git branch -d feat/x"):
        nxt = next((i for i, f in enumerate(flat) if i > at and f.startswith(expected)), None)
        assert nxt is not None, f"{expected} missing after position {at}: {flat}"
        at = nxt


# ---- cold-review findings F1/F2/F3 ------------------------------------------------------

def test_merge_reports_when_the_abort_could_not_clean_the_checkout(tmp_path, monkeypatch):
    """F1: `git merge --abort` can exit non-zero and leave an AM merge state. The refusal
    must SAY so instead of implying a clean restore."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha, dirty_after_abort=" M src/app.py\n",
        overrides={
            ("/bin/bash", "-lc"): _completed(1, "boom"),
            ("git", "merge", "--abort"): _completed(128, "fatal: could not abort"),
            ("git", "rev-parse", "-q", "--verify", "MERGE_HEAD"): _completed(0, "e" * 40 + "\n"),
        })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "checks-failed"
    assert "exited 128" in exc.value.hint, "the abort's own exit code is reported (F11)"
    assert "MERGE_HEAD" in exc.value.hint
    assert "src/app.py" in exc.value.hint, "the leftover state is named, not just hinted at"
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("git reset") for f in flat), "jaxflow never resets --hard on its own"


def test_merge_abort_that_succeeds_cleanly_adds_no_state_note(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha,
        overrides={
            ("/bin/bash", "-lc"): _completed(1, "boom"),
            ("git", "rev-parse", "-q", "--verify", "MERGE_HEAD"): _completed(1, ""),
        })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "checks-failed"
    assert "MERGE_HEAD" not in exc.value.hint


def test_merge_abort_that_exits_zero_still_reports_a_modified_tracked_file(tmp_path, monkeypatch):
    """F1, the second real case (verified on git 2026-09-07): when the checks modified a
    tracked file that was NOT part of the merge, `git merge --abort` exits 0 and leaves
    that modification in place. Exit 0 is not proof of a clean checkout, so the refusal
    still has to name what survived."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha, dirty_after_abort=" M untouched.py\n",
        overrides={
            ("/bin/bash", "-lc"): _completed(1, "boom"),
            ("git", "merge", "--abort"): _completed(0, ""),          # the abort SUCCEEDED
            ("git", "rev-parse", "-q", "--verify", "MERGE_HEAD"): _completed(1, ""),
        })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "checks-failed"
    assert "MERGE_HEAD" not in exc.value.hint, "there is no merge state left"
    assert "tracked changes remain" in exc.value.hint and "untouched.py" in exc.value.hint


def test_merge_refuses_when_the_checks_stage_a_tracked_change(tmp_path, monkeypatch):
    """F5: `git diff --quiet` is blind to a check that ran `git add` (verified on real git
    2026-09-07: it exits 0 while `git diff --cached --quiet` exits 1, and the following
    commit carried the staged change). The index fingerprint is what catches it."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        # fixture fix (execution rule): both values must be real 40-hex shapes -- the
        # implementation's shape check (round-5 F3) would otherwise reject the FIRST one
        # and refuse `merge-failed` before the staged-change comparison this test targets
        # is ever reached. Two DIFFERENT valid tree ids is what "the index moved under the
        # checks" actually means.
        tmp_path, sha, trees=["3" * 40, "4" * 40],
        overrides={("git", "rev-parse", "-q", "--verify", "MERGE_HEAD"): _completed(1, "")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "checks-dirtied-tree"
    assert "(staged)" in exc.value.hint
    flat = [" ".join(c) for c in calls]
    assert "git merge --abort" in flat
    assert not any(f.startswith("git commit") or f.startswith("git push") for f in flat)


@pytest.mark.parametrize("branch,code", [
    ("x" * 513, "branch-invalid"),   # a valid git ref, over the audit payload's bound
    ("", "branch-invalid"),
    ("feat/ x", "branch-invalid"),
])
def test_merge_refuses_a_branch_the_audit_event_cannot_carry(tmp_path, monkeypatch,
                                                             branch, code):
    """`git check-ref-format` accepts a 2048-character ref; `LIMITS.target` is 512. An
    oversized-but-valid branch would reach the commit subject AND the payload, commit,
    and only then fail its POST -- the stranded-merge shape the phase bound already
    prevents, one field over (round-5 F11). Bounded before any mutation."""
    monkeypatch.chdir(tmp_path)
    # the toplevel read must SUCCEED, or the refusal is `not-a-git-toplevel` and the gate
    # under test is never reached (round-6 F15)
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(branch=branch), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == code
    _assert_no_mutation(calls)


def test_merge_refuses_a_delivery_target_the_audit_event_cannot_carry(tmp_path, monkeypatch):
    """The target is payload too, and it comes from a hand-written `AGENTS.md` line or
    from a symbolic-ref read -- neither validates a shape (round-5 F8/F11)."""
    monkeypatch.chdir(tmp_path)
    # the parser wants the real heading shape, bold + backticks, or the block is never
    # recognised and the long target is never extracted (round-6 F15)
    _agents(tmp_path, "## Deploy policy\n\n**Preset: `dual-branch`** — notes.\n\n"
                      "Delivery target: `" + "y" * 513 + "`.\n")
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "preset-unknown"
    _assert_no_mutation(calls)


def test_merge_resume_refuses_when_the_subject_read_fails_but_prints_the_right_text(
        tmp_path, monkeypatch):
    """#27c covers EVERY value reader, not just one (diff review 2026-09-07). A failed
    `git log` whose stdout happens to be the expected subject must not authorise a resume
    -- that read is the identity binding that stops an alias branch being deleted."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        # non-zero, but printing exactly what the caller wanted to see
        ("git", "log", "-1", "--format=%s"): _completed(128, "feat: Phase X (merge feat/x)\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "sha-mismatch"
    _assert_no_mutation(calls)


def test_merge_refuses_when_the_post_switch_branch_read_fails_but_prints_the_target(
        tmp_path, monkeypatch):
    """The other #27c reader: `git branch --show-current` failing while echoing the
    target would let the merge proceed on a checkout it never confirmed."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha,
        overrides={("git", "branch", "--show-current"): _completed(128, "main\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "target-mismatch"
    assert not any(" ".join(c).startswith("git merge --no-ff") for c in calls), \
        "nothing is merged onto a checkout the tool could not confirm"


def test_merge_refuses_when_a_git_read_fails_but_prints_a_plausible_value(tmp_path,
                                                                          monkeypatch):
    """The whole point of `_git_read` (round-5 F2): git writes to stdout before it fails,
    so a NON-ZERO call whose output happens to look like a sha must not be trusted. Two
    of the eight call sites had exactly this bug -- checking the shape but not the exit."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha,
        overrides={("git", "rev-parse", "HEAD"): _completed(128, "c" * 40 + "\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "merge-failed"
    assert not any(" ".join(c).startswith("git push") for c in calls)


@pytest.mark.parametrize("trees,code", [
    (["not-a-tree-object", "not-a-tree-object"], "merge-failed"),
    (["b" * 40, "still-not-a-tree"], "checks-dirtied-tree"),
])
def test_merge_refuses_a_write_tree_value_that_is_not_a_tree_object(tmp_path, monkeypatch,
                                                                    trees, code):
    """The two fingerprints are compared to EACH OTHER, so two malformed-but-equal values
    would report `staged_changed = False` -- and `git diff --quiet` is blind to a staged
    change, which is the whole reason the fingerprints exist (round-5 F3)."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha, trees=trees)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == code
    assert not any(" ".join(c).startswith("git commit") for c in calls)


def test_merge_refuses_when_the_remote_probe_itself_fails(tmp_path, monkeypatch):
    """Exit 2 is `No such remote` and nothing else is (verified on real git). Treating
    every non-zero result as `no remote` would report a complete local delivery because
    the repository was broken or unreadable (round-5 F4). The commit is durable here, so
    the refusal is `push-failed` and the documented resume finishes the job."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    posted = []
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha,
        overrides={("git", "remote", "get-url"): _completed(128, "", "fatal: not a git repo")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=posted.append,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "push-failed"
    assert "could not probe origin" in exc.value.hint
    assert len(posted) == 1, "the audit event is posted BEFORE the push, and stays posted"
    assert not any(" ".join(c).startswith(("git push", "git worktree")) for c in calls)


def test_merge_resume_refuses_when_the_branch_read_fails_rather_than_being_gone(
        tmp_path, monkeypatch):
    """`rev-parse --verify` returns 128 both for `no such ref` and for a broken
    repository; `show-ref --verify` separates them (exit 1 vs 128, verified on real git).
    Only `gone` may skip cleanup -- a failed read must not post an audit for a delivery
    whose source branch could not be checked (round-5 F6)."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        ("git", "show-ref", "--verify"): _completed(128, "", "fatal: not a git repository"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "sha-mismatch"
    assert "could not read feat/x" in exc.value.hint
    _assert_no_mutation(calls)


def test_merge_reports_a_branch_that_could_not_be_deleted(tmp_path, monkeypatch, capsys):
    """Best-effort, but never silent (round-5 F10): the merge still succeeds, and a
    branch that outlived it is state the tech lead has to know about."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    worktree = _worktree_path(tmp_path)
    (worktree / ".local" / "reports").mkdir(parents=True)
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha,
        overrides={("git", "branch", "-d"): _completed(1, "", "error: not fully merged")})
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    out = capsys.readouterr().out
    assert "branch feat/x kept" in out and "not fully merged" in out
    assert "merged " in out, "the delivery still reports success"


@pytest.mark.parametrize("phase", [
    "Release\nsecond line",          # `git commit -m` takes it; `%s` flattens it (F15)
    "x" * 201,                       # over the event validator's LIMITS.summary bound (F16)
    "",
    "   ",
    " Phase X ",                     # untrimmed: the commit subject would not round-trip
    "\ufeffPhase X",                  # the ONE character JS `\s` strips and Python's
                                     # `strip()` does not, so only refusing it here keeps
                                     # the producer from being the looser side (part-2
                                     # cold review round 4)
    pytest.param("\U0001f680" * 200,      # 200 emoji = 400 UTF-16 units; `len()` would accept it
                 id="utf16-code-units-over-the-bound"),
    pytest.param("Phase \udcff X",        # POSIX argv surrogateescape: `encode("utf-16-le")` raises
                 id="undecodable-argv-byte"),
])
def test_merge_refuses_an_unusable_phase_title(tmp_path, monkeypatch, phase):
    """F15/F16: one canonical title serves the commit subject, the audit payload and the
    resume comparison, so anything that cannot survive all three is refused BEFORE any git
    state is touched."""
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(
        tmp_path, script={("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(phase=phase), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "phase-invalid"
    _assert_no_mutation(calls)


def test_merge_phase_title_is_identical_in_the_commit_and_the_audit_event(tmp_path, monkeypatch):
    """F16: the commit subject and the `merge-approved` payload carry the SAME title —
    the earlier draft committed the raw one and posted a truncated one."""
    monkeypatch.chdir(tmp_path)
    sha, posted = "a" * 40, []
    title = "Phase C.2 — Jax Rules (canonical rule-file management)"
    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    jaxflow.cmd_merge(_MergeArgs(sha=sha, phase=title), run=fake_run, post=posted.append,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    commit = next(c for c in calls if c[:2] == ["git", "commit"])
    assert commit[-1] == f"feat: {title} (merge feat/x)"
    assert posted[0]["payload"]["phase"] == title


def test_merge_refuses_when_the_merged_index_cannot_be_fingerprinted(tmp_path, monkeypatch):
    """F9: a PRE-check `git write-tree` failure is a broken merge, not a dirty check. It
    refuses `merge-failed` and the checks command never runs at all."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha, trees=[_completed(128, "", "fatal: unable to write new index file")])
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "merge-failed"
    assert "unable to write new index file" in exc.value.hint
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("/bin/bash -lc") for f in flat), "the checks never ran"
    assert "git merge --abort" in flat


def test_merge_report_copy_skips_a_symlinked_entry(tmp_path, monkeypatch, capsys):
    """F17: everything under the builder's `.local/reports/` is builder-controlled.
    `shutil.copytree` follows symlinks, which would copy whatever one points at — outside
    the allowlist included — into the control repo (spec §6/I4)."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    secret = tmp_path.parent / "outside-the-allowlist.txt"
    secret.write_text("SECRET", encoding="utf-8")
    reports = _worktree_path(tmp_path) / ".local" / "reports"
    reports.mkdir(parents=True)
    (reports / "abc123abc123.md").write_text("real report", encoding="utf-8")
    # An ALLOWED name (diff review 2026-09-07): `evil.md` was refused by the name filter
    # before the open, so this test passed without ever exercising `O_NOFOLLOW` -- it
    # would have stayed green if symlinked entries were followed.
    (reports / "def456def456.md").symlink_to(secret)
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    copied = tmp_path / ".local" / "reports"
    assert (copied / "abc123abc123.md").read_text(encoding="utf-8") == "real report"
    assert not (copied / "def456def456.md").exists(), "a symlinked report is never copied"
    assert "def456def456.md skipped" in capsys.readouterr().out
    assert secret.read_text(encoding="utf-8") == "SECRET"


def test_merge_accepts_a_phase_at_exactly_the_utf16_bound(tmp_path, monkeypatch):
    """The bound is 200 code units, not 200 bytes and not 100 characters: 100 emoji are
    exactly 200 units and must pass."""
    monkeypatch.chdir(tmp_path)
    sha, posted = "a" * 40, []
    title = "\U0001f680" * 100
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    jaxflow.cmd_merge(_MergeArgs(sha=sha, phase=title), run=fake_run, post=posted.append,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert posted[0]["payload"]["phase"] == title


@pytest.mark.parametrize("plant,expected_skip", [
    ("symlinked-dir", "reports copy skipped"),          # the reports dir itself
    ("symlinked-ancestor", "reports copy skipped"),     # `.local`, one level up
    ("dest-symlinked-ancestor", "reports copy skipped"),  # the CONTROL repo's `.local`
    ("hardlink", "not a single-linked regular file"),
    ("wrong-name", "not a run report"),
    ("dest-symlink", "existing destination unreadable"),
    ("oversize", "over the"),
    ("fifo", "not a single-linked regular file"),
])
def test_merge_report_copy_refuses_every_escape(tmp_path, monkeypatch, capsys, plant, expected_skip):
    """The four escapes a name-only check let through, each reproduced. `merge` reads a
    builder-written file exactly here and nowhere else, so this is the whole attack
    surface (spec §6/I4)."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    secret = tmp_path.parent / "outside-the-allowlist.txt"
    secret.write_text("SECRET", encoding="utf-8")
    worktree = _worktree_path(tmp_path)
    reports = worktree / ".local" / "reports"

    outside = tmp_path.parent / "elsewhere"
    # fixture fix (execution rule): `tmp_path.parent` is the SAME session tmp root for
    # every one of this test's 8 parametrized invocations (pytest nests each `tmp_path`
    # directly under one shared session dir), so a fixed name here collides with the
    # previous case's leftover directory. `exist_ok=True` is enough: the decoy content is
    # identical across cases, so re-using the same directory changes nothing under test.
    (outside / "reports").mkdir(parents=True, exist_ok=True)
    (outside / "reports" / "abc123abc123.md").write_text("SECRET", encoding="utf-8")

    if plant == "symlinked-dir":
        # the reports directory itself is a symlink pointing outside the worktree
        (worktree / ".local").mkdir(parents=True)
        reports.symlink_to(outside / "reports")
    elif plant == "symlinked-ancestor":
        # `.local` is the symlink — one level ABOVE the component O_NOFOLLOW would see
        # if only the leaf were checked (part-1 cold review round 2)
        worktree.mkdir(parents=True)
        (worktree / ".local").symlink_to(outside)
    elif plant == "dest-symlinked-ancestor":
        # the escape pointing the other way: the CONTROL repo's `.local` redirects writes
        reports.mkdir(parents=True)
        (reports / "abc123abc123.md").write_text("real report", encoding="utf-8")
        (tmp_path / ".local").symlink_to(outside)
    else:
        reports.mkdir(parents=True)
        if plant == "hardlink":
            os.link(secret, reports / "abc123abc123.md")
        elif plant == "wrong-name":
            (reports / ".env").write_text("TOKEN=1", encoding="utf-8")
        elif plant == "oversize":
            (reports / "abc123abc123.md").write_bytes(b"x" * (jaxflow_common.REPORT_COPY_MAX + 1))
        elif plant == "dest-symlink":
            (reports / "abc123abc123.md").write_text("real report", encoding="utf-8")
            (tmp_path / ".local" / "reports").mkdir(parents=True)
            (tmp_path / ".local" / "reports" / "abc123abc123.md").symlink_to(secret)
        elif plant == "fifo":
            # opening a FIFO for reading blocks until a writer shows up, and this copy
            # runs AFTER the merge commit -- a regression would hang the merge, not fail
            # it, so the alarm below turns the hang into an ordinary test failure.
            os.mkfifo(reports / "abc123abc123.md")

    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    if plant == "dest-symlinked-ancestor":
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                              env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
        assert exc.value.code == "agent-settings-symlink"
        assert secret.read_text(encoding="utf-8") == "SECRET"
        assert not list((outside / "reports").glob("*.tmp"))
        return
    previous = None
    if plant == "fifo":
        def _blocked(signum, frame):
            raise AssertionError("the report copy blocked on the FIFO")
        # Save BOTH the handler and any timer already armed: the autouse fixture at the
        # top of this file restores only SIGTERM and SIGHUP, so leaving `_blocked`
        # installed would make a later alarm anywhere in the suite fail in this test's
        # name (round-4 F2).
        previous = (signal.signal(signal.SIGALRM, _blocked),
                    signal.setitimer(signal.ITIMER_REAL, 5))
    try:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    finally:
        if previous is not None:
            handler, (delay, interval) = previous
            signal.setitimer(signal.ITIMER_REAL, delay, interval)
            signal.signal(signal.SIGALRM, handler)
    assert expected_skip in capsys.readouterr().out
    assert secret.read_text(encoding="utf-8") == "SECRET", "the outside file is never written"
    assert not list((outside / "reports").glob("*.tmp")), "nothing is written outside either"
    if plant == "dest-symlink":
        copied = tmp_path / ".local" / "reports" / "abc123abc123.md"
        assert copied.is_symlink() and copied.readlink() == secret, "the plant is left alone"
    elif plant != "dest-symlinked-ancestor":
        copied = tmp_path / ".local" / "reports" / "abc123abc123.md"
        assert not copied.exists() or copied.read_text(encoding="utf-8") != "SECRET"
    assert not (tmp_path / ".local" / "reports" / ".env").exists()


def test_merge_report_copy_continues_past_one_unreadable_entry(tmp_path, monkeypatch, capsys):
    """One bad report must not cost the others: the delivery still reports success, so a
    loop that aborts would lose evidence silently."""
    if os.geteuid() == 0:
        pytest.skip("mode 000 is readable for root, so the skip path never fires")
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    reports = _worktree_path(tmp_path) / ".local" / "reports"
    reports.mkdir(parents=True)
    (reports / "aaa111aaa111.md").write_text("first", encoding="utf-8")
    (reports / "bbb222bbb222.md").write_text("unreadable", encoding="utf-8")
    (reports / "bbb222bbb222.md").chmod(0o000)
    (reports / "ccc333ccc333.md").write_text("third", encoding="utf-8")
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    try:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
        dest = tmp_path / ".local" / "reports"
        assert (dest / "aaa111aaa111.md").read_text(encoding="utf-8") == "first"
        assert (dest / "ccc333ccc333.md").read_text(encoding="utf-8") == "third"
        assert "bbb222bbb222.md skipped" in capsys.readouterr().out
    finally:
        (reports / "bbb222bbb222.md").chmod(0o600)


def test_merge_refuses_when_the_commit_sha_cannot_be_read(tmp_path, monkeypatch):
    """`merge-approved` requires a full 40-hex sha and the validator rejects anything
    else, so an unchecked `rev-parse HEAD` would turn a durable commit into an
    un-auditable one (round-4 F5). Nothing is aborted -- the commit landed, and the same
    command resumes from it."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha, overrides={("git", "rev-parse", "HEAD"): _completed(128, "")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "merge-failed"
    assert "re-run the same jaxflow merge to resume" in exc.value.hint
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("git merge --abort") for f in flat), \
        "there is nothing to abort after a successful commit"
    assert not any(f.startswith(("git push", "git worktree", "git branch -d")) for f in flat)


@pytest.mark.parametrize("mutated,expected_len", [("ori", 3), ("original-grown", 14)])
def test_merge_report_copy_skips_a_report_that_changed_under_the_read(
        tmp_path, monkeypatch, capsys, mutated, expected_len):
    """`st_size` is a snapshot taken before the read. A report that shrinks would be
    copied truncated and one that grows would be copied as a stale prefix -- both are a
    silent evidence loss reported as a successful copy. Deterministic rather than
    timing-based: the first `os.read` rewrites the file, exactly as a concurrent writer
    would. The claim is about THIS path, not about `os.read` globally (round-4 F3):
    `subprocess` does call it when capturing pipe output, but `run` is injected here so no
    subprocess exists, and `real_read` is captured before the patch so the fake cannot
    recurse. `monkeypatch` restores the attribute at teardown."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    reports = _worktree_path(tmp_path) / ".local" / "reports"
    reports.mkdir(parents=True)
    victim = reports / "abc123abc123.md"
    victim.write_text("original", encoding="utf-8")          # 8 bytes at fstat time
    real_read, fired = os.read, []

    def racing_read(fd, n):
        if not fired:                                        # only the first read races
            fired.append(True)
            victim.write_text(mutated, encoding="utf-8")
        return real_read(fd, n)

    monkeypatch.setattr(os, "read", racing_read)
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert fired, "the race never fired -- the copy loop no longer calls os.read"
    assert len(victim.read_bytes()) == expected_len
    assert "abc123abc123.md skipped (changed while being read)" in capsys.readouterr().out
    assert not (tmp_path / ".local" / "reports" / "abc123abc123.md").exists(), \
        "a report whose bytes could not be trusted is not copied at all"


def test_merge_keeps_the_worktree_when_a_report_copy_fails(tmp_path, monkeypatch, capsys):
    """F3 (spec §4 guard 7 / §11): `cmd_merge` inherits the shared helper's stricter
    copy-then-remove rule -- a report that exists on disk but cannot be preserved now
    keeps the worktree instead of today's catch-and-continue removal."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    reports = _worktree_path(tmp_path) / ".local" / "reports"
    reports.mkdir(parents=True)
    (reports / "abc123abc123.md").write_bytes(b"x" * (jaxflow_common.REPORT_COPY_MAX + 1))
    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert "kept (report copy failed)" in capsys.readouterr().out
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("git worktree remove") for f in flat), \
        "a copy failure aborts before removal"
    assert not any(f == "git branch -d feat/x" for f in flat)


def test_merge_refuses_an_agents_md_that_exists_but_cannot_be_read(tmp_path, monkeypatch):
    """A policy that cannot be read is not the same as no policy: treating it as empty
    would fall through to the default branch and could direct-merge a `dual-branch-pr` repo."""
    monkeypatch.chdir(tmp_path)
    policy = tmp_path / "AGENTS.md"
    policy.write_bytes(b"**Preset: `dual-branch-pr`**\n\xff\xfe not utf-8\n")
    fake_run, calls = _merge_runner(
        tmp_path, script={("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "preset-unknown"
    assert "could not be read" in exc.value.hint
    _assert_no_mutation(calls)


def test_merge_resume_writes_the_target_as_the_status_branch(tmp_path, monkeypatch):
    """A resume never switches to the target, so `_current_branch()` would report whatever
    the tech lead was standing on — or nothing, from a detached HEAD."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    (tmp_path / ".jax-os").mkdir()
    (tmp_path / ".jax-os" / "status.md").write_text(
        # fixture fix (execution rule): see the same note in
        # test_merge_writes_status_md_stage_ship_and_drops_the_gate -- `_fixed_now()`
        # returns 2026-01-01, so `updated` must predate it or the staleness guard skips
        # the write.
        "---\nproject: Demo\nstage: review\nbuilder: codex\nbranch: feat/x\n"
        "updated: 2020-01-01T00:00:00-03:00\n---\n\n## Now\nOld.\n",
        encoding="utf-8")
    fake_run, _ = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        # fixture fix (execution rule): without this entry the fake's default (exit 0)
        # reads as "the branch is present", so `branch_present` never flips to False and
        # the flow falls through to the `feat/x^{commit}` probe below -- which THIS
        # fixture also scripts as gone (128), correctly raising sha-mismatch instead of
        # exercising this test's actual resume-completes path. Exit 1 is "no such ref".
        ("git", "show-ref", "--verify"): _completed(1, ""),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(128, ""),
        ("git", "remote", "get-url"): _completed(2, ""),   # 2 == no origin
        # a detached HEAD: `git branch --show-current` prints nothing
        ("git", "branch", "--show-current"): _completed(0, "\n"),
    })
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    text = (tmp_path / ".jax-os" / "status.md").read_text(encoding="utf-8")
    assert "branch: main" in text, "the delivery target, not the caller's checkout"
    assert "stage: ship" in text


def test_merge_refuses_when_the_switch_to_the_target_fails(tmp_path, monkeypatch):
    """F8: a failed `git switch` returns non-zero while leaving the previous branch checked
    out, so the return code has to be checked and not just the branch afterwards. This is
    the verb's ONLY switch (round-4 F12)."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha, switch_fails="main")
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "target-mismatch"
    assert "git switch main failed" in exc.value.hint
    _assert_no_mutation(calls)


def test_merge_resume_refuses_a_subject_that_merely_contains_the_branch(tmp_path, monkeypatch):
    """F7: `--phase` is free text. A real merge of `other` titled `Release (merge feat/x)`
    writes the subject `feat: Release (merge feat/x) (merge other)`; a substring check
    would accept a resume for feat/x and delete it. The exact subject shape does not."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        ("git", "log", "-1", "--format=%s"): _completed(
            0, "feat: Release (merge feat/x) (merge other)\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{sha}\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha, phase="Release (merge feat/x)"), run=fake_run,
                          post=_never_post, env=_merge_env(), now=_fixed_now,
                          allowlist_root=tmp_path.parent)
    assert exc.value.code == "sha-mismatch"
    _assert_no_mutation(calls)


def test_merge_resume_refuses_a_branch_the_merge_commit_does_not_name(tmp_path, monkeypatch):
    """F4: an alias branch pointing at the approved sha passes every sha check there is --
    reproduced on real git, where the alias and its registered worktree were both removed
    while the delivered branch survived. Only the merge commit's own subject says which
    branch the approval delivered."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        # jaxflow's own merge names feat/x; the caller asked to resume `alias`
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        ("git", "rev-parse", "--verify", "alias^{commit}"): _completed(0, f"{sha}\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(branch="alias", sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "sha-mismatch"
    assert "does not record alias" in exc.value.hint
    _assert_no_mutation(calls)


def test_merge_resume_refuses_when_the_branch_no_longer_names_the_approved_sha(tmp_path, monkeypatch):
    """F2: a resume must not delete a branch it never verified."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        # the caller-named branch points somewhere else entirely
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, "9" * 40 + "\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "sha-mismatch"
    _assert_no_mutation(calls)


def test_merge_resume_skips_cleanup_when_the_branch_is_already_gone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha, posted = "a" * 40, []
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        # fixture fix (execution rule): without this, the fake's default (exit 0) reads
        # as "branch present" and the flow never reaches the "already gone" path this
        # test is named for. Exit 1 is "no such ref".
        ("git", "show-ref", "--verify"): _completed(1, ""),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(128, ""),
        ("git", "remote", "get-url"): _completed(2, ""),   # 2 == no origin
    })
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=posted.append,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("git worktree remove") or f.startswith("git branch -d") for f in flat)
    assert posted, "an already-cleaned resume still completes its audit"


def test_merge_never_removes_a_worktree_git_does_not_register_for_that_branch(tmp_path, monkeypatch):
    """F2, second half: the derived path must be the REGISTERED worktree of that branch."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    worktree = tmp_path.parent / f"{jaxflow_common.slugify_project(tmp_path.name)}-feat-x"
    worktree.mkdir()
    # git knows this path, but as the worktree of a DIFFERENT branch
    fake_run, calls = _switch_aware_runner(tmp_path, sha, overrides={
        ("git", "worktree", "list"): _completed(
            0, f"worktree {worktree}\nHEAD {'f' * 40}\nbranch refs/heads/other\n\n")})
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("git worktree remove") for f in flat)
    assert not any(f == "git branch -d feat/x" for f in flat)
    assert worktree.is_dir(), "an unregistered path is never removed"


def test_merge_commit_failure_aborts_and_never_audits_or_pushes(tmp_path, monkeypatch):
    """F3: this repo's own commit-msg hook is exactly the failure class this guards."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha, overrides={
        ("git", "commit",): _completed(1, "hook refused: Co-Authored-By is banned")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "merge-failed"
    assert "hook refused" in exc.value.hint
    flat = [" ".join(c) for c in calls]
    assert "git merge --abort" in flat
    # ZERO cleanup, branch deletion included (cold review F3): nothing was delivered
    assert not any(f.startswith("git push") or f.startswith("git worktree")
                   or f.startswith("git branch -d") for f in flat)


_RELEASE_POLICY = (
    "## Deploy policy\n\n**Preset: `dual-branch`** — notes.\n\n"
    "Delivery target: `release`.\n"
)
_TARGET_512 = "ab/" * 170 + "cd"


@pytest.mark.parametrize("release_carries_merge", [True, False])
def test_merge_refuses_when_approved_target_differs_from_policy(
        tmp_path, monkeypatch, release_carries_merge):
    """MOA-458: approve main, policy now release — refuse before resume or fresh delivery."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    status = tmp_path / ".jax-os" / "status.md"
    status.parent.mkdir()
    original = "---\nproject: Demo\nstage: review\n---\n\n## Now\nOld.\n"
    status.write_text(original, encoding="utf-8")
    script = {
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "release^2"): _completed(
            0 if release_carries_merge else 128,
            f"{sha}\n" if release_carries_merge else ""),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{sha}\n"),
        ("git", "status", "--porcelain"): _completed(0, ""),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/release\n"),
    }
    fake_run, calls = _merge_runner(tmp_path, script=script, agents=_RELEASE_POLICY)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha, target="main"), run=fake_run,
                          post=_never_post, env=_merge_env(), now=_fixed_now,
                          allowlist_root=tmp_path.parent)
    assert exc.value.code == "target-mismatch"
    assert "main" in exc.value.hint and "release" in exc.value.hint
    _assert_no_delivery(calls, policy_target="release")
    assert status.read_text(encoding="utf-8") == original


def test_merge_happy_path_with_matching_non_main_policy_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha, posted = "a" * 40, []
    _agents(tmp_path, _TEMPLATE_DUAL_BRANCH_BLOCK)
    fake_run, calls = _switch_aware_runner(tmp_path, sha, target="staging")
    jaxflow.cmd_merge(_MergeArgs(sha=sha, target="staging"), run=fake_run,
                      post=posted.append, env=_merge_env(), now=_fixed_now,
                      allowlist_root=tmp_path.parent)
    assert ["git", "switch", "staging"] in calls
    assert posted[0]["payload"]["target"] == "staging"


@pytest.mark.parametrize("target", [
    None, "", " main", "main ", "ma in", "-main", "x" * 513, "a\udcff",
])
def test_merge_refuses_an_unusable_approved_target(tmp_path, monkeypatch, target):
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(
        tmp_path, script={("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(target=target), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "target-invalid"
    _assert_no_delivery(calls, policy_target="main")
    assert not any(c[:2] == ["git", "check-ref-format"] for c in calls)


def test_merge_direct_call_missing_target_is_target_invalid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(
        tmp_path, script={("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_merge_args_without_target(), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "target-invalid"
    _assert_no_delivery(calls, policy_target="main")


@pytest.mark.parametrize("target,code", [
    ("main^", "target-invalid"),
    ("main..x", "target-invalid"),
    ("main@{1}", "target-invalid"),
    ("feat/foo/bar", "target-mismatch"),
    ("feat/café", "target-mismatch"),
    (_TARGET_512, "target-mismatch"),
])
def test_merge_approved_target_uses_real_check_ref_format(tmp_path, monkeypatch, target, code):
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_run_real_ref_format(
        tmp_path, script={("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(target=target), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == code
    assert ["git", "check-ref-format", f"refs/heads/{target}"] in calls
    _assert_no_delivery(calls, policy_target="main")


@pytest.mark.parametrize("override,code", [
    pytest.param({"sha": "abc1234"}, "sha-mismatch", id="sha"),
    pytest.param({"branch": ""}, "branch-invalid", id="branch"),
    pytest.param({"phase": ""}, "phase-invalid", id="phase"),
])
def test_merge_gates_win_over_an_invalid_target(tmp_path, monkeypatch, override, code):
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(
        tmp_path, script={("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(target="-nope", **override), run=fake_run,
                          post=_never_post, env=_merge_env(), now=_fixed_now,
                          allowlist_root=tmp_path.parent)
    assert exc.value.code == code
    _assert_no_mutation(calls)
    assert not any(c[:2] == ["git", "check-ref-format"] for c in calls)


def test_merge_identical_retry_resumes_after_push_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    posted = []
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha, overrides={("git", "push"): _completed(1, "rejected")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=posted.append,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "push-failed"
    assert any(" ".join(c).startswith("git commit") for c in calls)

    posted2 = []
    fake_run2, calls2 = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{sha}\n"),
        ("git", "remote", "get-url"): _completed(0, "git@github:x/y.git\n"),
    })
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run2, post=posted2.append,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    flat2 = [" ".join(c) for c in calls2]
    for banned in ("git merge --no-ff", "git commit", "/bin/bash -lc", "git write-tree",
                   "git switch"):
        assert not any(f.startswith(banned) for f in flat2), f"{banned} ran on a resume"
    assert posted2 and posted2[0]["type"] == "merge-approved"
    assert any(f.startswith("git push") for f in flat2)


def test_merge_identical_retry_resumes_after_audit_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40

    def failing_post(event):
        raise RuntimeError("hub down")

    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=failing_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "hub-unreachable"
    assert any(" ".join(c).startswith("git commit") for c in calls)

    posted2 = []
    fake_run2, calls2 = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{sha}\n"),
        ("git", "remote", "get-url"): _completed(2, ""),
    })
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run2, post=posted2.append,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    flat2 = [" ".join(c) for c in calls2]
    for banned in ("git merge --no-ff", "git commit", "/bin/bash -lc", "git write-tree"):
        assert not any(f.startswith(banned) for f in flat2), f"{banned} ran on a resume"
    assert posted2 and posted2[0]["type"] == "merge-approved"


def test_merge_subparser_requires_every_flag_including_target_and_rejects_message():
    args = jaxflow.parse_args(["merge", "feat/x", "--sha", "a" * 40,
                              "--phase", "Phase X", "--checks", "pytest -q",
                              "--target", "main"])
    assert (args.command, args.branch, args.sha) == ("merge", "feat/x", "a" * 40)
    assert (args.phase, args.checks, args.target) == ("Phase X", "pytest -q", "main")
    for missing in (["merge", "feat/x", "--sha", "a" * 40, "--phase", "P", "--target", "main"],
                    ["merge", "feat/x", "--sha", "a" * 40, "--checks", "true", "--target", "main"],
                    ["merge", "--sha", "a" * 40, "--phase", "P", "--checks", "true",
                     "--target", "main"],
                    ["merge", "feat/x", "--sha", "a" * 40, "--phase", "P", "--checks", "true"]):
        with pytest.raises(SystemExit):
            jaxflow.parse_args(missing)
    with pytest.raises(SystemExit):
        jaxflow.parse_args(["merge", "feat/x", "--sha", "a" * 40, "--phase", "P",
                            "--checks", "true", "--target", "main", "--message", "x"])


def test_merge_rejects_a_blank_checks_command():
    # `required=True` only demands the flag; `bash -lc ""` exits 0, so a blank value would
    # let the merge commit and audit with nothing verified (whole-branch review, F1).
    for blank in ("", "   ", "\t", "\n", " \t\n "):
        with pytest.raises(SystemExit):
            jaxflow.parse_args(["merge", "feat/x", "--sha", "a" * 40,
                                "--phase", "P", "--checks", blank, "--target", "main"])
    # A command that merely LOOKS trivial is the caller's business, not the parser's.
    assert jaxflow.parse_args(["merge", "feat/x", "--sha", "a" * 40,
                               "--phase", "P", "--checks", "true",
                               "--target", "main"]).checks == "true"


# ---- MOA-510: merge reuses the test evidence a diff review recorded ----

_REUSE_SHA = "a" * 40


def _tests_txt(*frames):
    """One COMMAND/output/EXIT frame per (command, exit_code), the shape
    `_write_verify_tests_file` writes and `jr.parse_tests_frames` reads."""
    return "".join(f"COMMAND: {cmd}\nsome output\nEXIT: {code}\n" for cmd, code in frames)


def _seed_reusable_review(tmp_path, db, *, branch="feat/x", verify="pnpm test", build="pnpm build",
                          tests="ok", review_head=_REUSE_SHA, review_id="e" * 12, review_ts="t2",
                          review_branch=None, manifest=True, with_builder=True):
    """Seeds everything `_merge_checks_reuse` reads: a finished builder run (started payload
    carries `verify`/`build`), a diff review run-started row for `branch`, that review's
    manifest (`kind: diff`, `builder_run_id`, `verify.head_sha`) under
    `<tmp_path>/.local/runs/<review_id>/`, and the build worktree's tests.txt under
    `<worktree>/.local/reports/<builder>.tests.txt`.
    `tests`: "ok" = both frames EXIT 0; None = no evidence file; any other str = written raw.
    `manifest`: True = regular file; False = none; "symlink" = `manifest.json` is a symlink to a
    valid manifest OUTSIDE the run directory (the escape the helper must refuse)."""
    project = jaxflow_common.slugify_project(tmp_path.name)
    builder_id = "b" * 12
    con = sqlite3.connect(db)
    if not con.execute("SELECT 1 FROM sqlite_master WHERE name = 'workflow_events'").fetchone():
        con.close()
        con = _fresh_db(db)
    if with_builder:
        started = {"kind": "build", "target": branch, "repo": str(tmp_path), "verify": verify}
        if build:
            started["build"] = build
        _insert(con, builder_id, project, "builder", "run-started", started, ts="t0")
        _insert(con, builder_id, project, "builder", "run-finished",
                {"exit_code": 0, "contract_status": "ok", "head_sha": _REUSE_SHA}, ts="t0")
    _insert(con, review_id, project, "reviewer", "run-started",
            {"kind": "diff", "target": review_branch or branch, "repo": str(tmp_path),
             "builder_run_id": builder_id}, ts=review_ts)
    con.close()
    if manifest:
        mdir = tmp_path / ".local" / "runs" / review_id
        mdir.mkdir(parents=True, exist_ok=True)
        body = json.dumps({
            "kind": "diff", "builder_run_id": builder_id,
            "verify": {"mode": "rerun", "reason": "head-sha-changed",
                       "source_build_run_id": builder_id, "head_sha": review_head},
        })
        if manifest == "symlink":
            outside = tmp_path / "outside-manifest.json"
            outside.write_text(body, encoding="utf-8")
            (mdir / "manifest.json").symlink_to(outside)
        else:
            (mdir / "manifest.json").write_text(body, encoding="utf-8")
    worktree = _worktree_path(tmp_path, branch)
    if tests is not None:
        reports = worktree / ".local" / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        text = _tests_txt((verify, 0), (build, 0)) if tests == "ok" else tests
        (reports / f"{builder_id}.tests.txt").write_text(text, encoding="utf-8")
    else:
        worktree.mkdir(parents=True, exist_ok=True)


def _reuse_run(worktree, *, porcelain="", registered=True):
    listing = f"worktree {worktree}\nbranch refs/heads/feat/x\n" if registered else ""

    def run(argv, cwd=None):
        if argv[:3] == ["git", "worktree", "list"]:
            return _completed(0, listing)
        if argv == ["git", "status", "--porcelain", "--untracked-files=all"]:
            return _completed(0, porcelain)
        raise AssertionError(f"unexpected git call: {argv}")
    return run


def _call_reuse(tmp_path, db, *, checks="pnpm test", sha=_REUSE_SHA, recheck=False, **run_kw):
    worktree = _worktree_path(tmp_path)
    return jaxflow._merge_checks_reuse(
        _reuse_run(worktree, **run_kw), tmp_path, jaxflow_common.slugify_project(tmp_path.name),
        worktree, "feat/x", sha, checks, allowlist_root=tmp_path.parent, recheck=recheck,
        db_path=db)


def test_merge_checks_reuse_accepts_matching_evidence(tmp_path):
    db = tmp_path / "jaxos.db"
    _seed_reusable_review(tmp_path, db)
    assert _call_reuse(tmp_path, db) == (True, "all-conditions-met", "e" * 12)


def test_merge_checks_reuse_accepts_a_legacy_build_without_a_build_command(tmp_path):
    db = tmp_path / "jaxos.db"
    _seed_reusable_review(tmp_path, db, build=None, tests=_tests_txt(("pnpm test", 0)))
    assert _call_reuse(tmp_path, db) == (True, "all-conditions-met", "e" * 12)


@pytest.mark.parametrize("seed,call,reason", [
    ({}, {"recheck": True}, "recheck-forced"),
    ({"review_head": "9" * 40}, {}, "head-sha-changed"),
    ({}, {"sha": "9" * 40}, "head-sha-changed"),
    ({}, {"checks": "pnpm other"}, "verify-command-changed"),
    ({}, {"porcelain": "?? scratch.txt\n"}, "worktree-dirty"),
    ({}, {"porcelain": " M src/a.ts\n"}, "worktree-dirty"),
    ({"tests": None}, {}, "tests-file-missing"),
    ({"tests": "garbage\n"}, {}, "frame-1-missing"),
    ({"tests": _tests_txt(("pnpm test", 0), ("pnpm build", 1))}, {}, "prior-verify-failed"),
    ({"manifest": False}, {}, "manifest-unreadable"),
    ({"manifest": "symlink"}, {}, "lookup-failed"),
    ({"with_builder": False}, {}, "no-build-run"),
    ({"review_branch": "feat/other"}, {}, "no-diff-review"),
    ({}, {"registered": False}, "worktree-missing"),
])
def test_merge_checks_reuse_reasons_when_not_reusable(tmp_path, seed, call, reason):
    db = tmp_path / "jaxos.db"
    _seed_reusable_review(tmp_path, db, **seed)
    assert _call_reuse(tmp_path, db, **call) == (False, reason, None)


def test_merge_checks_reuse_refuses_a_failed_verify_masked_by_a_passing_build(tmp_path):
    # Review Focus 1 (spec F1): the review's own failed rerun leaves `verify EXIT 1` then
    # `build EXIT 0` before `verify-failed`. The frames must be parsed with BOTH commands, so
    # the build frame's EXIT 0 can never stand in for the verify frame's EXIT 1.
    db = tmp_path / "jaxos.db"
    _seed_reusable_review(
        tmp_path, db, tests=_tests_txt(("pnpm test", 1), ("pnpm build", 0)))
    assert _call_reuse(tmp_path, db) == (False, "prior-verify-failed", None)


def test_merge_checks_reuse_uses_only_the_newest_diff_review_of_the_branch(tmp_path):
    # Review Focus 5: the OLDER review matches the SHA, the NEWER one does not -> not reusable.
    db = tmp_path / "jaxos.db"
    _seed_reusable_review(tmp_path, db, review_id="d" * 12, review_ts="t1")
    _seed_reusable_review(tmp_path, db, review_id="f" * 12, review_ts="t2", review_head="9" * 40)
    assert _call_reuse(tmp_path, db) == (False, "head-sha-changed", None)
    # And the other way round: an older stale review does not hide a newer matching one.
    db2 = tmp_path / "jaxos2.db"
    _seed_reusable_review(tmp_path, db2, review_id="d" * 12, review_ts="t1", review_head="9" * 40)
    _seed_reusable_review(tmp_path, db2, review_id="f" * 12, review_ts="t2")
    assert _call_reuse(tmp_path, db2) == (True, "all-conditions-met", "f" * 12)


def test_merge_checks_reuse_never_raises_on_an_unreadable_ledger(tmp_path):
    # Review Focus 3: a lookup failure is "not reusable", never an exception or a refusal.
    worktree = _worktree_path(tmp_path)
    worktree.mkdir(parents=True)
    assert _call_reuse(tmp_path, tmp_path / "missing.db") == (False, "lookup-failed", None)
    bad = tmp_path / "bad.db"
    bad.write_text("not a sqlite file", encoding="utf-8")
    assert _call_reuse(tmp_path, bad) == (False, "lookup-failed", None)


def test_merge_checks_reuse_asks_git_for_untracked_files_explicitly(tmp_path):
    # Review F2: plain `git status --porcelain` obeys `status.showUntrackedFiles`; with it off a
    # worktree holding untracked files would read clean. The argv must override the config.
    db = tmp_path / "jaxos.db"
    _seed_reusable_review(tmp_path, db)
    worktree = _worktree_path(tmp_path)
    seen = []
    inner = _reuse_run(worktree)

    def run(argv, cwd=None):
        seen.append(list(argv))
        return inner(argv, cwd)

    jaxflow._merge_checks_reuse(
        run, tmp_path, jaxflow_common.slugify_project(tmp_path.name), worktree, "feat/x", _REUSE_SHA,
        "pnpm test", allowlist_root=tmp_path.parent, db_path=db)
    assert ["git", "status", "--porcelain", "--untracked-files=all"] in seen


def test_merge_checks_reuse_treats_a_non_dict_verify_field_as_an_unreadable_manifest(tmp_path):
    db = tmp_path / "jaxos.db"
    _seed_reusable_review(tmp_path, db)
    (tmp_path / ".local" / "runs" / ("e" * 12) / "manifest.json").write_text(
        json.dumps({"kind": "diff", "builder_run_id": "b" * 12, "verify": "oops"}),
        encoding="utf-8")
    assert _call_reuse(tmp_path, db) == (False, "manifest-unreadable", None)


def _pr_stateful_run(fake_run, calls, open_view):
    """`gh pr view` answers OPEN once, then MERGED (what the real merge does); every other
    call goes to the scripted fake."""
    seen = {"n": 0}

    def run(argv, cwd=None):
        if tuple(argv[:3]) == ("gh", "pr", "view"):
            calls.append(list(argv))
            seen["n"] += 1
            if seen["n"] == 1:
                return _completed(0, json.dumps(open_view))
            return _completed(0, json.dumps(
                {**open_view, "state": "MERGED", "mergeCommit": {"oid": "d" * 40}}))
        return fake_run(argv, cwd)
    return run


def _run_pr_merge(tmp_path, monkeypatch, *, seed=None, setup_kw=None, dirty_untracked=False,
                  dirty_tracked=False, **args_kw):
    """Runs the PR-path merge of feat/x. Returns (rc_or_Refusal, calls, events)."""
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db, **(setup_kw or {}))
    if seed is not None:
        _seed_reusable_review(tmp_path, db, **seed)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    inner = _pr_stateful_run(fake_run, calls, json.loads(script[("gh", "pr", "view")].stdout))

    def run(argv, cwd=None):
        if argv[:3] == ["git", "status", "--porcelain"]:
            if argv[3:] == ["--untracked-files=no"] and dirty_tracked:
                return _completed(0, " M tracked.txt\n")
            if argv[3:] == ["--untracked-files=all"] and dirty_untracked:
                return _completed(0, "?? scratch.txt\n")
        return inner(argv, cwd)

    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        try:
            rc = jaxflow.cmd_merge(
                _MergeArgs(target="staging", checks="pnpm test", **args_kw), run=run,
                post=_ledger_post(db, events), env=_merge_env(), now=_fixed_now,
                allowlist_root=tmp_path.parent)
        except ji.Refusal as exc:
            rc = exc
    return rc, calls, events


def _bash_ran(calls):
    return any(c[:2] == ["/bin/bash", "-lc"] for c in calls)


def test_merge_pr_reuses_the_review_evidence_and_audits_it(tmp_path, monkeypatch, capsys):
    rc, calls, events = _run_pr_merge(tmp_path, monkeypatch, seed={})
    assert rc == jaxflow_common.OK
    assert not _bash_ran(calls), "the checks command must not run when the evidence is reusable"
    assert events[-1]["payload"]["checks"] == {
        "mode": "reused", "source_review_run_id": "e" * 12, "head_sha": _REUSE_SHA}
    assert any(c[:3] == ["gh", "pr", "merge"] for c in calls), "the merge itself still happens"
    assert capsys.readouterr().out.strip().splitlines()[-1] == f"checks: reused from review {'e' * 12} at {'a' * 12}"


@pytest.mark.parametrize("seed,args_kw", [
    pytest.param(None, {}, id="no-diff-review"),
    pytest.param({"review_head": "9" * 40}, {}, id="head-moved-since-review"),
    pytest.param({"verify": "pnpm other"}, {}, id="checks-differ-from-recorded-verify"),
    pytest.param({"tests": None}, {}, id="evidence-missing"),
    pytest.param({"manifest": "symlink"}, {}, id="manifest-symlink-escape"),
    pytest.param({"tests": _tests_txt(("pnpm test", 0), ("pnpm build", 1))}, {}, id="build-frame-failed"),
    pytest.param({"tests": _tests_txt(("pnpm test", 1), ("pnpm build", 0))}, {}, id="masked-verify"),
    pytest.param({}, {"recheck": True}, id="recheck"),
])
def test_merge_pr_runs_the_checks_when_reuse_does_not_hold(tmp_path, monkeypatch, capsys, seed, args_kw):
    rc, calls, events = _run_pr_merge(tmp_path, monkeypatch, seed=seed, **args_kw)
    assert rc == jaxflow_common.OK
    assert _bash_ran(calls)
    assert events[-1]["payload"]["checks"] == {"mode": "run"}
    assert capsys.readouterr().out.strip().splitlines()[-1] == "checks: pnpm test exit 0"


def test_merge_pr_untracked_dirt_runs_the_checks_without_a_refusal(tmp_path, monkeypatch):
    # Review Focus 3: the PR guard ignores untracked files, so untracked dirt only disables reuse.
    rc, calls, events = _run_pr_merge(tmp_path, monkeypatch, seed={}, dirty_untracked=True)
    assert rc == jaxflow_common.OK
    assert _bash_ran(calls)
    assert events[-1]["payload"]["checks"] == {"mode": "run"}


def test_merge_pr_tracked_dirt_still_refuses_even_with_reusable_evidence(tmp_path, monkeypatch):
    rc, calls, _ = _run_pr_merge(tmp_path, monkeypatch, seed={}, dirty_tracked=True)
    assert isinstance(rc, ji.Refusal) and rc.code == "dirty-tracked-tree"
    assert not _bash_ran(calls)


def test_merge_pr_missing_build_worktree_still_refuses_checks_failed(tmp_path, monkeypatch):
    rc, calls, _ = _run_pr_merge(tmp_path, monkeypatch, seed={}, setup_kw={"worktree_registered": False})
    assert isinstance(rc, ji.Refusal) and rc.code == "checks-failed"


def test_merge_pr_failing_checks_still_refuse_when_reuse_does_not_hold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    script[("/bin/bash", "-lc")] = _completed(1, "boom")
    _seed_reusable_review(tmp_path, db, review_head="9" * 40)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_merge(_MergeArgs(target="staging", checks="pnpm test"), run=fake_run,
                              post=_never_post, env=_merge_env(), now=_fixed_now,
                              allowlist_root=tmp_path.parent)
    assert exc.value.code == "checks-failed"
    assert not any(c[:3] == ["gh", "pr", "merge"] for c in calls)


@pytest.mark.parametrize("recheck", [False, True])
def test_merge_pr_already_merged_recovery_audits_checks_resumed(tmp_path, monkeypatch, capsys, recheck):
    # Review Focus 4: nothing runs and nothing is reused on the recovery arm; --recheck is inert.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    script[("gh", "pr", "view")] = _completed(0, json.dumps(_GH_PR_VIEW_MERGED))
    _seed_reusable_review(tmp_path, db)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        rc = jaxflow.cmd_merge(
            _MergeArgs(target="staging", checks="pnpm test", recheck=recheck), run=fake_run,
            post=_ledger_post(db, events), env=_merge_env(), now=_fixed_now,
            allowlist_root=tmp_path.parent)
    assert rc == jaxflow_common.OK
    assert not _bash_ran(calls)
    assert events[-1]["payload"]["checks"] == {"mode": "resumed"}
    assert not any(line.startswith("checks:") for line in capsys.readouterr().out.splitlines())


def test_merge_pr_release_head_never_consults_the_reuse_lookup(tmp_path, monkeypatch):
    # D3: a release/* head has no build worktree and no diff review; it always runs.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jaxflow, "_merge_checks_reuse",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("reuse consulted")))
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db, branch="release/x", target="main",
                                worktree_registered=False)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    inner = _pr_stateful_run(fake_run, calls, json.loads(script[("gh", "pr", "view")].stdout))
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        rc = jaxflow.cmd_merge(
            _MergeArgs(branch="release/x", target="main", checks="pnpm test"), run=inner,
            post=_ledger_post(db, events), env=_merge_env(), now=_fixed_now,
            allowlist_root=tmp_path.parent)
    assert rc == jaxflow_common.OK
    assert _bash_ran(calls)
    assert events[-1]["payload"]["checks"] == {"mode": "run"}


def test_merge_subparser_accepts_recheck_and_defaults_it_off():
    base = ["merge", "feat/x", "--sha", "a" * 40, "--phase", "P", "--checks", "true",
            "--target", "main"]
    assert jaxflow.parse_args(base).recheck is False
    assert jaxflow.parse_args(base + ["--recheck"]).recheck is True


def test_merge_still_requires_checks_even_with_recheck():
    with pytest.raises(SystemExit):
        jaxflow.parse_args(["merge", "feat/x", "--sha", "a" * 40, "--phase", "P",
                            "--target", "main", "--recheck"])



def _run_local_ff(tmp_path, monkeypatch, *, seed=None, make_worktree=True, dirty=None, bash=None,
                  review_target_moved=False, **args_kw):
    """Local-preset merge where the target has not moved (`merge-base --is-ancestor` says yes).
    Returns (rc_or_Refusal, calls, events). `dirty` is the `git status --porcelain` text the
    BUILD worktree reports (the control repo's own preflight status stays clean)."""
    monkeypatch.chdir(tmp_path)
    sha = _REUSE_SHA
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    worktree = _worktree_path(tmp_path)
    if make_worktree:
        worktree.mkdir(parents=True)
    if seed is not None:
        _seed_reusable_review(tmp_path, db, **seed)
    overrides = {("git", "merge-base", "--is-ancestor"):
                 _completed(1 if review_target_moved else 0, "")}
    if bash is not None:
        overrides[("/bin/bash", "-lc")] = bash
    base, calls = _switch_aware_runner(tmp_path, sha, overrides=overrides)

    def run(argv, cwd=None):
        if (dirty and argv[:3] == ["git", "status", "--porcelain"] and cwd is not None
                and Path(cwd) == worktree):
            return _completed(0, dirty)
        return base(argv, cwd=cwd)

    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        try:
            rc = jaxflow.cmd_merge(
                _MergeArgs(sha=sha, checks="pnpm test", **args_kw), run=run,
                post=lambda e: events.append(e) or {"ok": True}, env=_merge_env(),
                now=_fixed_now, allowlist_root=tmp_path.parent)
        except ji.Refusal as exc:
            rc = exc
    return rc, calls, events


def test_merge_local_fast_forward_reuses_the_review_evidence_and_prints_why(tmp_path, monkeypatch, capsys):
    rc, calls, events = _run_local_ff(tmp_path, monkeypatch, seed={})
    assert rc == jaxflow_common.OK
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("/bin/bash -lc") for f in flat), "no checks call on reuse"
    assert not any(f.startswith("git diff --quiet") for f in flat)
    assert any(f.startswith("git commit") for f in flat), "the merge commit must still be made"
    assert capsys.readouterr().out.strip().splitlines()[-2:] == [
        f"merged {'c' * 40} pushed origin/main",
        f"checks: reused from review {'e' * 12} at {'a' * 12}",
    ]
    assert events[-1]["payload"]["checks"] == {
        "mode": "reused", "source_review_run_id": "e" * 12, "head_sha": _REUSE_SHA}


@pytest.mark.parametrize("kw", [
    pytest.param({"seed": None}, id="no-diff-review"),
    pytest.param({"seed": {"review_head": "9" * 40}}, id="head-moved-since-review"),
    pytest.param({"seed": {"verify": "pnpm other"}}, id="checks-differ-from-recorded-verify"),
    pytest.param({"seed": {"tests": None}}, id="evidence-missing"),
    pytest.param({"seed": {"manifest": "symlink"}}, id="manifest-symlink-escape"),
    pytest.param({"seed": {"tests": _tests_txt(("pnpm test", 1), ("pnpm build", 0))}}, id="masked-verify"),
    pytest.param({"seed": {}, "dirty": " M src/a.ts\n"}, id="dirty-tracked"),
    pytest.param({"seed": {}, "dirty": "?? scratch.txt\n"}, id="dirty-untracked"),
    pytest.param({"seed": {}, "recheck": True}, id="recheck"),
    pytest.param({"seed": {}, "review_target_moved": True}, id="target-moved"),
])
def test_merge_local_fast_forward_runs_the_checks_when_reuse_does_not_hold(tmp_path, monkeypatch, capsys, kw):
    rc, calls, events = _run_local_ff(tmp_path, monkeypatch, **kw)
    assert rc == jaxflow_common.OK, "a failed reuse lookup must never refuse the merge"
    assert any(" ".join(c).startswith("/bin/bash -lc pnpm test") for c in calls)
    assert events[-1]["payload"]["checks"] == {"mode": "run"}
    assert capsys.readouterr().out.strip().splitlines()[-1] == "checks: pnpm test exit 0"


def test_merge_local_fast_forward_runs_the_checks_when_the_build_worktree_is_missing(tmp_path, monkeypatch):
    # Review Focus 3: the local path has no build-worktree prerequisite; a missing one only
    # disables reuse. (`_seed_reusable_review` would create the directory, so seed nothing.)
    rc, calls, events = _run_local_ff(tmp_path, monkeypatch, seed=None, make_worktree=False)
    assert rc == jaxflow_common.OK
    assert any(" ".join(c).startswith("/bin/bash -lc") for c in calls)
    assert events[-1]["payload"]["checks"] == {"mode": "run"}


def test_merge_local_fast_forward_after_a_post_review_commit_runs_and_refuses_on_failure(tmp_path, monkeypatch):
    # Review Focus 2 / D4 acceptance: a branch with a commit after its last diff review no
    # longer merges untested.
    rc, calls, events = _run_local_ff(
        tmp_path, monkeypatch, seed={"review_head": "9" * 40}, bash=_completed(1, "boom"))
    assert isinstance(rc, ji.Refusal) and rc.code == "checks-failed"
    flat = [" ".join(c) for c in calls]
    assert any(f.startswith("git merge --abort") for f in flat), "the merge must be aborted"
    assert not any(f.startswith("git commit") for f in flat)
    assert events == [], "nothing may be audited for a refused merge"

def test_main_routes_merge_and_maps_a_refusal_to_exit_2(monkeypatch, capsys):
    seen = {}

    def fake_cmd_merge(args, **kwargs):
        seen["branch"] = args.branch
        return jaxflow_common.OK

    monkeypatch.setattr(jaxflow, "cmd_merge", fake_cmd_merge)
    argv = ["merge", "feat/x", "--sha", "a" * 40, "--phase", "P", "--checks", "true",
            "--target", "main"]
    assert jaxflow.main(argv) == jaxflow_common.OK
    assert seen["branch"] == "feat/x"

    def refusing(args, **kwargs):
        exc = ji.Refusal("preset-unsupported")
        exc.hint = "hint: PR preset"
        raise exc

    monkeypatch.setattr(jaxflow, "cmd_merge", refusing)
    assert jaxflow.main(argv) == jaxflow_common.REFUSED
    err = capsys.readouterr().err
    assert "preset-unsupported" in err and "hint: PR preset" in err


# ---- MOA-470: child.log capture/finalize helpers, isolated from any worker ----

def test_open_child_log_creates_the_file_mode_0600_before_anything_is_written():
    with TemporaryDirectory() as raw:
        log_path = Path(raw) / "runs" / "r1" / "child.log"
        fd = jaxflow_workerkit._open_child_log(log_path)
        try:
            assert stat.S_IMODE(os.fstat(fd).st_mode) == 0o600
            assert log_path.stat().st_size == 0
        finally:
            os.close(fd)


def test_open_child_log_refuses_to_reuse_an_existing_file():
    with TemporaryDirectory() as raw:
        log_path = Path(raw) / "runs" / "r1" / "child.log"
        fd = jaxflow_workerkit._open_child_log(log_path)
        os.close(fd)
        with pytest.raises(FileExistsError):
            jaxflow_workerkit._open_child_log(log_path)


def test_capture_child_output_writes_head_marker_and_tail_when_over_cap():
    with TemporaryDirectory() as raw:
        log_path = Path(raw) / "child.log"
        fd = jaxflow_workerkit._open_child_log(log_path)
        # head_cap=10, tail_cap=5: 20 raw bytes total, 5 over the 15-byte cap.
        stream = io.BytesIO(b"0123456789abcdefghij")
        jaxflow_workerkit._capture_child_output(stream, fd, head_cap=10, tail_cap=5)
        jaxflow_workerkit._finalize_child_log(fd, log_path)
        assert log_path.read_bytes() == b"0123456789[jaxflow: 5 bytes omitted]\nfghij"


def test_capture_child_output_boundary_exact_cap_has_no_marker():
    with TemporaryDirectory() as raw:
        log_path = Path(raw) / "child.log"
        fd = jaxflow_workerkit._open_child_log(log_path)
        stream = io.BytesIO(b"0123456789ABCDE")  # exactly head(10) + tail(5)
        jaxflow_workerkit._capture_child_output(stream, fd, head_cap=10, tail_cap=5)
        jaxflow_workerkit._finalize_child_log(fd, log_path)
        assert log_path.read_bytes() == b"0123456789ABCDE"


def test_capture_child_output_boundary_one_byte_over_marks_n_equal_1():
    with TemporaryDirectory() as raw:
        log_path = Path(raw) / "child.log"
        fd = jaxflow_workerkit._open_child_log(log_path)
        stream = io.BytesIO(b"0123456789ABCDEF")  # head(10) + tail(6): one byte over
        jaxflow_workerkit._capture_child_output(stream, fd, head_cap=10, tail_cap=5)
        jaxflow_workerkit._finalize_child_log(fd, log_path)
        assert log_path.read_bytes() == b"0123456789[jaxflow: 1 bytes omitted]\nBCDEF"


def test_capture_child_output_tolerates_a_none_stream():
    # spec: Claude's popen call passes only stderr= as PIPE; a fake (or a runtime with
    # no stderr wired at all) may hand back `None` for the other stream.
    with TemporaryDirectory() as raw:
        log_path = Path(raw) / "child.log"
        fd = jaxflow_workerkit._open_child_log(log_path)
        jaxflow_workerkit._capture_child_output(None, fd)
        jaxflow_workerkit._finalize_child_log(fd, log_path)
        assert log_path.read_bytes() == b""


def test_finalize_child_log_redacts_secret_shaped_text_and_keeps_surrounding_bytes():
    with TemporaryDirectory() as raw:
        log_path = Path(raw) / "child.log"
        fd = jaxflow_workerkit._open_child_log(log_path)
        secret = b"Authorization: Bearer sk-abcdefghijklmnopqrstuvwx0123456789ABCDEF\n"
        stream = io.BytesIO(b"before\n" + secret + b"after\n")
        jaxflow_workerkit._capture_child_output(stream, fd)
        text = jaxflow_workerkit._finalize_child_log(fd, log_path)
        assert "sk-abcdefghijklmnopqrstuvwx0123456789ABCDEF" not in text
        assert "before\n" in text and "after\n" in text
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o600


def test_finalize_child_log_is_a_noop_rewrite_when_nothing_is_secret_shaped():
    with TemporaryDirectory() as raw:
        log_path = Path(raw) / "child.log"
        fd = jaxflow_workerkit._open_child_log(log_path)
        stream = io.BytesIO(b"nothing secret here\n")
        jaxflow_workerkit._capture_child_output(stream, fd)
        text = jaxflow_workerkit._finalize_child_log(fd, log_path)
        assert text == "nothing secret here\n"
        assert log_path.read_bytes() == b"nothing secret here\n"


def test_finalize_child_log_is_lossless_for_non_secret_bytes_crlf_and_invalid_utf8():
    # cold review F1 (a6054f373f53): text-mode read/write translates CRLF -> LF and
    # substitutes invalid UTF-8 bytes, corrupting bytes the raw-byte cap (spec §4.2)
    # promises to preserve. Only the secret-shaped substring may differ after finalize.
    with TemporaryDirectory() as raw:
        log_path = Path(raw) / "child.log"
        fd = jaxflow_workerkit._open_child_log(log_path)
        secret = b"Authorization: Bearer sk-abcdefghijklmnopqrstuvwx0123456789ABCDEF"
        raw_bytes = (
            b"line one\r\n"
            b"line two \xff\xfe more\r\n"
            + secret + b"\r\n"
            b"end\r\n"
        )
        stream = io.BytesIO(raw_bytes)
        jaxflow_workerkit._capture_child_output(stream, fd)
        jaxflow_workerkit._finalize_child_log(fd, log_path)
        result = log_path.read_bytes()
        assert b"sk-abcdefghijklmnopqrstuvwx0123456789ABCDEF" not in result
        expected = raw_bytes.replace(
            b"sk-abcdefghijklmnopqrstuvwx0123456789ABCDEF", b"[REDACTED]"
        )
        assert result == expected


# ---- MOA-470: FakePopen/FakeBuilderPopen accept piped stdout/stderr ----

def test_fake_popen_exposes_stdout_as_a_readable_stream_when_piped():
    fake = FakePopen(["codex"], stdout=subprocess.PIPE)
    assert fake.stdout.read() == b""  # CHILD_BYTES defaults to empty
    assert fake.stderr is None


def test_fake_popen_still_writes_report_to_an_opened_stdout_file_for_claude():
    with TemporaryDirectory() as raw:
        out = Path(raw) / "last-message.md"
        with out.open("wb") as f:
            FakePopen(["claude"], stdout=f)
        assert out.read_text(encoding="utf-8") == FakePopen.REPORT_TEXT


def test_fake_popen_exposes_stderr_as_a_readable_stream_when_piped():
    class _WithStderr(FakePopen):
        CHILD_BYTES = b"provider error text\n"
    fake = _WithStderr(["claude"], stderr=subprocess.PIPE)
    assert fake.stderr.read() == b"provider error text\n"
    assert fake.stdout is None


def test_fake_builder_popen_exposes_stdout_as_a_readable_stream_when_piped():
    class _WithOutput(FakeBuilderPopen):
        CHILD_BYTES = b"opencode chatter\n"
    with TemporaryDirectory() as raw:
        prompt = Path(raw) / "prompt.txt"
        prompt.write_text(f"report: {raw}/report.md\n", encoding="utf-8")
        with prompt.open("rb") as stdin_source:
            fake = _WithOutput(["opencode"], stdin=stdin_source, stdout=subprocess.PIPE)
        assert fake.stdout.read() == b"opencode chatter\n"


def test_fault_fakes_still_expose_none_stdout_without_setting_it_themselves():
    # FailingReportPopen/NoReportPopen/etc. never set self.stdout -- proving the
    # CLASS-LEVEL default on FakeBuilderPopen is what a reader loop sees, via plain
    # attribute inheritance, with zero code changed in any of those five classes.
    assert FakeBuilderPopen.stdout is None
    assert FakeBuilderPopen.stderr is None
    assert FakePopen.stdout is None
    assert FakePopen.stderr is None


# ---- MOA-470: builder worker captures child.log ----

DEEPSEEK_CLEAN_EXIT_NO_REPORT_WINDOW = (
    '{"type":"message.updated","properties":{"info":{"finish":"length"}}}\n'
)


def test_worker_builder_captures_stdout_stderr_merged_into_child_log():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)

        class ChattyBuilderPopen(FakeBuilderPopen):
            CHILD_BYTES = b"opencode: starting build\nopencode: done\n"

        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=ChattyBuilderPopen, allowlist_root=root.parent,
        )
        assert code == 0
        log_path = jr.child_log_path(root, manifest["run_id"])
        assert log_path.read_bytes() == b"opencode: starting build\nopencode: done\n"


def test_worker_builder_call_order_is_open_then_popen_then_capture_then_wait_then_finalize(monkeypatch):
    # Assert the WIRING, not just the resulting file (house rule: a cut wire can still
    # leave a plausible-looking child.log behind if some other code path wrote it).
    # Recording `popen` too (not just open/capture/wait/finalize) pins that the child
    # is launched BEFORE the drain starts -- the exact ordering a wait-before-read
    # regression would break.
    calls = []
    real_open, real_capture, real_finalize = (
        jaxflow_workerkit._open_child_log, jaxflow_workerkit._capture_child_output, jaxflow_workerkit._finalize_child_log,
    )
    monkeypatch.setattr(jaxflow_workerkit, "_open_child_log", lambda p: (calls.append("open"), real_open(p))[1])
    monkeypatch.setattr(jaxflow_workerkit, "_capture_child_output", lambda s, f, **kw: (calls.append("capture"), real_capture(s, f, **kw))[1])
    monkeypatch.setattr(jaxflow_workerkit, "_finalize_child_log", lambda f, p: (calls.append("finalize"), real_finalize(f, p))[1])

    class RecordingPopen(FakeBuilderPopen):
        def __init__(self, *args, **kwargs):
            calls.append("popen")
            super().__init__(*args, **kwargs)

        def wait(self, timeout=None):
            calls.append("wait")
            return super().wait(timeout)

    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, _ = _write_manifest_for_builder_worker(root, worktree)
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=RecordingPopen, allowlist_root=root.parent,
        )
    assert calls == ["open", "popen", "capture", "wait", "finalize"]


def test_worker_builder_passes_merged_pipe_kwargs_to_popen():
    captured = {}

    class CapturePopen(FakeBuilderPopen):
        def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
            captured["stdout"] = stdout
            captured["stderr"] = stderr
            super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)

    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, _ = _write_manifest_for_builder_worker(root, worktree)
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=CapturePopen, allowlist_root=root.parent,
        )
    assert captured["stdout"] is subprocess.PIPE
    assert captured["stderr"] is subprocess.STDOUT


def test_worker_builder_missing_report_gets_reason_and_tail_together():
    class NoReportPopen(FakeBuilderPopen):
        """Never writes report.md -- the DeepSeek clean-exit-no-report shape (spec
        §2.1), self-contained here rather than reusing the file's OTHER, function-
        nested `NoReportPopen` (defined inside
        `test_worker_builder_missing_report_is_missing_not_invalid_and_omits_result`,
        not reachable from this test)."""
        CHILD_BYTES = DEEPSEEK_CLEAN_EXIT_NO_REPORT_WINDOW.encode("utf-8")

        def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
            self.argv = argv
            self.cwd = cwd
            self.pid = 5253
            self.returncode = 0
            self.stdin_bytes = stdin.read() if stdin is not None else None
            if stdout is subprocess.PIPE:
                self.stdout = io.BytesIO(self.CHILD_BYTES)
            if stderr is subprocess.PIPE:
                self.stderr = io.BytesIO(self.CHILD_BYTES)

        def wait(self, timeout=None):
            return self.returncode

    posted = []
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, _ = _write_manifest_for_builder_worker(root, worktree)
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=posted.append, popen=NoReportPopen, allowlist_root=root.parent,
        )
    payload = posted[0]["payload"]
    assert payload["contract_status"] == "missing"
    assert payload["reason"] == "no-report"
    assert "message.updated" in payload["tail"]


def test_worker_builder_ok_row_has_no_tail_or_reason():
    posted = []
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        _write_plan(worktree)
        manifest_path, _ = _write_manifest_for_builder_worker(root, worktree)
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=posted.append, popen=FakeBuilderPopen, allowlist_root=root.parent,
        )
    payload = posted[0]["payload"]
    assert payload["contract_status"] == "ok"
    assert "reason" not in payload
    assert "tail" not in payload


# ---- MOA-470: diff-reviewer and doc-reviewer workers capture child.log ----

def test_worker_diff_review_codex_merges_stdout_stderr_into_child_log():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        p1_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        (worktree / "f.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "f.txt")
        _git(worktree, "commit", "-m", "change")
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, runtime="codex", base_sha=p1_sha, head_sha=head_sha,
        )

        class ChattyCodexPopen(FakePopen):
            CHILD_BYTES = b"codex: reviewing diff\ncodex: wrote last-message\n"

        jaxflow.run_worker(
            str(manifest_path), run=_run_real, post=lambda e: {"ok": True},
            popen=ChattyCodexPopen, allowlist_root=root.parent,
        )
        log_path = jr.child_log_path(root, manifest["run_id"])
        assert log_path.read_bytes() == b"codex: reviewing diff\ncodex: wrote last-message\n"


def test_worker_diff_review_claude_captures_stderr_only_stdout_stays_the_report():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        p1_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        (worktree / "f.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "f.txt")
        _git(worktree, "commit", "-m", "change")
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, runtime="claude", base_sha=p1_sha, head_sha=head_sha,
        )
        captured = {}

        class CaptureClaudePopen(FakePopen):
            CHILD_BYTES = b"a provider error on stderr\n"
            def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                captured["stdout"] = stdout
                captured["stderr"] = stderr
                super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)

        jaxflow.run_worker(
            str(manifest_path), run=_run_real, post=lambda e: {"ok": True},
            popen=CaptureClaudePopen, allowlist_root=root.parent,
        )
        assert captured["stdout"] not in (None, subprocess.PIPE)  # the opened file, unchanged
        assert captured["stderr"] is subprocess.PIPE
        log_path = jr.child_log_path(root, manifest["run_id"])
        assert log_path.read_bytes() == b"a provider error on stderr\n"
        # `reviewer_output` is `jr.safe_run_paths`'s own path, `.local/scratch/<run_id>/last-message.md`
        # (jaxflow_run.py:250) -- NOT under `.local/runs/`, which is child.log's home.
        reviewer_output = root / ".local" / "scratch" / manifest["run_id"] / "last-message.md"
        assert reviewer_output.read_text(encoding="utf-8") == FakePopen.REPORT_TEXT


def test_worker_diff_review_missing_or_invalid_row_gets_tail_and_reason():
    class InvalidReportPopen(FakePopen):
        """Writes a report with an out-of-enum verdict -- fails both parse stages,
        so contract_status comes back "invalid" (the file DOES exist). Mirrors
        the builder-side fixture of the same name at
        `test_worker_builder_invalid_report_is_invalid_not_missing`, but that one is
        nested inside its own test function and not reachable from here."""
        REPORT_TEXT = "verdict: banana\nsummary: looks fine\n"
        CHILD_BYTES = b"codex wrote a malformed report\n"

        def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
            self.argv = argv
            self.cwd = cwd
            self.pid = 5254
            self.returncode = 0
            self.stdin_bytes = stdin.read() if stdin is not None else None
            if "--output-last-message" in argv:
                out_path = Path(argv[argv.index("--output-last-message") + 1])
                out_path.write_text(InvalidReportPopen.REPORT_TEXT, encoding="utf-8")
            if stdout is subprocess.PIPE:
                self.stdout = io.BytesIO(self.CHILD_BYTES)
            if stderr is subprocess.PIPE:
                self.stderr = io.BytesIO(self.CHILD_BYTES)

        def wait(self, timeout=None):
            return self.returncode

    posted = []
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        p1_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        (worktree / "f.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "f.txt")
        _git(worktree, "commit", "-m", "change")
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        manifest_path, _ = _write_manifest_for_diff_worker(
            root, worktree, runtime="codex", base_sha=p1_sha, head_sha=head_sha,
        )
        jaxflow.run_worker(
            str(manifest_path), run=_run_real, post=posted.append,
            popen=InvalidReportPopen, allowlist_root=root.parent,
        )
    payload = posted[0]["payload"]
    assert payload["contract_status"] == "invalid"
    assert payload["reason"] in jr.REASONS
    assert "malformed report" in payload["tail"]


def test_worker_doc_review_wires_the_same_codex_claude_branching():
    # Mirrors the existing `test_worker_pipes_prompt_on_stdin_for_both_runtimes_and_finalizes`
    # fixture shape exactly (`root` IS the allowlist root for a doc review -- there is
    # no separate worktree/control-repo split here, unlike build/diff).
    for runtime in ("codex", "claude"):
        with TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            _init_repo(root)
            target = _spec_file(root)
            manifest_path, manifest = _write_manifest_for_worker(root, target, runtime=runtime)

            class ChattyPopen(FakePopen):
                CHILD_BYTES = b"reviewer chatter\n"

            jaxflow.run_worker(
                str(manifest_path), run=_run_with_tmux(FakeTmux()), post=lambda e: {"ok": True},
                popen=ChattyPopen, allowlist_root=root,
            )
            log_path = jr.child_log_path(root, manifest["run_id"])
            assert log_path.read_bytes() == b"reviewer chatter\n"


def test_worker_doc_review_popen_kwargs_are_wired_per_runtime():
    # cold review F2 (a6054f373f53): the log-content check above passes even if the
    # doc-review Codex launch loses `stderr=subprocess.STDOUT` -- an identical-bytes
    # fake can't tell interleaved-into-stdout from piped-separately -- so assert the
    # actual Popen kwargs at this launch site directly, per runtime.
    for runtime in ("codex", "claude"):
        with TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            _init_repo(root)
            target = _spec_file(root)
            manifest_path, _ = _write_manifest_for_worker(root, target, runtime=runtime)
            captured = {}

            class CapturePopen(FakePopen):
                def __init__(self, argv, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None, env=None):
                    captured["stdout"] = stdout
                    captured["stderr"] = stderr
                    super().__init__(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session, env=env)

            jaxflow.run_worker(
                str(manifest_path), run=_run_with_tmux(FakeTmux()), post=lambda e: {"ok": True},
                popen=CapturePopen, allowlist_root=root,
            )
            if runtime == "codex":
                assert captured["stdout"] is subprocess.PIPE
                assert captured["stderr"] is subprocess.STDOUT
            else:
                assert captured["stdout"] not in (None, subprocess.PIPE)  # opened last-message.md file
                assert captured["stderr"] is subprocess.PIPE


def test_worker_diff_review_call_order_is_open_then_popen_then_capture_then_wait_then_finalize(monkeypatch):
    # Same wiring assertion as the builder's (Task 4, cold review 786debd43314 F4):
    # a plausible-looking child.log can still hide a wait-before-read regression at
    # THIS site specifically, since the real-pipe test below only proves the deadlock
    # risk itself, not per-site ordering (see that test's own comment).
    calls = []
    real_open, real_capture, real_finalize = (
        jaxflow_workerkit._open_child_log, jaxflow_workerkit._capture_child_output, jaxflow_workerkit._finalize_child_log,
    )
    monkeypatch.setattr(jaxflow_workerkit, "_open_child_log", lambda p: (calls.append("open"), real_open(p))[1])
    monkeypatch.setattr(jaxflow_workerkit, "_capture_child_output", lambda s, f, **kw: (calls.append("capture"), real_capture(s, f, **kw))[1])
    monkeypatch.setattr(jaxflow_workerkit, "_finalize_child_log", lambda f, p: (calls.append("finalize"), real_finalize(f, p))[1])

    class RecordingPopen(FakePopen):
        def __init__(self, *args, **kwargs):
            calls.append("popen")
            super().__init__(*args, **kwargs)

        def wait(self, timeout=None):
            calls.append("wait")
            return super().wait(timeout)

    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root)
        p1_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        (worktree / "f.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "f.txt")
        _git(worktree, "commit", "-m", "change")
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        manifest_path, _ = _write_manifest_for_diff_worker(
            root, worktree, runtime="codex", base_sha=p1_sha, head_sha=head_sha,
        )
        jaxflow.run_worker(
            str(manifest_path), run=_run_real, post=lambda e: {"ok": True},
            popen=RecordingPopen, allowlist_root=root.parent,
        )
    assert calls == ["open", "popen", "capture", "wait", "finalize"]


def test_worker_doc_review_call_order_is_open_then_popen_then_capture_then_wait_then_finalize(monkeypatch):
    # Same wiring assertion, third launch site (cold review 786debd43314 F4).
    calls = []
    real_open, real_capture, real_finalize = (
        jaxflow_workerkit._open_child_log, jaxflow_workerkit._capture_child_output, jaxflow_workerkit._finalize_child_log,
    )
    monkeypatch.setattr(jaxflow_workerkit, "_open_child_log", lambda p: (calls.append("open"), real_open(p))[1])
    monkeypatch.setattr(jaxflow_workerkit, "_capture_child_output", lambda s, f, **kw: (calls.append("capture"), real_capture(s, f, **kw))[1])
    monkeypatch.setattr(jaxflow_workerkit, "_finalize_child_log", lambda f, p: (calls.append("finalize"), real_finalize(f, p))[1])

    class RecordingPopen(FakePopen):
        def __init__(self, *args, **kwargs):
            calls.append("popen")
            super().__init__(*args, **kwargs)

        def wait(self, timeout=None):
            calls.append("wait")
            return super().wait(timeout)

    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        manifest_path, _ = _write_manifest_for_worker(root, target, runtime="codex")
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()), post=lambda e: {"ok": True},
            popen=RecordingPopen, allowlist_root=root,
        )
    assert calls == ["open", "popen", "capture", "wait", "finalize"]


# ---- MOA-470 §7: the ONE test that must not fake Popen (real OS pipe, real deadlock
# risk if drain-before-wait ever regresses back to wait-then-read) ----

CHILD_FLOOD_SCRIPT = (
    "import sys\n"
    "sys.stdout.buffer.write(b'O' * {stdout_n})\n"
    "sys.stdout.buffer.flush()\n"
    "sys.stderr.buffer.write(b'E' * {stderr_n})\n"
    "sys.stderr.buffer.flush()\n"
)


def _alarm_handler(signum, frame):
    raise TimeoutError("worker did not return -- drain-before-wait regressed to wait-then-read")


def test_worker_diff_review_real_subprocess_drains_before_wait(monkeypatch):
    # codex: both streams merge into the pipe THIS worker reads (>1 MiB combined).
    # claude: stdout stays the opened report file (untouched); only stderr is piped,
    # so ONLY stderr's byte count is what must appear in child.log.
    for runtime, stdout_n, stderr_n in (("codex", 600_000, 600_000), ("claude", 10, 1_200_000)):
        with TemporaryDirectory() as raw:
            root = Path(raw).resolve() / "demo"
            _init_repo(root)
            worktree = _init_worktree(root)
            base_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
            _git(worktree, "commit", "--allow-empty", "-m", "head")
            head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
            manifest_path, manifest = _write_manifest_for_diff_worker(
                root, worktree, runtime=runtime, base_sha=base_sha, head_sha=head_sha,
            )
            script = CHILD_FLOOD_SCRIPT.format(stdout_n=stdout_n, stderr_n=stderr_n)
            monkeypatch.setattr(jr, "runtime_argv", lambda *a, **k: [sys.executable, "-c", script])

            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(15)
            try:
                code = jaxflow.run_worker(
                    str(manifest_path), run=_run_real, post=lambda e: {"ok": True},
                    popen=subprocess.Popen, allowlist_root=root.parent,
                )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            assert code == 0
            log_path = jr.child_log_path(root, manifest["run_id"])
            expected = stdout_n + stderr_n if runtime == "codex" else stderr_n
            assert log_path.stat().st_size == expected


# ---- MOA-470: cmd_status/cmd_result render reason/tail; merge leaves child.log alone ----

def test_status_prints_failed_reason_and_exit_when_reason_is_present():
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        fake = FakeTmux()
        # MOA-474 §12.2: the legacy `reason` branch (branch 4) is builder-only -- for a
        # reviewer row, branch 3 (`no verdict`) fires first. This test's own subject is
        # the reason branch, so the fixture stays a builder row.
        _insert(con, "r9", "demo", "builder", "run-started", {"session": "jax-demo-spec-r9"})
        _insert(con, "r9", "demo", "builder", "run-finished", {
            "contract_status": "invalid", "exit_code": 1, "reason": "crash",
        })
        assert jaxflow.cmd_status("r9", run=_run_with_tmux(fake), db_path=db) == "finished(failed: crash, exit 1)"
        con.close()


def test_status_unaffected_for_ok_and_cancelled_rows():
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        fake = FakeTmux()
        _insert(con, "r10", "demo", "reviewer", "run-started", {"session": "jax-demo-spec-r10"})
        _insert(con, "r10", "demo", "reviewer", "run-finished", {"contract_status": "ok"})
        # MOA-474 §12.2 branch 2's own note: this row has no verdict key at all (a
        # pre-verdict-tracking fixture) -- branch 3 fires (role reviewer, neither
        # result nor verdict present), the literal "no verdict", not the old bare-ok
        # fallback.
        assert jaxflow.cmd_status("r10", run=_run_with_tmux(fake), db_path=db) == "finished(no verdict)"
        _insert(con, "r11", "demo", "reviewer", "run-started", {"session": "jax-demo-spec-r11"})
        _insert(con, "r11", "demo", "reviewer", "run-finished", {"contract_status": "cancelled"})
        assert jaxflow.cmd_status("r11", run=_run_with_tmux(fake), db_path=db) == "finished(cancelled)"
        con.close()


def test_result_renders_from_ledger_after_child_log_deleted(capsys):
    # Final review 1f119d5d1420 F1: the spec's actual `missing` case -- report.md
    # never existed, no child.log survives -- must still print the block and refuse.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        root.mkdir(parents=True)
        report = root / ".local" / "reports" / "r5.md"
        child_log = jr.child_log_path(root, "r5")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _insert(con, "r5", "demo", "reviewer", "run-started", {"repo": str(root)})
        _insert(con, "r5", "demo", "reviewer", "run-finished", {
            "contract_status": "missing", "exit_code": 1, "report_path": str(report),
            "summary": "no report written", "reason": "crash", "tail": "line one\nline two",
        })
        assert not report.exists() and not child_log.exists()
        try:
            jaxflow.cmd_result("r5", allowlist_root=allow_root, db_path=db)
        except ji.Refusal as exc:
            assert exc.code == "report-missing"
        else:
            raise AssertionError("report-missing not raised")
        out = capsys.readouterr().out
        assert out == (
            "exit_code: 1\n"
            "reason: crash\n"
            "tail:\nline one\nline two\n"
            f"child.log: {child_log}\n\n"
        )
        con.close()


def test_result_invalid_row_prints_block_then_report_body(capsys):
    # Companion to the `missing` case above: an `invalid` row's report DOES exist, so
    # both the printed block and the returned body appear -- block first.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        root.mkdir(parents=True)
        (root / ".local" / "reports").mkdir(parents=True)
        report = root / ".local" / "reports" / "r7.md"
        report.write_text("INVALID REPORT\n", encoding="utf-8")
        child_log = jr.child_log_path(root, "r7")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _insert(con, "r7", "demo", "reviewer", "run-started", {"repo": str(root)})
        _insert(con, "r7", "demo", "reviewer", "run-finished", {
            "contract_status": "invalid", "exit_code": 1, "report_path": str(report),
            "summary": "bad report", "reason": "unknown", "tail": "line one\nline two",
        })
        returned = jaxflow.cmd_result("r7", allowlist_root=allow_root, db_path=db)
        printed = capsys.readouterr().out
        expected_block = (
            "exit_code: 1\n"
            "reason: unknown\n"
            "tail:\nline one\nline two\n"
            f"child.log: {child_log}\n\n"
        )
        # `printed` happens during the call, before `returned` exists -- together they
        # are what `main()` puts on one stream (block, then body).
        assert printed == expected_block
        assert returned == f"{report}\nINVALID REPORT\n"
        con.close()


def test_result_older_missing_invalid_row_with_neither_key_is_unaffected():
    # Backward compatibility (spec §4.3): a pre-MOA-470 row has neither key -- the
    # extra block must simply not appear, not raise a KeyError reading `tail`/`reason`.
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        root.mkdir(parents=True)
        (root / ".local" / "reports").mkdir(parents=True)
        report = root / ".local" / "reports" / "r6.md"
        report.write_text("OLD INVALID REPORT\n", encoding="utf-8")
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        _insert(con, "r6", "demo", "reviewer", "run-started", {"repo": str(root)})
        _insert(con, "r6", "demo", "reviewer", "run-finished", {
            "contract_status": "invalid", "exit_code": 1, "report_path": str(report),
            "summary": "bad report",
        })
        out = jaxflow.cmd_result("r6", allowlist_root=allow_root, db_path=db)
        assert out == f"{report}\nOLD INVALID REPORT\n"
        con.close()


def test_merge_leaves_child_log_in_control_repo_and_excludes_it_from_report_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    worktree = tmp_path.parent / f"{jaxflow_common.slugify_project(tmp_path.name)}-feat-x"
    (worktree / ".local" / "reports").mkdir(parents=True)
    (worktree / ".local" / "reports" / "abc123abc123.md").write_text("report", encoding="utf-8")
    # child.log for this run already lives in the CONTROL repo -- written there by the
    # tmux-detached worker before the worktree even existed (spec §4.1: sibling of
    # manifest.json, rooted on the control repo via `child_log_path`, never the
    # worktree). Merge's reports-copy step only ever reads the WORKTREE's
    # `.local/reports/` (spec §5) -- it has no path to this file at all.
    child_log = jr.child_log_path(tmp_path, "abc123abc123")
    child_log.parent.mkdir(parents=True, exist_ok=True)
    child_log.write_text("child output\n", encoding="utf-8")
    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert child_log.read_text(encoding="utf-8") == "child output\n"
    assert not any(p.name == "child.log" for p in (tmp_path / ".local" / "reports").iterdir())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all tests passed")


class FindingsPopen(FakePopen):
    REPORT_TEXT = (
        "---\nrun_id: aaaabbbbcccc\nproject: demo\nrole: reviewer\nphase: PHASE\n"
        "verdict: approve-with-changes\nsummary: approve-with-changes -- 2 MEDIUM, 1 LOW\n---\n"
        "F1. MEDIUM — something\nF2. MEDIUM — something else\nF3. LOW — a style nit\n"
    )


def test_worker_reviewer_ok_report_posts_findings_counts_beside_verdict():
    # Round-3 cold review F1 (MOA-480): findings travels beside verdict on an ok report,
    # for both reviewer worker paths (doc review and --diff review call the same
    # finalize_reviewer_report -- representative test on the doc-review path here).
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        manifest_path, manifest = _write_manifest_for_worker(root, target)
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()),
            post=lambda e: events.append(e) or {"ok": True},
            popen=FindingsPopen, allowlist_root=root,
        )
        payload = events[0]["payload"]
        assert payload["contract_status"] == "ok"
        assert payload["verdict"] == "approve-with-changes"
        assert payload["findings"] == {"high": 0, "medium": 2, "low": 1}


def test_worker_reviewer_report_with_no_numbered_findings_omits_the_field():
    # FakePopen's default REPORT_TEXT has "verdict: approve\nsummary: looks fine" --
    # no numbered lines, no parseable summary counts -> findings is omitted, never a zero.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        target = _spec_file(root)
        manifest_path, manifest = _write_manifest_for_worker(root, target)
        events = []
        jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()),
            post=lambda e: events.append(e) or {"ok": True},
            popen=FakePopen, allowlist_root=root,
        )
        payload = events[0]["payload"]
        assert payload["contract_status"] == "ok"
        assert "findings" not in payload


def test_worker_diff_reviewer_ok_report_posts_findings_counts_beside_verdict():
    # Round-1 plan review F6: the OTHER finalize_reviewer_report call site
    # (_run_diff_reviewer_worker, the `review --diff` path) must post findings exactly
    # like the document-review path above -- same fake report, same assertion shape.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "demo"
        _init_repo(root)
        worktree = _init_worktree(root, "feat/x")
        (worktree / "changed.txt").write_text("x\n", encoding="utf-8")
        _git(worktree, "add", "changed.txt")
        _git(worktree, "commit", "-m", "change")
        base_sha = _run_real(["git", "rev-parse", "HEAD~1"], worktree).stdout.strip()
        head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
        tests_path = worktree / ".local" / "reports" / "b1b1b1b1b1b1.tests.txt"
        tests_path.parent.mkdir(parents=True)
        tests_path.write_text("COMMAND: true\n\nEXIT: 0\n", encoding="utf-8")
        manifest_path, manifest = _write_manifest_for_diff_worker(
            root, worktree, base_sha=base_sha, head_sha=head_sha, tests_path=str(tests_path),
        )
        events = []
        code = jaxflow.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: events.append(e) or {"ok": True},
            popen=FindingsPopen, allowlist_root=root.parent,
        )
        assert code == 0
        payload = events[0]["payload"]
        assert payload["contract_status"] == "ok"
        assert payload["verdict"] == "approve-with-changes"
        assert payload["findings"] == {"high": 0, "medium": 2, "low": 1}


def test_parse_worktree_porcelain_branch_detached_locked_and_empty():
    output = (
        "worktree /home/rafa/repos/jax-os\n"
        "HEAD c9213d5668654ab926494aa281000047710e43cf\n"
        "branch refs/heads/main\n"
        "\n"
        "worktree /home/rafa/repos/jax-os-locked\n"
        "HEAD aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "branch refs/heads/feat/locked-thing\n"
        "locked working on it\n"
        "\n"
        "worktree /home/rafa/repos/jax-os-detached\n"
        "HEAD bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
        "detached\n"
        "\n"
        "worktree /r/p\nHEAD " + "c" * 40 + "\nbranch refs/heads/x\nlocked\n"
    )
    entries = jaxflow._parse_worktree_porcelain(output)
    assert entries == [
        {"path": "/home/rafa/repos/jax-os", "branch": "main", "detached": False, "locked": False},
        {"path": "/home/rafa/repos/jax-os-locked", "branch": "feat/locked-thing", "detached": False, "locked": True},
        {"path": "/home/rafa/repos/jax-os-detached", "branch": None, "detached": True, "locked": False},
        {"path": "/r/p", "branch": "x", "detached": False, "locked": True},
    ]
    assert jaxflow._parse_worktree_porcelain("") == []


def _finished(contract_status, **extra):
    return {"contract_status": contract_status, **extra}


@pytest.mark.parametrize("finished,expected", [
    (None, "open"),
    (_finished("interrupted"), "interrupted"),
    (_finished("cancelled"), "cancelled"),
    (_finished("ok", result="success"), "success"),
    (_finished("ok", result="failure"), "other-terminal"),
    (_finished("ok", result="blocked"), "other-terminal"),
    (_finished("missing"), "other-terminal"),
    (_finished("invalid"), "other-terminal"),
    (_finished("some-future-status"), "unknown"),
])
def test_classify_gc_state(finished, expected):
    assert jaxflow._classify_gc_state(finished) == expected


def _entry(path="/home/rafa/repos/demo-feat-x", branch="feat/x", detached=False, locked=False):
    return {"path": path, "branch": branch, "detached": detached, "locked": locked}


_NOW = datetime(2026, 9, 19, tzinfo=timezone.utc)
_OLD_TS = "2026-09-01T00:00:00+00:00"   # 18 days before _NOW
_YOUNG_TS = "2026-09-18T00:00:00+00:00"  # 1 day before _NOW


@pytest.mark.parametrize("kwargs,expected", [
    # control repo / detached / locked / out-of-allowlist short-circuit before any ledger lookup
    (dict(entry=_entry(path="/home/rafa/repos/demo"), latest_build_run_id="x", finished_payload=_finished("ok", result="success"), finished_ts=_OLD_TS), {"action": "skip", "state": "control-repo"}),
    (dict(entry=_entry(detached=True), latest_build_run_id="x", finished_payload=_finished("ok", result="success"), finished_ts=_OLD_TS), {"action": "skip", "state": "detached"}),
    (dict(entry=_entry(locked=True), latest_build_run_id="x", finished_payload=_finished("ok", result="success"), finished_ts=_OLD_TS), {"action": "skip", "state": "locked"}),
    (dict(entry=_entry(path="/tmp/outside"), latest_build_run_id="x", finished_payload=_finished("ok", result="success"), finished_ts=_OLD_TS), {"action": "skip", "state": "out-of-allowlist"}),
    # unowned / open / unknown always skip, even old+clean+force
    (dict(entry=_entry(), latest_build_run_id=None, finished_payload=None, finished_ts=None, force=True, dirty=False), {"action": "skip", "state": "unowned"}),
    (dict(entry=_entry(), latest_build_run_id="x", finished_payload=None, finished_ts=None, force=True, dirty=False), {"action": "skip", "state": "open"}),
    (dict(entry=_entry(), latest_build_run_id="x", finished_payload=_finished("weird"), finished_ts=_OLD_TS, force=True, dirty=False), {"action": "skip", "state": "unknown"}),
    # terminal states: too young always skips
    (dict(entry=_entry(), latest_build_run_id="x", finished_payload=_finished("cancelled"), finished_ts=_YOUNG_TS), {"action": "skip", "state": "cancelled"}),
    # terminal states: old + clean removes, for every terminal category
    (dict(entry=_entry(), latest_build_run_id="x", finished_payload=_finished("interrupted"), finished_ts=_OLD_TS), {"action": "remove", "state": "interrupted"}),
    (dict(entry=_entry(), latest_build_run_id="x", finished_payload=_finished("cancelled"), finished_ts=_OLD_TS), {"action": "remove", "state": "cancelled"}),
    (dict(entry=_entry(), latest_build_run_id="x", finished_payload=_finished("ok", result="failure"), finished_ts=_OLD_TS), {"action": "remove", "state": "other-terminal"}),
    (dict(entry=_entry(), latest_build_run_id="x", finished_payload=_finished("ok", result="success"), finished_ts=_OLD_TS), {"action": "remove", "state": "success"}),
    # old + dirty skips unless --force; old + dirty + force removes
    (dict(entry=_entry(), latest_build_run_id="x", finished_payload=_finished("ok", result="success"), finished_ts=_OLD_TS, dirty=True), {"action": "skip", "state": "success"}),
    (dict(entry=_entry(), latest_build_run_id="x", finished_payload=_finished("ok", result="success"), finished_ts=_OLD_TS, dirty=True, force=True), {"action": "remove", "state": "success"}),
])
def test_gc_decide(kwargs, expected):
    kwargs.setdefault("dirty", False)
    kwargs.setdefault("force", False)
    result = jaxflow._gc_decide(
        kwargs["entry"], control_repo=Path("/home/rafa/repos/demo"), allowlist_root=Path("/home/rafa/repos"),
        latest_build_run_id=kwargs["latest_build_run_id"], finished_payload=kwargs["finished_payload"],
        finished_ts=kwargs["finished_ts"], now=_NOW, older_than_days=7, dirty=kwargs["dirty"], force=kwargs["force"])
    assert result == expected


def _plant_gc_manifest(repo, run_id, worktree):
    """F2 (round 4): the on-disk manifest a real `dispatch_build` writes, minimal down
    to the one field `_gc_worktree_is_reserved` reads."""
    path = repo / ".local" / "runs" / run_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"worktree": str(worktree)}), encoding="utf-8")


def test_latest_builder_run_for_gc_reuses_latest_builder_attempt_and_unowned(tmp_path):
    repo = tmp_path / "demo"
    repo.mkdir()
    worktree = tmp_path / "demo-feat-x"
    _plant_gc_manifest(repo, "aaaabbbbcccc", worktree)
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    con.row_factory = sqlite3.Row
    _insert(con, "aaaabbbbcccc", "demo", "builder", "run-started", {"target": "feat/x", "repo": str(repo)})
    _insert(con, "aaaabbbbcccc", "demo", "builder", "run-finished", {"contract_status": "ok", "result": "success"}, ts="2026-09-01T00:00:00+00:00")
    assert jaxflow._latest_builder_run_for_gc(con, "demo", repo, "feat/x", worktree) == ("aaaabbbbcccc", {"contract_status": "ok", "result": "success"}, "2026-09-01T00:00:00+00:00")
    assert jaxflow._latest_builder_run_for_gc(con, "demo", repo, "feat/nothing", worktree) == (None, None, None)
    # F2: same branch, but the manifest reserved a DIFFERENT path -- not ownership.
    assert jaxflow._latest_builder_run_for_gc(con, "demo", repo, "feat/x", tmp_path / "elsewhere") == (None, None, None)
    con.close()


def test_latest_builder_attempt_breaks_ts_tie_by_id(tmp_path):
    """Cross-language fixture (F6, round 4): mirrors vitest's `worktrees.test.ts`
    "picks the latest build per branch by (startedAt, id)" case -- two builder attempts
    share the exact SAME ts; the higher `id` must win, the same rule the TS collector's
    `collectRetainedWorktrees` uses."""
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    con.row_factory = sqlite3.Row
    same_ts = "2026-09-01T00:00:00+00:00"
    _insert(con, "aaaa", "demo", "builder", "run-started", {"target": "feat/x", "repo": "/home/rafa/repos/demo"}, ts=same_ts)
    _insert(con, "bbbb", "demo", "builder", "run-started", {"target": "feat/x", "repo": "/home/rafa/repos/demo"}, ts=same_ts)
    assert jaxflow_common._latest_builder_attempt(con, "demo", "/home/rafa/repos/demo", "feat/x") == "bbbb"
    con.close()


@pytest.mark.parametrize("plant,expected_failed", [
    ("missing-dir", False),    # nothing to preserve at all -- not a failure
    ("unreadable-dir", True),  # exists but wrong type (permission denied looks the same)
    ("invalid-shape", True),   # FIFO: not a regular, single-linked file (jaxflow.py:4896)
    ("oversized", True),
])
def test_copy_run_reports_failure_paths(tmp_path, plant, expected_failed):
    worktree = tmp_path / "wt"
    repo = tmp_path / "repo"
    repo.mkdir()
    if plant == "missing-dir":
        worktree.mkdir()  # no .local at all
    elif plant == "unreadable-dir":
        # ".local" is a FILE, not a directory -- opening it with O_DIRECTORY fails with
        # NotADirectoryError, an OSError that is NOT FileNotFoundError: "exists but
        # could not be listed", as opposed to "missing-dir"'s "genuinely nothing there".
        worktree.mkdir()
        (worktree / ".local").write_text("not a directory\n", encoding="utf-8")
    else:
        (worktree / ".local" / "reports").mkdir(parents=True)
        if plant == "invalid-shape":
            os.mkfifo(worktree / ".local" / "reports" / "aaaabbbbcccc.md")
        elif plant == "oversized":
            (worktree / ".local" / "reports" / "aaaabbbbcccc.md").write_bytes(b"x" * (jaxflow_common.REPORT_COPY_MAX + 1))
    result = jaxflow_common._copy_run_reports(worktree, repo, "aaaabbbbcccc")
    assert result == {"failed": expected_failed, "existed": False}
    assert not (repo / ".local" / "reports" / "aaaabbbbcccc.md").exists()


def test_copy_run_reports_fails_closed_on_existing_file_copy_error(tmp_path, monkeypatch):
    worktree = tmp_path / "wt"
    repo = tmp_path / "repo"
    (worktree / ".local" / "reports").mkdir(parents=True)
    (repo / ".local" / "reports").mkdir(parents=True)
    report = worktree / ".local" / "reports" / "aaaabbbbcccc.md"
    report.write_text("report body\n", encoding="utf-8")
    real_open = os.open

    def failing_open(path, flags, *a, **kw):
        # simulate a read failure on the specific report file, dir_fd opens pass through
        if path == "aaaabbbbcccc.md" and not (flags & os.O_CREAT):
            raise OSError("simulated read failure")
        return real_open(path, flags, *a, **kw)

    monkeypatch.setattr(os, "open", failing_open)
    result = jaxflow_common._copy_run_reports(worktree, repo, "aaaabbbbcccc")
    assert result == {"failed": True, "existed": True}
    assert not (repo / ".local" / "reports" / "aaaabbbbcccc.md").exists()


@pytest.mark.parametrize("plant,expected_failed", [
    ("directory", True),        # F1 (round 2, HIGH): a directory at the destination fails closed.
    ("identical-file", False),  # byte-identical existing file is genuinely preserved.
])
def test_copy_run_reports_existing_destination_shape(tmp_path, plant, expected_failed):
    worktree = tmp_path / "wt"
    repo = tmp_path / "repo"
    (worktree / ".local" / "reports").mkdir(parents=True)
    (repo / ".local" / "reports").mkdir(parents=True)
    body = "report body\n"
    (worktree / ".local" / "reports" / "aaaabbbbcccc.md").write_text(body, encoding="utf-8")
    dest = repo / ".local" / "reports" / "aaaabbbbcccc.md"
    dest.mkdir() if plant == "directory" else dest.write_text(body, encoding="utf-8")
    result = jaxflow_common._copy_run_reports(worktree, repo, "aaaabbbbcccc")
    assert result == {"failed": expected_failed, "existed": True}


def test_copy_run_reports_no_report_is_not_a_failure_then_copies_once_written(tmp_path):
    worktree = tmp_path / "wt"
    repo = tmp_path / "repo"
    (worktree / ".local" / "reports").mkdir(parents=True)
    (repo / ".local" / "reports").mkdir(parents=True)
    assert jaxflow_common._copy_run_reports(worktree, repo, "aaaabbbbcccc") == {"failed": False, "existed": False}
    (worktree / ".local" / "reports" / "aaaabbbbcccc.md").write_text("x\n", encoding="utf-8")
    result = jaxflow_common._copy_run_reports(worktree, repo)  # merge's own call shape: no run_id
    assert result == {"failed": False, "existed": False}
    assert (repo / ".local" / "reports" / "aaaabbbbcccc.md").read_text(encoding="utf-8") == "x\n"


def test_cmd_gc_dry_run_prints_and_mutates_nothing_then_yes_removes(tmp_path, monkeypatch, capsys):
    allow_root = tmp_path / "repos"
    root = allow_root / "demo"
    _init_repo(root)
    worktree = _init_worktree(root, branch="feat/old")
    # Plan deviation (spec §4 guard 7 / F3): a successful run's report must exist for
    # removal to be allowed at all -- the plan's own fixture omitted it while asserting
    # removal, which contradicts the implementation and spec acceptance criterion
    # "absent report needs contract_status: missing".
    (worktree / ".local" / "reports").mkdir(parents=True)
    (worktree / ".local" / "reports" / "aaaabbbbcccc.md").write_text("report body\n", encoding="utf-8")
    open_worktree = _init_worktree(root, branch="feat/open")  # no run-finished row at all
    _plant_gc_manifest(root, "aaaabbbbcccc", worktree)  # F2: reserves this exact path
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    old_ts = "2026-09-01T00:00:00+00:00"
    _insert(con, "aaaabbbbcccc", "demo", "builder", "run-started",
            {"target": "feat/old", "repo": str(root), "phase": "p", "caller": "claude", "session": "s",
             "runtime": "opencode-builder", "model": "m", "effort": "e", "verify": "true"})
    _insert(con, "aaaabbbbcccc", "demo", "builder", "run-finished", {"contract_status": "ok", "result": "success"}, ts=old_ts)
    _insert(con, "bbbbccccdddd", "demo", "builder", "run-started",
            {"target": "feat/open", "repo": str(root), "phase": "p", "caller": "claude", "session": "s",
             "runtime": "opencode-builder", "model": "m", "effort": "e", "verify": "true"}, ts="2026-01-01T00:00:00+00:00")
    con.close()
    monkeypatch.chdir(root)
    now = lambda: datetime(2026, 9, 19, tzinfo=timezone.utc)
    posted = []
    post = lambda event: posted.append(event) or {"ok": True}

    args = SimpleNamespace(older_than="7d", dry_run=True, yes=False, force=False)
    jaxflow.cmd_gc(args, run=_run_real, post=post, now=now, db_path=db, allowlist_root=allow_root)
    assert posted == []
    assert worktree.is_dir() and open_worktree.is_dir()
    # F3 (round 4): the removable candidate's dry-run line names its age and warns that
    # removal discards `--resume` eligibility; the skipped `open` one gets neither.
    out_lines = capsys.readouterr().out.splitlines()
    old_line = next(line for line in out_lines if str(worktree) in line)
    assert "age 18d" in old_line and "removal discards --resume eligibility" in old_line
    open_line = next(line for line in out_lines if str(open_worktree) in line)
    assert "age" not in open_line and "--resume" not in open_line

    # --force must not touch feat/open (no override for `open`); feat/old is removed.
    args_yes = SimpleNamespace(older_than="7d", dry_run=False, yes=True, force=True)
    jaxflow.cmd_gc(args_yes, run=_run_real, post=post, now=now, db_path=db, allowlist_root=allow_root)
    assert not worktree.exists()
    assert open_worktree.is_dir()
    assert len(posted) == 1
    assert posted[0]["type"] == "gc-removed"
    assert "run_id" not in posted[0]  # not runScoped -- lives in payload instead
    assert posted[0]["payload"]["branch"] == "feat/old"
    assert posted[0]["payload"]["reason"] == "success"
    assert posted[0]["payload"]["run_id"] == "aaaabbbbcccc"


def test_cmd_gc_out_of_allowlist_candidate_never_triggers_probe(tmp_path, monkeypatch):
    """F1 (round 4): every skip guard runs before any `is_dir`/`git status` probe -- a
    candidate outside the allowlist must never trigger one, whatever its ledger/state."""
    root = tmp_path / "repos" / "demo"
    _init_repo(root)
    worktree = _init_worktree(root, branch="feat/old")
    allow_root = tmp_path / "elsewhere"  # `worktree` lives OUTSIDE this allowlist root
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, "aaaabbbbcccc", "demo", "builder", "run-started",
            {"target": "feat/old", "repo": str(root), "phase": "p", "caller": "claude", "session": "s",
             "runtime": "opencode-builder", "model": "m", "effort": "e", "verify": "true"})
    _insert(con, "aaaabbbbcccc", "demo", "builder", "run-finished", {"contract_status": "ok", "result": "success"},
            ts="2026-09-01T00:00:00+00:00")
    con.close()
    monkeypatch.chdir(root)
    now = lambda: datetime(2026, 9, 19, tzinfo=timezone.utc)
    probed = []

    def recording_run(argv, cwd=None):
        if argv[:3] == ["git", "status", "--porcelain"]:
            probed.append(cwd)
        return _run_real(argv, cwd=cwd)

    args = SimpleNamespace(older_than="7d", dry_run=True, yes=False, force=False)
    jaxflow.cmd_gc(args, run=recording_run, post=_never_post, now=now, db_path=db, allowlist_root=allow_root)
    assert probed == [], "an out-of-allowlist candidate must never trigger the dirty-tree probe"


def _gc_single_candidate(tmp_path):
    """Shared fixture for the two F2 (plan review round 1) tests below: one repo, one
    old/finished/clean worktree that WOULD be removed absent a fresh dirty-tree probe."""
    allow_root = tmp_path / "repos"
    root = allow_root / "demo"
    _init_repo(root)
    worktree = _init_worktree(root, branch="feat/old")
    _plant_gc_manifest(root, "aaaabbbbcccc", worktree)  # F2 (round 4): reserves this exact path
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, "aaaabbbbcccc", "demo", "builder", "run-started",
            {"target": "feat/old", "repo": str(root), "phase": "p", "caller": "claude", "session": "s",
             "runtime": "opencode-builder", "model": "m", "effort": "e", "verify": "true"})
    _insert(con, "aaaabbbbcccc", "demo", "builder", "run-finished", {"contract_status": "ok", "result": "success"},
            ts="2026-09-01T00:00:00+00:00")
    con.close()
    return allow_root, root, worktree, db


def _probe_override_run(worktree, *, plant_dirt=False, override_result=None):
    """A `run` fake for the F2 test below: every "git status" probe on `worktree` passes
    through to `_run_real` except the SECOND one (the removal-time reprobe) -- the first
    (listing-time) probe always sees the worktree genuinely clean. `plant_dirt` writes an
    uncommitted file right before that second probe runs for real; `override_result`
    instead returns a canned result without touching the filesystem (a probe failure)."""
    seen = []
    def fake_run(argv, cwd=None):
        is_probe = (argv[:3] == ["git", "status", "--porcelain"] and cwd is not None
                    and Path(cwd).resolve() == worktree.resolve())
        if is_probe:
            seen.append(True)
            if len(seen) == 2:
                if override_result is not None:
                    return override_result
                if plant_dirt:
                    (worktree / "surprise.txt").write_text("uncommitted work\n", encoding="utf-8")
        return _run_real(argv, cwd=cwd)
    return fake_run


@pytest.mark.parametrize("kwargs,force", [
    (dict(plant_dirt=True), False),  # dirt appearing between listing and removal
    (dict(override_result=_completed(1, "", err="git: fatal error\n")), True),  # probe failure, --force included
])
def test_cmd_gc_yes_reprobes_before_removal_and_never_removes_when_unsafe(tmp_path, monkeypatch, kwargs, force):
    allow_root, root, worktree, db = _gc_single_candidate(tmp_path)
    monkeypatch.chdir(root)
    now = lambda: datetime(2026, 9, 19, tzinfo=timezone.utc)
    posted = []
    args_yes = SimpleNamespace(older_than="7d", dry_run=False, yes=True, force=force)
    jaxflow.cmd_gc(args_yes, run=_probe_override_run(worktree, **kwargs),
                   post=lambda e: posted.append(e) or {"ok": True}, now=now, db_path=db, allowlist_root=allow_root)
    assert worktree.is_dir(), "F2: unsafe to remove -- dirty or unprobeable, --force never overrides a probe failure"
    assert posted == []


def test_cmd_gc_yes_force_rechecks_registration_and_lock_before_removal(tmp_path, monkeypatch):
    """Round 3 F1 / spec §10: even with `--force`, a candidate locked (or no longer
    registered) between the listing and the removal is aborted by the porcelain re-list."""
    allow_root, root, worktree, db = _gc_single_candidate(tmp_path)
    monkeypatch.chdir(root)
    now = lambda: datetime(2026, 9, 19, tzinfo=timezone.utc)
    posted = []
    real_run = _run_real
    seen = []

    def locking_run(argv, cwd=None):
        if argv[:3] == ["git", "worktree", "list", "--porcelain"]:
            seen.append(True)
            if len(seen) == 2:  # the removal-time re-list sees the entry locked
                real = real_run(argv, cwd=cwd)
                locked = real.stdout.replace(
                    f"worktree {worktree}\n", f"worktree {worktree}\nlocked held\n", 1)
                return _completed(0, locked)
        return real_run(argv, cwd=cwd)

    args_yes = SimpleNamespace(older_than="7d", dry_run=False, yes=True, force=True)
    jaxflow.cmd_gc(args_yes, run=locking_run, post=lambda e: posted.append(e) or {"ok": True},
                   now=now, db_path=db, allowlist_root=allow_root)
    assert worktree.is_dir(), "a locked entry is never removed, --force included"
    assert posted == []


def test_cmd_gc_refuses_both_modes_and_bad_duration(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    now = lambda: datetime(2026, 9, 19, tzinfo=timezone.utc)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_gc(SimpleNamespace(older_than="7d", dry_run=True, yes=True, force=False), run=_run_real, post=_never_post, now=now, db_path=tmp_path / "x.db")
    assert exc.value.code == "gc-both-modes"
    with pytest.raises(ji.Refusal) as exc2:
        jaxflow.cmd_gc(SimpleNamespace(older_than="7hours", dry_run=True, yes=False, force=False), run=_run_real, post=_never_post, now=now, db_path=tmp_path / "x.db")
    assert exc2.value.code == "gc-duration-invalid"


def test_redeliver_gc_spools_retries_only_gc_removed_and_deletes_on_success(tmp_path):
    repo = tmp_path / "demo"
    repo.mkdir()
    gc_event = {"project": "demo", "role": "lead", "type": "gc-removed", "source": "deterministic", "emitter": "wrapper", "payload": {"run_id": "aaaabbbbcccc", "branch": "feat/x", "worktree": "/x", "reason": "success", "age_days": 9}}
    other_event = {"run_id": "ffffeeeedddd", "project": "demo", "role": "builder", "type": "run-finished", "source": "deterministic", "emitter": "wrapper", "payload": {"contract_status": "ok"}}
    jaxflow_common._write_spool(repo, "aaaabbbbcccc", gc_event)
    jaxflow_common._write_spool(repo, "ffffeeeedddd", other_event)  # NOT a gc-removed spool -- must survive
    posted = []
    jaxflow._redeliver_gc_spools(repo, post=lambda e: posted.append(e) or {"ok": True})
    assert posted == [gc_event]
    assert not jaxflow_common._spool_path(repo, "aaaabbbbcccc").exists()
    assert jaxflow_common._spool_path(repo, "ffffeeeedddd").exists()  # untouched
    jaxflow._redeliver_gc_spools(tmp_path / "no-such-repo", post=_never_post)  # must not raise


def test_redeliver_gc_spools_leaves_spool_on_post_failure_and_retries_next_call(tmp_path):
    repo = tmp_path / "demo"
    repo.mkdir()
    gc_event = {"project": "demo", "role": "lead", "type": "gc-removed", "source": "deterministic", "emitter": "wrapper", "payload": {"run_id": "aaaabbbbcccc", "branch": "feat/x", "worktree": "/x", "reason": "success", "age_days": 9}}
    jaxflow_common._write_spool(repo, "aaaabbbbcccc", gc_event)

    def failing_post(event):
        raise RuntimeError("event post failed: 500")

    jaxflow._redeliver_gc_spools(repo, post=failing_post)
    assert jaxflow_common._spool_path(repo, "aaaabbbbcccc").exists()

    # F6: at-least-once -- the spool survives and the NEXT call redelivers; no
    # idempotency key, a duplicate is a harmless repeat (ledger row is the source of truth).
    posted = []
    jaxflow._redeliver_gc_spools(repo, post=lambda e: posted.append(e) or {"ok": True})
    assert posted == [gc_event]
    assert not jaxflow_common._spool_path(repo, "aaaabbbbcccc").exists()


def test_dispatch_diff_review_started_payload_carries_builder_run_id(tmp_path, monkeypatch):
    allow_root = tmp_path / "repos"
    root = allow_root / "demo"
    _init_repo(root)
    worktree = _init_worktree(root, branch="feat/x")
    head = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    # Plan deviation: the plan's fixture omitted `kind: "build"` from the run-started row,
    # which makes dispatch_diff_review refuse with `unknown-run` before any post -- the
    # test could never reach its assertion.
    _insert(con, "aaaabbbbcccc", "demo", "builder", "run-started",
            {"kind": "build", "target": "feat/x", "repo": str(root), "phase": "p", "verify": "true"})
    _insert(con, "aaaabbbbcccc", "demo", "builder", "run-finished", {"contract_status": "ok", "result": "success", "head_sha": head})
    con.close()
    manifest_dir = root / ".local" / "runs" / "aaaabbbbcccc"
    manifest_dir.mkdir(parents=True)
    plan = _plan_file(root)
    (manifest_dir / "manifest.json").write_text(json.dumps({"plan_path": str(plan)}), encoding="utf-8")
    events = []
    args = SimpleNamespace(diff="aaaabbbbcccc", since=None, full=None, phase=None, model=None, effort=None,
                            from_caller="claude", no_callback=True, focus=None, reverify=False)
    monkeypatch.chdir(root)
    with contextlib.suppress(ji.Refusal):  # a subsequent tmux-launch failure is irrelevant -- post already fired
        jaxflow.dispatch_diff_review(
            args, run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: events.append(e) or {"ok": True},
            env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "sess"}, now=_fixed_now,
            allowlist_root=allow_root, db_path=db,
        )
    started = next(e for e in events if e["type"] == "run-started")
    assert started["payload"]["builder_run_id"] == "aaaabbbbcccc"


@pytest.mark.parametrize("raw,expected", [
    ("moa-467", "moa467"), ("moa467-claude-diff", "moa467claudediff"),
    ("2026-09-11-moa-472-loosen-formats-spec", "20260911moa472loosenformatsspec"),
])
def test_normalize_phase(raw, expected):
    assert jaxflow._normalize_phase(raw) == expected


def test_loop_refuses_short_prefix(tmp_path):
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_loop("abc", db_path=tmp_path / "x.db")
    assert exc.value.code == "loop-prefix-too-short"


def _loop_db(tmp_path):
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, "b1", "demo", "builder", "run-started", {"phase": "moa-474-a2", "kind": "build", "target": "feat/moa-474-a2"}, ts="2026-09-10T00:00:00+00:00")
    _insert(con, "b1", "demo", "builder", "run-finished", {"contract_status": "ok", "result": "success"})
    _insert(con, "d1", "demo", "reviewer", "run-started", {"phase": "moa-474-a2-diff-r1", "kind": "diff", "target": "feat/moa-474-a2", "builder_run_id": "b1"}, ts="2026-09-11T00:00:00+00:00")
    _insert(con, "d1", "demo", "reviewer", "run-finished", {"contract_status": "ok", "verdict": "reject"})
    _insert(con, "d2", "demo", "reviewer", "run-started", {"phase": "moa-474-a2-diff-r2", "kind": "diff", "target": "feat/moa-474-a2", "builder_run_id": "b1"}, ts="2026-09-12T00:00:00+00:00")
    _insert(con, "d2", "demo", "reviewer", "run-finished", {"contract_status": "ok", "verdict": "approve"})
    con.close()
    return db


def test_loop_anchor_builds_and_diff_linking_by_builder_run_id(tmp_path):
    db = _loop_db(tmp_path)
    ro = jaxflow_common._open_ro(db)
    ro.row_factory = sqlite3.Row
    anchors = jaxflow._anchor_builds_for_prefix(ro, "moa474")
    assert [a["run_id"] for a in anchors] == ["b1"]
    diffs = jaxflow._diff_rows_for_builds(ro, anchors)
    assert [d["run_id"] for d in diffs] == ["d1", "d2"]
    rounds, approved = jaxflow._rounds_to_approve(ro, diffs)
    assert (rounds, approved) == (2, "d2")
    assert jaxflow._wall_time_days(ro, anchors, [], []) is None  # not merged yet
    ro.close()


def test_loop_reviewer_rows_break_same_timestamp_ties_by_id(tmp_path):
    """F3 (round 5): two reviews sharing one `ts` must still resolve in insertion
    (id) order, matching the TypeScript side's `ORDER BY ts, id` -- a reversed
    tie-break would count only 1 round and approve early instead of 2."""
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    same_ts = "2026-09-10T00:00:00+00:00"
    _insert(con, "b1", "demo", "builder", "run-started", {"phase": "moa-600", "kind": "build", "target": "feat/moa-600"}, ts="2026-09-01T00:00:00+00:00")
    _insert(con, "d1", "demo", "reviewer", "run-started", {"phase": "moa-600-diff-r1", "kind": "diff", "target": "feat/moa-600", "builder_run_id": "b1"}, ts=same_ts)
    _insert(con, "d1", "demo", "reviewer", "run-finished", {"contract_status": "ok", "verdict": "reject"}, ts=same_ts)
    _insert(con, "d2", "demo", "reviewer", "run-started", {"phase": "moa-600-diff-r2", "kind": "diff", "target": "feat/moa-600", "builder_run_id": "b1"}, ts=same_ts)
    _insert(con, "d2", "demo", "reviewer", "run-finished", {"contract_status": "ok", "verdict": "approve"}, ts=same_ts)
    con.close()
    ro = jaxflow_common._open_ro(db)
    ro.row_factory = sqlite3.Row
    anchors = jaxflow._anchor_builds_for_prefix(ro, "moa600")
    diffs = jaxflow._diff_rows_for_builds(ro, anchors)
    assert [d["run_id"] for d in diffs] == ["d1", "d2"]
    rounds, approved = jaxflow._rounds_to_approve(ro, diffs)
    assert (rounds, approved) == (2, "d2")
    ro.close()


def test_loop_diff_linking_falls_back_to_branch_for_legacy_rows_without_builder_run_id(tmp_path):
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, "b1", "demo", "builder", "run-started", {"phase": "moa-467", "kind": "build", "target": "feat/moa-467"})
    _insert(con, "b1", "demo", "builder", "run-finished", {"contract_status": "ok", "result": "success"})
    _insert(con, "d1", "demo", "reviewer", "run-started", {"phase": "moa467-claude-diff", "kind": "diff", "target": "feat/moa-467"})  # legacy: no builder_run_id
    con.close()
    ro = jaxflow_common._open_ro(db)
    ro.row_factory = sqlite3.Row
    anchors = jaxflow._anchor_builds_for_prefix(ro, "moa467")
    diffs = jaxflow._diff_rows_for_builds(ro, anchors)
    assert [d["run_id"] for d in diffs] == ["d1"]
    ro.close()


def test_cmd_loop_produces_nonzero_counts(tmp_path):
    db = _loop_db(tmp_path)
    output = jaxflow.cmd_loop("moa-474", db_path=db)
    assert "build 1" in output
    assert "diff 2" in output
    assert "rounds to approve: 2" in output


def test_cmd_loop_family_matches_ts_getLoopSummary_fixture(tmp_path):
    """Cross-language fixture (F4, round 4): the SAME events as vitest's
    `workflows.test.ts` "counts the whole anchored family..." case -- a resumed build
    sharing its phase with the first attempt, plus one spec review -- must produce the
    same counts (build 2, spec 1) from both `cmd_loop` and TypeScript's
    `getLoopSummary`."""
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, "s1", "family", "reviewer", "run-started", {"phase": "moa510", "kind": "spec", "target": "docs/spec.md"}, ts="2026-09-01T00:00:00+00:00")
    _insert(con, "b1", "family", "builder", "run-started", {"phase": "moa510", "kind": "build", "target": "feat/z"}, ts="2026-09-02T00:00:00+00:00")
    _insert(con, "b2", "family", "builder", "run-started", {"phase": "moa510", "kind": "build", "target": "feat/z"}, ts="2026-09-03T00:00:00+00:00")
    con.close()
    output = jaxflow.cmd_loop("moa510", db_path=db)
    assert "spec 1" in output
    assert "build 2" in output


def test_wall_time_falls_back_short_path_no_spec_review(tmp_path):
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, "b1", "demo", "builder", "run-started", {"phase": "x", "kind": "build", "target": "feat/x"}, ts="2026-09-10T00:00:00+00:00")
    _insert(con, "m1", "demo", "wrapper", "merge-approved", {"branch": "feat/x"}, ts="2026-09-13T00:00:00+00:00")
    con.close()
    ro = jaxflow_common._open_ro(db)
    ro.row_factory = sqlite3.Row
    anchors = jaxflow._anchor_builds_for_prefix(ro, "x")
    days = jaxflow._wall_time_days(ro, anchors, [], [])  # no spec/plan matched -- falls back to the build itself
    assert days == 3.0
    ro.close()


def test_wall_time_days_windows_the_merge_match_to_the_anchored_build(tmp_path):
    """F4 (plan review round 1): the merge match is windowed to the anchored build's own
    dispatch, never "the first merge-approved event that ever matched this branch". Two
    branches, two scenarios in one DB: `feat/reused` was merged once for OLD, unrelated
    work (b0/m0), then reused (b1/m1) -- the second build's wall time must use the
    second merge; `feat/x` was merged once (m2) before its OWN build (b2) even existed --
    with no merge inside b2's window, wall time is null/absent, never negative."""
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, "b0", "demo", "builder", "run-started", {"phase": "old-work", "kind": "build", "target": "feat/reused"}, ts="2026-01-01T00:00:00+00:00")
    _insert(con, "m0", "demo", "lead", "merge-approved", {"branch": "feat/reused"}, ts="2026-01-03T00:00:00+00:00")
    _insert(con, "b1", "demo", "builder", "run-started", {"phase": "moa-500", "kind": "build", "target": "feat/reused"}, ts="2026-09-10T00:00:00+00:00")
    _insert(con, "m1", "demo", "lead", "merge-approved", {"branch": "feat/reused"}, ts="2026-09-15T00:00:00+00:00")
    _insert(con, "m2", "demo", "lead", "merge-approved", {"branch": "feat/x"}, ts="2026-01-01T00:00:00+00:00")
    _insert(con, "b2", "demo", "builder", "run-started", {"phase": "moa-501", "kind": "build", "target": "feat/x"}, ts="2026-09-10T00:00:00+00:00")
    con.close()
    ro = jaxflow_common._open_ro(db)
    ro.row_factory = sqlite3.Row
    reused = jaxflow._anchor_builds_for_prefix(ro, "moa500")
    assert jaxflow._wall_time_days(ro, reused, [], []) == 5.0  # never the 2026-01 merge
    predates = jaxflow._anchor_builds_for_prefix(ro, "moa501")
    assert jaxflow._wall_time_days(ro, predates, [], []) is None  # m2 predates b2's dispatch
    ro.close()


def test_wall_time_days_null_when_trailing_review_is_after_the_merge(tmp_path):
    """F1 (diff review 79cbcce4, MEDIUM). Cross-language fixture: same events as vitest's
    "a trailing plan review after a build already exists..." case -- build, diff and merge
    happen first, then a plan re-review is dispatched LAST. The matched plan ts is AFTER
    merge_ts, so the wall time must be null, never negative."""
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, "b1", "trailing", "builder", "run-started", {"phase": "moa610", "kind": "build", "target": "feat/moa-610"}, ts="2026-09-01T00:00:00+00:00")
    _insert(con, "m1", "trailing", "lead", "merge-approved", {"branch": "feat/moa-610"}, ts="2026-09-03T00:00:00+00:00")
    _insert(con, "p1", "trailing", "reviewer", "run-started", {"phase": "moa610", "kind": "plan", "target": "docs/plan.md"}, ts="2026-09-04T00:00:00+00:00")
    con.close()
    ro = jaxflow_common._open_ro(db)
    ro.row_factory = sqlite3.Row
    anchors = jaxflow._anchor_builds_for_prefix(ro, "moa610")
    build_phases_norm = [jaxflow._normalize_phase(b["payload"]["phase"]) for b in anchors]
    plan_rows = jaxflow._spec_or_plan_rows_for_build_phases(ro, "plan", build_phases_norm)
    assert jaxflow._wall_time_days(ro, anchors, [], plan_rows) is None
    ro.close()
# ---- Phase 2: --from jaxos (spec §8, Decisions 6/23) ----

def test_resolve_caller_accepts_jaxos_from_flag_without_env_markers():
    assert jaxflow_common.resolve_caller({}, "jaxos") == "jaxos"
    assert jaxflow_common._CALLER_SESSION_VAR["jaxos"] == "JAXOS_CALLER_SESSION"


def test_main_accepts_from_jaxos_on_review_build_and_merge(monkeypatch):
    seen = []
    monkeypatch.setattr(jaxflow, "dispatch_review", lambda args, **kw: seen.append(("review", args.from_caller)) or "r1")
    monkeypatch.setattr(jaxflow, "dispatch_build", lambda args, **kw: seen.append(("build", args.from_caller)) or "b1")
    monkeypatch.setattr(jaxflow, "cmd_merge", lambda args, **kw: seen.append(("merge", args.from_caller)) or jaxflow_common.OK)
    assert jaxflow.main(["review", "--spec", "x.md", "--from", "jaxos"]) == jaxflow_common.OK
    assert jaxflow.main([
        "build", "--plan", "p.md", "--phase", "P", "--branch", "feat/x", "--whitelist", "a",
        "--verify", "true", "--from", "jaxos",
    ]) == jaxflow_common.OK
    assert jaxflow.main([
        "merge", "feat/x", "--sha", "a" * 40, "--phase", "P", "--checks", "true",
        "--target", "main", "--from", "jaxos",
    ]) == jaxflow_common.OK
    assert seen == [("review", "jaxos"), ("build", "jaxos"), ("merge", "jaxos")]


def test_jaxos_review_dispatch_forces_no_callback_and_records_caller(monkeypatch):
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

        with pytest.raises(ji.Refusal) as exc:
            jaxflow.dispatch_review(
                _review_args(spec=str(target), from_caller="jaxos"),
                run=_run_with_tmux(fake, real_cwd=root), post=post,
                env={}, now=_fixed_now, allowlist_root=allow_root,
            )
        assert exc.value.code == "caller-session-missing"
        assert exc.value.hint == "hint: set JAXOS_CALLER_SESSION"
        assert events == []

        run_id = jaxflow.dispatch_review(
            _review_args(spec=str(target), from_caller="jaxos"),   # NO no_callback=True
            run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"JAXOS_CALLER_SESSION": "jaxos"}, now=_fixed_now, allowlist_root=allow_root,
        )
        payload = events[0]["payload"]
        assert payload["caller"] == "jaxos" and payload["caller_session"] == "jaxos"
        assert events[0]["emitter"] == "wrapper"
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["no_callback"] is True
        assert manifest["runtime"] == "codex"   # the same reviewer runtime a claude caller gets


def test_jaxos_build_dispatch_forces_no_callback_and_records_caller(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        events = []

        def post(event):
            events.append(event)
            return {"ok": True}

        run_id = jaxflow.dispatch_build(
            _build_args(plan=str(plan), from_caller="jaxos"),
            run=_run_with_tmux(fake, real_cwd=root), post=post,
            env={"JAXOS_CALLER_SESSION": "jaxos"}, now=_fixed_now, allowlist_root=allow_root,
        )
        payload = events[0]["payload"]
        assert payload["caller"] == "jaxos" and payload["caller_session"] == "jaxos"
        manifest = json.loads((root / ".local" / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["no_callback"] is True


def _status_md_fixture(tmp_path):
    (tmp_path / ".jax-os").mkdir()
    status = tmp_path / ".jax-os" / "status.md"
    status.write_text(
        "---\nproject: Demo\nstage: review\nbuilder: codex\nbranch: feat/x\n"
        "gate: awaiting-approval\nupdated: 2020-01-01T00:00:00-03:00\n---\n\n## Now\nOld text.\n",
        encoding="utf-8",
    )
    return status


def test_merge_from_jaxos_keeps_the_existing_builder_field(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    status = _status_md_fixture(tmp_path)
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    rc = jaxflow.cmd_merge(_MergeArgs(sha=sha, from_caller="jaxos"), run=fake_run, post=lambda e: None,
                           env={}, now=_fixed_now, allowlist_root=tmp_path.parent)
    assert rc == jaxflow_common.OK
    text = status.read_text(encoding="utf-8")
    assert "stage: ship" in text
    assert "builder: codex" in text
    assert "jaxos" not in text


def test_merge_from_claude_still_writes_the_caller_as_builder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    status = _status_md_fixture(tmp_path)
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    jaxflow.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    text = status.read_text(encoding="utf-8")
    assert "stage: ship" in text and "builder: claude" in text


def _recording_post(status, body):
    calls = []
    def post(url, payload):
        calls.append((url, payload))
        return status, body
    return post, calls


def test_mission_start_posts_the_assembled_body_and_succeeds_on_ok():
    post, calls = _recording_post(200, {"ok": True, "data": {"id": 1}})
    args = SimpleNamespace(name="Ship it", goal="6 phases tonight", milestones=["Phase 1", "Phase 2"])
    jaxflow.cmd_mission_start(args, post=post)
    assert calls == [(jaxflow.MISSION_BASE_URL, {"name": "Ship it", "goal": "6 phases tonight", "milestones": ["Phase 1", "Phase 2"]})]


def test_mission_start_refuses_with_no_milestones_before_any_post():
    def post(url, payload):
        raise AssertionError("must not post with zero milestones")
    args = SimpleNamespace(name="Ship it", goal="g", milestones=[])
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_mission_start(args, post=post)
    assert exc.value.code == "malformed milestone"


def test_mission_start_reraises_the_route_error_verbatim():
    post, _ = _recording_post(200, {"ok": False, "error": "mission-active"})
    args = SimpleNamespace(name="Ship it", goal="g", milestones=["m1"])
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_mission_start(args, post=post)
    assert exc.value.code == "mission-active"


def test_mission_start_reraises_every_field_shape_refusal_verbatim():
    for code in ("too-many-milestones", "malformed name", "malformed goal", "malformed milestone"):
        post, _ = _recording_post(200, {"ok": False, "error": code})
        args = SimpleNamespace(name="Ship it", goal="g", milestones=["m1"])
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_mission_start(args, post=post)
        assert exc.value.code == code


def test_cancel_spool_redelivery_hub_rejected_keeps_spool(monkeypatch):
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        repo = Path(raw) / "demo"
        repo.mkdir()
        _insert(con, "aaaabbbbcccc", "demo", "builder", "run-started", {
            "session": "jax-demo-build-aaaabbbbcccc", "phase": "P", "caller": "claude", "repo": str(repo),
        })
        con.close()
        jaxflow_common._write_spool(repo, "aaaabbbbcccc", {"run_id": "aaaabbbbcccc", "payload": {"result": "failure"}})
        monkeypatch.setattr(time, "sleep", lambda s: None)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_cancel(
                "aaaabbbbcccc", run=_run_with_tmux(FakeTmux()),
                post=lambda url, body: (500, {}), now=lambda: None, db_path=db,
            )
        assert exc.value.code == "hub-rejected: 500"
        assert jaxflow_common._spool_path(repo, "aaaabbbbcccc").exists()


def test_cancel_spool_redelivery_transport_error_keeps_spool(monkeypatch):
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = _fresh_db(db)
        repo = Path(raw) / "demo"
        repo.mkdir()
        _insert(con, "aaaabbbbcccc", "demo", "builder", "run-started", {
            "session": "jax-demo-build-aaaabbbbcccc", "phase": "P", "caller": "claude", "repo": str(repo),
        })
        con.close()
        jaxflow_common._write_spool(repo, "aaaabbbbcccc", {"run_id": "aaaabbbbcccc", "payload": {"result": "failure"}})
        monkeypatch.setattr(time, "sleep", lambda s: None)

        def post(url, body):
            raise OSError("down")

        with pytest.raises(ji.Refusal) as exc:
            jaxflow.cmd_cancel(
                "aaaabbbbcccc", run=_run_with_tmux(FakeTmux()), post=post, now=lambda: None, db_path=db,
            )
        assert exc.value.code == "hub-unreachable"
        assert jaxflow_common._spool_path(repo, "aaaabbbbcccc").exists()


def test_mission_post_500_empty_body_is_mission_hub_rejected_status():
    with pytest.raises(ji.Refusal) as exc:
        jaxflow._mission_post("http://127.0.0.1:9/mission", {"name": "n"}, post=lambda url, payload: (500, {}))
    assert exc.value.code == "mission-hub-rejected: 500"
    with pytest.raises(ji.Refusal) as exc:
        jaxflow._mission_post(
            "http://127.0.0.1:9/mission", {"name": "n"},
            post=lambda url, payload: (400, {"ok": False, "error": "too-many-milestones"}),
        )
    assert exc.value.code == "too-many-milestones"

    def down(url, payload):
        raise OSError("down")

    with pytest.raises(ji.Refusal) as exc:
        jaxflow._mission_post("http://127.0.0.1:9/mission", {"name": "n"}, post=down)
    assert exc.value.code == "mission-hub-unreachable"


def test_mission_start_transport_failure_is_mission_hub_unreachable():
    def post(url, payload):
        raise OSError("connection refused")
    args = SimpleNamespace(name="Ship it", goal="g", milestones=["m1"])
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_mission_start(args, post=post)
    assert exc.value.code == "mission-hub-unreachable"


def test_mission_status_posts_status_line_and_reraises_no_active_mission():
    post, calls = _recording_post(200, {"ok": True, "data": {}})
    jaxflow.cmd_mission_status(SimpleNamespace(text="phase 1 merged"), post=post)
    assert calls == [(f"{jaxflow.MISSION_BASE_URL}/status", {"status_line": "phase 1 merged"})]
    post2, _ = _recording_post(200, {"ok": False, "error": "no-active-mission"})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_mission_status(SimpleNamespace(text="x"), post=post2)
    assert exc.value.code == "no-active-mission"


def test_mission_status_reraises_malformed_status_verbatim():
    post, _ = _recording_post(200, {"ok": False, "error": "malformed status"})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_mission_status(SimpleNamespace(text=""), post=post)
    assert exc.value.code == "malformed status"


def test_mission_mark_resolves_index_and_title_forms_and_reraises_unknown_milestone():
    post, calls = _recording_post(200, {"ok": True, "data": {}})
    jaxflow.cmd_mission_mark(SimpleNamespace(milestone="2", state="done"), post=post)
    jaxflow.cmd_mission_mark(SimpleNamespace(milestone="Phase 1", state="in-progress"), post=post)
    assert calls == [
        (f"{jaxflow.MISSION_BASE_URL}/milestone", {"milestone": "2", "state": "done"}),
        (f"{jaxflow.MISSION_BASE_URL}/milestone", {"milestone": "Phase 1", "state": "in-progress"}),
    ]
    post2, _ = _recording_post(200, {"ok": False, "error": "unknown-milestone"})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_mission_mark(SimpleNamespace(milestone="nope", state="done"), post=post2)
    assert exc.value.code == "unknown-milestone"


def test_mission_mark_refuses_malformed_state_before_any_post():
    def post(url, payload):
        raise AssertionError("must not post an invalid state")
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_mission_mark(SimpleNamespace(milestone="1", state="whatever"), post=post)
    assert exc.value.code == "malformed state"


def test_mission_done_and_cancel_post_the_matching_outcome_and_reraise_no_active_mission():
    post, calls = _recording_post(200, {"ok": True, "data": {}})
    jaxflow.cmd_mission_finish("done", post=post)
    jaxflow.cmd_mission_finish("cancelled", post=post)
    assert calls == [
        (f"{jaxflow.MISSION_BASE_URL}/finish", {"outcome": "done"}),
        (f"{jaxflow.MISSION_BASE_URL}/finish", {"outcome": "cancelled"}),
    ]
    post2, _ = _recording_post(200, {"ok": False, "error": "no-active-mission"})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_mission_finish("done", post=post2)
    assert exc.value.code == "no-active-mission"


def test_mission_show_prints_no_active_mission():
    def get(url):
        return 200, {"ok": True, "data": None}
    assert jaxflow.cmd_mission_show(get=get) == "no active mission"


def test_mission_show_formats_the_active_mission():
    def get(url):
        assert url == f"{jaxflow.MISSION_BASE_URL}/current"
        return 200, {"ok": True, "data": {
            "name": "Ship it", "goal": "6 phases tonight", "statusLine": "phase 1 merged",
            "milestones": [{"title": "Phase 1", "state": "done"}, {"title": "Phase 2", "state": "in-progress"}, {"title": "Phase 3", "state": "pending"}],
        }}
    assert jaxflow.cmd_mission_show(get=get) == (
        "Ship it: 6 phases tonight\nstatus: phase 1 merged\n  [x] Phase 1\n  [~] Phase 2\n  [ ] Phase 3"
    )


def test_mission_show_transport_failure_is_mission_hub_unreachable():
    def get(url):
        raise OSError("unreachable")
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_mission_show(get=get)
    assert exc.value.code == "mission-hub-unreachable"


def test_mission_argparse_wiring_end_to_end(monkeypatch):
    calls = []
    def fake_post(url, payload):
        calls.append((url, payload))
        return 200, {"ok": True, "data": {"id": 1}}
    monkeypatch.setattr(jaxflow_common, "_post", fake_post)
    rc = jaxflow.main(["mission", "start", "--name", "Ship it", "--goal", "g", "--milestone", "M1", "--milestone", "M2"])
    assert rc == jaxflow_common.OK
    assert calls == [(jaxflow.MISSION_BASE_URL, {"name": "Ship it", "goal": "g", "milestones": ["M1", "M2"]})]

    def fake_get(url):
        return 200, {"ok": True, "data": None}
    monkeypatch.setattr(jaxflow, "_get", fake_get)
    rc2 = jaxflow.main(["mission", "show"])
    assert rc2 == jaxflow_common.OK


def test_mission_argparse_requires_at_least_one_milestone_flag():
    args = jaxflow.parse_args(["mission", "start", "--name", "A", "--goal", "g"])
    assert args.milestones == []


def test_mission_refusal_through_main_exits_refused_with_the_exact_stderr_code(monkeypatch, capsys):
    # Cold review F8: every other mission test above calls a cmd_mission_* helper directly, or
    # drives main() only on the success path -- this is the one test proving a refusal survives
    # the FULL main() path (parse_args -> dispatch -> except Refusal) with the documented exit
    # code and stderr shape (main():5769-5774 -- `print(exc.code, file=sys.stderr); return REFUSED`).
    def fake_post(url, payload):
        return 200, {"ok": False, "error": "no-active-mission"}
    monkeypatch.setattr(jaxflow_common, "_post", fake_post)
    rc = jaxflow.main(["mission", "status", "phase 1 merged"])
    assert rc == jaxflow_common.REFUSED
    assert capsys.readouterr().err == "no-active-mission\n"

# ---- slice e: model defaults (spec: One source of model defaults) ----

_MODEL_DEFAULTS = json.loads((Path(__file__).resolve().parents[1] / "workflow" / "fixtures" / "model-defaults-v1.json").read_text())


def test_jaxflow_reviewer_defaults_matches_jaxflow_settings_with_no_drift():
    assert jaxflow.REVIEWER_DEFAULTS == {
        "claude": {"runtime": "codex", **_MODEL_DEFAULTS["reviewers"]["codex"]},
        "codex": {"runtime": "claude", **_MODEL_DEFAULTS["reviewers"]["claude"]},
    }
    assert jaxflow.REVIEWER_DEFAULTS == {
        k: v for k, v in jset.REVIEWER_DEFAULTS.items() if k in ("claude", "codex")
    }

# ---- slice f: missing-cli refusal (spec: Clean error on missing CLI) ----

class _MissingCliPopen:
    def __init__(self, argv, **_kwargs):
        raise FileNotFoundError(argv[0])


def _prepare_builder_worker_fixture(tmp_path):
    allow_root = tmp_path / "repos"
    root = allow_root / "demo"
    _init_repo(root)
    worktree = _init_worktree(root)
    _write_plan(worktree)
    manifest_path, manifest = _write_manifest_for_builder_worker(root, worktree)
    return allow_root, root, worktree, manifest_path, manifest


def _prepare_diff_reviewer_worker_fixture(tmp_path):
    root = tmp_path / "demo"
    _init_repo(root)
    worktree = _init_worktree(root, "feat/x")
    base_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
    (worktree / "change.txt").write_text("change\n", encoding="utf-8")
    _git(worktree, "add", "change.txt")
    _git(worktree, "commit", "-m", "change")
    head_sha = _run_real(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
    manifest_path, manifest = _write_manifest_for_diff_worker(
        root, worktree, base_sha=base_sha, head_sha=head_sha,
    )
    return manifest_path, manifest, root, worktree


def _prepare_doc_review_worker_fixture(tmp_path):
    root = tmp_path / "demo"
    _init_repo(root)
    target = _spec_file(root)
    manifest_path, manifest = _write_manifest_for_worker(root, target)
    return manifest_path, manifest, root


def test_builder_worker_refuses_missing_cli_with_no_traceback_and_cleans_up_the_reservation(tmp_path):
    allow_root, root, worktree, manifest_path, manifest = _prepare_builder_worker_fixture(tmp_path)
    events = []
    code = jaxflow.run_worker(
        str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
        post=lambda e: events.append(e) or {"ok": True}, popen=_MissingCliPopen,
        allowlist_root=allow_root,
    )
    assert code == jaxflow_common.REFUSED
    finished = [e for e in events if e["type"] == "run-finished"][0]
    assert finished["payload"]["contract_status"] == "cancelled"
    assert "missing-cli" in finished["payload"]["summary"]
    assert not worktree.exists()  # _refuse_builder_run's existing cleanup ran


def test_diff_reviewer_worker_refuses_missing_cli_with_no_worktree_cleanup_attempted(tmp_path):
    manifest_path, manifest, repo, worktree = _prepare_diff_reviewer_worker_fixture(tmp_path)
    events = []
    code = jaxflow.run_worker(
        str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
        post=lambda e: events.append(e) or {"ok": True}, popen=_MissingCliPopen,
        allowlist_root=repo.parent,
    )
    assert code == jaxflow_common.REFUSED
    finished = [e for e in events if e["type"] == "run-finished"][0]
    assert finished["payload"]["contract_status"] == "cancelled"
    assert "missing-cli" in finished["payload"]["summary"]
    assert worktree.is_dir()  # _refuse_diff_run never owns/cleans the worktree


def test_doc_review_worker_refuses_missing_cli(tmp_path):
    manifest_path, manifest, repo = _prepare_doc_review_worker_fixture(tmp_path)
    events = []
    code = jaxflow.run_worker(
        str(manifest_path), run=_run_with_tmux(FakeTmux()),
        post=lambda e: events.append(e) or {"ok": True}, popen=_MissingCliPopen,
        allowlist_root=repo.parent,
    )
    assert code == jaxflow_common.REFUSED
    finished = [e for e in events if e["type"] == "run-finished"][0]
    assert "missing-cli" in finished["payload"]["summary"]


# ------------------------------------------------------------------ slice f: doctor

def _doctor_which(repo_root, present):
    """which() stub: names in `present` resolve to real wrapper files created under
    repo_root/bin (never a fake path outside repo_root, which would itself fail the
    containment check `_doctor_wrapper` performs) -- every other name is absent."""
    bin_dir = repo_root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in present:
        wrapper = bin_dir / name
        if not wrapper.exists():
            wrapper.write_text("#!/bin/sh\n")
    return lambda name, path=None: (str(bin_dir / name) if name in present else None)


def _doctor_fixtures(tmp_path):
    """Every doctor test injects these so none of them ever reads Rafa's real
    ~/.claude, ~/.codex, ~/.agents skill/hook files or the real $JAXOS_HOME/.env
    and the OpenCode installer fallback under `$HOME` (an isolated, empty temporary home)."""
    return dict(
        skill_dirs={
            "claude": tmp_path / "fake-claude-skills",
            "codex": tmp_path / "fake-codex-skills",
            "opencode": tmp_path / "fake-opencode-skills",
        },
        hook_files={
            "claude": tmp_path / "fake-claude-hooks.json",
            "codex": tmp_path / "fake-codex-hooks.json",
        },
        env_status=lambda: {"state": "ok"},
        home=tmp_path / "home",
    )


def test_doctor_wrapper_checks_ok_when_which_resolves_under_this_checkout(tmp_path):
    repo_root = tmp_path / "checkout"
    (repo_root / "bin").mkdir(parents=True)
    for name in ("jaxflow", "jaxflow-hook", "jax-init"):
        (repo_root / "bin" / name).write_text("#!/bin/sh\n")
    which = lambda name, path=None: str(repo_root / "bin" / name) if name in ("jaxflow", "jaxflow-hook", "jax-init") else None
    lines, code = jaxflow.cmd_doctor(repo_root=repo_root, which=which,
                                      read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}},
                                      get=lambda url: (200, {}), **_doctor_fixtures(tmp_path))
    assert any(l == "wrapper:jaxflow ok" for l in lines)
    assert any(l == "wrapper:jaxflow-hook ok" for l in lines)
    assert any(l == "wrapper:jax-init ok" for l in lines)


def test_doctor_wrapper_missing_is_required_and_fails_the_exit_code(tmp_path):
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    lines, code = jaxflow.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
                                      read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}},
                                      get=lambda url: (200, {}), **_doctor_fixtures(tmp_path))
    assert any(l.startswith("wrapper:jaxflow missing") for l in lines)
    assert code != 0


def test_doctor_wrapper_resolving_outside_this_checkout_is_missing(tmp_path):
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    elsewhere = tmp_path / "elsewhere" / "jaxflow"
    elsewhere.parent.mkdir()
    elsewhere.write_text("#!/bin/sh\n")
    which = lambda name, path=None: str(elsewhere) if name == "jaxflow" else None
    lines, code = jaxflow.cmd_doctor(repo_root=repo_root, which=which,
                                      read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}},
                                      get=lambda url: (200, {}), **_doctor_fixtures(tmp_path))
    assert any(l.startswith("wrapper:jaxflow missing") for l in lines)


def test_doctor_skill_ok_only_when_it_resolves_inside_this_checkouts_workflow_skills(tmp_path):
    repo_root = tmp_path / "checkout"
    (repo_root / "workflow" / "skills" / "jaxflow").mkdir(parents=True)
    claude_skills = tmp_path / "fake-claude-skills"
    claude_skills.mkdir()
    # An unrelated pre-existing dir with the right name, but NOT under this checkout's
    # workflow/skills/, must not count as installed.
    (claude_skills / "jaxflow").mkdir()
    fixtures = _doctor_fixtures(tmp_path)
    fixtures["skill_dirs"]["claude"] = claude_skills
    lines, _ = jaxflow.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
                                   read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}},
                                   get=lambda url: (200, {}), **fixtures)
    assert any(l.startswith("skill:claude missing") for l in lines)

    # A symlink resolving to a DIFFERENT skill under workflow/skills/ (swapped target) must
    # not count as installed -- descendant-of-skills_root is not enough, it must match `name`.
    (repo_root / "workflow" / "skills" / "jax-init").mkdir(parents=True)
    (claude_skills / "jaxflow").rmdir()
    (claude_skills / "jaxflow").symlink_to(repo_root / "workflow" / "skills" / "jax-init")
    lines, _ = jaxflow.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
                                   read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}},
                                   get=lambda url: (200, {}), **fixtures)
    assert any(l.startswith("skill:claude missing") for l in lines)

    # A symlink resolving into this checkout's workflow/skills/ counts as installed.
    (claude_skills / "jaxflow").unlink()
    (claude_skills / "jaxflow").symlink_to(repo_root / "workflow" / "skills" / "jaxflow")
    lines, _ = jaxflow.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
                                   read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}},
                                   get=lambda url: (200, {}), **fixtures)
    assert any(l == "skill:claude ok" for l in lines)


def test_doctor_custom_codex_home_note_does_not_count_toward_the_checks_total(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "custom-codex-home"))
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    which = _doctor_which(repo_root, {"jaxflow", "jaxflow-hook", "jax-init", "tmux", "git", "rg", "gh"})
    lines, code = jaxflow.cmd_doctor(repo_root=repo_root, which=which,
                                      read_settings=lambda: {"ok": True, "data": {"integrations": {"github": True, "agents": _ALL_AGENTS_ON}}},
                                      get=lambda url: (200, {}), **_doctor_fixtures(tmp_path))
    assert any(l.startswith("harness-home: custom CODEX_HOME") for l in lines)
    summary = [l for l in lines if l.endswith(" checks passed")][0]
    _passed, total = summary.split(" checks passed")[0].split("/")
    check_lines = [l for l in lines if l != summary and not l.startswith("harness-home:")]
    assert int(total) == len(check_lines)  # the info line must not inflate the total


def test_doctor_cli_gh_is_missing_but_optional_when_github_integration_is_off(tmp_path):
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    lines, code = jaxflow.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
                                      read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}},
                                      get=lambda url: (200, {}), **_doctor_fixtures(tmp_path))
    assert any(l.startswith("cli:gh missing") for l in lines)
    # gh being absent must not, by itself, fail the exit code when github is off. All three
    # wrappers plus tmux/git/rg resolve for real, under repo_root/bin, so gh is the only gap:
    only_gh_missing = jaxflow.cmd_doctor(
        repo_root=repo_root,
        which=_doctor_which(repo_root, {"jaxflow", "jaxflow-hook", "jax-init", "tmux", "git", "rg", "claude", "codex", "opencode"}),
        read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}}, get=lambda url: (200, {}),
        **_doctor_fixtures(tmp_path))
    assert only_gh_missing[1] == 0


def test_doctor_cli_gh_is_required_when_github_integration_is_on(tmp_path):
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    lines, code = jaxflow.cmd_doctor(
        repo_root=repo_root,
        which=_doctor_which(repo_root, {"jaxflow", "jaxflow-hook", "jax-init", "tmux", "git", "rg"}),
        read_settings=lambda: {"ok": True, "data": {"integrations": {"github": True, "agents": _ALL_AGENTS_ON}}}, get=lambda url: (200, {}),
        **_doctor_fixtures(tmp_path))
    assert any(l.startswith("cli:gh missing") for l in lines)
    assert code != 0


def test_doctor_settings_readable_reflects_the_injected_reader(tmp_path):
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    lines, _ = jaxflow.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
                                   read_settings=lambda: {"ok": False, "error": "settings-malformed"},
                                   get=lambda url: (200, {}), **_doctor_fixtures(tmp_path))
    assert any(l.startswith("settings:readable missing") for l in lines)


def test_doctor_api_reachable_true_on_any_http_response_false_only_on_transport_error(tmp_path):
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    ok_lines, _ = jaxflow.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
                                      read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}},
                                      get=lambda url: (503, {"ok": False}), **_doctor_fixtures(tmp_path))
    assert any(l == "api:reachable ok" for l in ok_lines)
    def down(url):
        raise ConnectionRefusedError()
    down_lines, code = jaxflow.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
                                           read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}},
                                           get=down, **_doctor_fixtures(tmp_path))
    assert any(l.startswith("api:reachable missing") for l in down_lines)
    assert code != 0


def test_doctor_makes_no_write_and_no_gh_or_git_mutation(tmp_path, monkeypatch):
    """Doctor is read-only (spec MOA-502 Decision 8): patch the actual mutation boundary --
    subprocess.run/Popen, the only way jaxflow.py could shell out to git/gh -- to raise if
    called at all, instead of asserting on a `which`-call log that never records a mutation
    (the old assertion was `... or True`, unconditionally true)."""
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()

    def _forbidden(*args, **kwargs):
        raise AssertionError("cmd_doctor must never invoke subprocess.run/Popen")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(subprocess, "Popen", _forbidden)

    which = _doctor_which(repo_root, set())  # creates repo_root/bin fixtures BEFORE the snapshot
    before = set(repo_root.rglob("*"))
    jaxflow.cmd_doctor(repo_root=repo_root, which=which,
                        read_settings=lambda: {"ok": True, "data": {"integrations": {"github": True, "agents": _ALL_AGENTS_ON}}},
                        get=lambda url: (200, {}), **_doctor_fixtures(tmp_path))
    after = set(repo_root.rglob("*"))
    assert after == before  # no write anywhere under repo_root


_DOCTOR_BASE = {"jaxflow", "jaxflow-hook", "jax-init", "tmux", "git", "rg"}


def _doctor(tmp_path, agents, present=(), home=None, settings_ok=True):
    repo_root = tmp_path / "checkout"
    repo_root.mkdir(exist_ok=True)
    settings = ({"ok": True, "data": {"integrations": {"github": False, "agents": agents}}} if settings_ok
                else {"ok": False, "error": "settings-malformed"})
    fixtures = _doctor_fixtures(tmp_path)  # already carries the isolated `home`
    if home is not None:
        fixtures["home"] = home
    return jaxflow.cmd_doctor(
        repo_root=repo_root, which=_doctor_which(repo_root, _DOCTOR_BASE | set(present)),
        read_settings=lambda: settings, get=lambda url: (200, {}), **fixtures)


def _line(lines, prefix):
    return next((l for l in lines if l.startswith(prefix)), None)


def test_doctor_all_agents_off_flags_agents_any_and_checks_no_agent(tmp_path):
    lines, code = _doctor(tmp_path, dict(claude=False, codex=False, opencode=False))
    assert _line(lines, "agents:any missing — jaxflow will not run: it needs at least one agent configured")
    assert _line(lines, "agents:reviewer") is None
    assert all(_line(lines, p) is None for p in ("cli:claude", "cli:codex", "cli:opencode", "skill:claude", "hook:codex"))
    assert code != 0


def test_doctor_opencode_only_flags_agents_reviewer(tmp_path):
    lines, code = _doctor(tmp_path, dict(claude=False, codex=False, opencode=True), present={"opencode"})
    assert "agents:any ok" in lines
    assert _line(lines, "agents:reviewer missing — jaxflow review will not run: it needs Claude Code or Codex, builds still work")
    assert "cli:opencode ok" in lines and _line(lines, "cli:claude") is None
    assert code != 0


def test_doctor_one_reviewer_agent_with_its_cli_passes(tmp_path):
    lines, code = _doctor(tmp_path, dict(claude=True, codex=False, opencode=False), present={"claude"})
    assert "agents:any ok" in lines and "agents:reviewer ok" in lines and "cli:claude ok" in lines
    assert _line(lines, "cli:codex") is None and _line(lines, "skill:opencode") is None
    assert code == 0


def test_doctor_cli_of_an_enabled_agent_is_required(tmp_path):
    lines, code = _doctor(tmp_path, dict(claude=True, codex=False, opencode=False))
    assert _line(lines, "cli:claude missing")
    assert code != 0


def test_doctor_cli_opencode_accepts_the_installer_fallback_under_the_injected_home(tmp_path):
    agents = dict(claude=False, codex=False, opencode=True)
    home = tmp_path / "home"
    lines, _ = _doctor(tmp_path, agents, home=home)
    assert _line(lines, "cli:opencode missing")
    fallback = home / ".opencode" / "bin" / "opencode"
    fallback.parent.mkdir(parents=True)
    fallback.write_text("#!/bin/sh\n")
    fallback.chmod(0o755)
    lines, _ = _doctor(tmp_path, agents, home=home)
    assert "cli:opencode ok" in lines


def test_doctor_unreadable_settings_checks_all_agents_non_required_and_emits_no_agents_check(tmp_path):
    lines, _ = _doctor(tmp_path, None, settings_ok=False)
    assert _line(lines, "settings:readable missing")
    assert all(_line(lines, f"cli:{a} missing") for a in ("claude", "codex", "opencode"))
    assert _line(lines, "agents:any") is None and _line(lines, "agents:reviewer") is None


# ---- hub rejection vs outage (MOA-507) --------------------------------------------------


def _json_opener(status, body):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()

    def opener(req, timeout):
        class Resp:
            def read(self_):
                return raw

            def __enter__(self_):
                return self_

            def __exit__(self_, *a):
                return False

        resp = Resp()
        resp.status = status
        return resp

    return opener


_EVENTS_URL = "http://127.0.0.1:9/events"


def test_post_event_200_ok_false_raises_hub_rejected_with_error_text():
    with pytest.raises(jaxflow_common.HubRejected) as exc:
        jaxflow_common._post_event(
            {"x": 1}, url=_EVENTS_URL,
            opener=_json_opener(200, {"ok": False, "error": "invalid event: verify too long"}),
        )
    assert exc.value.status == 200
    assert "invalid event: verify too long" in str(exc.value)


def test_post_event_http_error_with_json_error_raises_hub_rejected_with_status():
    for status in (400, 500):
        def opener(req, timeout, status=status):
            raise urllib.error.HTTPError(
                _EVENTS_URL, status, "err", None,
                io.BytesIO(json.dumps({"ok": False, "error": "nope"}).encode()),
            )

        with pytest.raises(jaxflow_common.HubRejected) as exc:
            jaxflow_common._post_event({"x": 1}, url=_EVENTS_URL, opener=opener)
        assert exc.value.status == status
        assert str(exc.value).startswith(f"{status} ")
        assert "nope" in str(exc.value)


def test_post_event_http_error_undecodable_body_raises_hub_rejected_status_only():
    def opener(req, timeout):
        raise urllib.error.HTTPError(_EVENTS_URL, 502, "bad", None, io.BytesIO(b"\xff"))

    with pytest.raises(jaxflow_common.HubRejected) as exc:
        jaxflow_common._post_event({"x": 1}, url=_EVENTS_URL, opener=opener)
    assert exc.value.status == 502
    assert str(exc.value) == "502"


def test_post_event_409_still_returns_claimed_marker():
    result = jaxflow_common._post_event(
        {"x": 1}, url=_EVENTS_URL,
        opener=_json_opener(409, {"ok": False, "error": "already finished"}),
    )
    assert result == {"ok": False, "claimed": True}


def test_post_event_transport_error_propagates_not_hub_rejected():
    def url_opener(req, timeout):
        raise urllib.error.URLError("timed out")

    with pytest.raises(urllib.error.URLError):
        jaxflow_common._post_event({"x": 1}, url=_EVENTS_URL, opener=url_opener)

    def os_opener(req, timeout):
        raise OSError("refused")

    with pytest.raises(OSError) as exc:
        jaxflow_common._post_event({"x": 1}, url=_EVENTS_URL, opener=os_opener)
    assert not isinstance(exc.value, getattr(jaxflow, "HubRejected", ()))


def test_hub_refusal_maps_hub_rejected_and_everything_else():
    refused = jaxflow_common._hub_refusal(jaxflow_common.HubRejected(200, "invalid event: x"))
    assert refused.code.startswith("hub-rejected: 200 invalid event")
    assert jaxflow_common._hub_refusal(RuntimeError("x")).code == "hub-unreachable"
    assert jaxflow_common._hub_refusal(OSError("x")).code == "hub-unreachable"


def _dispatch_tail_kwargs(raw, post, fake):
    repo = Path(raw) / "demo"
    repo.mkdir()
    run_id = "aaaabbbbcccc"
    manifest_dir = repo / ".local" / "runs" / run_id
    cleanup = repo / ".local" / "reports" / f"{run_id}.tests.txt"
    cleanup.parent.mkdir(parents=True)
    cleanup.write_text("x", encoding="utf-8")
    return dict(
        run=_run_with_tmux(fake), post=post, repo=repo, project="demo", phase="P",
        kind="build", role="builder", run_id=run_id, session="jax-demo-build-aaaabbbbcccc",
        manifest={"caller": "codex", "run_id": run_id, "no_callback": True},
        manifest_dir=manifest_dir, started_payload={"phase": "P"}, cleanup_paths=[cleanup],
    ), manifest_dir, cleanup


def test_dispatch_run_hub_rejected_refuses_cleans_manifest_dir_and_never_calls_tmux():
    with TemporaryDirectory() as raw:
        fake = FakeTmux()

        def post(event):
            raise jaxflow_common.HubRejected(200, "invalid event: verify too long")

        kw, manifest_dir, cleanup = _dispatch_tail_kwargs(raw, post, fake)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_common._dispatch_run(**kw)
        assert exc.value.code.startswith("hub-rejected: 200")
        assert not manifest_dir.exists()
        assert not cleanup.exists()
        assert fake.calls == []


def test_dispatch_run_oserror_is_hub_unreachable():
    with TemporaryDirectory() as raw:
        fake = FakeTmux()

        def post(event):
            raise OSError("connection refused")

        kw, manifest_dir, cleanup = _dispatch_tail_kwargs(raw, post, fake)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_common._dispatch_run(**kw)
        assert exc.value.code == "hub-unreachable"
        assert not manifest_dir.exists()
        assert not cleanup.exists()
        assert fake.calls == []


def _review_hub(monkeypatch, post):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        fake = FakeTmux()
        try:
            jaxflow.dispatch_review(
                _review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root), post=post,
                env={"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd"},
                now=_fixed_now, allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            reports = root / ".local" / "reports"
            runs = root / ".local" / "runs"
            return (
                exc,
                not runs.exists() or not any(runs.iterdir()),
                not any(reports.glob("*.tests.txt")),
                [c[1] for c in fake.calls] == ["has-session"],
            )
        raise AssertionError("refusal not raised")


def test_dispatch_review_hub_rejected_refuses_and_cleans(monkeypatch):
    def post(event):
        raise jaxflow_common.HubRejected(200, "invalid event: verify too long")

    exc, runs_gone, tests_gone, preflight_only = _review_hub(monkeypatch, post)
    assert exc.code.startswith("hub-rejected: 200")
    assert runs_gone and tests_gone and preflight_only


def test_dispatch_review_oserror_is_hub_unreachable(monkeypatch):
    exc, runs_gone, tests_gone, preflight_only = _review_hub(
        monkeypatch, lambda event: (_ for _ in ()).throw(OSError("down")),
    )
    assert exc.code == "hub-unreachable"
    assert runs_gone and tests_gone and preflight_only


def _event_post_rejected(event):
    return jaxflow_common._post_event(
        event, url=_EVENTS_URL,
        opener=_json_opener(200, {"ok": False, "error": "invalid event: verify too long"}),
    )


def test_pr_open_and_merge_audit_hub_rejected_has_no_resume_advice_unreachable_keeps_it(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    script = {
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("git", "push", "origin", f"{'a' * 40}:refs/heads/feat/x"): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, "[]"),
        ("gh", "pr", "create"): _completed(0, "https://github.com/acme/x/pull/7\n"),
    }

    def pr_open(post):
        fake_run, _calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
        with monkeypatch.context() as m:
            m.setattr(jr, "DB_PATH", db)
            jaxflow.cmd_pr_open(
                _pr_open_args(), run=fake_run, post=post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )

    with pytest.raises(ji.Refusal) as exc:
        pr_open(_event_post_rejected)
    assert exc.value.code.startswith("hub-rejected:")
    assert "re-run" not in capsys.readouterr().out
    assert getattr(exc.value, "hint", None) is None

    with pytest.raises(ji.Refusal) as exc:
        pr_open(lambda event: (_ for _ in ()).throw(OSError("down")))
    assert exc.value.code == "hub-unreachable"
    out = capsys.readouterr().out
    assert "re-run" in out
    assert exc.value.hint

    (tmp_path / "AGENTS.md").unlink(missing_ok=True)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(
            _MergeArgs(sha=sha), run=fake_run, post=_event_post_rejected,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code.startswith("hub-rejected:")
    assert "re-run" not in capsys.readouterr().out
    assert getattr(exc.value, "hint", None) is None
    assert any(" ".join(c).startswith("git commit") for c in calls)

    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(
            _MergeArgs(sha=sha), run=fake_run, post=lambda event: (_ for _ in ()).throw(OSError("down")),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "hub-unreachable"
    assert "re-run" in capsys.readouterr().out
    assert exc.value.hint
    assert any(" ".join(c).startswith("git commit") for c in calls)

    # A 5xx is a hub-side fault that may clear: keep the status, but still advise a retry.
    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow.cmd_merge(
            _MergeArgs(sha=sha), run=fake_run,
            post=lambda event: (_ for _ in ()).throw(jaxflow_common.HubRejected(500, "boom")),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "hub-rejected: 500 boom"
    assert "re-run" in capsys.readouterr().out
    assert exc.value.hint
    assert any(" ".join(c).startswith("git commit") for c in calls)


_CAP_HINT = "hint: move long commands into a package.json script and pass the script (e.g. pnpm run verify)"
_CALLER_ENV = {
    "CLAUDECODE": "1",
    "CLAUDE_CODE_SESSION_ID": "01234567-89ab-4cde-8f01-23456789abcd",
}


def _no_runs_or_tests(root):
    assert not (root / ".local" / "runs").exists()
    assert list(root.glob(".local/reports/*.tests.txt")) == []


def _probe_review(root, allow_root, target, env=None, **flags):
    events = []
    try:
        jaxflow.dispatch_review(
            _review_args(spec=str(target), **flags),
            run=_run_with_tmux(FakeTmux(), real_cwd=root),
            post=lambda e: events.append(e) or {"ok": True},
            env=dict(_CALLER_ENV if env is None else env),
            now=_fixed_now, allowlist_root=allow_root,
        )
    except ji.Refusal as exc:
        return events, exc
    return events, None


def test_build_verify_501_chars_refused_before_any_side_effect(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)

        def probe(**flags):
            return _probe_dispatch_build(monkeypatch, root, allow_root, plan, **flags)

        _run_id, events, git_calls, mkdir_calls, exc = probe(verify="v" * 501)
        assert exc is not None and exc.code == "verify-too-long (501 > 500)"
        assert exc.hint == _CAP_HINT
        _assert_no_reservation(allow_root, git_calls, mkdir_calls, events)
        _no_runs_or_tests(root)

        _run_id, events, _git, _mkdir, exc = probe(verify="v" * 500, branch="feat/verify-ok")
        assert exc is None and events

        _run_id, events, git_calls, mkdir_calls, exc = probe(build="b" * 501, branch="feat/build-long")
        assert exc is not None and exc.code == "build-too-long (501 > 500)"
        assert exc.hint == _CAP_HINT
        _assert_no_reservation(allow_root, git_calls, mkdir_calls, events)

        _run_id, events, _git, _mkdir, exc = probe(build="b" * 500, branch="feat/build-ok")
        assert exc is None and events

        env = dict(_CALLER_ENV)
        env["TMUX_PANE"] = "p" * 129
        fake = FakeTmux()
        git_calls, events, mkdir_calls = [], [], []
        real_mkdir = os.mkdir

        def wrapped_mkdir(path, *a, **k):
            mkdir_calls.append(path)
            return real_mkdir(path, *a, **k)

        monkeypatch.setattr(os, "mkdir", wrapped_mkdir)
        pane_exc = None
        try:
            jaxflow.dispatch_build(
                _build_args(plan=str(plan), branch="feat/pane"),
                run=_run_with_tmux_and_log(fake, git_calls, real_cwd=root),
                post=lambda e: events.append(e) or {"ok": True},
                env=env, now=_fixed_now, allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            pane_exc = exc
        assert pane_exc is not None and pane_exc.code == "caller-pane-too-long (129 > 128)"
        _assert_no_reservation(allow_root, git_calls, mkdir_calls, events)

        long_session = dict(_CALLER_ENV)
        long_session["CLAUDE_CODE_SESSION_ID"] = "s" * 129
        events, git_calls, mkdir_calls = [], [], []
        session_exc = None
        try:
            jaxflow.dispatch_build(
                _build_args(plan=str(plan), branch="feat/session"),
                run=_run_with_tmux_and_log(fake, git_calls, real_cwd=root),
                post=lambda e: events.append(e) or {"ok": True},
                env=long_session, now=_fixed_now, allowlist_root=allow_root,
            )
        except ji.Refusal as exc:
            session_exc = exc
        assert session_exc is not None and session_exc.code == "caller-session-too-long (129 > 128)"
        _assert_no_reservation(allow_root, git_calls, mkdir_calls, events)


def test_check_hub_caps_ignores_uncapped_keys():
    jaxflow_common._check_hub_caps({
        "phase": "P", "runtime": "opencode-grok", "kind": "build", "target": "feat/demo",
        "caller": "claude", "caller_session": "abc", "model": "m", "effort": "n/a",
        "session": "jax-demo-build-abc", "repo": "/tmp/demo", "verify": "true",
    })


def test_review_target_path_over_512_refused_early(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        cursor = root / ".local" / "docs" / "specs"
        while True:
            cursor = cursor / ("d" * 80)
            candidate = cursor / "spec.md"
            if jaxflow_common._utf16_len(str(candidate)) > 512:
                break
        candidate.parent.mkdir(parents=True)
        candidate.write_text("# Demo spec\n", encoding="utf-8")
        target = candidate.resolve()
        n = jaxflow_common._utf16_len(str(target))
        assert n > 512
        monkeypatch.chdir(root)
        events, exc = _probe_review(root, allow_root, target, phase="P")
        assert exc is not None and exc.code == f"target-too-long ({n} > 512)"
        assert events == []
        _no_runs_or_tests(root)


def test_build_backstop_session_too_long_cleans_worktree(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        plan = _plan_file(root)
        monkeypatch.chdir(root)
        monkeypatch.setattr(
            jr, "_alloc_run",
            lambda project, role, repo, run, run_id=None: ("abcdef123456", "s" * 81, {}),
        )
        events = []
        with pytest.raises(ji.Refusal) as exc:
            jaxflow.dispatch_build(
                _build_args(plan=str(plan)),
                run=_run_with_tmux(FakeTmux(), real_cwd=root),
                post=lambda e: events.append(e) or {"ok": True},
                env=dict(_CALLER_ENV), now=_fixed_now, allowlist_root=allow_root,
            )
        assert exc.value.code == "session-too-long (81 > 80)"
        assert events == []
        assert not (allow_root / "demo-feat-demo").exists()
        listed = subprocess.run(
            ["git", "branch", "--list", "feat/demo"], cwd=root,
            capture_output=True, text=True, check=False,
        )
        assert "feat/demo" not in listed.stdout


def test_check_hub_caps_counts_utf16_units():
    text = "\U0001F600" * 251
    assert jaxflow_common._utf16_len(text) == 502
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_common._check_hub_caps({"verify": text})
    assert exc.value.code == "verify-too-long (502 > 500)"


def test_review_phase_65_chars_refused_before_manifest_dir(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        events, exc = _probe_review(root, allow_root, target, phase="a" * 65)
        assert exc is not None and exc.code == "phase-too-long (65 > 64)"
        assert events == []
        _no_runs_or_tests(root)


def test_review_early_check_oversized_repo_or_model_leaves_nothing(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / ("r" * 180)
        root = allow_root / "demo"
        _init_repo(root)
        n = jaxflow_common._utf16_len(str(root.resolve()))
        assert n > 200
        target = _spec_file(root)
        monkeypatch.chdir(root)
        events, exc = _probe_review(root, allow_root, target, phase="P")
        assert exc is not None and exc.code == f"repo-too-long ({n} > 200)"
        assert events == []
        _no_runs_or_tests(root)


def test_review_backstop_oversized_caller_pane_leaves_no_manifest_dir_or_test_output(monkeypatch):
    with TemporaryDirectory() as raw:
        allow_root = Path(raw) / "repos"
        root = allow_root / "demo"
        _init_repo(root)
        target = _spec_file(root)
        monkeypatch.chdir(root)
        env = dict(_CALLER_ENV)
        env["TMUX_PANE"] = "p" * 129
        events, exc = _probe_review(root, allow_root, target, env=env, phase="P")
        assert exc is not None and exc.code == "caller-pane-too-long (129 > 128)"
        assert events == []
        _no_runs_or_tests(root)


def test_dispatch_run_backstop_repo_too_long_refuses_before_mkdir(tmp_path):
    repo = tmp_path / "demo"
    repo.mkdir()
    run_id = "abc123def456"
    manifest_dir = repo / ".local" / "runs" / run_id
    posted = []
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_common._dispatch_run(
            run=lambda *a, **k: None,
            post=lambda e: posted.append(e) or {"ok": True},
            repo=repo, project="demo", phase="P", kind="build", role="builder",
            run_id=run_id, session="jax-demo-build-" + run_id,
            manifest={"kind": "build"},
            manifest_dir=manifest_dir,
            started_payload={"repo": "r" * 201, "phase": "P", "session": "short"},
            cleanup_paths=[],
        )
    assert exc.value.code == "repo-too-long (201 > 200)"
    assert not manifest_dir.exists()
    assert posted == []


def test_check_hub_caps_skips_none_and_maps_caller_labels():
    jaxflow_common._check_hub_caps({"verify": None, "build": None, "callerPane": None})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_common._check_hub_caps({"callerSession": "s" * 129})
    assert exc.value.code == "caller-session-too-long (129 > 128)"
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_common._check_hub_caps({"callerPane": "p" * 129})
    assert exc.value.code == "caller-pane-too-long (129 > 128)"


def test_hub_caps_match_workflow_ts_limits():
    text = (Path(jaxflow.__file__).resolve().parent.parent / "src" / "lib" / "workflow.ts").read_text(
        encoding="utf-8",
    )
    start = text.index("export const LIMITS = {")
    block = text[start:text.index("}", start)]
    limits = {}
    for match in re.finditer(r"^\s+(\w+): (\d+)(?: \* (\d+))?", block, re.M):
        value = int(match.group(2))
        if match.group(3):
            value *= int(match.group(3))
        limits[match.group(1)] = value
    for key, cap in jaxflow_common.HUB_CAPS.items():
        assert key in limits
        assert cap == limits[key]


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
        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_diff_review(
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
        run_id = jaxflow.dispatch_review(
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
        run_id = jaxflow.dispatch_review(
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
            jaxflow.dispatch_review(
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
            lambda: jaxflow.dispatch_review(_review_args(spec=str(target)), run=_run_with_tmux(fake, real_cwd=root),
                                            post=lambda e: {"ok": True}, env=_CLAUDE_ENV, now=_fixed_now, allowlist_root=allow_root),
            lambda: jaxflow.dispatch_build(_build_args(plan=str(plan)), run=_run_with_tmux(fake, real_cwd=root),
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
        run_id = jaxflow.dispatch_diff_review(
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
        returned = jaxflow.cmd_result("r1", allowlist_root=allow_root, db_path=db)
        printed = capsys.readouterr().out
        assert printed == _FALLBACK_LINE + "tally: 1 findings — 0 repeated, 1 new\n\n"  # line BEFORE the tally (cmd_result prints `tally + "\n"` with print)
        assert returned == f"{report}\nBODY\n"
        assert (printed + returned).count("reviewer_runtime") == 1
        assert "reviewer_runtime" not in jaxflow.cmd_result("r2", allowlist_root=allow_root, db_path=db)
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
        returned = jaxflow.cmd_result("aaaaaaaaaaa1", allowlist_root=allow_root, db_path=db)
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
            jaxflow.cmd_result("r1", allowlist_root=allow_root, db_path=db)
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
            jaxflow.dispatch_build(
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
        run_id = jaxflow.dispatch_build(
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
            jaxflow.dispatch_build(
                _build_args(resume=seeded.prior), run=_run_with_tmux_and_log(fake, [], real_cwd=root),
                post=lambda e: posts.append(e), env=_CLAUDE_ENV, now=_fixed_now, allowlist_root=allow_root,
            )
        assert caught.value.code == "agent-disabled: opencode"
        assert posts == [] and fake.calls == []


def test_merge_help_documents_checks_reuse_and_recheck(capsys):
    with pytest.raises(SystemExit):
        jaxflow.parse_args(["merge", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "--recheck" in out
    assert "unless an up-to-date diff review already proved the same command on this exact SHA" in out
