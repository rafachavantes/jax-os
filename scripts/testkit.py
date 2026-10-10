"""Shared fakes, helpers and autouse fixtures for the jaxflow tests (P2 split)."""
import io
import json
import os
import signal
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import pytest
import general_settings
import jaxflow_worker
import jaxflow_build
import jaxflow_common
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
    code = jaxflow_worker.run_worker(
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


def _review_args(**kw):
    return SimpleNamespace(
        spec=kw.get("spec"), plan=kw.get("plan"), focus=kw.get("focus"),
        model=kw.get("model"), effort=kw.get("effort"), phase=kw.get("phase"),
        from_caller=kw.get("from_caller"), no_callback=kw.get("no_callback", False),
    )


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


_CAPTURED_THREAD = "11111111-2222-4333-8444-555555555555"


def _forbidden_tmux(argv, cwd=None):
    pytest.fail(f"Codex callback attempted tmux: {argv}")


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
        run_id = jaxflow_build.dispatch_build(
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


def _write_status_md(root, text=_STATUS_TEMPLATE):
    path = root / ".jax-os" / "status.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


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


def _checkpoint_path(root, run_id="bbbbccccdddd"):
    return root / ".local" / "runs" / run_id / "resume-checkpoint.json"


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


def _run_diff_worker_and_capture_prompt(manifest_path, worktree, root):
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
    return captured["stdin_bytes"].decode("utf-8")


# ---------------------------------------------------------------- slice c: preset parser

def _agents(tmp_path, text):
    """Writes an AGENTS.md into tmp_path and returns the repo path."""
    (tmp_path / "AGENTS.md").write_text(text, encoding="utf-8")
    return tmp_path


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
