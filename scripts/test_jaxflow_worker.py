"""jaxflow worker tests (P2 split of test_jaxflow.py)."""
import io
import json
import os
import re
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import time
from subprocess import CompletedProcess
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import pytest
import jaxflow_cli
import jaxflow_worker
import jaxflow_build
import jaxflow_review
import jaxflow_workerkit
import jaxflow_common
import jaxflow_run as jr
import jaxflow_settings as jset
import jax_init as ji

from testkit import (  # noqa: F401  (autouse fixtures included on purpose)
    FakeBuilderPopen,
    FakePopen,
    FakeTmux,
    _CAPTURED_THREAD,
    _TEST_CLAUDE_SESSION_ID,
    _build_args,
    _capture_builder_popen,
    _checkpoint_path,
    _diff_args,
    _fixed_now,
    _forbidden_tmux,
    _fresh_db,
    _git,
    _init_repo,
    _init_separate_git_dir_repo,
    _init_worktree,
    _insert,
    _integration_on_by_default,
    _isolate_agent_settings,
    _isolate_callbacks,
    _managed_worker_repo,
    _native_config,
    _plan_file,
    _restore_signal_handlers,
    _rewrite_json,
    _run_builder_worker_test,
    _run_diff_worker_and_capture_prompt,
    _run_real,
    _run_with_tmux,
    _seed_finished_build,
    _spec_file,
    _stub_launch_paths,
    _write_manifest_for_builder_worker,
    _write_manifest_for_diff_worker,
    _write_manifest_for_worker,
    _write_plan,
    _write_status_md,
)


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

            code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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
        assert jaxflow_cli.REVIEWER_DEFAULTS[caller]["runtime"] == runtime
        captured = {}

        class CapturePopen(FakePopen):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured["child"] = self

        monkeypatch.setattr(
            jaxflow_common, "_update_status_md",
            lambda state, **kwargs: captured.update(gate=state.get("spec_gate_required")),
        )
        assert jaxflow_worker.run_worker(
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

        assert jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
    jaxflow_worker.run_worker(
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
            jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
            str(manifest_path), run=_run_with_tmux(fake), post=lambda e: {"ok": True}, popen=FakePopen,
            allowlist_root=root,
        )
        assert not any(c[1] == "send-keys" for c in fake.calls)
        assert not jaxflow_common.CALLBACKS_ROOT.exists() or not any(jaxflow_common.CALLBACKS_ROOT.rglob("*.line"))


_WORKER_THREAD = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


_CALLBACK_SECRET = "sentinel-secret-not-for-logs"


_QUEUE_KW = dict(stdin=subprocess.DEVNULL, capture_output=True, timeout=10, check=False)


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
            return jaxflow_worker._refuse_builder_run(
                "boom", manifest=manifest, run=jr.run_command, post=post_fn,
                control_repo=root, worktree=worktree, branch="feat/x",
            )
        if refuse_kind == "unvalidated":
            return jaxflow_worker._refuse_unvalidated_builder_run(
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
        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
    jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        run_id = jaxflow_build.dispatch_build(
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
        assert jaxflow_cli.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == (
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
        assert jaxflow_cli.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == "finished(no result)"
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
        assert jaxflow_cli.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == (
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
        assert jaxflow_cli.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == (
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
        assert jaxflow_cli.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == (
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
        out = jaxflow_cli.cmd_result("r1", db_path=db)
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
            jaxflow_cli.cmd_result("r5", allowlist_root=allow_root, db_path=db)
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

        jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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

    monkeypatch.setattr(jaxflow_worker, "_persist_launch_selection", boom)

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

    monkeypatch.setattr(jaxflow_worker, "_persist_launch_selection", boom)
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

    monkeypatch.setattr(jaxflow_worker, "_persist_launch_selection", boom)
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
    monkeypatch.setattr(jaxflow_worker, "_builder_started_payload", lambda *a, **k: None)
    cleaned = _spy_cleanup(monkeypatch)

    def forbidden(*a, **k):
        raise AssertionError("git must not run")

    jaxflow_worker._refuse_builder_run(
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

    monkeypatch.setattr(jaxflow_worker, "_persist_launch_selection", boom)

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
        text = jaxflow_cli.cmd_status(run_id, run=_run_with_tmux(fake), db_path=db)
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
        out = jaxflow_cli.cmd_result(run_id, allowlist_root=allow_root, db_path=db)
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
        text = jaxflow_cli.cmd_status(run_id, run=_run_with_tmux(fake), db_path=db)
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
        text = jaxflow_cli.cmd_status(run_id, run=_run_with_tmux(fake), db_path=db)
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
            text = jaxflow_cli.cmd_status(run_id, run=_run_with_tmux(fake), db_path=db)
            assert "preview model: fixture/wire-real-model" in text
            assert "resolved selection" not in text
            assert "launched " not in text
        manifest_path.write_text(json.dumps({
            "nested": {"deep": {"x": 1}},
            "launch_selection": "not-an-object",
        }), encoding="utf-8")
        text = jaxflow_cli.cmd_status(run_id, run=_run_with_tmux(fake), db_path=db)
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
        text = jaxflow_cli.cmd_status(run_id, run=_run_with_tmux(fake), db_path=db)
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
        jaxflow_worker._persist_launch_selection(
            copied, dict(manifest), _LAUNCH_SELECTION, control_repo=repo,
        )
    assert exc.value.code in ("path-outside-allowlist", "agent-settings-permissions")
    assert "launch_selection" not in json.loads(copied.read_text(encoding="utf-8"))
    assert "launch_selection" not in json.loads(canonical.read_text(encoding="utf-8"))
    linked_parent = tmp_path / "via-runs"
    linked_parent.symlink_to(canonical.parent.parent)
    linked = linked_parent / run_id / "manifest.json"
    with pytest.raises(ji.Refusal):
        jaxflow_worker._persist_launch_selection(
            linked, dict(manifest), _LAUNCH_SELECTION, control_repo=repo,
        )
    assert "launch_selection" not in json.loads(canonical.read_text(encoding="utf-8"))
    jaxflow_worker._persist_launch_selection(
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
        jaxflow_worker._persist_launch_selection(
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
    under = jaxflow_worker._pure_config_run(
        [exe, "-c", "import sys; sys.stdout.buffer.write(b'{}')"], cap=64,
    )
    assert under.returncode == 0
    assert under.stdout == b"{}"
    exact = jaxflow_worker._pure_config_run(
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
        jaxflow_worker._pure_config_run([exe, "-c", over_script], cap=64)
    assert exc.value.code == "agent-profile-conflict"
    pid = int(pid_file.read_text(encoding="utf-8"))
    with pytest.raises(OSError):
        os.kill(pid, 0)

    err = jaxflow_worker._pure_config_run(
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
        jaxflow_worker._pure_config_run([exe, "-c", stall_script], deadline=0.2, cap=64)
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
    text = jaxflow_cli.cmd_status(
        manifest["run_id"], run=_run_with_tmux(fake), db_path=jr.DB_PATH,
    )
    assert "resolved selection model: fixture/wire-real-model" in text
    assert "launched model:" not in text


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
        text = jaxflow_cli.cmd_status(run_id, run=_run_with_tmux(FakeTmux()), db_path=db)
        assert "checkpoint: present" in text
        out = jaxflow_cli.cmd_result(run_id, allowlist_root=allow_root, db_path=db)
        assert "checkpoint: present" in out
        dest.unlink()
        text = jaxflow_cli.cmd_status(run_id, run=_run_with_tmux(FakeTmux()), db_path=db)
        assert "checkpoint: absent" in text
        con.close()


# ---- worker: builder handoff is contract-valid (fixes cold review F2) ----

def test_builder_handoff_carries_both_commands_and_never_hardcodes_none():
    """MOA-454, the defect this issue is named for: `commands.build` was the literal
    `none` no matter what, so a tech lead who chained a build into --verify got a handoff
    that told the builder it had no build command."""
    handoff = jaxflow_worker._build_builder_handoff(
        plan_dest=Path("/p/plan.md"),
        spec_dest=Path("/p/spec.md"), agents_path=Path("/p/AGENTS.md"),
        branch="feat/x", head_sha="a" * 40, whitelist=["src"],
        verify_cmd="pnpm test", build_cmd="pnpm build",
    )
    assert "  test: pnpm test\n" in handoff
    assert "  build: pnpm build\n" in handoff
    assert "build: none" not in handoff


def test_builder_handoff_says_none_only_when_there_is_no_build_command():
    handoff = jaxflow_worker._build_builder_handoff(
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

        jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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

        assert jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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
        run_id = jaxflow_build.dispatch_build(
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

        assert jaxflow_worker.run_worker(
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
        diff_run_id = jaxflow_review.dispatch_diff_review(
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

        assert jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=FakeBuilderPopen, allowlist_root=root.parent,
        )
        text = status_path.read_text(encoding="utf-8")
        assert "stage: build" in text
        assert "branch: feat/x" in text
        assert "## Now\nbuilt the thing" in text


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
        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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
            code = jaxflow_worker.run_worker(
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

            code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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
            jaxflow_worker.run_worker(
                str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
                post=lambda e: events.append(e) or {"ok": True}, popen=FakePopen,
                allowlist_root=root.parent,
            )
        except RuntimeError as exc:
            assert "git diff" in str(exc)
        else:
            raise AssertionError("git diff failure did not raise")
        assert events == []


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

        code = jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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

        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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

            jaxflow_worker.run_worker(
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

            jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
                code = jaxflow_worker.run_worker(
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
    code = jaxflow_worker.run_worker(
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
    code = jaxflow_worker.run_worker(
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
    code = jaxflow_worker.run_worker(
        str(manifest_path), run=_run_with_tmux(FakeTmux()),
        post=lambda e: events.append(e) or {"ok": True}, popen=_MissingCliPopen,
        allowlist_root=repo.parent,
    )
    assert code == jaxflow_common.REFUSED
    finished = [e for e in events if e["type"] == "run-finished"][0]
    assert "missing-cli" in finished["payload"]["summary"]
