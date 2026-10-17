"""jaxflow common tests (P2 split of test_jaxflow.py)."""
import io
import json
import os
import re
import subprocess
import urllib.error
from datetime import datetime, timedelta, timezone
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
import jaxflow_workerkit
import jaxflow_common
import jaxflow_run as jr
import jax_init as ji

from testkit import (  # noqa: F401  (autouse fixtures included on purpose)
    FakePopen,
    FakeTmux,
    _DUAL_PR_AGENTS,
    _MergeArgs,
    _STATUS_TEMPLATE,
    _assert_no_reservation,
    _build_args,
    _completed,
    _fixed_now,
    _fresh_db,
    _git,
    _init_repo,
    _init_separate_git_dir_repo,
    _init_worktree,
    _integration_on_by_default,
    _isolate_agent_settings,
    _isolate_callbacks,
    _merge_env,
    _merge_runner,
    _plan_file,
    _pr_open_args,
    _probe_dispatch_build,
    _restore_signal_handlers,
    _review_args,
    _run_real,
    _run_with_tmux,
    _run_with_tmux_and_log,
    _spec_file,
    _switch_aware_runner,
    _write_manifest_for_worker,
    _write_status_md,
)


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

        code = jaxflow_cli.main(
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

        jaxflow_review.dispatch_review(
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
        code = jaxflow_cli.main(
            ["review", "--spec", str(root / "x.md")], run=_run_real, post=lambda e: {"ok": True},
            env={"CLAUDECODE": "1"},
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "not-a-git-toplevel"


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
        code = jaxflow_cli.main(
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
            code = jaxflow_cli.main(
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
        code = jaxflow_cli.main(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
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

        code = jaxflow_worker.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux()), post=post, popen=FakePopen,
            allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "path-outside-allowlist"
        assert events == []
        assert not (root / ".local" / "reports").exists()


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
        code = jaxflow_cli.main(
            ["review", "--spec", str(target)], run=_run_with_tmux(fake, real_cwd=root),
            post=lambda e: {"ok": True}, env={"CLAUDECODE": "1"}, allowlist_root=allow_root,
        )
        assert code == jaxflow_common.REFUSED
        assert capsys.readouterr().err.strip() == "detached-head"


# Every "a write should happen" test below dispatches its manifest AFTER the template's own
# `updated` value (2020) -- fixes cold review F5's own finding that the first draft's
# fixtures had dispatch_start (2000) BEFORE updated (2020), which made the (correct) stale
# guard skip every write the tests then asserted had happened.
_AFTER_TEMPLATE_UPDATED = "2026-01-01T00:00:00-03:00"


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
        roots = jaxflow_worker._builder_read_roots(
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
        code = jaxflow_worker.run_worker(
            str(manifest_path), run=_run_with_tmux(FakeTmux(), real_cwd=worktree),
            post=lambda e: {"ok": True}, popen=FakePopen, allowlist_root=allow_root,
        )
        assert code == 0
        control_text = control_status.read_text(encoding="utf-8")
        assert "stage: spec" in control_text
        assert "builder: claude" in control_text
        assert sentinel.read_text(encoding="utf-8") == before_sentinel


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
            jaxflow_review.dispatch_review(
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
            jaxflow_merge.cmd_pr_open(
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
        jaxflow_merge.cmd_merge(
            _MergeArgs(sha=sha), run=fake_run, post=_event_post_rejected,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code.startswith("hub-rejected:")
    assert "re-run" not in capsys.readouterr().out
    assert getattr(exc.value, "hint", None) is None
    assert any(" ".join(c).startswith("git commit") for c in calls)

    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(
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
        jaxflow_merge.cmd_merge(
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
        jaxflow_review.dispatch_review(
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
            jaxflow_build.dispatch_build(
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
            jaxflow_build.dispatch_build(
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
            jaxflow_build.dispatch_build(
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
