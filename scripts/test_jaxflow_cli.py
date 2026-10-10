"""jaxflow cli tests (P2 split of test_jaxflow.py)."""
import contextlib
import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import pytest
import general_settings
import jaxflow_cli
import jaxflow_worker
import jaxflow_build
import jaxflow_review
import jaxflow_merge
import jaxflow_workerkit
import jaxflow_common
import jaxflow_run as jr
import jaxflow_settings as jset
import jax_init as ji

from testkit import (  # noqa: F401  (autouse fixtures included on purpose)
    FakePopen,
    FakeTmux,
    _ALL_AGENTS_ON,
    _MergeArgs,
    _build_args,
    _completed,
    _fixed_now,
    _fresh_db,
    _git,
    _init_repo,
    _init_worktree,
    _insert,
    _integration_on_by_default,
    _isolate_agent_settings,
    _isolate_callbacks,
    _merge_env,
    _never_post,
    _plan_file,
    _restore_signal_handlers,
    _review_args,
    _run_real,
    _run_with_tmux,
    _spec_file,
    _switch_aware_runner,
    _write_manifest_for_diff_worker,
    _write_manifest_for_worker,
)


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

        code = jaxflow_cli.main(
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

        code = jaxflow_cli.main(
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
        code = jaxflow_worker.run_worker(
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
    code = jaxflow_worker.run_worker(
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
        assert jaxflow_cli.cmd_status("nope", run=_run_with_tmux(fake), db_path=db) == "unknown"

        _insert(con, "r1", "demo", "reviewer", "run-started", {"session": "jax-demo-spec-r1"})
        fake.existing.add("jax-demo-spec-r1")
        assert jaxflow_cli.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == "running"

        fake.existing.discard("jax-demo-spec-r1")
        assert jaxflow_cli.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == "dead"

        _insert(con, "r1", "demo", "reviewer", "run-finished", {"contract_status": "ok"})
        # MOA-474 §12.2: a reviewer row with no verdict now renders the literal
        # "no verdict" instead of the old bare-ok fold.
        assert jaxflow_cli.cmd_status("r1", run=_run_with_tmux(fake), db_path=db) == "finished(no verdict)"

        _insert(con, "r2", "demo", "reviewer", "run-started", {"session": "jax-demo-spec-r2"})
        _insert(con, "r2", "demo", "reviewer", "run-finished", {"contract_status": "invalid"})
        assert jaxflow_cli.cmd_status("r2", run=_run_with_tmux(fake), db_path=db) == "finished(no verdict, report invalid)"

        _insert(con, "r3", "demo", "reviewer", "run-started", {"session": "jax-demo-spec-r3"})
        _insert(con, "r3", "demo", "reviewer", "run-finished", {"contract_status": "cancelled"})
        assert jaxflow_cli.cmd_status("r3", run=_run_with_tmux(fake), db_path=db) == "finished(cancelled)"
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
            jaxflow_cli.cmd_result("nope", allowlist_root=allow_root, db_path=db)
        except ji.Refusal as exc:
            assert exc.code == "unknown-run"
        else:
            raise AssertionError("unknown-run not raised")

        _insert(con, "r1", "demo", "reviewer", "run-started", {"repo": str(root)})
        _insert(con, "r1", "demo", "reviewer", "run-finished", {
            "contract_status": "ok", "report_path": str(report), "summary": "ok",
        })
        out = jaxflow_cli.cmd_result("r1", allowlist_root=allow_root, db_path=db)
        assert out == f"{report}\nREPORT BODY\n"

        _insert(con, "r2", "demo", "reviewer", "run-started", {"repo": str(root)})
        _insert(con, "r2", "demo", "reviewer", "run-finished", {
            "contract_status": "ok", "report_path": "/tmp/not-the-real-path.md", "summary": "ok",
        })
        try:
            jaxflow_cli.cmd_result("r2", allowlist_root=allow_root, db_path=db)
        except ji.Refusal as exc:
            assert exc.code == "report-path-mismatch"
        else:
            raise AssertionError("report-path-mismatch not raised")

        _insert(con, "r3", "demo", "reviewer", "run-started", {"repo": str(root)})
        _insert(con, "r3", "demo", "reviewer", "run-finished", {
            "contract_status": "cancelled", "report_path": None, "summary": "cancelled by claude at t",
        })
        assert jaxflow_cli.cmd_result("r3", allowlist_root=allow_root, db_path=db) == "cancelled — cancelled by claude at t"
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
            jaxflow_cli.cmd_result("r4", allowlist_root=allow_root, db_path=db)
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
            jaxflow_cli.cmd_result("r5", allowlist_root=allow_root, db_path=db)
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
            out = jaxflow_cli.cmd_result("b1", allowlist_root=allow_root, db_path=db)
            assert out == f"{expected.resolve()}\nBUILDER BODY\n"
            assert not (control / ".local" / "runs").exists()
        else:
            try:
                jaxflow_cli.cmd_result("b1", allowlist_root=allow_root, db_path=db)
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
            jaxflow_cli.cmd_cancel(
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
            jaxflow_cli.cmd_cancel(
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
        out = jaxflow_cli.cmd_cancel(
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
            jaxflow_cli.cmd_cancel("r1", run=_run_with_tmux(fake), post=lambda url, body: (409, {"ok": False}), now=now, db_path=db)
        except ji.Refusal as exc:
            assert exc.code == "already-finished"
        else:
            raise AssertionError("already-finished not raised on 409")

        try:
            jaxflow_cli.cmd_cancel("r1", run=_run_with_tmux(fake), post=lambda url, body: (500, {"ok": False}), now=now, db_path=db)
        except ji.Refusal as exc:
            assert exc.code == "hub-rejected: 500"
        else:
            raise AssertionError("hub-rejected not raised on 500")

        def raising_post(url, body):
            raise OSError("connection refused")

        try:
            jaxflow_cli.cmd_cancel("r1", run=_run_with_tmux(fake), post=raising_post, now=now, db_path=db)
        except ji.Refusal as exc:
            assert exc.code == "hub-unreachable"
        else:
            raise AssertionError("hub-unreachable not raised on transport error")

        # fixes cold review F5: a 200 that decodes to {ok: false} is the route's own
        # deliberate non-claim-failure shape (§ route.ts), not a success -- must not be
        # read as "cancelled".
        try:
            jaxflow_cli.cmd_cancel(
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
            jaxflow_cli.cmd_cancel(
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
        out = jaxflow_cli.cmd_cancel(
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
        out = jaxflow_cli.cmd_cancel(
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

        out = jaxflow_cli.cmd_cancel("aaaabbbbcccc", run=_run_with_tmux(fake), post=post, now=now, db_path=db)
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
            jaxflow_cli.cmd_cancel("aaaabbbbcccc", run=_run_with_tmux(FakeTmux()), post=post, now=now, db_path=db)
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

        out = jaxflow_cli.cmd_cancel("aaaabbbbcccc", run=_run_with_tmux(fake), post=post, now=now, db_path=db)
        assert out == "cancelled"
        assert events[0]["payload"]["contract_status"] == "cancelled"


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
        assert jaxflow_cli.cmd_status("r9", run=_run_with_tmux(fake), db_path=db) == "finished(failed: crash, exit 1)"
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
        assert jaxflow_cli.cmd_status("r10", run=_run_with_tmux(fake), db_path=db) == "finished(no verdict)"
        _insert(con, "r11", "demo", "reviewer", "run-started", {"session": "jax-demo-spec-r11"})
        _insert(con, "r11", "demo", "reviewer", "run-finished", {"contract_status": "cancelled"})
        assert jaxflow_cli.cmd_status("r11", run=_run_with_tmux(fake), db_path=db) == "finished(cancelled)"
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
            jaxflow_cli.cmd_result("r5", allowlist_root=allow_root, db_path=db)
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
        returned = jaxflow_cli.cmd_result("r7", allowlist_root=allow_root, db_path=db)
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
        out = jaxflow_cli.cmd_result("r6", allowlist_root=allow_root, db_path=db)
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
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert child_log.read_text(encoding="utf-8") == "child output\n"
    assert not any(p.name == "child.log" for p in (tmp_path / ".local" / "reports").iterdir())


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
        jaxflow_worker.run_worker(
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
        jaxflow_worker.run_worker(
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
        code = jaxflow_worker.run_worker(
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
    entries = jaxflow_cli._parse_worktree_porcelain(output)
    assert entries == [
        {"path": "/home/rafa/repos/jax-os", "branch": "main", "detached": False, "locked": False},
        {"path": "/home/rafa/repos/jax-os-locked", "branch": "feat/locked-thing", "detached": False, "locked": True},
        {"path": "/home/rafa/repos/jax-os-detached", "branch": None, "detached": True, "locked": False},
        {"path": "/r/p", "branch": "x", "detached": False, "locked": True},
    ]
    assert jaxflow_cli._parse_worktree_porcelain("") == []


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
    assert jaxflow_cli._classify_gc_state(finished) == expected


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
    result = jaxflow_cli._gc_decide(
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
    assert jaxflow_cli._latest_builder_run_for_gc(con, "demo", repo, "feat/x", worktree) == ("aaaabbbbcccc", {"contract_status": "ok", "result": "success"}, "2026-09-01T00:00:00+00:00")
    assert jaxflow_cli._latest_builder_run_for_gc(con, "demo", repo, "feat/nothing", worktree) == (None, None, None)
    # F2: same branch, but the manifest reserved a DIFFERENT path -- not ownership.
    assert jaxflow_cli._latest_builder_run_for_gc(con, "demo", repo, "feat/x", tmp_path / "elsewhere") == (None, None, None)
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
    ("invalid-shape", True),   # FIFO: not a regular, single-linked file (`_copy_run_reports`, `scripts/jaxflow_common.py`)
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
    jaxflow_cli.cmd_gc(args, run=_run_real, post=post, now=now, db_path=db, allowlist_root=allow_root)
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
    jaxflow_cli.cmd_gc(args_yes, run=_run_real, post=post, now=now, db_path=db, allowlist_root=allow_root)
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
    jaxflow_cli.cmd_gc(args, run=recording_run, post=_never_post, now=now, db_path=db, allowlist_root=allow_root)
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
    jaxflow_cli.cmd_gc(args_yes, run=_probe_override_run(worktree, **kwargs),
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
    jaxflow_cli.cmd_gc(args_yes, run=locking_run, post=lambda e: posted.append(e) or {"ok": True},
                   now=now, db_path=db, allowlist_root=allow_root)
    assert worktree.is_dir(), "a locked entry is never removed, --force included"
    assert posted == []


def test_cmd_gc_refuses_both_modes_and_bad_duration(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    now = lambda: datetime(2026, 9, 19, tzinfo=timezone.utc)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_cli.cmd_gc(SimpleNamespace(older_than="7d", dry_run=True, yes=True, force=False), run=_run_real, post=_never_post, now=now, db_path=tmp_path / "x.db")
    assert exc.value.code == "gc-both-modes"
    with pytest.raises(ji.Refusal) as exc2:
        jaxflow_cli.cmd_gc(SimpleNamespace(older_than="7hours", dry_run=True, yes=False, force=False), run=_run_real, post=_never_post, now=now, db_path=tmp_path / "x.db")
    assert exc2.value.code == "gc-duration-invalid"


def test_redeliver_gc_spools_retries_only_gc_removed_and_deletes_on_success(tmp_path):
    repo = tmp_path / "demo"
    repo.mkdir()
    gc_event = {"project": "demo", "role": "lead", "type": "gc-removed", "source": "deterministic", "emitter": "wrapper", "payload": {"run_id": "aaaabbbbcccc", "branch": "feat/x", "worktree": "/x", "reason": "success", "age_days": 9}}
    other_event = {"run_id": "ffffeeeedddd", "project": "demo", "role": "builder", "type": "run-finished", "source": "deterministic", "emitter": "wrapper", "payload": {"contract_status": "ok"}}
    jaxflow_common._write_spool(repo, "aaaabbbbcccc", gc_event)
    jaxflow_common._write_spool(repo, "ffffeeeedddd", other_event)  # NOT a gc-removed spool -- must survive
    posted = []
    jaxflow_cli._redeliver_gc_spools(repo, post=lambda e: posted.append(e) or {"ok": True})
    assert posted == [gc_event]
    assert not jaxflow_common._spool_path(repo, "aaaabbbbcccc").exists()
    assert jaxflow_common._spool_path(repo, "ffffeeeedddd").exists()  # untouched
    jaxflow_cli._redeliver_gc_spools(tmp_path / "no-such-repo", post=_never_post)  # must not raise


def test_redeliver_gc_spools_leaves_spool_on_post_failure_and_retries_next_call(tmp_path):
    repo = tmp_path / "demo"
    repo.mkdir()
    gc_event = {"project": "demo", "role": "lead", "type": "gc-removed", "source": "deterministic", "emitter": "wrapper", "payload": {"run_id": "aaaabbbbcccc", "branch": "feat/x", "worktree": "/x", "reason": "success", "age_days": 9}}
    jaxflow_common._write_spool(repo, "aaaabbbbcccc", gc_event)

    def failing_post(event):
        raise RuntimeError("event post failed: 500")

    jaxflow_cli._redeliver_gc_spools(repo, post=failing_post)
    assert jaxflow_common._spool_path(repo, "aaaabbbbcccc").exists()

    # F6: at-least-once -- the spool survives and the NEXT call redelivers; no
    # idempotency key, a duplicate is a harmless repeat (ledger row is the source of truth).
    posted = []
    jaxflow_cli._redeliver_gc_spools(repo, post=lambda e: posted.append(e) or {"ok": True})
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
        jaxflow_review.dispatch_diff_review(
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
    assert jaxflow_cli._normalize_phase(raw) == expected


def test_loop_refuses_short_prefix(tmp_path):
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_cli.cmd_loop("abc", db_path=tmp_path / "x.db")
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
    anchors = jaxflow_cli._anchor_builds_for_prefix(ro, "moa474")
    assert [a["run_id"] for a in anchors] == ["b1"]
    diffs = jaxflow_cli._diff_rows_for_builds(ro, anchors)
    assert [d["run_id"] for d in diffs] == ["d1", "d2"]
    rounds, approved = jaxflow_cli._rounds_to_approve(ro, diffs)
    assert (rounds, approved) == (2, "d2")
    assert jaxflow_cli._wall_time_days(ro, anchors, [], []) is None  # not merged yet
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
    anchors = jaxflow_cli._anchor_builds_for_prefix(ro, "moa600")
    diffs = jaxflow_cli._diff_rows_for_builds(ro, anchors)
    assert [d["run_id"] for d in diffs] == ["d1", "d2"]
    rounds, approved = jaxflow_cli._rounds_to_approve(ro, diffs)
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
    anchors = jaxflow_cli._anchor_builds_for_prefix(ro, "moa467")
    diffs = jaxflow_cli._diff_rows_for_builds(ro, anchors)
    assert [d["run_id"] for d in diffs] == ["d1"]
    ro.close()


def test_cmd_loop_produces_nonzero_counts(tmp_path):
    db = _loop_db(tmp_path)
    output = jaxflow_cli.cmd_loop("moa-474", db_path=db)
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
    output = jaxflow_cli.cmd_loop("moa510", db_path=db)
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
    anchors = jaxflow_cli._anchor_builds_for_prefix(ro, "x")
    days = jaxflow_cli._wall_time_days(ro, anchors, [], [])  # no spec/plan matched -- falls back to the build itself
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
    reused = jaxflow_cli._anchor_builds_for_prefix(ro, "moa500")
    assert jaxflow_cli._wall_time_days(ro, reused, [], []) == 5.0  # never the 2026-01 merge
    predates = jaxflow_cli._anchor_builds_for_prefix(ro, "moa501")
    assert jaxflow_cli._wall_time_days(ro, predates, [], []) is None  # m2 predates b2's dispatch
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
    anchors = jaxflow_cli._anchor_builds_for_prefix(ro, "moa610")
    build_phases_norm = [jaxflow_cli._normalize_phase(b["payload"]["phase"]) for b in anchors]
    plan_rows = jaxflow_cli._spec_or_plan_rows_for_build_phases(ro, "plan", build_phases_norm)
    assert jaxflow_cli._wall_time_days(ro, anchors, [], plan_rows) is None
    ro.close()


# ---- Phase 2: --from jaxos (spec §8, Decisions 6/23) ----

def test_resolve_caller_accepts_jaxos_from_flag_without_env_markers():
    assert jaxflow_common.resolve_caller({}, "jaxos") == "jaxos"
    assert jaxflow_common._CALLER_SESSION_VAR["jaxos"] == "JAXOS_CALLER_SESSION"


def test_main_accepts_from_jaxos_on_review_build_and_merge(monkeypatch):
    seen = []
    monkeypatch.setattr(jaxflow_review, "dispatch_review", lambda args, **kw: seen.append(("review", args.from_caller)) or "r1")
    monkeypatch.setattr(jaxflow_build, "dispatch_build", lambda args, **kw: seen.append(("build", args.from_caller)) or "b1")
    monkeypatch.setattr(jaxflow_merge, "cmd_merge", lambda args, **kw: seen.append(("merge", args.from_caller)) or jaxflow_common.OK)
    assert jaxflow_cli.main(["review", "--spec", "x.md", "--from", "jaxos"]) == jaxflow_common.OK
    assert jaxflow_cli.main([
        "build", "--plan", "p.md", "--phase", "P", "--branch", "feat/x", "--whitelist", "a",
        "--verify", "true", "--from", "jaxos",
    ]) == jaxflow_common.OK
    assert jaxflow_cli.main([
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
            jaxflow_review.dispatch_review(
                _review_args(spec=str(target), from_caller="jaxos"),
                run=_run_with_tmux(fake, real_cwd=root), post=post,
                env={}, now=_fixed_now, allowlist_root=allow_root,
            )
        assert exc.value.code == "caller-session-missing"
        assert exc.value.hint == "hint: set JAXOS_CALLER_SESSION"
        assert events == []

        run_id = jaxflow_review.dispatch_review(
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

        run_id = jaxflow_build.dispatch_build(
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
    rc = jaxflow_merge.cmd_merge(_MergeArgs(sha=sha, from_caller="jaxos"), run=fake_run, post=lambda e: None,
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
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
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
    jaxflow_cli.cmd_mission_start(args, post=post)
    assert calls == [(jaxflow_cli.MISSION_BASE_URL, {"name": "Ship it", "goal": "6 phases tonight", "milestones": ["Phase 1", "Phase 2"]})]


def test_mission_start_refuses_with_no_milestones_before_any_post():
    def post(url, payload):
        raise AssertionError("must not post with zero milestones")
    args = SimpleNamespace(name="Ship it", goal="g", milestones=[])
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_cli.cmd_mission_start(args, post=post)
    assert exc.value.code == "malformed milestone"


def test_mission_start_reraises_the_route_error_verbatim():
    post, _ = _recording_post(200, {"ok": False, "error": "mission-active"})
    args = SimpleNamespace(name="Ship it", goal="g", milestones=["m1"])
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_cli.cmd_mission_start(args, post=post)
    assert exc.value.code == "mission-active"


def test_mission_start_reraises_every_field_shape_refusal_verbatim():
    for code in ("too-many-milestones", "malformed name", "malformed goal", "malformed milestone"):
        post, _ = _recording_post(200, {"ok": False, "error": code})
        args = SimpleNamespace(name="Ship it", goal="g", milestones=["m1"])
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_cli.cmd_mission_start(args, post=post)
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
            jaxflow_cli.cmd_cancel(
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
            jaxflow_cli.cmd_cancel(
                "aaaabbbbcccc", run=_run_with_tmux(FakeTmux()), post=post, now=lambda: None, db_path=db,
            )
        assert exc.value.code == "hub-unreachable"
        assert jaxflow_common._spool_path(repo, "aaaabbbbcccc").exists()


def test_mission_post_500_empty_body_is_mission_hub_rejected_status():
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_cli._mission_post("http://127.0.0.1:9/mission", {"name": "n"}, post=lambda url, payload: (500, {}))
    assert exc.value.code == "mission-hub-rejected: 500"
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_cli._mission_post(
            "http://127.0.0.1:9/mission", {"name": "n"},
            post=lambda url, payload: (400, {"ok": False, "error": "too-many-milestones"}),
        )
    assert exc.value.code == "too-many-milestones"

    def down(url, payload):
        raise OSError("down")

    with pytest.raises(ji.Refusal) as exc:
        jaxflow_cli._mission_post("http://127.0.0.1:9/mission", {"name": "n"}, post=down)
    assert exc.value.code == "mission-hub-unreachable"


def test_mission_start_transport_failure_is_mission_hub_unreachable():
    def post(url, payload):
        raise OSError("connection refused")
    args = SimpleNamespace(name="Ship it", goal="g", milestones=["m1"])
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_cli.cmd_mission_start(args, post=post)
    assert exc.value.code == "mission-hub-unreachable"


def test_mission_status_posts_status_line_and_reraises_no_active_mission():
    post, calls = _recording_post(200, {"ok": True, "data": {}})
    jaxflow_cli.cmd_mission_status(SimpleNamespace(text="phase 1 merged"), post=post)
    assert calls == [(f"{jaxflow_cli.MISSION_BASE_URL}/status", {"status_line": "phase 1 merged"})]
    post2, _ = _recording_post(200, {"ok": False, "error": "no-active-mission"})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_cli.cmd_mission_status(SimpleNamespace(text="x"), post=post2)
    assert exc.value.code == "no-active-mission"


def test_mission_status_reraises_malformed_status_verbatim():
    post, _ = _recording_post(200, {"ok": False, "error": "malformed status"})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_cli.cmd_mission_status(SimpleNamespace(text=""), post=post)
    assert exc.value.code == "malformed status"


def test_mission_mark_resolves_index_and_title_forms_and_reraises_unknown_milestone():
    post, calls = _recording_post(200, {"ok": True, "data": {}})
    jaxflow_cli.cmd_mission_mark(SimpleNamespace(milestone="2", state="done"), post=post)
    jaxflow_cli.cmd_mission_mark(SimpleNamespace(milestone="Phase 1", state="in-progress"), post=post)
    assert calls == [
        (f"{jaxflow_cli.MISSION_BASE_URL}/milestone", {"milestone": "2", "state": "done"}),
        (f"{jaxflow_cli.MISSION_BASE_URL}/milestone", {"milestone": "Phase 1", "state": "in-progress"}),
    ]
    post2, _ = _recording_post(200, {"ok": False, "error": "unknown-milestone"})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_cli.cmd_mission_mark(SimpleNamespace(milestone="nope", state="done"), post=post2)
    assert exc.value.code == "unknown-milestone"


def test_mission_mark_refuses_malformed_state_before_any_post():
    def post(url, payload):
        raise AssertionError("must not post an invalid state")
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_cli.cmd_mission_mark(SimpleNamespace(milestone="1", state="whatever"), post=post)
    assert exc.value.code == "malformed state"


def test_mission_done_and_cancel_post_the_matching_outcome_and_reraise_no_active_mission():
    post, calls = _recording_post(200, {"ok": True, "data": {}})
    jaxflow_cli.cmd_mission_finish("done", post=post)
    jaxflow_cli.cmd_mission_finish("cancelled", post=post)
    assert calls == [
        (f"{jaxflow_cli.MISSION_BASE_URL}/finish", {"outcome": "done"}),
        (f"{jaxflow_cli.MISSION_BASE_URL}/finish", {"outcome": "cancelled"}),
    ]
    post2, _ = _recording_post(200, {"ok": False, "error": "no-active-mission"})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_cli.cmd_mission_finish("done", post=post2)
    assert exc.value.code == "no-active-mission"


def test_mission_show_prints_no_active_mission():
    def get(url):
        return 200, {"ok": True, "data": None}
    assert jaxflow_cli.cmd_mission_show(get=get) == "no active mission"


def test_mission_show_formats_the_active_mission():
    def get(url):
        assert url == f"{jaxflow_cli.MISSION_BASE_URL}/current"
        return 200, {"ok": True, "data": {
            "name": "Ship it", "goal": "6 phases tonight", "statusLine": "phase 1 merged",
            "milestones": [{"title": "Phase 1", "state": "done"}, {"title": "Phase 2", "state": "in-progress"}, {"title": "Phase 3", "state": "pending"}],
        }}
    assert jaxflow_cli.cmd_mission_show(get=get) == (
        "Ship it: 6 phases tonight\nstatus: phase 1 merged\n  [x] Phase 1\n  [~] Phase 2\n  [ ] Phase 3"
    )


def test_mission_show_transport_failure_is_mission_hub_unreachable():
    def get(url):
        raise OSError("unreachable")
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_cli.cmd_mission_show(get=get)
    assert exc.value.code == "mission-hub-unreachable"


def test_mission_argparse_wiring_end_to_end(monkeypatch):
    calls = []
    def fake_post(url, payload):
        calls.append((url, payload))
        return 200, {"ok": True, "data": {"id": 1}}
    monkeypatch.setattr(jaxflow_common, "_post", fake_post)
    rc = jaxflow_cli.main(["mission", "start", "--name", "Ship it", "--goal", "g", "--milestone", "M1", "--milestone", "M2"])
    assert rc == jaxflow_common.OK
    assert calls == [(jaxflow_cli.MISSION_BASE_URL, {"name": "Ship it", "goal": "g", "milestones": ["M1", "M2"]})]

    def fake_get(url):
        return 200, {"ok": True, "data": None}
    monkeypatch.setattr(jaxflow_cli, "_get", fake_get)
    rc2 = jaxflow_cli.main(["mission", "show"])
    assert rc2 == jaxflow_common.OK


def test_mission_argparse_requires_at_least_one_milestone_flag():
    args = jaxflow_cli.parse_args(["mission", "start", "--name", "A", "--goal", "g"])
    assert args.milestones == []


def test_mission_refusal_through_main_exits_refused_with_the_exact_stderr_code(monkeypatch, capsys):
    # Cold review F8: every other mission test above calls a cmd_mission_* helper directly, or
    # drives main() only on the success path -- this is the one test proving a refusal survives
    # the FULL main() path (parse_args -> dispatch -> except Refusal) with the documented exit
    # code and stderr shape (main():5769-5774 -- `print(exc.code, file=sys.stderr); return REFUSED`).
    def fake_post(url, payload):
        return 200, {"ok": False, "error": "no-active-mission"}
    monkeypatch.setattr(jaxflow_common, "_post", fake_post)
    rc = jaxflow_cli.main(["mission", "status", "phase 1 merged"])
    assert rc == jaxflow_common.REFUSED
    assert capsys.readouterr().err == "no-active-mission\n"


# ---- slice e: model defaults (spec: One source of model defaults) ----

_MODEL_DEFAULTS = json.loads((Path(__file__).resolve().parents[1] / "workflow" / "fixtures" / "model-defaults-v1.json").read_text())


def test_jaxflow_reviewer_defaults_matches_jaxflow_settings_with_no_drift():
    assert jaxflow_cli.REVIEWER_DEFAULTS == {
        "claude": {"runtime": "codex", **_MODEL_DEFAULTS["reviewers"]["codex"]},
        "codex": {"runtime": "claude", **_MODEL_DEFAULTS["reviewers"]["claude"]},
    }
    assert jaxflow_cli.REVIEWER_DEFAULTS == {
        k: v for k, v in jset.REVIEWER_DEFAULTS.items() if k in ("claude", "codex")
    }


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
    lines, code = jaxflow_cli.cmd_doctor(repo_root=repo_root, which=which,
                                      read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}},
                                      get=lambda url: (200, {}), **_doctor_fixtures(tmp_path))
    assert any(l == "wrapper:jaxflow ok" for l in lines)
    assert any(l == "wrapper:jaxflow-hook ok" for l in lines)
    assert any(l == "wrapper:jax-init ok" for l in lines)


def test_doctor_wrapper_missing_is_required_and_fails_the_exit_code(tmp_path):
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    lines, code = jaxflow_cli.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
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
    lines, code = jaxflow_cli.cmd_doctor(repo_root=repo_root, which=which,
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
    lines, _ = jaxflow_cli.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
                                   read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}},
                                   get=lambda url: (200, {}), **fixtures)
    assert any(l.startswith("skill:claude missing") for l in lines)

    # A symlink resolving to a DIFFERENT skill under workflow/skills/ (swapped target) must
    # not count as installed -- descendant-of-skills_root is not enough, it must match `name`.
    (repo_root / "workflow" / "skills" / "jax-init").mkdir(parents=True)
    (claude_skills / "jaxflow").rmdir()
    (claude_skills / "jaxflow").symlink_to(repo_root / "workflow" / "skills" / "jax-init")
    lines, _ = jaxflow_cli.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
                                   read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}},
                                   get=lambda url: (200, {}), **fixtures)
    assert any(l.startswith("skill:claude missing") for l in lines)

    # A symlink resolving into this checkout's workflow/skills/ counts as installed.
    (claude_skills / "jaxflow").unlink()
    (claude_skills / "jaxflow").symlink_to(repo_root / "workflow" / "skills" / "jaxflow")
    lines, _ = jaxflow_cli.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
                                   read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}},
                                   get=lambda url: (200, {}), **fixtures)
    assert any(l == "skill:claude ok" for l in lines)


def test_doctor_custom_codex_home_note_does_not_count_toward_the_checks_total(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "custom-codex-home"))
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    which = _doctor_which(repo_root, {"jaxflow", "jaxflow-hook", "jax-init", "tmux", "git", "rg", "gh"})
    lines, code = jaxflow_cli.cmd_doctor(repo_root=repo_root, which=which,
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
    lines, code = jaxflow_cli.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
                                      read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}},
                                      get=lambda url: (200, {}), **_doctor_fixtures(tmp_path))
    assert any(l.startswith("cli:gh missing") for l in lines)
    # gh being absent must not, by itself, fail the exit code when github is off. All three
    # wrappers plus tmux/git/rg resolve for real, under repo_root/bin, so gh is the only gap:
    only_gh_missing = jaxflow_cli.cmd_doctor(
        repo_root=repo_root,
        which=_doctor_which(repo_root, {"jaxflow", "jaxflow-hook", "jax-init", "tmux", "git", "rg", "claude", "codex", "opencode"}),
        read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}}, get=lambda url: (200, {}),
        **_doctor_fixtures(tmp_path))
    assert only_gh_missing[1] == 0


def test_doctor_cli_gh_is_required_when_github_integration_is_on(tmp_path):
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    lines, code = jaxflow_cli.cmd_doctor(
        repo_root=repo_root,
        which=_doctor_which(repo_root, {"jaxflow", "jaxflow-hook", "jax-init", "tmux", "git", "rg"}),
        read_settings=lambda: {"ok": True, "data": {"integrations": {"github": True, "agents": _ALL_AGENTS_ON}}}, get=lambda url: (200, {}),
        **_doctor_fixtures(tmp_path))
    assert any(l.startswith("cli:gh missing") for l in lines)
    assert code != 0


def test_doctor_settings_readable_reflects_the_injected_reader(tmp_path):
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    lines, _ = jaxflow_cli.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
                                   read_settings=lambda: {"ok": False, "error": "settings-malformed"},
                                   get=lambda url: (200, {}), **_doctor_fixtures(tmp_path))
    assert any(l.startswith("settings:readable missing") for l in lines)


def test_doctor_api_reachable_true_on_any_http_response_false_only_on_transport_error(tmp_path):
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    ok_lines, _ = jaxflow_cli.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
                                      read_settings=lambda: {"ok": True, "data": {"integrations": {"github": False, "agents": _ALL_AGENTS_ON}}},
                                      get=lambda url: (503, {"ok": False}), **_doctor_fixtures(tmp_path))
    assert any(l == "api:reachable ok" for l in ok_lines)
    def down(url):
        raise ConnectionRefusedError()
    down_lines, code = jaxflow_cli.cmd_doctor(repo_root=repo_root, which=_doctor_which(repo_root, set()),
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
    jaxflow_cli.cmd_doctor(repo_root=repo_root, which=which,
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
    return jaxflow_cli.cmd_doctor(
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
