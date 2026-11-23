#!/usr/bin/env python3
"""Tests for the one-shot builder launch-record backfill. Run: python3 -m pytest scripts"""
import json
import sqlite3
from pathlib import Path

import backfill_launch_records as bl


def _selection(model="fixture/real-model"):
    return {
        "profile_name": "default",
        "settings_revision": "x" * 32,
        "model": model,
        "runtime_model": "fixture/jaxflow-builder-default",
        "effort": "high",
        "credential": {"kind": "env", "env": "KEY"},
    }


def _manifest(repos_root, *, name="demo", run_id="aaaabbbbcccc", project="demo",
              selection=None, dispatch_start="2026-01-01T03:00:00Z"):
    root = Path(repos_root) / name
    path = root / ".local" / "runs" / run_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id, "project": project,
        "worktree": str(Path(repos_root) / f"{name}-feat-x"),
        "dispatch_start": dispatch_start,
    }
    if selection is not None:
        manifest["launch_selection"] = selection
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _db(path):
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE workflow_events (id INTEGER PRIMARY KEY, ts TEXT, run_id TEXT, "
        "project TEXT, role TEXT, type TEXT, payload TEXT)"
    )
    con.commit()
    con.close()
    return path


def test_collect_records_yields_one_event_from_a_launch_selection(tmp_path):
    _manifest(tmp_path / "repos", selection=_selection())
    records = bl.collect_records(tmp_path / "repos")
    assert len(records) == 1
    event = records[0]
    assert event["run_id"] == "aaaabbbbcccc"
    assert event["project"] == "demo"
    assert (event["type"], event["role"], event["source"], event["emitter"]) == (
        "builder-launched", "builder", "deterministic", "wrapper",
    )
    assert event["payload"] == {
        "launched_at": "2026-01-01T03:00:00Z",
        "worktree": str(tmp_path / "repos" / "demo-feat-x"),
        "runtime_model": "fixture/jaxflow-builder-default",
        "model": "fixture/real-model",
        "profile": "default",
    }


def test_collect_records_normalizes_dispatch_start_to_utc_z(tmp_path):
    _manifest(tmp_path / "repos", dispatch_start="2026-01-01T00:00:00-03:00", selection=_selection())
    records = bl.collect_records(tmp_path / "repos")
    assert records[0]["payload"]["launched_at"] == "2026-01-01T03:00:00Z"


def test_collect_records_skips_legacy_manifests(tmp_path):
    _manifest(tmp_path / "repos", selection=None)
    assert bl.collect_records(tmp_path / "repos") == []


def test_collect_records_skips_unreadable_and_partial_manifests(tmp_path):
    root = tmp_path / "repos" / "demo" / ".local" / "runs"
    (root / "bad1").mkdir(parents=True)
    (root / "bad1" / "manifest.json").write_text("{not json", encoding="utf-8")
    (root / "bad2").mkdir(parents=True)
    (root / "bad2" / "manifest.json").write_text(json.dumps({
        "run_id": "aaaabbbbcccc", "project": "demo", "worktree": "/w",
        "dispatch_start": "2026-01-01T00:00:00Z",
        "launch_selection": {"model": "fixture/real-model"},
    }), encoding="utf-8")
    assert bl.collect_records(tmp_path / "repos") == []


def test_resumed_run_manifest_yields_exactly_one_event_with_the_final_model(tmp_path):
    # The manifest keeps only the last launch_selection (spec D5).
    _manifest(tmp_path / "repos", selection=_selection("fixture/final-model"))
    records = bl.collect_records(tmp_path / "repos")
    assert len(records) == 1
    assert records[0]["payload"]["model"] == "fixture/final-model"


def test_main_dry_run_posts_nothing_and_returns_0(tmp_path, capsys):
    _manifest(tmp_path / "repos", selection=_selection())
    db = _db(tmp_path / "jaxos.db")
    posted = []
    code = bl.main(
        ["--repos-root", str(tmp_path / "repos"), "--db", str(db)],
        post=lambda e: posted.append(e) or {"ok": True},
    )
    assert code == 0
    assert posted == []
    assert "would post 1, skipping 0" in capsys.readouterr().out


def test_main_apply_posts_each_event(tmp_path, capsys):
    _manifest(tmp_path / "repos", selection=_selection())
    db = _db(tmp_path / "jaxos.db")
    posted = []
    code = bl.main(
        ["--repos-root", str(tmp_path / "repos"), "--db", str(db), "--apply"],
        post=lambda e: posted.append(e) or {"ok": True},
    )
    assert code == 0
    assert [e["payload"]["model"] for e in posted] == ["fixture/real-model"]
    assert "posted 1, skipping 0" in capsys.readouterr().out


def test_main_apply_skips_run_ids_already_in_the_db(tmp_path, capsys):
    _manifest(tmp_path / "repos", selection=_selection())
    db = _db(tmp_path / "jaxos.db")
    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO workflow_events (run_id, project, role, type, payload) VALUES (?,?,?,?,?)",
        ("aaaabbbbcccc", "demo", "builder", "builder-launched", "{}"),
    )
    con.commit()
    con.close()
    posted = []
    code = bl.main(
        ["--repos-root", str(tmp_path / "repos"), "--db", str(db), "--apply"],
        post=lambda e: posted.append(e) or {"ok": True},
    )
    assert code == 0
    assert posted == []
    assert "posted 0, skipping 1" in capsys.readouterr().out


def test_existing_run_ids_reads_only_launch_records(tmp_path):
    db = _db(tmp_path / "jaxos.db")
    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO workflow_events (run_id, project, role, type, payload) VALUES (?,?,?,?,?)",
        ("aaaabbbbcccc", "demo", "builder", "builder-launched", "{}"),
    )
    con.execute(
        "INSERT INTO workflow_events (run_id, project, role, type, payload) VALUES (?,?,?,?,?)",
        ("ddddeeeeffff", "demo", "builder", "run-started", "{}"),
    )
    con.commit()
    con.close()
    assert bl.existing_run_ids(db) == {"aaaabbbbcccc"}
