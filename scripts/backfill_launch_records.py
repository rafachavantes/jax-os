#!/usr/bin/env python3
"""One-shot backfill: one `builder-launched` event per managed build run manifest.

Usage attribution reads these records from `workflow_events`; runs dispatched before the
worker emitted them still carry `launch_selection` in their manifest. Dry-run by default
(prints `would post N, skipping M`); `--apply` posts. Idempotent: run_ids already present
as `builder-launched` rows are skipped. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import jaxflow_common as jc
import jaxflow_run as jr


def _utc_z(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def collect_records(repos_root: Path) -> list[dict]:
    """Every manifest under `<repos_root>/*/.local/runs/*/` with a usable `launch_selection`."""
    records = []
    for path in sorted(Path(repos_root).glob("*/.local/runs/*/manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        selection = manifest.get("launch_selection")
        if not isinstance(selection, dict):
            continue
        if not all(
            isinstance(selection.get(key), str) and selection.get(key)
            for key in ("runtime_model", "model", "profile_name")
        ):
            continue
        launched_at = _utc_z(manifest.get("dispatch_start"))
        worktree = manifest.get("worktree")
        run_id = manifest.get("run_id")
        project = manifest.get("project")
        if not launched_at or not isinstance(worktree, str) or not worktree:
            continue
        if not isinstance(run_id, str) or not run_id or not isinstance(project, str) or not project:
            continue
        records.append({
            "run_id": run_id, "project": project, "role": "builder",
            "type": "builder-launched", "source": "deterministic", "emitter": "wrapper",
            "payload": {
                "launched_at": launched_at,
                "worktree": worktree,
                "runtime_model": selection["runtime_model"],
                "model": selection["model"],
                "profile": selection["profile_name"],
            },
        })
    return records


def existing_run_ids(db_path: Path) -> set[str]:
    con = jc._open_ro(db_path)
    try:
        rows = con.execute(
            "SELECT DISTINCT run_id FROM workflow_events WHERE type = 'builder-launched'"
        ).fetchall()
    finally:
        con.close()
    return {row[0] for row in rows if isinstance(row[0], str) and row[0]}


def main(argv, *, post=jc._post_event) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos-root", type=Path, default=Path.home() / "repos")
    parser.add_argument("--db", type=Path, default=jr.DB_PATH)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    records = collect_records(args.repos_root)
    try:
        existing = existing_run_ids(args.db)
    except Exception as e:
        print(f"backfill-launch: cannot read {args.db}: {e}", file=sys.stderr)
        return 1
    todo = [record for record in records if record["run_id"] not in existing]
    skipped = len(records) - len(todo)
    if not args.apply:
        print(f"would post {len(todo)}, skipping {skipped}")
        return 0

    posted = 0
    for record in todo:
        try:
            post(record)
        except Exception as e:
            print(f"backfill-launch: post failed for {record['run_id']}: {e}", file=sys.stderr)
            continue
        posted += 1
    print(f"posted {posted}, skipping {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
