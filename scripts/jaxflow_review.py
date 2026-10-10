"""jaxflow review module -- P2 split of scripts/jaxflow.py (lean spec, Decision 18)."""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sqlite3
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
import jaxflow_run as jr
import jaxflow_settings as jset
from jax_init import ALLOWLIST_ROOT_DEFAULT, Refusal, _contained, _is_secret_path, canonicalize_target
from jaxflow_hook import redact
import jaxflow_common as jc
import jaxflow_workerkit as jk


# Acceptance smoke 2026-09-06: `--spec`/`--plan` reviews have no diff and no builder
# handoff, so `parse_reviewer_handoff`'s `test-output:` line never exists for them. The
# reviewer contract still treats "the redacted test/build output file" as required
# evidence-of-is unless the file is exactly this marker sentence (status-report-contract's
# own wording for "no command was required"). The dispatcher writes this file up front so
# a doc review always has valid evidence on disk, and the worker's prompt (below) points
# the reviewer at it.
DOC_REVIEW_TEST_MARKER = "No test/build command was required by this handoff.\n"


def _diff_reviews_for_branch(con, project, branch):
    """Every `kind: diff` `run-started` row for `project` targeting `branch` (F3:
    `project` is part of the identity), with its `run-finished` row (or None) and
    dispatch `session` (probed via tmux directly, not `cmd_status`). Item 7 guard."""
    rows = con.execute(
        "SELECT run_id, ts, payload FROM workflow_events WHERE project = ? AND role = 'reviewer' "
        "AND type = 'run-started' ORDER BY ts",
        (project,),
    ).fetchall()
    result = []
    for row in rows:
        payload = json.loads(row["payload"])
        if payload.get("kind") != "diff" or payload.get("target") != branch:
            continue
        finished = con.execute(
            "SELECT ts, payload FROM workflow_events WHERE run_id = ? AND role = 'reviewer' "
            "AND type = 'run-finished' LIMIT 1",
            (row["run_id"],),
        ).fetchone()
        result.append({
            "run_id": row["run_id"],
            "session": payload.get("session"),
            "finished_ts": finished["ts"] if finished else None,
            "finished_payload": json.loads(finished["payload"]) if finished else None,
        })
    return result


def _validate_since_chain(chain, *, branch, current_merge_base, run, worktree):
    """Applies lock (a) (verdict/target/lineage; kind is already implied by a non-None
    `_load_diff_review_node`) plus lock (c)'s continuity/root/ancestry checks to EVERY
    node in a root-first `chain`. Returns None on success, else a refusal code."""
    for i, node in enumerate(chain):
        if node["target"] != branch or node["builder_target"] != branch:
            return "since-lineage-mismatch"
        if node["contract_status"] != "ok" or node["verdict"] not in jr.VERDICTS:
            return "since-not-terminal"
        if not node["base_sha"] or not node["head_sha"]:
            return "since-chain-broken"
        ancestry = run(
            ["git", "merge-base", "--is-ancestor", node["base_sha"], node["head_sha"]], cwd=worktree,
        )
        if ancestry.returncode != 0:
            return "since-ancestry-broken"
        if i == 0:
            if node["base_sha"] != current_merge_base:
                return "since-root-stale"
        elif node["base_sha"] != chain[i - 1]["head_sha"]:
            return "since-continuity-broken"
    return None


def _log_refusal(repo, *, reason, target, phase, builder_run_id, prior_review_run_id, now):
    # ponytail: `dispatch-refused` isn't a valid workflow_events type (EVENT_TYPES in
    # src/lib/workflow.ts) -- refusals are local telemetry: one JSON line appended to
    # a control-repo file. Never blocks the refusal: any OSError is swallowed.
    line = json.dumps({
        "ts": jc._iso8601(now()), "verb": "review", "kind": "diff", "reason": reason,
        "target": target, "phase": phase, "builder_run_id": builder_run_id,
        "prior_review_run_id": prior_review_run_id,
    })
    try:
        path = repo / ".local" / "runs" / "refusals.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, (line + "\n").encode("utf-8"))
        finally:
            os.close(fd)
    except OSError:
        pass


def dispatch_review(args, *, run, post, env, now, allowlist_root=ALLOWLIST_ROOT_DEFAULT):
    # `repo` is the resolved git toplevel of the cwd, not the cwd itself (fixes
    # branch review F1) -- only a failed rev-parse (cwd outside any git repo) refuses.
    cwd = Path.cwd().resolve()
    repo = jc._require_toplevel(run, cwd)
    caller = jc.resolve_caller(env, args.from_caller)
    kind = "spec" if args.spec else "plan"
    raw_target = args.spec if args.spec else args.plan
    target = canonicalize_target(Path(raw_target), allowlist_root)
    # Defense against reading a secret file into the reviewer prompt (fixes branch
    # review F3) -- same guard and refusal text jax-init uses for a staged secret path. The
    # worker re-validates its own manifest's target the same way before reading it.
    if _is_secret_path(target):
        raise Refusal(f"secret-detected: {target}")
    project = jc.slugify_project(repo.name)
    phase = args.phase or target.stem
    selected = jset.select_reviewer(
        jset.read_settings(), caller, model=args.model, effort=args.effort, agents=jc._enabled_agents(),
    )
    runtime = selected["runtime"]
    model = selected["model"]
    effort = selected["effort"]
    fallback = selected["fallback"]
    jc._check_hub_caps({
        "phase": phase, "target": str(target), "model": model, "effort": effort,
        "repo": str(repo),
    })

    args_ns = SimpleNamespace(
        role="reviewer", project=project, phase=phase, repo=str(repo),
        prompt_file=None, runtime=runtime, callback=None,
    )
    try:
        jr.preflight(args_ns, run=run, allow_untracked=True, require_feat_branch=False, skip_handoff=True)
    except ValueError as exc:
        raise Refusal(jc._map_refusal(str(exc))) from exc

    try:
        run_id, session, paths = jr._alloc_run(project, kind, repo, run, run_id=uuid.uuid4().hex[:12])
    except ValueError as exc:
        raise Refusal(jc._map_refusal(str(exc))) from exc

    # Fixes acceptance smoke 2026-09-06 (fix 4): refuse BEFORE the manifest dir exists and
    # before any POST -- see `_require_caller_session`.
    caller_session = jc._require_caller_session(env, caller)
    caller_pane = env.get("TMUX_PANE")
    started_payload = {
        "phase": phase, "runtime": runtime, "kind": kind, "target": str(target),
        "caller": caller, "caller_session": caller_session, "model": model,
        "effort": effort, "session": session, "repo": str(repo),
    }
    if caller_pane:
        started_payload["caller_pane"] = caller_pane
    jc._check_hub_caps(jc._caps_from_started(started_payload))

    # Fixes acceptance smoke 2026-09-06 (fix 2): doc-review evidence, written before the
    # manifest/dispatch so it is already on disk when the worker composes its prompt.
    paths["tests"].write_text(DOC_REVIEW_TEST_MARKER, encoding="utf-8")

    manifest = {
        "kind": kind, "role": "reviewer", "project": project, "phase": phase,
        "repo": str(repo), "target": str(target), "caller": caller,
        "caller_session": caller_session, "model": model, "effort": effort,
        "runtime": runtime, "run_id": run_id, "session": session,
        "no_callback": jc._no_callback(args, caller), "focus": args.focus,
        "threat_model": _threat_model_for(repo),
        "dispatch_start": jc._iso8601(now()),
    }
    if fallback:
        manifest["fallback"] = fallback

    manifest_dir = jc._manifest_dir(repo, run_id)
    manifest_path = manifest_dir / "manifest.json"
    manifest_dir.mkdir(parents=True, exist_ok=False)
    jc._write_manifest(manifest_path, manifest)

    jc._warn_if_claude_callback_hook_missing(manifest)

    started_event = {
        "run_id": run_id, "project": project, "role": "reviewer", "type": "run-started",
        "source": "deterministic", "emitter": "wrapper", "payload": started_payload,
    }
    try:
        post(started_event)
    except Exception as exc:
        shutil.rmtree(manifest_dir, ignore_errors=True)
        paths["tests"].unlink(missing_ok=True)
        raise jc._hub_refusal(exc) from exc

    # The worker PROCESS is sealed from birth via an `env` prefix (fixes branch review F4)
    # -- not by writing os.environ once the process is already running.
    cmd = " ".join(
        shlex.quote(part) for part in (
            "env", "HONCHO_ENABLED=false", "JAXFLOW_ONESHOT=1",
            sys.executable, str(jc.SCRIPT_PATH), "--run-worker", str(manifest_path),
        )
    )
    try:
        result = run(["tmux", "new-session", "-d", "-s", session, "-c", str(repo), cmd])
        launch_failed = result.returncode != 0
    except Exception:
        launch_failed = True
    if launch_failed:
        cancelled_event = {
            "run_id": run_id, "project": project, "role": "reviewer", "type": "run-finished",
            "source": "deterministic", "emitter": "wrapper",
            "payload": {
                "phase": phase, "exit_code": None, "contract_status": "cancelled",
                "report_path": None, "summary": "tmux new-session failed at dispatch",
            },
        }
        try:
            post(cancelled_event)
        except Exception:
            pass
        shutil.rmtree(manifest_dir, ignore_errors=True)
        paths["tests"].unlink(missing_ok=True)
        raise Refusal("tmux-failed")
    jc._write_callback_pointer(manifest)
    if fallback:
        print(jc._reviewer_runtime_line(runtime, fallback))
    return run_id


def _load_finished_build(db_path, builder_run_id):
    """The finished build a `review --diff` targets, as `(project, started_payload,
    finished_payload)`. Every refusal is `unknown-run` with its own hint (spec 2.9). The
    unfiltered `run-started` lookup only words the hint for a missing builder row; it never
    decides whether the run is usable."""
    con = jc._open_ro(db_path)
    con.row_factory = sqlite3.Row
    try:
        # fixes Part 2 diff-review F3: `started` is filtered to `role = 'builder'` again
        # (spec §4.3 step 1's explicit requirement) -- without this filter, a reviewer's
        # own `run-started` row (forged or not, `kind: build`) could be paired with an
        # UNRELATED builder's `run-finished` row sharing the same run id and accepted as a
        # build. When this filtered query misses, a SEPARATE unfiltered lookup below is
        # used ONLY to word the `unknown-run` hint -- never to decide whether the run is
        # usable.
        started, finished = jc._builder_run_rows(con, builder_run_id)
        if not started:
            any_started = con.execute(
                "SELECT role, payload FROM workflow_events WHERE run_id = ? "
                "AND type = 'run-started' LIMIT 1",
                (builder_run_id,),
            ).fetchone()
            if not any_started:
                hint = f"hint: no run-started row for {builder_run_id}"
            elif any_started["role"] == "reviewer":
                hint = f"hint: run {builder_run_id} is a reviewer run, not a build"
            else:
                any_kind = json.loads(any_started["payload"]).get("kind", "unknown")
                hint = f"hint: run {builder_run_id} is a {any_kind} run, not a build"
            raise jc._refuse("unknown-run", hint)
    finally:
        con.close()
    # fixes cold review F1: a build that has not finished yet is treated exactly like an
    # unknown run -- the alternative (a KeyError on a missing "verify" field, or reviewing
    # a diff that is not final) is worse than one shared refusal code (spec §2.9's
    # extended unknown-run semantics). fixes cold review round 2 F4: each branch below
    # sets its own .hint so §4.3's required second stderr line names the actual reason.
    started_payload = json.loads(started["payload"])
    if started_payload.get("kind") != "build":
        # Defense-in-depth: `started` is already filtered to `role = 'builder'`, so this
        # should never fire for a legitimate row (only `build` dispatches ever write
        # `role: builder`) -- kept in case a malformed/hand-crafted row slips past the
        # role filter with the wrong `kind`.
        raise jc._refuse("unknown-run", f"hint: run {builder_run_id} is a {started_payload.get('kind', 'unknown')} run, not a build")
    if not finished:
        raise jc._refuse("unknown-run", f"hint: build {builder_run_id} has not finished yet")
    finished_payload = json.loads(finished["payload"])
    # A build whose REPORT is missing/invalid still has a real diff: the wrapper records
    # `head_sha` regardless of `contract_status`, and verify is re-run below anyway.
    # Refusing it threw away whole builds over a report-format slip (MOA-471, b756335468ba).
    if not finished_payload.get("head_sha"):
        raise jc._refuse("unknown-run", f"hint: build {builder_run_id} finished without a head_sha")
    return started["project"], started_payload, finished_payload


def _load_build_inputs(repo, builder_run_id, worktree, allowlist_root):
    """The build's own manifest, plan and resolved spec, as `(builder_manifest, plan_path,
    spec_dest)`. Runs AFTER the worktree check and BEFORE the verify-first re-run, so a build
    whose manifest/plan cannot be resolved never spends a verify run or rewrites `.tests.txt`.
    `_resolve_handoff_spec_path` is a module global (tests spy on it)."""
    # fixes S2 (real smoke b12a3db3da4b): the diff handoff must name the SAME plan/spec
    # the build's own handoff named -- read from the build's own manifest, using the SAME
    # `_manifest_dir` helper `build` writes to. Runs AFTER the worktree check but BEFORE
    # the verify-first re-run below (spec §4.3 step 1 amended), so a build whose own
    # manifest/plan can't be resolved never spends a verify run or rewrites `.tests.txt`.
    builder_manifest_path = jc._manifest_dir(repo, builder_run_id) / "manifest.json"

    def _unusable_builder_manifest():
        return jc._refuse("unknown-run", f"hint: build manifest for {builder_run_id} missing or invalid: {builder_manifest_path}")

    try:
        builder_manifest = jc._load_manifest(jc._manifest_dir(repo, builder_run_id))
    except (OSError, ValueError):
        raise _unusable_builder_manifest()
    raw_plan_path = builder_manifest.get("plan_path")
    try:
        plan_path = Path(raw_plan_path).resolve() if raw_plan_path else None
    except OSError:
        plan_path = None
    # MOA-467: the plan is the ORIGINAL document -- it lives in the control repo (new
    # manifests), in the worktree (legacy copied manifests), or in another explicitly
    # allowed location under the allowlist (external documents `build` already
    # validated at dispatch). The check below mirrors `dispatch_build`'s own: canonical,
    # non-secret, readable, inside the allowlist. Secret-shaped paths, missing files and
    # outside-allowlist paths all refuse here, before any verify run is ever spent.
    if jc._plan_path_defect(plan_path, allowlist_root):
        raise _unusable_builder_manifest()
    plan_text = plan_path.read_text(encoding="utf-8")
    spec_dest, spec_refusal_code = jc._resolve_handoff_spec_path(plan_text, plan_path, worktree, allowlist_root)
    if spec_refusal_code:
        raise Refusal(spec_refusal_code)
    return builder_manifest, plan_path, spec_dest


def _evidence_path(worktree, builder_run_id):
    """The build's `.tests.txt` evidence path with its reports directory created. Refuses a
    symlink or a non-regular file BEFORE the path is ever snapshotted, read or written
    through (F3). Only the file itself is checked: parent dirs are the operator's own
    worktree, trusted (diff review 4884f63bdd16 F2 rejected)."""
    tests_path = worktree / ".local" / "reports" / f"{builder_run_id}.tests.txt"
    # fixes cold review F1: the reports directory may not exist yet for a fresh worktree.
    tests_path.parent.mkdir(parents=True, exist_ok=True)

    # fixes F3 (HIGH, NEW): a builder-planted symlink (or any non-regular file) at the
    # evidence path must be refused BEFORE it is ever snapshotted, read, or written
    # through -- checked here, before ANY of that happens.
    # ponytail: parent dirs are the operator's own worktree, trusted; only the evidence
    # file itself is checked (diff review 4884f63bdd16 F2 rejected).
    if tests_path.is_symlink() or (tests_path.exists() and not tests_path.is_file()):
        raise jc._refuse("path-outside-allowlist", f"hint: evidence path is a symlink or not a regular file: {tests_path}")
    return tests_path


def _review_guards(args, *, run, repo, db_path, project, branch, phase, builder_run_id, now):
    """The two re-review locks of `review --diff`, in order: the CONCURRENCY lock (never
    bypassed) and the terminal-verdict lock (bypassed by --full/--since). Returns the
    manifest's `guard` dict. `run` is only used for `tmux has-session`."""
    # MOA-471 item 7: the CONCURRENCY lock always runs first and is never bypassed.
    guard_con = jc._open_ro(db_path)
    guard_con.row_factory = sqlite3.Row
    try:
        prior_reviews = _diff_reviews_for_branch(guard_con, project, branch)
    finally:
        guard_con.close()
    # ponytail: best-effort, no reservation (a duplicate reviewer costs minutes, a
    # reservation costs a schema) -- cr F1, rejected by the tech lead.
    running = next(
        (r for r in prior_reviews if r["finished_payload"] is None and r["session"]
         and run(["tmux", "has-session", "-t", r["session"]]).returncode == 0),
        None,
    )
    if running:
        _log_refusal(
            repo, reason="review-running", target=branch, phase=phase,
            builder_run_id=builder_run_id, prior_review_run_id=running["run_id"], now=now,
        )
        raise jc._refuse("review-running", f"hint: run {running['run_id']} is still reviewing this branch")

    # The terminal-verdict lock: only among prior reviews with a real terminal verdict.
    terminal = [
        r for r in prior_reviews
        if r["finished_payload"] is not None
        and r["finished_payload"].get("contract_status") == "ok"
        and r["finished_payload"].get("verdict") in jr.VERDICTS
    ]
    selected = max(terminal, key=lambda r: r["finished_ts"], default=None)
    # F4 (diff review 4884f63bdd16): the override reflects the flags actually PASSED,
    # independently of whether a terminal prior review was found to bypass.
    flag_override = "full" if args.full else ("since" if args.since else "none")
    if selected is None:
        return {"prior_review_run_id": None, "prior_verdict": None, "override": flag_override}
    prior_run_id = selected["run_id"]
    verdict = selected["finished_payload"]["verdict"]
    if args.full:
        override = "full"
    elif args.since:
        override = "since"
    elif verdict in ("approve", "approve-with-changes"):
        prior_head = jc._read_manifest_field(repo, prior_run_id, "head_sha")
        _log_refusal(
            repo, reason="prior-review-accepted", target=branch, phase=phase,
            builder_run_id=builder_run_id, prior_review_run_id=prior_run_id, now=now,
        )
        raise jc._refuse("prior-review-accepted", f"hint: run {prior_run_id} already {verdict} at {prior_head}; "
            f"pass --full \"<reason>\" or --since {prior_run_id}")
    else:  # reject, no flags: proceeds, advisory only
        override = "none"
        print(f"hint: prior review {prior_run_id} rejected this branch; consider --since {prior_run_id}")
    return {"prior_review_run_id": prior_run_id, "prior_verdict": verdict, "override": override}


def _snapshot_evidence(tests_path):
    """Snapshot any PRE-EXISTING evidence before the verify re-run overwrites it. Returns
    `(restore, existing)`: `existing` is the bytes (None when there was no file) and
    `restore()` puts the snapshot back (or removes the new file), leaving a symlink planted
    after the guard alone (F3). The caller invokes `restore` itself: from the
    path-outside-allowlist refusal in `_verify_or_reuse` and from the parent's single
    `except Refusal`. `verify-failed` never restores: the fresh evidence IS its point."""
    # fixes Part 2 diff-review F5: snapshot any PRE-EXISTING evidence before the verify
    # re-run overwrites it, so a refusal further down this function -- after verify has
    # already PASSED -- can restore it instead of leaving a stale/unlogged artifact behind
    # in the builder's worktree. `verify-failed` itself is excluded on purpose: that
    # refusal's whole point IS the fresh evidence this write is about to produce (spec
    # §4.3 step 2/3 -- "fresher... since it was just re-run"), so it is never undone.
    evidence_existed = tests_path.is_file()
    evidence_before = tests_path.read_bytes() if evidence_existed else None

    def restore():
        # Re-checked immediately before acting (fixes F3): a symlink planted after the
        # guard above must never be followed for the restore write/unlink either -- a
        # symlinked path here is simply left alone.
        if tests_path.is_symlink():
            return
        if evidence_existed:
            tests_path.write_bytes(evidence_before)
        else:
            tests_path.unlink(missing_ok=True)
    return restore, evidence_before


def _verify_or_reuse(run, worktree, tests_path, existing, restore, *, builder_run_id,
                     verify_cmd, build_cmd, finished_payload, reverify):
    """The verification step of `review --diff`: reuse the builder's own evidence when
    HEAD/tree/commands/tests.txt still match (MOA-471 item 8), else re-run and rewrite
    `.tests.txt`. Returns the manifest's `verify` field. `existing` is the snapshotted
    evidence bytes (None when absent). The path-outside-allowlist refusal restores the
    snapshot (F3); `verify-failed` raises WITHOUT restoring, and this helper is called
    OUTSIDE the parent's `try`, so no verify-step refusal can ever reach the parent's
    `except Refusal` restore (Decision 10)."""
    # MOA-471 item 8: skip the foreground re-run when HEAD/tree/commands/tests.txt
    # evidence all still match the builder's own finished run (see spec item 8).
    head_probe = run(["git", "rev-parse", "HEAD"], cwd=worktree)
    current_head = head_probe.stdout.strip() if head_probe.returncode == 0 else None
    porcelain = run(["git", "status", "--porcelain"], cwd=worktree)
    requested_cmds = [verify_cmd] + ([build_cmd] if build_cmd else [])
    frames_parsed, parse_error = None, "tests-file-missing"
    if existing is not None:
        try:
            existing_text = existing.decode("utf-8")
        except UnicodeDecodeError:
            frames_parsed, parse_error = None, "tests-file-not-utf8"
        else:
            frames_parsed, parse_error = jr.parse_tests_frames(
                existing_text, [redact(c) for c in requested_cmds],
            )
    should_reuse, verify_reason = jc._decide_verify_reuse(
        head_matches=bool(current_head) and current_head == finished_payload.get("head_sha"),
        porcelain_clean=(porcelain.returncode == 0 and porcelain.stdout == ""),
        # ponytail: always True on every real call (verify_cmd/build_cmd ARE
        # started_payload's own fields); kept as an explicit input, its False branch
        # exercised directly by `_decide_verify_reuse`'s own unit tests.
        commands_match=True,
        frames=frames_parsed, parse_error=parse_error,
    )
    if reverify:
        should_reuse, verify_reason = False, "reverify-forced"
    if should_reuse:
        print(f"verification reused from build `{builder_run_id}` at `{current_head}`")
        return {
            "mode": "reused", "reason": verify_reason,
            "source_build_run_id": builder_run_id, "head_sha": current_head,
        }
    print(f"verification rerun: {verify_reason}")
    verify_frames = jk._run_verify_commands(run, worktree, verify_cmd, build_cmd)
    if not jk._write_verify_tests_file(tests_path, verify_frames, worktree=worktree):
        # fixes F3 (TOCTOU symlink race): refuse, restoring the snapshotted evidence.
        restore()
        raise jc._refuse("path-outside-allowlist", f"hint: evidence path is a symlink or not a regular file: {tests_path}")
    if any(r.returncode != 0 for _, r in verify_frames):
        raise Refusal("verify-failed")
    return {
        "mode": "rerun", "reason": verify_reason,
        "source_build_run_id": builder_run_id, "head_sha": current_head,
    }


def _resolve_review_range(run, repo, worktree, branch, builder_manifest):
    """`(base_sha, head_sha)` of the range a `review --diff` covers: worktree HEAD, and the
    base the build itself recorded (MOA-503), falling back to merge-base. Refuses
    `reviewer head mismatch` when either end is not a 40-hex SHA. Called INSIDE the parent's
    `try`: its refusals restore the evidence."""
    head = run(["git", "rev-parse", "HEAD"], cwd=worktree)
    head_sha = head.stdout.strip() if head.returncode == 0 else None
    # MOA-503: review the range the build itself produced -- from its recorded base
    # (a stacked build's base is its parent branch, not the default branch). A missing,
    # malformed or non-ancestor base falls back to merge-base; present-but-unusable hints.
    base_sha, recorded = None, builder_manifest.get("base_sha")
    if recorded is not None:
        if not (isinstance(recorded, str) and jr.SHA_RE.match(recorded)):
            why = "not a 40-hex SHA"
        elif run(["git", "merge-base", "--is-ancestor", recorded, "HEAD"], cwd=worktree).returncode != 0:
            why = "not an ancestor of HEAD"
        else:
            why, base_sha = None, recorded
        if why:
            print(f"hint: recorded build base {recorded} unusable ({why}); reviewing merge-base..HEAD")
    if base_sha is None:
        base = run(["git", "merge-base", branch, jr._default_branch(repo, run)], cwd=worktree)
        base_sha = base.stdout.strip() if base.returncode == 0 else None
    if not base_sha or not jr.SHA_RE.match(base_sha) or not head_sha or not jr.SHA_RE.match(head_sha):
        raise Refusal("reviewer head mismatch")
    return base_sha, head_sha


def _resolve_since(args, *, run, repo, worktree, branch, db_path, head_sha, base_sha):
    """The `--since` correction round: `(since_review_run_id, since_verdict, base_sha)`, the
    base moved to the prior review's head. Without `--since` it returns `(None, None,
    base_sha)` untouched. A `Refusal` propagates unchanged; any malformed chain data
    refuses `since-chain-missing`."""
    if not args.since:
        return None, None, base_sha
    # F3 (diff review 4884f63bdd16): a malformed chain node (a non-string
    # since_review_run_id or SHA in some manifest along the chain) can raise a
    # TypeError/AttributeError/etc. deep in the walk or in the git calls below.
    # ponytail: one guard instead of per-field validation -- any such crash
    # refuses exactly like a broken/unreadable chain node (Refusal itself
    # propagates unchanged, never caught here).
    try:
        since_con = jc._open_ro(db_path)
        since_con.row_factory = sqlite3.Row
        try:
            chain, walk_error = jc._walk_review_chain(
                lambda rid: jc._load_diff_review_node(since_con, repo, rid), args.since,
            )
        finally:
            since_con.close()
        if walk_error is not None:
            kind, bad_run_id = walk_error
            raise jc._refuse(f"since-chain-{kind}", f"hint: chain walk failed at {bad_run_id} ({kind}); run a full review instead")
        target_node = chain[-1]
        if not target_node["head_sha"]:
            raise jc._refuse("since-chain-broken", f"hint: {target_node['run_id']} has no head_sha; run a full review instead")
        ancestor = run(
            ["git", "merge-base", "--is-ancestor", target_node["head_sha"], head_sha], cwd=worktree,
        )
        if ancestor.returncode != 0:
            raise jc._refuse("since-not-ancestor", "hint: the --since target's head_sha is not an ancestor of HEAD; run a full review instead")
        lock_error = _validate_since_chain(
            chain, branch=branch, current_merge_base=base_sha, run=run, worktree=worktree,
        )
        if lock_error:
            raise jc._refuse(lock_error, "hint: run a full review instead")
        return args.since, target_node["verdict"], target_node["head_sha"]
    except Refusal:
        raise
    except (TypeError, AttributeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise jc._refuse("since-chain-missing",
                      f"hint: malformed chain data ({exc.__class__.__name__}); run a full review instead") from exc


def _preflight_diff(*, run, db_path, worktree, builder_run_id, project, phase, runtime,
                    base_sha, head_sha, spec_dest, plan_path, tests_path):
    """Route `--diff` through the SAME handoff-grammar/evidence validation every other
    reviewer dispatch gets (spec 4.3 step 3). A throwaway stub file carries the grammar
    block `jr.preflight` parses; the diff text itself is rendered later, by the worker.
    Raises `Refusal` through `_map_refusal` on a `ValueError`."""
    # fixes cold review round 2 F3(a): route --diff through the SAME handoff-grammar/
    # evidence validation every other reviewer dispatch gets (spec §4.3 step 3), instead of
    # the standalone jr.reviewer_head_matches(...) call this replaces plus skip_handoff=True.
    # base_sha/head_sha/tests_path are all already known here, so a small throwaway stub file
    # carries exactly the grammar block preflight()'s handoff parser expects -- the actual
    # diff text is rendered later, by the worker. repo=str(worktree), not str(repo) (the
    # control repo), so the dirty-tree/detached-HEAD checks validate the worktree under
    # review, not the control repo's own unrelated branch/dirty state. fixes S2: the extra
    # `spec:`/`plan:` lines (legal, unparsed `paths:` keys -- `parse_reviewer_handoff` only
    # ever looks for the one `test-output:` line) name the SAME plan/spec `build` itself
    # resolved, above.
    stub_path = worktree / ".local" / "runs" / f"{builder_run_id}-diff-handoff-stub.txt"
    stub_path.parent.mkdir(parents=True, exist_ok=True)
    stub_path.write_text(
        f"diff: {base_sha}..{head_sha}\npaths:\n"
        f"  spec: {spec_dest}\n  plan: {plan_path}\n  test-output: {tests_path}\n",
        encoding="utf-8",
    )
    args_ns = SimpleNamespace(
        role="reviewer", project=project, phase=phase, repo=str(worktree),
        prompt_file=str(stub_path), runtime=runtime, callback=None,
    )
    try:
        jr.preflight(
            args_ns, run=run, allow_untracked=True, require_feat_branch=False,
            skip_handoff=False, db_path=db_path,
        )
    except ValueError as exc:
        raise Refusal(jc._map_refusal(str(exc))) from exc
    finally:
        stub_path.unlink(missing_ok=True)


def _diff_manifests(args, *, repo, worktree, project, phase, branch, builder_run_id, base_sha,
                    head_sha, tests_path, caller, caller_session, caller_pane, model, effort,
                    runtime, fallback, run_id, session, now, plan_path, spec_dest, verify_field,
                    guard_info, since_review_run_id, since_verdict):
    """`(manifest, started_event_payload)` of a diff-review run. Key order is part of the
    contract (manifest.json is written in insertion order). `caller_session` is passed in:
    `_require_caller_session` must keep running in the dispatcher, right after `_alloc_run`."""
    manifest = {
        "kind": "diff", "role": "reviewer", "project": project, "phase": phase,
        "repo": str(repo), "worktree": str(worktree), "target": branch,
        "builder_run_id": builder_run_id, "base_sha": base_sha, "head_sha": head_sha,
        "tests_path": str(tests_path), "caller": caller, "caller_session": caller_session,
        "model": model, "effort": effort, "runtime": runtime, "run_id": run_id,
        "session": session, "no_callback": jc._no_callback(args, caller), "focus": args.focus,
        "threat_model": _threat_model_for(repo),
        "plan_path": str(plan_path), "spec_path": str(spec_dest),
        "dispatch_start": jc._iso8601(now()),
        "verify": verify_field,
        "guard": guard_info,
        "full_reason": args.full,
    }
    if since_review_run_id:
        manifest["since_review_run_id"] = since_review_run_id
        manifest["since_verdict"] = since_verdict
    if fallback:
        manifest["fallback"] = fallback

    started_event_payload = {
        "phase": phase, "runtime": runtime, "kind": "diff", "target": branch,
        "caller": caller, "caller_session": caller_session, "model": model, "effort": effort,
        "session": session, "repo": str(repo), "builder_run_id": builder_run_id,
    }
    if caller_pane:
        started_event_payload["caller_pane"] = caller_pane
    return manifest, started_event_payload


def dispatch_diff_review(args, *, run, post, env, now, allowlist_root=ALLOWLIST_ROOT_DEFAULT, db_path=None):
    if args.since is not None and args.full is not None:
        raise Refusal("since-full-conflict")
    if args.full is not None and not args.full.strip():
        raise Refusal("full-reason-blank")
    if args.since is not None and not jc._RUN_ID_RE.match(args.since):
        raise Refusal("since-invalid")
    cwd = Path.cwd().resolve()
    repo = jc._require_toplevel(run, cwd)
    caller = jc.resolve_caller(env, args.from_caller)
    selected = jset.select_reviewer(
        jset.read_settings(), caller, model=args.model, effort=args.effort, agents=jc._enabled_agents(),
    )
    runtime = selected["runtime"]
    model = selected["model"]
    effort = selected["effort"]
    fallback = selected["fallback"]
    builder_run_id = args.diff

    db_path = db_path or jr.DB_PATH
    project, started_payload, finished_payload = _load_finished_build(db_path, builder_run_id)
    branch = started_payload["target"]
    verify_cmd = started_payload["verify"]
    # `.get`, never a subscript: every build recorded before MOA-454 has no `build` key
    # (spec §2.6), and `review --diff` must keep working on those.
    build_cmd = started_payload.get("build")
    phase = args.phase or started_payload["phase"]

    # fixes cold review F2: resolve and confine the derived worktree path (§6 I10) and
    # require it to still exist -- a removed/never-created worktree must refuse cleanly
    # (a NEW code, distinct from `unknown-run`: the RUN is known and finished, only its
    # worktree is gone) instead of an uncaught FileNotFoundError from the verify call below.
    worktree = jc._branch_worktree_path(allowlist_root, project, branch)
    if not _contained(worktree, allowlist_root) or not worktree.is_dir():
        raise Refusal("worktree-missing")

    builder_manifest, plan_path, spec_dest = _load_build_inputs(repo, builder_run_id, worktree, allowlist_root)
    tests_path = _evidence_path(worktree, builder_run_id)
    guard_info = _review_guards(
        args, run=run, repo=repo, db_path=db_path, project=project, branch=branch,
        phase=phase, builder_run_id=builder_run_id, now=now,
    )
    restore_evidence, existing_evidence = _snapshot_evidence(tests_path)
    verify_field = _verify_or_reuse(
        run, worktree, tests_path, existing_evidence, restore_evidence,
        builder_run_id=builder_run_id, verify_cmd=verify_cmd, build_cmd=build_cmd,
        finished_payload=finished_payload, reverify=args.reverify,
    )
    try:
        base_sha, head_sha = _resolve_review_range(run, repo, worktree, branch, builder_manifest)
        since_review_run_id, since_verdict, base_sha = _resolve_since(
            args, run=run, repo=repo, worktree=worktree, branch=branch, db_path=db_path,
            head_sha=head_sha, base_sha=base_sha,
        )
        _preflight_diff(
            run=run, db_path=db_path, worktree=worktree, builder_run_id=builder_run_id,
            project=project, phase=phase, runtime=runtime, base_sha=base_sha, head_sha=head_sha,
            spec_dest=spec_dest, plan_path=plan_path, tests_path=tests_path,
        )
        try:
            run_id, session, paths = jr._alloc_run(project, "diff", repo, run, run_id=uuid.uuid4().hex[:12])
        except ValueError as exc:
            raise Refusal(jc._map_refusal(str(exc))) from exc

        caller_session = jc._require_caller_session(env, caller)
        caller_pane = env.get("TMUX_PANE")

        manifest, started_event_payload = _diff_manifests(
            args, repo=repo, worktree=worktree, project=project, phase=phase, branch=branch,
            builder_run_id=builder_run_id, base_sha=base_sha, head_sha=head_sha,
            tests_path=tests_path, caller=caller, caller_session=caller_session,
            caller_pane=caller_pane, model=model, effort=effort, runtime=runtime,
            fallback=fallback, run_id=run_id, session=session, now=now, plan_path=plan_path,
            spec_dest=spec_dest, verify_field=verify_field, guard_info=guard_info,
            since_review_run_id=since_review_run_id, since_verdict=since_verdict,
        )
        run_id = jc._dispatch_run(
            run=run, post=post, repo=repo, project=project, phase=phase, kind="diff",
            role="reviewer", run_id=run_id, session=session, manifest=manifest,
            manifest_dir=jc._manifest_dir(repo, run_id), started_payload=started_event_payload,
            cleanup_paths=[],
        )
        if fallback:
            print(jc._reviewer_runtime_line(runtime, fallback))
        return run_id
    except Refusal:
        # fixes Part 2 diff-review F5: every refusal reachable from here on (reviewer head
        # mismatch, preflight, allocation, caller-session, hub-unreachable, tmux-failed)
        # fires AFTER the verify re-run above already overwrote `.tests.txt` -- restore
        # whatever evidence existed before it (or remove the new file entirely) so a
        # refused dispatch never leaves a stale/unlogged artifact behind.
        restore_evidence()
        raise


# Blueprint decision D28: the project's own AGENTS.md declares a threat model ONCE, in the
# same `## <heading>` shape as the Deploy policy block above (see
# `workflow/templates/AGENTS-template.md`) -- `jaxflow review` injects one line into every
# reviewer handoff instead of the tech lead re-typing `--focus` on every dispatch (MOA-474
# took 9 spec rounds over a "hostile local attacker" HIGH finding on a single-user VPS).
_THREAT_MODEL_HEADING_RE = re.compile(r"^##[ \t]+Threat model\b.*$", re.I | re.M)

_THREAT_MODEL_MODE_RE = re.compile(r"^[ \t]*mode:[ \t]*(\S+)[ \t]*$", re.I | re.M)


def parse_threat_model(text):
    """Finds the `## Threat model` heading (case-insensitive, any trailing HTML comment
    allowed) and reads the first `mode:` line under it, bounded to the next `##` heading
    or end of file -- the same block-boundary idea `_resolve_delivery_target` uses for its
    Deploy policy block, so a `mode:` line belonging to a LATER section is never borrowed.
    No heading, no mode line under it, or a mode that is not a known value all return
    None; the caller treats None as "no line", never an error."""
    heading = _THREAT_MODEL_HEADING_RE.search(text)
    if not heading:
        return None
    tail = text[heading.end():]
    boundary = re.search(r"^##[ \t]", tail, re.M)
    block = tail[:boundary.start()] if boundary else tail
    mode_match = _THREAT_MODEL_MODE_RE.search(block)
    if not mode_match:
        return None
    mode = mode_match.group(1).lower()
    return mode if mode in jk._THREAT_MODEL_LINES else None


def _threat_model_for(repo):
    """Reads the CONTROL repo's own AGENTS.md for its `## Threat model` mode, to store on
    the manifest at review dispatch (D28). Missing/unreadable AGENTS.md and a missing/
    unknown mode are the SAME outcome -- None, plus one stderr note so the operator knows
    the reviewer handoff will carry no threat-model line."""
    try:
        text = (repo / "AGENTS.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        text = None
    mode = parse_threat_model(text) if text is not None else None
    if mode is None:
        print(
            "jaxflow: no threat model in AGENTS.md — reviewer gets no threat-model line",
            file=sys.stderr,
        )
    return mode


def _run_doc_reviewer_worker(manifest, *, run, post, popen, killpg, env, allowlist_root,
                             manifest_path):
    """The `--spec` / `--plan` document-review worker: validates its own manifest target
    (defense-in-depth), builds the doc-review handoff, runs the reviewer in `/tmp` and
    finishes through `_finish_reviewer_worker(status_first=True)`. Returns the exit code."""
    repo = Path(manifest["repo"]).resolve()
    if not _contained(repo, allowlist_root):
        print("path-outside-allowlist", file=sys.stderr)
        return jc.REFUSED
    run_id = manifest["run_id"]
    project = manifest["project"]
    phase = manifest["phase"]
    kind = manifest["kind"]
    runtime = manifest["runtime"]
    model = manifest["model"]
    effort = manifest["effort"]
    # Resolved before either check (fixes branch review G1a): the dispatcher already
    # canonicalizes its target, but the worker used to guard the raw manifest string, so a
    # manifest target that was itself a symlink (e.g. review.md -> .env) passed the check
    # on its innocent-looking name and then read the secret through the link. Resolving
    # first makes both checks -- and the printed refusal -- agree with what actually gets
    # read.
    target_path = Path(manifest["target"]).resolve()

    # Defense-in-depth (fixes branch review F3/G1a): the dispatcher already refuses a path
    # outside the allowlist or a secret target before ever writing the manifest, but the
    # worker re-validates its own manifest's target with the same helpers before reading
    # it -- no read, no post, on a hit.
    if not _contained(target_path, allowlist_root):
        print("path-outside-allowlist", file=sys.stderr)
        return jc.REFUSED
    if _is_secret_path(target_path):
        print(f"secret-detected: {target_path}", file=sys.stderr)
        return jc.REFUSED

    paths = jr.safe_run_paths(repo, run_id)
    paths["scratch"].mkdir(parents=True, exist_ok=True)

    # Fixes acceptance smoke 2026-09-06 (fix 5, real reviewer rejection e8a614331d43): a
    # doc-review handoff never says the document itself is the object under review, so a
    # literal reviewer applies the diff-range missing-input rule and refuses. This framing
    # names the target and the review kind; the reviewer reads the document from disk.
    framing = (
        f"Document under review: {target_path}\n"
        f"Review kind: {kind} — this is a document review. "
        "Read the named document and required evidence from disk before reviewing. "
        "The document is the only object under review: it is both the "
        "evidence-of-should and evidence-of-is. There is no diff range and no "
        "separate spec/plan; write the document path in place of the reviewed range.\n"
    )
    target_text = target_path.read_text(encoding="utf-8")
    # spec §4.5's status.md gate rule needs to know whether the reviewed doc's own header
    # opts into Rafa's gate -- captured here (the only place the worker has this text),
    # read back by `_update_status_md` via `manifest["spec_gate_required"]`.
    manifest["spec_gate_required"] = any(
        line.strip().lower().startswith("rafa gate: required")
        for line in target_text.splitlines()[:10]
    )
    handoff = framing
    if manifest.get("threat_model"):
        handoff = f"{handoff}\n\n{jk.threat_model_line(manifest['threat_model'])}\n"
    if manifest.get("focus"):
        handoff = f"{handoff}\n\n--focus: {manifest['focus']}\n"
    # Fixes acceptance smoke 2026-09-06 (fix 2): names the marker file the dispatcher
    # already wrote (DOC_REVIEW_TEST_MARKER) as this review's test-output evidence -- a
    # `--spec`/`--plan` review has no diff/handoff `test-output:` line of its own.
    handoff = (
        f"{handoff}\n\nTest-output evidence: {paths['tests']} "
        "(doc review — no commands were required).\n"
    )
    prompt_text = jk.REVIEWER_PROMPT_PREAMBLE + jr.assemble_prompt("reviewer", {
        "run_id": run_id, "project": project, "role": "reviewer", "phase": phase,
        "report": str(paths["report"]), "tests": str(paths["tests"]),
        "scratch": str(paths["scratch"]), "repo": str(repo),
    }, handoff)
    prompt_path = paths["scratch"] / "prompt.txt"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    # MOA-467: the read grants add the canonical control repo when this review started in
    # a linked worktree, and keep the existing explicitly validated external-document
    # parent grant (its parent only, never its repo) when the document lives outside
    # every granted root -- deduplicated, no sibling-repo grants. Codex argv unchanged.
    argv = jr.runtime_argv(
        runtime, "reviewer", repo, prompt_path, paths["reviewer_output"],
        model=model, effort=effort,
        extra_read_dirs=jk._review_read_roots(
            repo, repo, run=run, allowlist_root=allowlist_root, documents=(target_path,),
        ),
    )
    stdout_file, stdout_target, stderr_target = jk._reviewer_stdio(runtime, paths["reviewer_output"])

    # Handlers are installed BEFORE Popen (fixes cold review F6): `tmux kill-session`
    # delivers the signal to this pane's process, and a fast kill could otherwise land
    # before the child even exists. `child_box` (not a bare `child` closure variable) plus
    # the `is not None` check below is what lets the handler tolerate that race as a
    # plain no-op instead of an `UnboundLocalError`/`AttributeError`.
    child_box = {"child": None}
    terminated = {"flag": False}

    # MOA-474 cold review 887b1d3994d4 F1: bind `log_path` BEFORE registering the
    # handlers -- a signal in the old window raised NameError inside the handler and
    # exited with no terminal row.
    log_path = jr.child_log_path(repo, run_id)
    jk._install_worker_signal_handler(
        manifest, role="reviewer", kind=kind, outcome="no verdict", run=run, post=post,
        killpg=killpg, run_id=run_id, project=project, phase=phase, spool_repo=repo,
        log_path=log_path, child_box=child_box, terminated=terminated,
    )

    status, child, child_log_text = jk._spawn_and_drain(
        popen, argv, cwd="/tmp", prompt_path=prompt_path, env=env, log_path=log_path,
        child_box=child_box, terminated=terminated, stdout_target=stdout_target,
        stderr_target=stderr_target, stdout_file=stdout_file, drain_stderr=runtime == "claude",
    )
    if status == "missing-cli":
        return jk._refuse_diff_run(
            f"missing-cli: {argv[0]}", manifest=manifest, run=run, post=post, manifest_path=manifest_path,
        )
    if status == "terminated":
        return 0

    return jk._finish_reviewer_worker(
        manifest, run=run, post=post, repo=repo, paths=paths, run_id=run_id, project=project,
        phase=phase, kind=kind, child=child, child_log_text=child_log_text,
        status_first=True, allowlist_root=allowlist_root,
    )
