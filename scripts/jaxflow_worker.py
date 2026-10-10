"""jaxflow worker module -- P2 split of scripts/jaxflow.py (lean spec, Decision 18)."""
from __future__ import annotations

import json
import os
import selectors
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
import jaxflow_run as jr
import jaxflow_resume as jresume
import jaxflow_settings as jset
from jax_init import ALLOWLIST_ROOT_DEFAULT, Refusal, _contained, _is_secret_path
import jaxflow_common as jc
import jaxflow_workerkit as jk
import jaxflow_review as jv


def _reap_owned(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def _pure_config_run(argv, cwd=None, env=None, *, cap=None, deadline=None):
    cap = jset.CONFIG_STDOUT_CAP if cap is None else cap
    budget = 10 if deadline is None else deadline
    try:
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise Refusal("agent-profile-conflict") from exc
    chunks = []
    total = 0
    sel = None
    try:
        stdout = proc.stdout
        os.set_blocking(stdout.fileno(), False)
        sel = selectors.DefaultSelector()
        sel.register(stdout, selectors.EVENT_READ)
        end = time.monotonic() + budget
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise Refusal("agent-profile-conflict")
            ready = sel.select(timeout=remaining)
            if not ready:
                raise Refusal("agent-profile-conflict")
            try:
                data = stdout.read(min(65536, cap + 1 - total))
            except BlockingIOError:
                continue
            if not data:
                break
            if total + len(data) > cap:
                raise Refusal("agent-profile-conflict")
            chunks.append(data)
            total += len(data)
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise Refusal("agent-profile-conflict")
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise Refusal("agent-profile-conflict") from exc
        return SimpleNamespace(returncode=proc.returncode, stdout=b"".join(chunks), stderr=b"")
    except Exception as exc:
        _reap_owned(proc)
        if isinstance(exc, Refusal):
            raise
        raise Refusal("agent-profile-conflict") from exc
    finally:
        if sel is not None:
            sel.close()
        if proc.stdout is not None:
            proc.stdout.close()


def _persist_launch_selection(path, manifest, selection, *, control_repo):
    record = {key: selection[key] for key in jc._LAUNCH_KEYS}
    if set(record) != set(jc._LAUNCH_KEYS):
        raise Refusal("agent-profile-conflict")
    run_id = manifest.get("run_id")
    if type(run_id) is not str or not jc._RUN_ID_RE.match(run_id):
        raise Refusal("path-outside-allowlist")
    expected = jc._manifest_dir(Path(control_repo), run_id) / "manifest.json"
    path = Path(path)
    if os.path.abspath(str(path)) != os.path.abspath(str(expected)):
        raise Refusal("path-outside-allowlist")
    jset._reject_symlink_parents(path)
    try:
        info = path.lstat()
    except OSError as exc:
        raise Refusal("agent-settings-permissions") from exc
    if stat.S_ISLNK(info.st_mode):
        raise Refusal("agent-settings-symlink")
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise Refusal("agent-settings-permissions")
    parent_fd = jset._open_directory_nofollow(path.parent, create=False)
    tmp_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    fd = None
    try:
        fd = os.open(
            tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600,
            dir_fd=parent_fd,
        )
        os.fchmod(fd, 0o600)
        payload = dict(manifest)
        payload["launch_selection"] = record
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        fd = None
        jset._require_same_directory(parent_fd, path.parent)
        os.replace(tmp_name, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        manifest["launch_selection"] = record
    except Exception:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp_name, dir_fd=parent_fd)
        except OSError:
            pass
        raise
    finally:
        os.close(parent_fd)


def _resolve_managed_launch(manifest, worktree, manifest_path, env, control_repo):
    with jset.configuration_claim(jset.LOCK_PATH):
        settings = jset.read_settings()
        if settings is None:
            raise Refusal("agent-settings-uninitialized")
        jset.require_source_revisions(settings, jset.SOURCE_PATHS)
        profile_name = manifest.get("requested_profile")
        if profile_name not in ("default", "fallback"):
            raise Refusal("agent-profile-conflict")
        profile = settings["builders"][profile_name]
        child_env = jset.builder_environment(profile, env, env_path=jset.ENV_PATH)
        effective = jset.read_effective_opencode_config(
            run=_pure_config_run, env=child_env, repo=worktree,
        )
        selection = jset.resolve_profile(
            settings, profile_name, effective_config=effective,
        )
        _persist_launch_selection(
            manifest_path, manifest, selection, control_repo=control_repo,
        )
    return child_env, selection


def _builder_started_payload(run_id, *, db_path=None):
    db_path = db_path or jr.DB_PATH
    try:
        con = jc._open_ro(db_path)
    except sqlite3.Error:
        return None
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            "SELECT payload FROM workflow_events WHERE run_id = ? AND role = 'builder' "
            "AND type = 'run-started' LIMIT 1",
            (run_id,),
        ).fetchone()
    finally:
        con.close()
    if row is None:
        return None
    try:
        payload = json.loads(row["payload"])
    except Exception:
        return None
    return payload if type(payload) is dict else None


def _treats_as_managed(manifest):
    return (
        manifest.get("runtime") == "opencode-builder"
        or manifest.get("requested_profile") in ("default", "fallback")
    )


def _managed_lineage_agrees(manifest, started):
    if type(started) is not dict:
        return False
    try:
        if Path(started.get("repo", "")).resolve() != Path(manifest.get("repo", "")).resolve():
            return False
    except OSError:
        return False
    for key in (
        "runtime", "session", "requested_profile", "model", "effort",
        "root_build_run_id", "resumes_run_id",
    ):
        if started.get(key) != manifest.get(key):
            return False
    return started.get("target") == manifest.get("target")


def finalize_builder_report(paths, run_id, project, phase, *, worktree):
    # Unlike a reviewer's report (produced as raw LLM output text and finalized via
    # atomic_finalize, spec §2.4), a builder writes report.md itself, directly, at the
    # already-injected path -- there is nothing to move, only to validate and lock.
    path = paths["report"]

    def _open_nofollow():
        # Final canonical containment check plus a no-follow open, redone immediately
        # before EACH access below (fixes F5, RECURRENCE(part3-F2)): `resolve()` catches
        # an ancestor symlink that escaped the worktree; `O_NOFOLLOW` then refuses the
        # leaf itself being a symlink no matter where it points -- including a target
        # still INSIDE the worktree, which containment alone would miss (the original
        # Part 3 diff-review F2 probe). The final inode comparison narrows the TOCTOU
        # window between the open and this check. A symlink (ELOOP) or a vanished path
        # both surface as a plain `OSError` here -- never a crash.
        resolved = path.resolve()
        if not _contained(resolved, worktree):
            return None
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            return None
        if os.fstat(fd).st_ino != os.lstat(path).st_ino:
            os.close(fd)
            return None
        return fd

    if not path.exists() and not path.is_symlink():
        return "missing", "Report missing", None

    fd = _open_nofollow()
    if fd is None:
        return "invalid", None, None
    try:
        with os.fdopen(fd, "rb") as f:
            text = f.read().decode("utf-8")
    except (OSError, UnicodeError):
        return "missing", "Report missing", None

    status, parsed, outcome = jr.validate_report(text, run_id, project, "builder", phase)
    if status != "ok":
        return "invalid", "Report invalid", outcome

    # Re-opened, not reused: a symlink planted between the read above and the lock below
    # must be caught here too, not silently followed because the read's own check already
    # passed (fixes F5).
    fd2 = _open_nofollow()
    if fd2 is None:
        return "invalid", None, outcome
    try:
        os.fchmod(fd2, 0o444)
    except OSError:
        # fixes Part 3 diff-review F5: a report that fails to lock is not immutable, so
        # it is not a valid report per spec §2.8 -- a chmod failure must not silently
        # return ("ok", parsed) as if validation and locking both succeeded.
        return "invalid", None, outcome
    finally:
        os.close(fd2)
    return "ok", parsed, outcome


def _build_builder_handoff(*, plan_dest, spec_dest, agents_path, branch, head_sha,
                            whitelist, verify_cmd, build_cmd=None, read_roots=()):
    """Composes a builder-contract/orchestrator-handoff.md-shaped package. No
    goal/tasks/acceptance extraction any more (decision 7) -- the builder reads
    `paths.plan` whole and derives its own goal/task breakdown from it."""
    return (
        f"Project AGENTS.md: {agents_path}\n"
        f"Branch: {branch}\n"
        f"target: {branch}\n"
        f"HEAD: {head_sha}\n"
        "Never print secrets (API keys, tokens, credentials) anywhere in the report or output.\n\n"
        "read-scope:\n"
        f"{''.join(f'  - {root}\n' for root in read_roots)}"
        "paths:\n"
        f"  spec: {spec_dest}\n"
        f"  plan: {plan_dest}\n"
        "  refs: none\n"
        f"assumptions: The plan is the sole authority; the repo is at `{branch}` @ `{head_sha}`.\n"
        "commands:\n"
        f"  test: {verify_cmd}\n"
        f"  build: {build_cmd or 'none'}\n"
        "verification: rafa-verifies\n"
        "out-of-scope: Anything the plan does not name. No push, no merge, no rebase.\n\n"
        f"whitelist: {', '.join(whitelist)}\n"
    )


def _refuse_builder_run(message, *, manifest, run, post, control_repo, worktree, branch):
    """Every builder-worker refusal that fires BEFORE the child ever launches (fixes Part 3
    diff-review F3) used to just print to stderr and leave the dispatched reservation
    behind -- the worktree (scratch dir included, once past `safe_run_paths`), its branch,
    its Git worktree admin entry, and the run's manifest dir. This posts the same
    cancelled-payload shape `_dispatch_run`'s own tmux-failed path already posts as the
    run's terminal row, then undoes every one of those, so a pre-launch refusal leaves
    nothing reserved behind, exactly like a dispatch-side failure already does. Only ever
    called once `worktree`/`branch`/`control_repo` are already known-good (F1's own
    manifest-identity check ran first and passed) -- never on an unvalidated, possibly
    forged path. `_cleanup_worktree`'s own `git worktree remove --force` (plus its
    fallback `shutil.rmtree`) already deletes the ENTIRE worktree tree, scratch dir
    included -- only the manifest dir (under the CONTROL repo, a separate location) needs
    its own removal here. fixes S1: a pre-launch refusal is a FINISHED run for the caller
    too -- `run_worker`'s own `_update_status_md` call runs on `manifest` right after this
    returns, and the caller's pane gets the same `[JAXFLOW]` line a launched-then-finished
    run would have sent."""
    print(message, file=sys.stderr)
    run_id = manifest["run_id"]
    bounded = jr._bound(message, 200)
    payload = {
        "phase": manifest["phase"], "exit_code": None, "contract_status": "cancelled",
        "report_path": None, "summary": bounded, "head_sha": None,
    }
    finished_event = {
        "run_id": run_id, "project": manifest["project"], "role": "builder",
        "type": "run-finished", "source": "deterministic", "emitter": "wrapper", "payload": payload,
    }
    run_dir = jc._manifest_dir(control_repo, run_id)
    spool_path = run_dir / "finished.json"
    jc._write_spool_at(spool_path, finished_event)
    delivered = True
    try:
        post(finished_event)
    except Exception:
        delivered = False
    if delivered:
        spool_path.unlink(missing_ok=True)
    started = _builder_started_payload(run_id)
    fresh = (
        type(started) is dict
        and not started.get("resumes_run_id")
        and started.get("root_build_run_id") in (None, run_id)
    )
    if _treats_as_managed(manifest):
        owns_reservation = (
            fresh
            and _managed_lineage_agrees(manifest, started)
            and not (run_dir / "resume-checkpoint.json").exists()
        )
    else:
        owns_reservation = started is None or fresh
    if owns_reservation:
        jc._cleanup_worktree(worktree, branch, run=run, repo=control_repo)
        # MOA-474 F1: never remove the run dir while a pending spool still lives in it
        # -- a delivery failure above means the spool is A2's only way to recover this
        # terminal row (reaper/`jaxflow cancel` re-POST it later). Clean up the
        # directory only once the row is confirmed delivered.
        if delivered:
            shutil.rmtree(run_dir, ignore_errors=True)
    manifest["worker_summary"] = bounded
    manifest["worker_contract_status"] = "cancelled"
    manifest["worker_outcome"] = None
    jk._send_callback(
        manifest, run=run, kind=manifest["kind"], outcome="cancelled", summary=bounded,
        report_path=None, ledger_pending=not delivered,
    )
    return jc.REFUSED


def _refuse_unvalidated_builder_run(message, *, manifest, manifest_path, run, post):
    """A `_validate_run_paths` failure for the builder worker (fixes RECURRENCE(part3-F3),
    F1): unlike every refusal `_refuse_builder_run` handles, the manifest's own
    `worktree`/`branch`/`repo` fields cannot be trusted for cleanup here -- validation is
    exactly what just failed. The only trustworthy facts left are the run_id (whatever it
    is, well-formed or not) and the manifest's own on-disk location -- the argv path this
    worker was actually invoked with, not anything derived from the manifest body. Posts
    the same cancelled-payload terminal row every other pre-launch refusal posts, for
    this run_id regardless of its shape (the hub's own terminal-claim 409 on
    `POST /api/workflow/events` protects a real run if this forged id happens to collide
    with one) and sends the usual `[JAXFLOW]` callback (fixes S1). Never deletes a
    supplied argv parent, worktree, or branch -- an unvalidated path is not ownership."""
    print(message, file=sys.stderr)
    bounded = jr._bound(message, 200)
    payload = {
        "phase": manifest.get("phase"), "exit_code": None, "contract_status": "cancelled",
        "report_path": None, "summary": bounded, "head_sha": None,
    }
    finished_event = {
        "run_id": manifest.get("run_id"), "project": manifest.get("project"),
        "role": manifest.get("role", "builder"), "type": "run-finished",
        "source": "deterministic", "emitter": "wrapper", "payload": payload,
    }
    spool_path = Path(manifest_path).resolve().parent / "finished.json"
    jc._write_spool_at(spool_path, finished_event)
    delivered = True
    try:
        post(finished_event)
    except Exception:
        delivered = False
    if delivered:
        spool_path.unlink(missing_ok=True)
    manifest["worker_summary"] = bounded
    manifest["worker_contract_status"] = "cancelled"
    manifest["worker_outcome"] = None
    jk._send_callback(
        manifest, run=run, kind=manifest.get("kind"), outcome="cancelled", summary=bounded,
        report_path=None, ledger_pending=not delivered,
    )
    return jc.REFUSED


def _validate_run_paths(manifest, *, allowlist_root):
    """Identity/containment check shared by both worker kinds (fixes Part 2 diff-review
    F1, reusing Part 3's builder-only guard instead of duplicating it): `run_id` must be
    the exact `uuid.uuid4().hex[:12]` shape, and `repo`/`worktree` must both canonicalize
    under the allowlist root, with `worktree` equal to the SAME derivation `build`/
    `review --diff` itself computed from this manifest's own `project`/`target` (both
    manifest shapes record the branch under the `target` key). Also requires
    `manifest["branch"]` (fixes F4, REGRESSION), when present, to equal `target` -- a
    builder manifest with a `branch` naming some unrelated real branch must never reach
    a caller that trusts that field for cleanup identity. Returns `(control_repo,
    worktree)` -- both resolved -- on success, or `None` on any mismatch. No cleanup is
    attempted here on a mismatch: nothing about a forged manifest's paths can be trusted
    as a safe `_cleanup_worktree` target."""
    run_id = manifest["run_id"]
    if not jc._RUN_ID_RE.match(run_id):
        return None
    control_repo = Path(manifest["repo"]).resolve()
    worktree = Path(manifest["worktree"]).resolve()
    expected_worktree = jc._branch_worktree_path(allowlist_root, manifest["project"], manifest["target"])
    branch = manifest.get("branch")
    if (
        not _contained(control_repo, allowlist_root)
        or not _contained(worktree, allowlist_root)
        or worktree != expected_worktree
        or (branch is not None and branch != manifest["target"])
    ):
        return None
    return control_repo, worktree


def _builder_read_roots(control_repo, worktree, *, run, allowlist_root, documents=()):
    """The deduplicated read-only scope a builder handoff names (MOA-467 acceptance
    fixes): the validated control repo and assigned worktree, plus the parent of each
    named document (original plan, declared spec) that lives outside both -- its own
    parent only, never the whole allowlist or an unrelated sibling repo. The documents
    are already validated files by the time this runs; this helper only decides which
    directories to NAME in the handoff, never whether a read is allowed."""
    roots = []
    for candidate in (control_repo, worktree, jk._control_repo_of(control_repo, run, allowlist_root)):
        resolved = Path(candidate).resolve()
        if resolved not in roots:
            roots.append(resolved)
    for document in documents:
        parent = Path(document).resolve().parent
        if any(_contained(parent, owner) for owner in roots):
            continue
        roots.append(parent)
    return tuple(roots)


def _manifest_identity_error(manifest_path, control_repo, run_id):
    """Why `manifest_path` is not the genuine manifest of this run, as a refusal code, or
    None. It must be the path `build` wrote for `run_id` in the control repo, a regular file
    (never a symlink, never behind a symlinked parent) owned by this user."""
    expected_manifest = jc._manifest_dir(control_repo, run_id) / "manifest.json"
    if os.path.abspath(str(manifest_path)) != os.path.abspath(str(expected_manifest)):
        return "path-outside-allowlist"
    try:
        jset._reject_symlink_parents(Path(manifest_path))
        info = Path(manifest_path).lstat()
        if stat.S_ISLNK(info.st_mode):
            return "agent-settings-symlink"
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            return "agent-settings-permissions"
    except Refusal as exc:
        return exc.code
    except OSError:
        return "agent-settings-permissions"
    return None


def _prepare_builder_launch(manifest, *, run, env, allowlist_root, manifest_path, control_repo,
                            worktree, branch, whitelist, verify_cmd, build_cmd):
    """Everything a builder launch needs BEFORE `popen`: worker-side preflight, plan-path
    re-validation, resume-lineage checks, managed launch selection, the handoff and the
    prompt file. Returns `(argv, prompt_path, paths, child_env)`. Any pre-launch failure
    RAISES (`Refusal(code)`, `KeyError`, ...): the caller's single `except Exception` turns
    it into one `_refuse_builder_run` (Refusal -> its code, KeyError -> `manifest missing
    <key>`, anything else -> `str(exc)`)."""
    run_id = manifest["run_id"]
    project = manifest["project"]
    phase = manifest["phase"]
    runtime = manifest["runtime"]
    handoff_branch = manifest.get("branch", branch)
    child_env = env
    launch_selection = None

    # Worker-side preflight (fixes cold review round 2 F2, spec §2.4 worker step 2):
    # the dispatcher already preflighted the worktree once, moments earlier, but the
    # worker re-validates its own manifest independently -- the only guard against a
    # hand-edited manifest or a `--run-worker` invocation outside the normal dispatch
    # path.
    args_ns = SimpleNamespace(
        role="builder", project=project, phase=phase, repo=str(worktree),
        prompt_file=None, runtime=runtime, callback=None,
    )
    try:
        jr.preflight(args_ns, run=run, allow_untracked=True, require_feat_branch=False, skip_handoff=True)
    except ValueError as exc:
        raise Refusal(jc._map_refusal(str(exc))) from exc

    # Defense-in-depth (fixes cold review round 2 F1; amended by the MOA-467
    # acceptance fixes): the worker re-validates its own manifest's plan_path with
    # the SAME policy `dispatch_build` applied -- canonicalized, non-secret,
    # readable, inside the allowlist. A directly named plan path IS a read input,
    # so an explicitly named external document (another allowlist repo, a related
    # worktree) is legitimate: refusing it here while `build` accepted it was the
    # regression. Forged paths outside the allowlist, secrets, missing files and
    # symlink escapes all still refuse. Secret-shaped paths refuse FIRST -- a
    # symlink planted inside the worktree that resolves to a control-repo `.env`
    # must refuse even though it is "contained".
    plan_path = Path(manifest["plan_path"]).resolve()
    defect = jc._plan_path_defect(plan_path, allowlist_root)
    if defect == "secret-detected":
        raise Refusal(f"secret-detected: {plan_path}")
    if defect:
        raise Refusal(defect)
    resume_start = manifest.get("resume_start")
    if resume_start is not None:
        jresume.require_same_work_state(
            resume_start, jresume.capture_work_state(worktree, plan_path, run=run),
        )
    if _treats_as_managed(manifest):
        try:
            con = jc._open_ro(jr.DB_PATH)
        except sqlite3.Error as exc:
            raise Refusal("resume-ineligible") from exc
        con.row_factory = sqlite3.Row
        try:
            latest = jc._latest_builder_attempt(con, manifest["project"], control_repo, branch)
        finally:
            con.close()
        if latest != run_id:
            raise Refusal("resume-ineligible")

    paths = jr.safe_run_paths(worktree, run_id)
    paths["scratch"].mkdir(parents=True, exist_ok=True)

    if runtime == "opencode-builder":
        child_env, launch_selection = _resolve_managed_launch(
            manifest, worktree, manifest_path, env, control_repo,
        )

    # Contract-valid builder handoff, derived from the plan's own text (round-1 cold
    # review F2 -- MOA-467: the plan is the ORIGINAL document in the validated read
    # scope, neither copied into the worktree nor re-copied here).
    plan_text = plan_path.read_text(encoding="utf-8")
    spec_dest, spec_refusal = jc._resolve_handoff_spec_path(plan_text, plan_path, worktree, allowlist_root)
    if spec_refusal:
        # fixes cold review round 2 F5: a DECLARED but bad **Spec:** reference
        # refuses loudly instead of silently handing the builder the plan itself as
        # its own spec -- only the absence of a **Spec:** line at all falls back
        # that way.
        raise Refusal(spec_refusal)
    head_probe = run(["git", "rev-parse", "HEAD"], cwd=worktree)
    head_for_handoff = head_probe.stdout.strip() if head_probe.returncode == 0 else "unknown"
    handoff = _build_builder_handoff(
        plan_dest=plan_path, spec_dest=spec_dest,
        agents_path=worktree / "AGENTS.md", branch=handoff_branch, head_sha=head_for_handoff,
        whitelist=whitelist, verify_cmd=verify_cmd, build_cmd=build_cmd,
        read_roots=_builder_read_roots(
            control_repo, worktree, run=run, allowlist_root=allowlist_root,
            documents=(plan_path, spec_dest),
        ),
    )
    prompt_text = jr.assemble_prompt("builder", {
        "run_id": run_id, "project": project, "role": "builder", "phase": phase,
        "report": str(paths["report"]), "tests": str(paths["tests"]),
        "scratch": str(paths["scratch"]), "repo": str(worktree),
    }, handoff)
    prompt_path = paths["scratch"] / "prompt.txt"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    # `model`/`effort` are converted back from their ledger sentinels to a real
    # override-or-None here (fixes cold review F4) -- "default"/"n/a" mean "no
    # override was given", never a literal value to pass to the runtime.
    if launch_selection is not None:
        model_override = launch_selection["runtime_model"]
        effort_override = launch_selection["effort"]
    else:
        model = manifest["model"]
        effort = manifest["effort"]
        model_override = None if model == "default" else model
        effort_override = None if effort == "n/a" else effort
    argv = jr.runtime_argv(
        runtime, "builder", worktree, prompt_path, paths["reviewer_output"],
        model=model_override, effort=effort_override,
    )
    return argv, prompt_path, paths, child_env


def _builder_finish_payload(manifest, *, run, paths, run_id, project, phase, worktree,
                            verify_cmd, build_cmd, child, child_log_text):
    """Builder contract finalize, jaxflow's OWN post-build verification (both commands run;
    the build is never skipped because the test failed), outcome derivation and the
    run-finished payload. Returns `(payload, summary, contract_status, outcome,
    verify_contradicts)`. The resume checkpoint, delivery, manifest `worker_*` fields and
    the callback stay with the caller, in that order."""
    contract_status, summary, report_result = finalize_builder_report(
        paths, run_id, project, phase, worktree=worktree)

    head_sha = None
    head = run(["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=worktree)
    if head.returncode == 0 and jr.SHA_RE.match(head.stdout.strip()):
        head_sha = head.stdout.strip()

    # jaxflow's OWN post-build verification (spec §4.2 step 4) -- distinct from and in
    # addition to whatever the builder ran per its own handoff's commands. BOTH commands
    # run; the build is never skipped because the test failed.
    verify_frames = jk._run_verify_commands(run, worktree, verify_cmd, build_cmd)
    tests_written = jk._write_verify_tests_file(paths["tests"], verify_frames, worktree=worktree)

    exit_code = child.returncode if 0 <= child.returncode <= 255 else 1
    base_sha = manifest.get("base_sha")
    is_descendant = (
        head_sha is not None and isinstance(base_sha, str)
        and jc._is_strict_descendant(run, worktree, base_sha, head_sha)
    )
    # MOA-474 D5/§7: the last non-empty PHYSICAL line of the redacted child.log --
    # never the whole stream (criterion 2: a recovered mid-stream error is not
    # terminal evidence, only the LAST line can ever be).
    stream_last_line = next(
        (ln for ln in reversed(child_log_text.splitlines()) if ln.strip()), "",
    )
    outcome = jr.derive_outcome({
        "role": "builder", "report_result": report_result,
        "report_summary": summary, "contract_status": contract_status,
        "stream_last_line": stream_last_line, "exit_code": exit_code, "signal": None,
        "verify_frames": verify_frames, "tests_written": tests_written,
        "head_sha": head_sha, "base_sha": base_sha, "is_descendant": is_descendant,
        # MOA-474 cold review F4: wires _last_stream_text_line into rows 5/10's
        # diagnostic (§13) -- without this, evidence.get("last_stream_text") always
        # defaults to None and the exit/signal diagnostic never has a real cause.
        "last_stream_text": jr._last_stream_text_line(child_log_text),
    })
    payload = {
        "phase": phase, "exit_code": exit_code, "contract_status": contract_status,
        "report_path": str(paths["report"]), "summary": jr._bound(summary, 200),
        "head_sha": head_sha, "result": outcome["result"],
    }
    if outcome["stage"] is not None:
        payload["stage"] = outcome["stage"]
        payload["diagnostic"] = outcome["diagnostic"]
    # MOA-470 §4.3: both keys travel together, emitted iff contract_status needed a
    # diagnosis -- never for "ok" (verify's own result covers that).
    jk._log_diagnosis(payload, child_log_text, exit_code=exit_code, contract_status=contract_status)
    # Cold review F2: last-line-of-defense against the ingress validator's 16 KiB
    # payload byte ceiling -- drops log_context (then tally, N/A on this branch)
    # if the combination still overflows it despite the byte-bounded chunks above.
    jr.cap_payload_bytes(payload)
    # MOA-455, unaffected by this change: a claimed failure the verify chain did
    # NOT confirm is flagged on the callback line ONLY, never touching the ledger
    # `result`. Unlike the pre-MOA-474 code, this note is orthogonal to
    # derive_outcome's own veto: it only ever fires when the report claimed
    # failure and jaxflow's own checks disagree (the opposite direction from the
    # veto, which overrides a claimed SUCCESS).
    verify_failed_now = any(r.returncode != 0 for _, r in verify_frames)
    verify_contradicts = (
        report_result == "failure" and not verify_failed_now and tests_written
    )
    return payload, summary, contract_status, outcome, verify_contradicts


def _validate_diff_worker_inputs(manifest, *, run, allowlist_root, worktree, builder_run_id,
                                 base_sha, head_sha):
    """Every pre-launch check of the diff worker. Returns `(message, tests_path, plan_path,
    spec_path)`: `message` is the refusal text for `_refuse_diff_run` (None when all checks
    passed -- then the other three are the validated values). A failing `git diff
    --name-only` still RAISES RuntimeError (not a refusal), exactly as before."""
    # fixes F6 (HIGH, NEW): the evidence path is DERIVED from the validated worktree plus
    # `builder_run_id` -- exactly as `dispatch_diff_review` itself derives it (spec §4.3
    # step 1) -- never trusted verbatim from the manifest body. `builder_run_id` itself is
    # shape-checked first (it is embedded into a filesystem path below); a manifest
    # `tests_path` that disagrees with the derived path is a forged or stale value and
    # refuses rather than being echoed into the reviewer prompt.
    if not jc._RUN_ID_RE.match(builder_run_id):
        return "path-outside-allowlist", None, None, None
    tests_path = worktree / ".local" / "reports" / f"{builder_run_id}.tests.txt"
    manifest_tests_path = manifest.get("tests_path")
    if manifest_tests_path is not None and Path(manifest_tests_path).resolve() != tests_path.resolve():
        return "path-outside-allowlist", None, None, None

    # fixes S2, amended MOA-467 (acceptance fixes): re-validates plan_path/spec_path --
    # the build's own handoff inputs -- before any read (defense-in-depth against a
    # hand-edited manifest or a `--run-worker` invocation outside the normal dispatch
    # path). The plan must be a real readable non-secret file inside the allowlist --
    # the same policy `dispatch_build` applied, so an explicitly named external plan
    # (another allowlist repo, a related worktree) survives the build AND the later
    # diff review. The spec is NOT trusted verbatim from the manifest: it must equal the
    # one the plan itself declares (re-derived here from the validated plan text, so a
    # forged manifest can never inject an arbitrary spec path) -- except for the legacy
    # worktree-copy shape, which stays accepted exactly as it always was.
    raw_plan = manifest.get("plan_path")
    try:
        plan_path = Path(raw_plan).resolve() if raw_plan else None
    except OSError:
        plan_path = None
    defect = jc._plan_path_defect(plan_path, allowlist_root)
    if defect == "secret-detected":
        return f"secret-detected: {plan_path}", None, None, None
    if defect:
        return defect, None, None, None
    raw_spec = manifest.get("spec_path")
    try:
        spec_path = Path(raw_spec).resolve() if raw_spec else None
    except OSError:
        spec_path = None
    legacy_copy_ok = (
        spec_path is not None and not _is_secret_path(spec_path)
        and spec_path.is_file() and os.access(spec_path, os.R_OK)
        and _contained(spec_path, worktree)
    )
    if not legacy_copy_ok:
        # New-style manifests: re-derive the spec from the validated plan text and
        # require the manifest to agree -- a forged spec path can never be injected.
        plan_text = plan_path.read_text(encoding="utf-8")
        derived_spec, spec_refusal = jc._resolve_handoff_spec_path(plan_text, plan_path, worktree, allowlist_root)
        if spec_refusal:
            return spec_refusal, None, None, None
        # str()-compared, not Path-compared (MOA-471 item 2): `derived_spec` is a plain
        # str, not a Path, whenever the plan's **Spec:** line carries a resolved
        # `#<heading>` fragment -- comparing Path != str would always be unequal and
        # falsely refuse every fragmented spec even when both sides name the same value.
        if str(spec_path) != str(derived_spec):
            return "path-outside-allowlist", None, None, None
        spec_path = derived_spec

    # fixes Part 2 diff-review F2: the changed-path LIST is checked against the same
    # secret-path guard I4/jax_init uses BEFORE the full diff (which can carry secret file
    # CONTENTS, not just names) is ever read -- refusing loudly instead of embedding a
    # secret's value in the reviewer prompt.
    names_result = run(["git", "diff", "--name-only", f"{base_sha}..{head_sha}"], cwd=worktree)
    if names_result.returncode != 0:
        raise RuntimeError(f"git diff --name-only {base_sha}..{head_sha} failed: {names_result.stderr}")
    secret_hit = next(
        (p for p in names_result.stdout.splitlines() if p and _is_secret_path(p)), None,
    )
    if secret_hit:
        return f"secret-detected: {secret_hit}", None, None, None
    return None, tests_path, plan_path, spec_path


def _build_diff_handoff(manifest, *, repo, spec_path, plan_path, tests_path, base_sha, head_sha):
    """The reviewer handoff text of a `--diff` review (grammar `parse_reviewer_handoff`
    parses: one `diff:` line, one `paths:` line, one indented `  test-output:` line).
    The diff itself is NOT embedded: the reviewer runs `git diff base..head` in the
    worktree (its cwd; Codex has a read-only shell, Claude has `Bash(git diff:*)`). A 2.7M
    character diff exceeded Codex's 1M input limit (MOA-506 P2). Returns the handoff string."""
    # Grammar `parse_reviewer_handoff` parses: exactly one `diff:` line, exactly one
    # `paths:` line, and exactly one indented `  test-output:` line in the block right
    # after it -- everything else in this text (the framing, the `spec:`/`plan:` lines
    # below, the instruction, --focus) is free-form and never scanned by that parser
    # (spec §4.3 step 3). fixes S2: `spec:`/`plan:` name the SAME evidence-of-should the
    # builder's own handoff named -- the reviewer contract's required spec/plan inputs.
    since_run_id = manifest.get("since_review_run_id")
    correction_header = ""
    prior_review_line = ""
    if since_run_id:
        # `_safe_run_subpath` (not jr.safe_run_paths, which raises on an existing
        # report) validates since_run_id; a forged/escaping value renders NEITHER
        # line, same as no since_run_id (F2/F6).
        prior_report = jc._safe_run_subpath(repo, "reports", since_run_id, suffix=".md")
        if prior_report is not None:
            since_verdict = manifest.get("since_verdict", "unknown")
            correction_header = f"Correction review of {since_run_id} (verdict {since_verdict})\n"
            prior_review_line = f"  prior-review: {prior_report}\n"
    handoff = (
        f"{correction_header}"
        f"diff: {base_sha}..{head_sha}\n"
        "paths:\n"
        f"  spec: {spec_path}\n"
        f"  plan: {plan_path}\n"
        f"  test-output: {tests_path}\n"
        f"{prior_review_line}"
        "\n"
        "The diff under review is NOT embedded in this prompt. Produce it yourself: run "
        f"`git diff {base_sha}..{head_sha}` in your working directory (the worktree); for a "
        f"large change start with `git diff --stat {base_sha}..{head_sha}` and then diff "
        "file by file. That output is the evidence-of-is the reviewer contract names.\n"
    )
    if manifest.get("threat_model"):
        handoff = f"{handoff}\n{jk.threat_model_line(manifest['threat_model'])}\n"
    if manifest.get("focus"):
        handoff = f"{handoff}\n--focus: {manifest['focus']}\n"
    return handoff


def _run_builder_worker(manifest, *, run, post, popen, killpg, env, allowlist_root, manifest_path):
    run_id = manifest["run_id"]
    project = manifest["project"]
    phase = manifest["phase"]
    runtime = manifest["runtime"]
    whitelist = manifest["whitelist"]
    verify_cmd = manifest["verify"]
    build_cmd = manifest.get("build")

    # Binds every path this worker builds to the run `build` actually dispatched (fixes
    # Part 3 diff-review F1, now shared with the diff worker via `_validate_run_paths`,
    # Part 2 diff-review F1) -- BEFORE any of these manifest fields is used to construct a
    # path. fixes F1 (RECURRENCE part3-F3): a failure here can trust neither `worktree`
    # nor `branch` from the manifest, so it routes through the lightweight helper instead
    # of `_refuse_builder_run` -- posting a terminal row and removing only this worker's
    # own manifest dir (the argv path, never a path built from the untrusted manifest).
    validated = _validate_run_paths(manifest, allowlist_root=allowlist_root)
    if validated is None:
        return _refuse_unvalidated_builder_run(
            "path-outside-allowlist", manifest=manifest, manifest_path=manifest_path,
            run=run, post=post,
        )
    control_repo, worktree = validated
    identity_error = _manifest_identity_error(manifest_path, control_repo, run_id)
    if identity_error:
        return _refuse_unvalidated_builder_run(
            identity_error, manifest=manifest, manifest_path=manifest_path,
            run=run, post=post,
        )
    # fixes F4 (REGRESSION): every subsequent refusal below cleans up `target` -- the
    # VALIDATED branch this worktree is actually checked out on -- never the manifest's
    # own free-standing `branch` field (kept only for the informational handoff text
    # below, where `_validate_run_paths` already guarantees it agrees with `target`
    # whenever it is present at all).
    branch = manifest["target"]
    claim = None
    if _treats_as_managed(manifest):
        started = _builder_started_payload(manifest["run_id"])
        if not _managed_lineage_agrees(manifest, started):
            return _refuse_unvalidated_builder_run(
                "resume-ineligible", manifest=manifest, manifest_path=manifest_path,
                run=run, post=post,
            )
        claim = jresume.worktree_claim(control_repo, worktree)
        try:
            claim.__enter__()
        except Refusal as exc:
            return _refuse_unvalidated_builder_run(
                exc.code, manifest=manifest, manifest_path=manifest_path,
                run=run, post=post,
            )
    try:
        # fixes F1, second half: every pre-launch failure from here through argv assembly
        # (a `safe_run_paths` collision, an unreadable `plan_path`, any other exception)
        # routes through `_refuse_builder_run` with the exception's own text as the message,
        # instead of an uncaught crash leaving the reservation behind.
        try:
            argv, prompt_path, paths, child_env = _prepare_builder_launch(
                manifest, run=run, env=env, allowlist_root=allowlist_root,
                manifest_path=manifest_path, control_repo=control_repo, worktree=worktree,
                branch=branch, whitelist=whitelist, verify_cmd=verify_cmd, build_cmd=build_cmd,
            )
        except Exception as exc:
            # fixes F1 (RECURRENCE, tech-lead triage): anything that fails before the child
            # `popen` is a refusal by definition, a bare `KeyError` (e.g. a manifest missing
            # "plan_path") included -- narrowing to `(Refusal, OSError, ValueError)` let a
            # KeyError escape uncaught, skipping cleanup entirely.
            if isinstance(exc, Refusal):
                message = exc.code
            elif isinstance(exc, KeyError):
                message = f"manifest missing {exc.args[0]}"
            else:
                message = str(exc)
            return _refuse_builder_run(
                message, manifest=manifest, run=run, post=post,
                control_repo=control_repo, worktree=worktree, branch=branch,
            )

        child_box = {"child": None}
        terminated = {"flag": False}

        # MOA-474 cold review 887b1d3994d4 F1: `_on_signal` above reads `log_path`, so
        # it must be bound BEFORE the handlers are registered -- a signal landing in the
        # window between registration and the old assignment raised NameError inside the
        # handler and exited with no terminal row.
        log_path = jr.child_log_path(control_repo, run_id)
        jk._install_worker_signal_handler(
            manifest, role="builder", kind="build", outcome="failure", run=run, post=post,
            killpg=killpg, run_id=run_id, project=project, phase=phase,
            spool_repo=control_repo, log_path=log_path, child_box=child_box,
            terminated=terminated, worktree=worktree, base_sha=manifest.get("base_sha"),
            persist=True,
        )

        # MOA-470 §4.2: OpenCode has no --output-* flag, so its stdout was never the
        # report channel -- free to redirect to a pipe. stderr merges into the same pipe
        # (matches how both already appear interleaved in a tmux pane today).
        status, child, child_log_text = jk._spawn_and_drain(
            popen, argv, cwd=str(worktree), prompt_path=prompt_path, env=child_env,
            log_path=log_path, child_box=child_box, terminated=terminated,
            stdout_target=subprocess.PIPE, stderr_target=subprocess.STDOUT,
        )
        if status == "missing-cli":
            return _refuse_builder_run(
                f"missing-cli: {argv[0]}", manifest=manifest, run=run, post=post,
                control_repo=control_repo, worktree=worktree, branch=branch,
            )
        if status == "terminated":
            return 0

        payload, summary, contract_status, outcome, verify_contradicts = _builder_finish_payload(
            manifest, run=run, paths=paths, run_id=run_id, project=project, phase=phase,
            worktree=worktree, verify_cmd=verify_cmd, build_cmd=build_cmd,
            child=child, child_log_text=child_log_text,
        )
        finished_event = {
            "run_id": run_id, "project": project, "role": "builder", "type": "run-finished",
            "source": "deterministic", "emitter": "wrapper", "payload": payload,
        }
        jk._persist_resume_checkpoint(manifest, control_repo, worktree, payload, run=run)
        delivered = jk._deliver_terminal_event(post, control_repo, run_id, finished_event)

        manifest["worker_summary"] = summary
        manifest["worker_contract_status"] = contract_status
        manifest["worker_outcome"] = payload.get("result")

        result_text = payload["result"]
        if verify_contradicts:
            result_text = f"{result_text} (jaxflow verify passed — read the evidence)"
        jk._send_callback(
            manifest, run=run, kind="build", outcome=result_text, summary=summary,
            report_path=paths["report"], stage=outcome["stage"], diagnostic=outcome["diagnostic"],
            contract_status=contract_status, ledger_pending=not delivered,
        )
        return 0
    finally:
        if claim is not None:
            claim.__exit__(None, None, None)


def _run_diff_reviewer_worker(manifest, *, run, post, popen, killpg, env, allowlist_root,
                               manifest_path):
    run_id = manifest["run_id"]
    project = manifest["project"]
    phase = manifest["phase"]
    runtime = manifest["runtime"]
    model = manifest["model"]
    effort = manifest["effort"]
    base_sha = manifest["base_sha"]
    head_sha = manifest["head_sha"]
    builder_run_id = manifest["builder_run_id"]

    # Binds every path this worker builds to the run `review --diff` actually dispatched
    # (fixes Part 2 diff-review F1, via the SAME helper `_run_builder_worker` uses) --
    # BEFORE any mkdir/read/subprocess. A mismatch posts the same cancelled-payload shape
    # `_refuse_builder_run` posts for its own pre-launch refusals, as this run's terminal
    # row -- but never touches the worktree: a diff review never owns it.
    validated = _validate_run_paths(manifest, allowlist_root=allowlist_root)
    if validated is None:
        return jk._refuse_diff_run("path-outside-allowlist", manifest=manifest, run=run, post=post, manifest_path=manifest_path)
    repo, worktree = validated

    message, tests_path, plan_path, spec_path = _validate_diff_worker_inputs(
        manifest, run=run, allowlist_root=allowlist_root, worktree=worktree,
        builder_run_id=builder_run_id, base_sha=base_sha, head_sha=head_sha,
    )
    if message:
        return jk._refuse_diff_run(message, manifest=manifest, run=run, post=post, manifest_path=manifest_path)

    # Reviewer runs always use the CONTROL repo for report/scratch paths (spec §2.4 step
    # 2), even though the reviewer's own cwd (below) is the worktree.
    paths = jr.safe_run_paths(repo, run_id)
    paths["scratch"].mkdir(parents=True, exist_ok=True)

    handoff = _build_diff_handoff(
        manifest, repo=repo, spec_path=spec_path, plan_path=plan_path, tests_path=tests_path,
        base_sha=base_sha, head_sha=head_sha,
    )
    prompt_text = jk.REVIEWER_PROMPT_PREAMBLE + jr.assemble_prompt("reviewer", {
        "run_id": run_id, "project": project, "role": "reviewer", "phase": phase,
        "report": str(paths["report"]), "tests": str(paths["tests"]),
        "scratch": str(paths["scratch"]), "repo": str(repo),
    }, handoff)
    prompt_path = paths["scratch"] / "prompt.txt"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    # `codex_cwd=worktree` (fixes cold review F4) roots Codex's OWN -C flag at the real
    # repo for a --diff review; a doc review's own call site (unchanged, below in this
    # file) never passes this kwarg, so it keeps today's /tmp sandbox exactly. MOA-467:
    # the extra read dirs grant the canonical control repo (new original-path plans and
    # specs live there) and -- only when the declared spec lives outside every granted
    # root -- that spec's own parent; never an unrelated sibling repo. Codex's argv is
    # untouched by these grants.
    argv = jr.runtime_argv(
        runtime, "reviewer", worktree, prompt_path, paths["reviewer_output"],
        model=model, effort=effort, codex_cwd=worktree,
        extra_read_dirs=jk._review_read_roots(
            worktree, repo, run=run, allowlist_root=allowlist_root,
            documents=(plan_path, spec_path),
        ),
    )
    stdout_file, stdout_target, stderr_target = jk._reviewer_stdio(runtime, paths["reviewer_output"])

    child_box = {"child": None}
    terminated = {"flag": False}

    # MOA-474 cold review 887b1d3994d4 F1: bind `log_path` BEFORE registering the
    # handlers -- a signal in the old window raised NameError inside the handler and
    # exited with no terminal row.
    log_path = jr.child_log_path(repo, run_id)
    jk._install_worker_signal_handler(
        manifest, role="reviewer", kind="diff", outcome="no verdict", run=run, post=post,
        killpg=killpg, run_id=run_id, project=project, phase=phase, spool_repo=repo,
        log_path=log_path, child_box=child_box, terminated=terminated,
    )

    # cwd = the worktree (spec §4.3 step 4), not /tmp -- unlike a doc review, this
    # reviewer's cwd is the real repo whose commits it is reviewing.
    status, child, child_log_text = jk._spawn_and_drain(
        popen, argv, cwd=str(worktree), prompt_path=prompt_path, env=env, log_path=log_path,
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
        phase=phase, kind="diff", child=child, child_log_text=child_log_text,
        status_first=False, allowlist_root=allowlist_root,
    )


def run_worker(manifest_path, *, run=jr.run_command, post=jc._post_event, popen=subprocess.Popen,
                killpg=os.killpg, env=None, allowlist_root=ALLOWLIST_ROOT_DEFAULT):
    # Sealed env (§2.3) is set on a COPY, never on the real os.environ (fixes cold review
    # F4 — the old in-place os.environ mutation was untestable and left every later test
    # in the same process permanently sealed too). The copy is what actually reaches the
    # child via Popen(env=...) below.
    env = dict(os.environ if env is None else env)
    env["HONCHO_ENABLED"] = "false"
    env["JAXFLOW_ONESHOT"] = "1"
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    worker_kwargs = dict(run=run, post=post, popen=popen, killpg=killpg, env=env,
                         allowlist_root=allowlist_root, manifest_path=manifest_path)

    if manifest["role"] == "builder":
        code = _run_builder_worker(manifest, **worker_kwargs)
    elif manifest["kind"] == "diff":
        code = _run_diff_reviewer_worker(manifest, **worker_kwargs)
    else:
        # The doc path updates status.md itself, BEFORE its callback (Decision 7).
        return jv._run_doc_reviewer_worker(manifest, **worker_kwargs)
    jc._update_status_md(manifest, run=run, allowlist_root=allowlist_root)
    return code
