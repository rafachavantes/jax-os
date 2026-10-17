"""jaxflow build module -- P2 split of scripts/jaxflow.py (lean spec, Decision 18)."""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import stat
import uuid
from pathlib import Path
from types import SimpleNamespace
import jaxflow_run as jr
import jaxflow_resume as jresume
import jaxflow_settings as jset
from jax_init import ALLOWLIST_ROOT_DEFAULT, Refusal, _contained, canonicalize_target
import jaxflow_common as jc


MANIFEST_CAP = 1 << 20

BUILDER_DEFAULT = "opencode-grok"


def _require_opencode_on():
    """MOA-504 D9: every builder profile is OpenCode, so one switch gates build and --resume."""
    if not jc._enabled_agents()["opencode"]:
        raise jc._refuse("agent-disabled: opencode", "hint: turn OpenCode on in /settings -> General")


def _read_prior_manifest(path):
    path = Path(path)
    try:
        parent_fd = jset._open_directory_nofollow(path.parent, create=False)
    except Refusal as exc:
        raise Refusal("resume-ineligible") from exc
    try:
        try:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        except OSError as exc:
            raise Refusal("resume-ineligible") from exc
    finally:
        os.close(parent_fd)
    try:
        st = os.fstat(fd)
        if (not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid()
                or stat.S_IMODE(st.st_mode) & 0o022):
            raise Refusal("resume-ineligible")
        raw = jresume._read_capped(fd, MANIFEST_CAP)
    finally:
        os.close(fd)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Refusal("resume-ineligible") from exc
    try:
        data = json.loads(
            text,
            object_pairs_hook=jset._reject_duplicates,
            parse_constant=jset._reject_constant,
        )
    except Refusal:
        raise Refusal("resume-ineligible")
    except (json.JSONDecodeError, ValueError, TypeError, RecursionError) as exc:
        raise Refusal("resume-ineligible") from exc
    if type(data) is not dict:
        raise Refusal("resume-ineligible")
    return data


def _git_common_dir(run, path):
    probe = run(["git", "rev-parse", "--git-common-dir"], cwd=path)
    raw = (probe.stdout or "").strip()
    if probe.returncode != 0 or not raw:
        return None
    found = Path(raw)
    if not found.is_absolute():
        found = Path(path) / found
    return found.resolve()


def _resume_rows(db_path, resume_id, latest_of=None):
    """The builder `run-started`/`run-finished` rows of the run being resumed, as `(started,
    finished, latest)`; `resume-ineligible` when the DB is unreadable or either row is
    missing. `latest_of=(project, repo, branch)` also resolves the latest attempt, on the
    SAME connection and BEFORE the missing-row refusal (the order the inline code had)."""
    try:
        con = jc._open_ro(db_path)
    except sqlite3.Error as exc:
        raise Refusal("resume-ineligible") from exc
    con.row_factory = sqlite3.Row
    try:
        started, finished = jc._builder_run_rows(con, resume_id)
        latest = jc._latest_builder_attempt(con, *latest_of) if latest_of else None
    finally:
        con.close()
    if not started or not finished:
        raise Refusal("resume-ineligible")
    return started, finished, latest


def _resume_load_prior(repo, resume_id, project, started, finished):
    """Decode the recorded payloads, read the prior manifest and the resume checkpoint, and
    refuse `resume-ineligible` unless they all describe ONE finished-failed opencode build of
    `project`. Returns `(started_payload, finished_payload, prior_path, prior, checkpoint)`."""
    try:
        started_payload = json.loads(started["payload"])
        finished_payload = json.loads(finished["payload"])
        prior_path = jc._manifest_dir(repo, resume_id) / "manifest.json"
        prior = _read_prior_manifest(prior_path)
        checkpoint = jresume.read_checkpoint(jc._manifest_dir(repo, resume_id) / "resume-checkpoint.json")
    except Refusal:
        raise
    except Exception as exc:
        raise Refusal("resume-ineligible") from exc
    if (started["project"] != project or type(prior) is not dict
            or started_payload.get("runtime") != "opencode-builder"
            or started_payload.get("kind") != "build"
            or prior.get("runtime") != "opencode-builder"
            or prior.get("kind") != "build" or prior.get("role") != "builder"
            or prior.get("run_id") != resume_id
            or started_payload.get("requested_profile") not in ("default", "fallback")
            or prior.get("requested_profile") not in ("default", "fallback")
            or prior.get("requested_profile") != started_payload.get("requested_profile")
            # MOA-474 §5.1 conflict 2: the explicit cancelled check is dropped -- an
            # interrupted row was never excluded by it either way, and a cancelled row
            # stays ineligible on the very next clause (it never carries a `result`).
            or finished_payload.get("result") not in ("failure", "blocked")
            or checkpoint["run_id"] != resume_id
            or checkpoint["outcome"] not in ("failure", "blocked")):
        raise Refusal("resume-ineligible")
    return started_payload, finished_payload, prior_path, prior, checkpoint


def _resume_locate(repo, project, allowlist_root, started_payload, prior, checkpoint):
    """`(branch, worktree)` of the run being resumed, refusing `resume-ineligible` unless the
    recorded repos, branch and worktree agree with the checkpoint and the worktree is the
    one `build` reserves for that branch, inside the allowlist and still a directory."""
    try:
        if Path(started_payload.get("repo", "")).resolve() != repo:
            raise Refusal("resume-ineligible")
        if Path(prior.get("repo", "")).resolve() != repo:
            raise Refusal("resume-ineligible")
        if Path(checkpoint["repo"]).resolve() != repo:
            raise Refusal("resume-ineligible")
    except Refusal:
        raise
    except Exception as exc:
        raise Refusal("resume-ineligible") from exc
    branch = prior.get("branch") or prior.get("target")
    if not branch or started_payload.get("target") != branch or checkpoint["branch"] != branch:
        raise Refusal("resume-ineligible")
    try:
        worktree = Path(prior["worktree"]).resolve()
        if Path(checkpoint["worktree"]).resolve() != worktree:
            raise Refusal("resume-ineligible")
    except Refusal:
        raise
    except Exception as exc:
        raise Refusal("resume-ineligible") from exc
    expected = jc._branch_worktree_path(allowlist_root, project, branch)
    if worktree != expected or not _contained(worktree, allowlist_root) or not worktree.is_dir():
        raise Refusal("resume-ineligible")
    return branch, worktree


def _resume_plan(prior, allowlist_root):
    """`(whitelist, verify, phase, plan_path)` recorded by the prior attempt, refusing
    `resume-ineligible` unless they are well-formed and the plan is a canonical, non-secret,
    readable file (P1a's `_plan_path_defect`)."""
    whitelist = prior.get("whitelist")
    verify = prior.get("verify")
    phase = prior.get("phase")
    if type(whitelist) is not list or not whitelist or type(verify) is not str or not verify or not phase:
        raise Refusal("resume-ineligible")
    try:
        plan_path = canonicalize_target(Path(prior["plan_path"]), allowlist_root)
    except Refusal:
        raise
    except Exception as exc:
        raise Refusal("resume-ineligible") from exc
    if jc._plan_path_defect(plan_path, allowlist_root):
        raise Refusal("resume-ineligible")
    return whitelist, verify, phase, plan_path


def _resume_profile_name(args, prior):
    """The builder profile a resume runs under: `fallback` with `--fallback`, else the one the
    prior attempt recorded; anything but default/fallback refuses `resume-ineligible`."""
    profile_name = "fallback" if getattr(args, "fallback", False) else prior["requested_profile"]
    if profile_name not in ("default", "fallback"):
        raise Refusal("resume-ineligible")
    return profile_name


def _resume_profile(args, prior):
    """`(runtime, model, effort)` of the resumed builder from the saved settings profile,
    after the opencode-on and settings-initialized refusals."""
    _require_opencode_on()
    settings = jset.read_settings()
    if settings is None:
        raise Refusal("agent-settings-uninitialized")
    profile = settings["builders"][_resume_profile_name(args, prior)]
    runtime = "opencode-builder"
    model = f"{profile['connection']}/{profile['model']}"
    effort = profile["effort"] if profile["effort"] is not None else "n/a"
    if type(model) is not str or not 1 <= jc._utf16_len(model) <= jc.HUB_CAPS["model"]:
        raise Refusal("model-invalid")
    if type(effort) is not str or not 1 <= jc._utf16_len(effort) <= jc.HUB_CAPS["effort"]:
        raise Refusal("effort-invalid")
    return runtime, model, effort


def _resume_recheck(repo, worktree, branch, resume_id, prior_path, started_now, finished_now):
    """The re-validation under the worktree claim: re-read the prior manifest and checkpoint
    and refuse `resume-ineligible` unless every recorded identity still agrees. Returns
    `(prior, checkpoint, started_payload, base_sha, root_build_run_id)`. The statement order
    (including `prior.get` BEFORE the `type(prior)` check) is the inline code's."""
    try:
        prior = _read_prior_manifest(prior_path)
        checkpoint = jresume.read_checkpoint(
            jc._manifest_dir(repo, resume_id) / "resume-checkpoint.json")
        started_payload = json.loads(started_now["payload"])
        finished_now_payload = json.loads(finished_now["payload"])
        if Path(prior.get("repo", "")).resolve() != repo:
            raise Refusal("resume-ineligible")
        if Path(started_payload.get("repo", "")).resolve() != repo:
            raise Refusal("resume-ineligible")
        if Path(checkpoint["repo"]).resolve() != repo:
            raise Refusal("resume-ineligible")
        if Path(prior["worktree"]).resolve() != worktree:
            raise Refusal("resume-ineligible")
        if Path(checkpoint["worktree"]).resolve() != worktree:
            raise Refusal("resume-ineligible")
    except Refusal:
        raise
    except Exception as exc:
        raise Refusal("resume-ineligible") from exc
    base_sha = prior.get("base_sha")
    root_build_run_id = prior.get("root_build_run_id")
    if (type(prior) is not dict or prior.get("run_id") != resume_id
            or finished_now_payload.get("result") not in ("failure", "blocked")
            or checkpoint["run_id"] != resume_id
            or checkpoint["outcome"] not in ("failure", "blocked")
            or checkpoint["outcome"] != finished_now_payload.get("result")
            or checkpoint["branch"] != branch
            or (prior.get("branch") or prior.get("target")) != branch
            or type(base_sha) is not str or not re.fullmatch(r"[0-9a-f]{40}", base_sha)
            or checkpoint["base"] != base_sha
            or checkpoint["root_build_run_id"] != root_build_run_id
            or started_payload.get("root_build_run_id") != root_build_run_id
            or started_payload.get("requested_profile") not in ("default", "fallback")
            or prior.get("requested_profile") not in ("default", "fallback")
            or prior.get("requested_profile") != started_payload.get("requested_profile")):
        raise Refusal("resume-ineligible")
    return prior, checkpoint, started_payload, base_sha, root_build_run_id


def _resume_git_checks(run, repo, worktree, branch, resume_id, latest, started_payload,
                       base_sha, plan_path, checkpoint):
    """The live-state checks of a resume, in order: this IS the latest attempt, its tmux
    session is gone, repo and worktree share one git common dir and the worktree is
    registered on `branch`, HEAD is attached and descends from the recorded base, and the
    work state and plan revision still match the checkpoint."""
    if latest != resume_id:
        raise jc._refuse("resume-ineligible", f"hint: latest attempt is {latest}")
    prior_session = started_payload.get("session")
    if prior_session:
        live = run(["tmux", "has-session", "-t", prior_session])
        if live.returncode == 0:
            raise Refusal("resume-ineligible")
    common_repo = _git_common_dir(run, repo)
    common_wt = _git_common_dir(run, worktree)
    if common_repo is None or common_wt is None or common_repo != common_wt:
        raise Refusal("resume-ineligible")
    if not jc._is_registered_worktree(run, repo, worktree, branch):
        raise Refusal("resume-ineligible")
    attached = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=worktree)
    if attached.returncode != 0 or attached.stdout.strip() != branch:
        raise Refusal("resume-ineligible")
    ancestor = run(["git", "merge-base", "--is-ancestor", base_sha, "HEAD"], cwd=worktree)
    if ancestor.returncode != 0:
        raise Refusal("resume-ineligible")
    actual = jresume.capture_work_state(worktree, plan_path, run=run)
    jresume.require_same_work_state(checkpoint["work_state"], actual)
    if checkpoint["plan_revision"] != actual["plan_sha256"]:
        raise Refusal("resume-state-changed")


def _dispatch_resume_build(args, *, resume_id, repo, project, caller, caller_session,
                           run, post, env, now, allowlist_root, db_path):
    db_path = db_path or jr.DB_PATH
    started, finished, _ = _resume_rows(db_path, resume_id)
    started_payload, finished_payload, prior_path, prior, checkpoint = _resume_load_prior(
        repo, resume_id, project, started, finished)
    branch, worktree = _resume_locate(repo, project, allowlist_root, started_payload, prior, checkpoint)
    whitelist, verify, phase, plan_path = _resume_plan(prior, allowlist_root)
    plan_defects = _validate_plan_structure(plan_path.read_text(encoding="utf-8"), plan_path, allowlist_root)
    if plan_defects:
        raise Refusal("resume-ineligible")
    runtime, model, effort = _resume_profile(args, prior)
    with jresume.worktree_claim(repo, worktree):
        started_now, finished_now, latest = _resume_rows(db_path, resume_id, (project, repo, branch))
        prior, checkpoint, started_payload, base_sha, root_build_run_id = _resume_recheck(
            repo, worktree, branch, resume_id, prior_path, started_now, finished_now)
        whitelist, verify, phase, plan_path = _resume_plan(prior, allowlist_root)
        profile_name = _resume_profile_name(args, prior)
        _resume_git_checks(
            run, repo, worktree, branch, resume_id, latest, started_payload, base_sha,
            plan_path, checkpoint)
        args_ns = SimpleNamespace(
            role="builder", project=project, phase=phase, repo=str(worktree),
            prompt_file=None, runtime=runtime, callback=None,
        )
        try:
            jr.preflight(args_ns, run=run, allow_untracked=True, require_feat_branch=False, skip_handoff=True)
            run_id, session, _paths = jr._alloc_run(
                project, "build", worktree, run, run_id=uuid.uuid4().hex[:12])
        except ValueError as exc:
            raise Refusal(jc._map_refusal(str(exc))) from exc
        caller_pane = env.get("TMUX_PANE")
        manifest = _build_manifest_base(
            args, project=project, phase=phase, repo=repo, worktree=worktree, branch=branch,
            base_sha=base_sha, caller=caller, caller_session=caller_session, model=model,
            effort=effort, runtime=runtime, run_id=run_id, session=session, whitelist=whitelist,
            verify=verify, plan_path=plan_path, now=now,
        )
        manifest.update({
            "requested_profile": profile_name,
            "root_build_run_id": root_build_run_id,
            "resumes_run_id": resume_id,
            "reservation_owned": False,
            "resume_start": checkpoint["work_state"],
        })
        if prior.get("build"):
            manifest["build"] = prior["build"]
        started_out = _build_started_base(
            phase=phase, runtime=runtime, branch=branch, caller=caller,
            caller_session=caller_session, model=model, effort=effort, session=session,
            repo=repo, verify=verify,
        )
        started_out.update({
            "requested_profile": profile_name,
            "root_build_run_id": root_build_run_id,
            "resumes_run_id": resume_id,
        })
        if caller_pane:
            started_out["caller_pane"] = caller_pane
        if prior.get("build"):
            started_out["build"] = prior["build"]
        return jc._dispatch_run(
            run=run, post=post, repo=repo, project=project, phase=phase, kind="build",
            role="builder", run_id=run_id, session=session, manifest=manifest,
            manifest_dir=jc._manifest_dir(repo, run_id), started_payload=started_out,
            cleanup_paths=[],
        )


def _build_manifest_base(args, *, project, phase, repo, worktree, branch, base_sha, caller,
                         caller_session, model, effort, runtime, run_id, session, whitelist,
                         verify, plan_path, now):
    """The manifest keys a fresh build and a resumed build share, in their shared order
    (manifest.json is written in insertion order; the callers append their own tails).
    `_no_callback` and `now()` keep their dict-literal evaluation order."""
    return {
        "kind": "build", "role": "builder", "project": project, "phase": phase,
        "repo": str(repo), "worktree": str(worktree), "branch": branch,
        # target = the branch (spec §2.6/§4.3/§2.4's `result`), NOT the plan path --
        # see this plan's header for the §4.2/§7.1#14 conflict this resolves.
        "target": branch,
        "base_sha": base_sha,
        "caller": caller, "caller_session": caller_session, "model": model, "effort": effort,
        "runtime": runtime, "run_id": run_id, "session": session,
        "no_callback": jc._no_callback(args, caller), "whitelist": whitelist,
        "verify": verify, "plan_path": str(plan_path),
        "dispatch_start": jc._iso8601(now()),
    }


def _build_started_base(*, phase, runtime, branch, caller, caller_session, model, effort,
                        session, repo, verify):
    """The `run-started` payload keys a fresh and a resumed build share (the callers append
    their own tails in their own order)."""
    return {
        "phase": phase, "runtime": runtime, "kind": "build", "target": branch,
        "caller": caller, "caller_session": caller_session, "model": model, "effort": effort,
        "session": session, "repo": str(repo), "verify": verify,
    }


def _validated_build_plan(args, allowlist_root):
    """The canonical plan path of a fresh build, validated BEFORE any reservation: canonical,
    non-secret, readable file, structurally valid (MOA-471 item 10). No git call needed."""
    # Canonicalize + secret-check the plan path BEFORE any reservation (fixes cold review
    # F3 -- the first draft never called _is_secret_path at all for `build`).
    plan_path = canonicalize_target(Path(args.plan), allowlist_root)
    defect = jc._plan_path_defect(plan_path, allowlist_root)
    if defect == "secret-detected":
        raise Refusal(f"secret-detected: {plan_path}")
    # fixes Part 1 diff-review F1: a directory or unreadable plan path must refuse here,
    # before any reservation -- not after a worktree/branch already exist (MOA-467: the
    # original document is read in place later, so there is no copy step to fail in).
    # No git call needed for this check.
    if defect:
        raise Refusal("plan is not a readable file")

    # MOA-471 item 10: plan-structure validation runs BEFORE any reservation below --
    # `os.mkdir(worktree)` (further down this function) is the earliest filesystem/git
    # write this dispatch makes, and no plan defect may ever burn one.
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_defects = _validate_plan_structure(plan_text, plan_path, allowlist_root)
    if plan_defects:
        raise jc._refuse("plan-invalid", "hint: " + "; ".join(plan_defects))
    return plan_path


def _select_build_profile(args):
    """`(runtime, model, effort, requested_profile)` of a fresh build: the saved builder
    profile (`requested_profile` set) or, with no saved profile, the legacy runtime/model
    defaults (`requested_profile` None). Validates the hub caps on model and effort."""
    _require_opencode_on()
    settings = jset.read_settings()
    selected = jset.select_builder(
        settings,
        fallback=getattr(args, "fallback", False),
        builder=args.builder,
        model=args.model,
        effort=args.effort,
    )
    requested_profile = None
    if selected is None:
        runtime = args.builder or BUILDER_DEFAULT
        model = args.model or jr.MODEL_BY_RUNTIME.get(runtime, "default")
        effort = args.effort or "n/a"
        if args.effort and runtime in jr.MODEL_BY_RUNTIME:
            raise Refusal(f"effort-not-supported: {runtime}")
    else:
        runtime = selected["runtime"]
        requested_profile = selected["profile_name"]
        profile = settings["builders"][requested_profile]
        model = f"{profile['connection']}/{profile['model']}"
        effort = profile["effort"] if profile["effort"] is not None else "n/a"
    if type(model) is not str or not 1 <= jc._utf16_len(model) <= jc.HUB_CAPS["model"]:
        raise Refusal("model-invalid")
    if type(effort) is not str or not 1 <= jc._utf16_len(effort) <= jc.HUB_CAPS["effort"]:
        raise Refusal("effort-invalid")
    return runtime, model, effort, requested_profile


def _resolve_explicit_base(args, run, repo):
    """`--base` resolved to a 40-hex commit SHA BEFORE the reservation, or None when the flag
    is absent. It is the SHA, never the caller's string, that reaches `git worktree add`."""
    # `--base` is resolved to a SHA here, BEFORE the reservation, and it is the SHA -- never
    # the caller's string -- that reaches `git worktree add` further down. Two reasons, both
    # load-bearing:
    #   * `git worktree add -b <branch> <path> <base>` takes the base POSITIONALLY with no
    #     `--` guard, so a value like `--foo` would be read by git as an option. A resolved
    #     40-hex SHA cannot start with `-`, which closes that off structurally rather than by
    #     blacklisting shapes.
    #   * `rev-parse --verify <base>^{commit}` proves the ref exists and names a commit, so a
    #     typo refuses here instead of half-creating a worktree and unwinding it.
    # The explicit shape gate still runs first: `rev-parse` itself would read a leading `--`
    # as its own option, so the value has to be proven flag-shaped-safe before it is handed
    # to git at all.
    if args.base is None:
        return None
    if (args.base.startswith("-") or any(c.isspace() for c in args.base)
            or not 1 <= jc._utf16_len(args.base) <= 512):
        raise Refusal("base-invalid")
    probe = run(["git", "rev-parse", "--verify", f"{args.base}^{{commit}}"], cwd=repo)
    resolved = probe.stdout.strip()
    if probe.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise Refusal("base-invalid")
    return resolved


def _reserve_build_worktree(args, *, run, repo, worktree, base_sha):
    """Reserve `worktree` for a fresh build and create its branch; returns the 40-hex base SHA
    the branch starts from. Order is load-bearing: the exclusive `os.mkdir` claim is the first
    git-or-filesystem write; the default-branch probe (only without `--base`) comes after it;
    `git worktree add -b` last. A failure after the claim removes only the bare directory."""
    # Exclusive reservation FIRST (fixes cold review G8/RECURRENCE(F19), spec §4.2 step 1;
    # reordered ahead of `_default_branch()` per Part 1 diff-review F2): `git worktree add
    # -b` alone does not reliably refuse a pre-existing, empty directory at this path -- it
    # happily populates it -- so `os.mkdir` is the actual atomic claim, and it must be the
    # very first git-or-filesystem write this dispatch makes: an EEXIST refusal here makes
    # zero git calls of its own (beyond the toplevel resolve every verb already needs).
    try:
        os.mkdir(worktree)
    except FileExistsError:
        raise Refusal("branch-exists")

    # Read-only, has no bearing on the path/branch collision this dispatch just claimed --
    # resolved AFTER the reservation (Part 1 diff-review F2) so a failure here only ever
    # has to undo the bare directory this dispatch itself just reserved, never a git
    # worktree/branch that `git worktree add -b` has not created yet.
    try:
        # Only consulted when `--base` was NOT given: a resolved base makes the default
        # branch irrelevant, and probing for it anyway would fail a build in a repo whose
        # default branch is missing even though the caller named a perfectly good base.
        # New (spec §4.2 F5): resolved to its own commit SHA the same way `--base`
        # already is, so `manifest["base_sha"]` is always a real 40-hex SHA -- never a
        # branch name that can't equal a 40-hex `head_sha` even on an untouched
        # default-base build.
        if base_sha is None:
            default_branch = jr._default_branch(repo, run)
            default_probe = run(
                ["git", "rev-parse", "--verify", f"{default_branch}^{{commit}}"], cwd=repo)
            resolved_default = default_probe.stdout.strip()
            if default_probe.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", resolved_default):
                raise Refusal("base-invalid")
            base_sha = resolved_default
    except Exception as exc:
        shutil.rmtree(worktree, ignore_errors=True)
        raise Refusal(f"build failed: {exc}") from exc

    added = run(["git", "worktree", "add", "-b", args.branch, str(worktree), base_sha], cwd=repo)
    if added.returncode != 0:
        # Never deletes `args.branch` here (fixes cold review round 2 F5): dropping the
        # pre-add `git rev-parse <branch>` probe (itself a check-then-act on the very branch
        # this call is about to create) means this dispatch can no longer safely tell a
        # pre-existing branch apart from one `git worktree add -b` itself half-created
        # before failing to populate. The common case -- the branch name was already taken
        # by something unrelated -- must never be deleted; the rare git-half-created case is
        # accepted as a leftover orphan ref instead.
        shutil.rmtree(worktree, ignore_errors=True)
        raise jc._refuse("branch-exists", f"hint: {added.stderr.strip()}" if added.stderr else None)
    return base_sha


def _copy_agents_md(repo, worktree):
    """Copy the CONTROL repo's own AGENTS.md into the worktree root, when it has one. It is
    gitignored, so `git worktree add` never brings it along, and the handoff points the
    builder at `worktree / "AGENTS.md"`. The copy stays gitignored in the worktree too."""
    # Copy the CONTROL repo's own AGENTS.md into the worktree root, when it has one
    # (fixes cold review round 2 F2): AGENTS.md is gitignored (`.gitignore:10`), so
    # `git worktree add` never brings it along, and Part 3's handoff points the builder
    # at `worktree / "AGENTS.md"`. The copy stays gitignored in the worktree too --
    # never `git add`ed.
    control_agents_md = repo / "AGENTS.md"
    if control_agents_md.is_file():
        (worktree / "AGENTS.md").write_text(
            control_agents_md.read_text(encoding="utf-8"), encoding="utf-8",
        )


def dispatch_build(args, *, run, post, env, now, allowlist_root=ALLOWLIST_ROOT_DEFAULT, db_path=None):
    cwd = Path.cwd().resolve()
    repo = jc._require_toplevel(run, cwd)
    caller = jc.resolve_caller(env, args.from_caller)
    # Fires as early as possible -- BEFORE any worktree reservation -- so a missing
    # session variable never leaves a worktree/branch to clean up (a natural extension of
    # cold review F1's "fail cheaply, before side effects" fix, below).
    caller_session = jc._require_caller_session(env, caller)
    project = jc.slugify_project(repo.name)
    resume_id = getattr(args, "resume", None)
    if resume_id:
        forbidden = (
            "plan", "phase", "branch", "whitelist", "verify", "build", "base",
            "builder", "model", "effort",
        )
        if any(getattr(args, name, None) not in (None,) for name in forbidden):
            raise Refusal("resume-ineligible")
        if not jc._RUN_ID_RE.match(resume_id):
            raise Refusal("resume-ineligible")
        return _dispatch_resume_build(
            args, resume_id=resume_id, repo=repo, project=project,
            caller=caller, caller_session=caller_session,
            run=run, post=post, env=env, now=now,
            allowlist_root=allowlist_root, db_path=db_path,
        )
    required = ("plan", "phase", "branch", "whitelist", "verify")
    if any(not getattr(args, name, None) for name in required):
        raise Refusal("build-missing-required-flags")
    phase = args.phase

    plan_path = _validated_build_plan(args, allowlist_root)

    runtime, model, effort, requested_profile = _select_build_profile(args)
    jc._check_hub_caps({
        "phase": phase, "verify": args.verify, "build": args.build, "target": args.branch,
        "callerSession": caller_session, "callerPane": env.get("TMUX_PANE"),
        "model": model, "effort": effort, "repo": str(repo),
    })
    base_sha = _resolve_explicit_base(args, run, repo)
    whitelist = [p.strip() for p in args.whitelist.split(",") if p.strip()]
    worktree = jc._branch_worktree_path(allowlist_root, project, args.branch, resolve=False)

    base_sha = _reserve_build_worktree(args, run=run, repo=repo, worktree=worktree, base_sha=base_sha)
    # Past this point, `git worktree add` succeeded, so `worktree` is a registered Git
    # worktree admin entry and `args.branch` is a branch this dispatch itself just created --
    # every cleanup path below undoes both via `_cleanup_worktree` (fixes cold review round
    # 2 F1), never a raw `shutil.rmtree` (which leaves `.git/worktrees/<id>` behind).

    # Every step from here through dispatch runs under ONE guard (fixes Part 1 diff-review
    # F1): ANY exception -- not only `Refusal`/`ValueError` as before -- undoes the
    # worktree/branch via `_cleanup_worktree` before the refusal reaches the caller. A
    # `Refusal` (from `_dispatch_run`'s own hub-unreachable/tmux-failed paths, which already
    # clean up their OWN control-repo-side manifest dir) is re-raised as is; a `ValueError`
    # (`preflight`/`_alloc_run`'s own refusals) is mapped through `_map_refusal` exactly as
    # before; any other exception (an AGENTS.md copy read/write failure, e.g.) becomes a
    # plain-English refusal that still names what broke.
    try:
        # Builder preflight runs AGAINST THE NEW WORKTREE, never the control repo (fixes
        # cold review F1 -- checking the control repo instead falsely tripped
        # `builder-on-default-branch`, since the control repo is almost always on `main`
        # when the tech lead runs `build`). `repo` here is the worktree; its checked-out
        # branch is `args.branch`, freshly created above.
        args_ns = SimpleNamespace(
            role="builder", project=project, phase=phase, repo=str(worktree),
            prompt_file=None, runtime=runtime, callback=None,
        )
        jr.preflight(args_ns, run=run, allow_untracked=True, require_feat_branch=False, skip_handoff=True)

        run_id, session, paths = jr._alloc_run(project, "build", worktree, run, run_id=uuid.uuid4().hex[:12])

        # MOA-467: the plan is NOT copied into the worktree anymore -- the manifest stores
        # the ORIGINAL canonical plan path and the builder reads it in place from the
        # validated read scope (the old plan-copy step, and the copy it produced, are
        # gone). The AGENTS.md copy below stays the only control-repo file brought into
        # the worktree.

        _copy_agents_md(repo, worktree)
        caller_pane = env.get("TMUX_PANE")

        manifest = _build_manifest_base(
            args, project=project, phase=phase, repo=repo, worktree=worktree, branch=args.branch,
            base_sha=base_sha, caller=caller, caller_session=caller_session, model=model,
            effort=effort, runtime=runtime, run_id=run_id, session=session, whitelist=whitelist,
            verify=args.verify, plan_path=plan_path, now=now,
        )
        if args.build:
            manifest["build"] = args.build
        if requested_profile is not None:
            manifest["requested_profile"] = requested_profile
            manifest["root_build_run_id"] = run_id
            manifest["reservation_owned"] = True

        started_payload = _build_started_base(
            phase=phase, runtime=runtime, branch=args.branch, caller=caller,
            caller_session=caller_session, model=model, effort=effort, session=session,
            repo=repo, verify=args.verify,
        )
        if caller_pane:
            started_payload["caller_pane"] = caller_pane
        if args.build:
            started_payload["build"] = args.build
        if requested_profile is not None:
            started_payload["requested_profile"] = requested_profile
            started_payload["root_build_run_id"] = run_id

        return jc._dispatch_run(
            run=run, post=post, repo=repo, project=project, phase=phase, kind="build",
            role="builder", run_id=run_id, session=session, manifest=manifest,
            manifest_dir=jc._manifest_dir(repo, run_id), started_payload=started_payload,
            cleanup_paths=[],
        )
    except Refusal:
        # `_dispatch_run`'s own hub-unreachable/tmux-failed paths already leave nothing
        # behind on the CONTROL repo side (manifest dir); this run's WORKTREE and branch
        # are this function's own side effect, so this function also owns undoing them.
        jc._cleanup_worktree(worktree, args.branch, run=run, repo=repo)
        raise
    except ValueError as exc:
        jc._cleanup_worktree(worktree, args.branch, run=run, repo=repo)
        raise Refusal(jc._map_refusal(str(exc))) from exc
    except Exception as exc:
        # fixes Part 1 diff-review F1: an ordinary exception (an AGENTS.md copy write
        # failure, e.g., or any other unexpected error)
        # must undo the worktree/branch just like a `Refusal`/`ValueError` already did --
        # not escape uncaught and leave a successfully-created worktree behind.
        jc._cleanup_worktree(worktree, args.branch, run=run, repo=repo)
        raise Refusal(f"build failed: {exc}") from exc


def _validate_plan_structure(plan_text, plan_dest, allowlist_root):
    """The one check that remains after decision 7: the plan's OPTIONAL **Spec:** line,
    if present, must resolve. `builder-contract.md:11-13` already requires the builder
    to read `paths.plan` in full, so no title/Goal/Task-heading structure is checked
    here any more."""
    _, spec_refusal_code = jc._resolve_handoff_spec_path(plan_text, plan_dest, None, allowlist_root)
    if spec_refusal_code:
        return [f"unresolvable **Spec:** line: {spec_refusal_code}"]
    return []
