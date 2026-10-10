"""jaxflow merge module -- P2 split of scripts/jaxflow.py (lean spec, Decision 18)."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
import general_settings
import jaxflow_run as jr
import jaxflow_resume as jresume
from jax_init import ALLOWLIST_ROOT_DEFAULT, Refusal, _contained
from jaxflow_hook import redact
import jaxflow_common as jc


def _hub_retryable(exc):
    # Transport errors and hub-side 5xx faults may clear on a re-run; a 4xx / {ok:false}
    # rejection will not, so it gets no resume advice.
    return not isinstance(exc, jc.HubRejected) or (isinstance(exc.status, int) and exc.status >= 500)


def _refuse_nonterminal_builder(project, repo, branch, db_path=None):
    db_path = db_path or jr.DB_PATH
    try:
        con = jc._open_ro(db_path)
    except sqlite3.Error:
        return
    con.row_factory = sqlite3.Row
    try:
        try:
            latest = jc._latest_builder_attempt(con, project, repo, branch)
            if latest is None:
                return
            finished = con.execute(
                "SELECT 1 FROM workflow_events WHERE run_id = ? AND type = 'run-finished' LIMIT 1",
                (latest,),
            ).fetchone()
        except sqlite3.Error:
            return
    finally:
        con.close()
    if finished is None:
        raise Refusal("resume-ineligible")


def _merge_checks_reuse(run, repo, project, worktree, branch, sha, checks_cmd, *,
                        allowlist_root, recheck=False, db_path=None):
    """MOA-510 D1/D2/D3. May `jaxflow merge` skip its own `--checks` run because the branch's
    latest diff review already tested `sha`? Returns `(reuse, reason, review_run_id)`.
    Collector-style: `run` is injected. It NEVER raises and NEVER refuses -- every lookup
    failure is "not reusable", and the caller then runs the checks exactly as before.
    Reads the BUILD worktree's tests.txt and `git status` (the proof that `sha` passed lives
    with the build), never the control repo's merged index."""
    if recheck:
        return False, "recheck-forced", None
    try:
        if not (_contained(worktree, allowlist_root) and worktree.is_dir()
                and jc._is_registered_worktree(run, repo, worktree, branch)):
            return False, "worktree-missing", None
        con = jc._open_ro(db_path or jr.DB_PATH)
        con.row_factory = sqlite3.Row
        try:
            # Newest diff review of THIS branch (`ts DESC, id DESC`, the ordering
            # `_latest_builder_attempt` uses). Reviews link backward, so the newest is the
            # chain head: no `_walk_review_chain`. The verdict is irrelevant here.
            rows = con.execute(
                "SELECT run_id, payload FROM workflow_events "
                "WHERE project = ? AND role = 'reviewer' AND type = 'run-started' "
                "ORDER BY ts DESC, id DESC", (project,),
            ).fetchall()
            repo_r = Path(repo).resolve()
            review_id = None
            for row in rows:
                payload = json.loads(row["payload"])
                if (payload.get("kind") == "diff" and payload.get("target") == branch
                        and Path(payload.get("repo", "")).resolve() == repo_r):
                    review_id = row["run_id"]
                    break
            if review_id is None:
                return False, "no-diff-review", None
            mdir = jc._safe_run_subpath(repo, "runs", review_id)
            manifest = None
            if mdir is not None:
                mpath = (mdir / "manifest.json").resolve()
                if not _contained(mpath, mdir):  # review F1: a symlinked manifest.json escapes
                    return False, "lookup-failed", None
                try:
                    manifest = jc._load_manifest(mdir)
                except (OSError, ValueError):
                    manifest = None
            if not isinstance(manifest, dict) or manifest.get("kind") != "diff" \
                    or not isinstance(manifest.get("verify"), dict):
                return False, "manifest-unreadable", None
            verify = manifest["verify"]
            builder_id = manifest.get("builder_run_id")
            built = con.execute(
                "SELECT payload FROM workflow_events WHERE run_id = ? AND role = 'builder' "
                "AND type = 'run-started' LIMIT 1", (builder_id,),
            ).fetchone()
            if built is None:
                return False, "no-build-run", None
            built_payload = json.loads(built["payload"])
        finally:
            con.close()
        verify_cmd, build_cmd = built_payload["verify"], built_payload.get("build")
        # The FULL ordered, redacted command list, exactly as the review parses it: parsing
        # only the verify command would let a later passing build frame mask a failed verify.
        cmds = [redact(c) for c in [verify_cmd] + ([build_cmd] if build_cmd else [])]
        tests_path = jc._safe_run_subpath(worktree, "reports", builder_id, ".tests.txt")
        frames, parse_error = None, "tests-file-missing"
        if tests_path is not None and tests_path.is_file():
            try:
                text = tests_path.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                parse_error = "tests-file-not-utf8"
            else:
                frames, parse_error = jr.parse_tests_frames(text, cmds)
        # --untracked-files=all: do not let `status.showUntrackedFiles` hide untracked dirt
        porcelain = run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=worktree)
        reuse, reason = jc._decide_verify_reuse(
            head_matches=sha == verify.get("head_sha"),
            porcelain_clean=porcelain.returncode == 0 and porcelain.stdout == "",
            commands_match=checks_cmd == verify_cmd,
            frames=frames, parse_error=parse_error,
        )
        return reuse, reason, review_id if reuse else None
    except Exception:  # ponytail: advisory lookup -- any failure means "run the checks"
        return False, "lookup-failed", None


def _checks_audit(reused, review_id, sha):
    """The `checks` field of the `merge-approved` audit: `reused` records which diff review
    already tested this exact head; otherwise the checks ran. (A resume records
    `{"mode": "resumed"}` itself: it neither runs nor reuses checks.)"""
    if reused:
        return {"mode": "reused", "source_review_run_id": review_id, "head_sha": sha}
    return {"mode": "run"}


def _merge_checks_line(audit, checks_cmd):
    """The `checks:` outcome line (D5). A resumed delivery ran nothing, so it prints none."""
    if audit["mode"] == "reused":
        return f"checks: reused from review {audit['source_review_run_id']} at {audit['head_sha'][:12]}"
    return f"checks: {checks_cmd} exit 0" if audit["mode"] == "run" else None


# Spec §5.2 Step 0. The exact shape of the five preset headings in
# `workflow/templates/AGENTS-template.md` (e.g.
# "**Preset: `dual-branch-pr`** — GitLab Flow with PRs: ...").
_PRESET_LINE_RE = re.compile(r"^\*\*Preset: `([a-z0-9]+(?:-[a-z0-9]+)*)`\*\*", re.M)

# The first backtick-quoted token AFTER the literal phrase, so an earlier
# "Base branch: `main`" in the same block is never mistaken for the delivery target.
_DELIVERY_TARGET_RE = re.compile(r"Delivery target:[\s\S]*?`([a-z0-9][a-z0-9/_-]*)`")

_PRODUCTION_TARGET_RE = re.compile(r"Production target:[\s\S]*?`([a-z0-9][a-z0-9/_-]*)`")

# MOA-509: three sets decide behaviour. A call site asks "is this a PR preset", never "is this
# <name>". `bubble-buildprint` is unchanged; `release` exists only for `dual-branch-pr`.
_LOCAL_PRESETS = ("single-branch", "dual-branch", "bubble-buildprint")

_PR_PRESETS = ("single-branch-pr", "dual-branch-pr")

_RELEASE_PRESETS = ("dual-branch-pr",)


def _preset_block(repo):
    """AGENTS.md's Deploy policy: (preset_name, its own block text — from its heading to the
    next preset heading, the next `##` heading, or EOF). `(None, None)` means zero preset
    lines, the common case (most repos on this machine predate the preset system). Raises
    `preset-unknown` for 2+ preset lines. Shared by `_resolve_delivery_target` and
    `_resolve_production_target` so both parse the exact same block text the exact same way."""
    policy = repo / "AGENTS.md"
    try:
        text = policy.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    except (OSError, UnicodeError) as exc:
        refusal = Refusal("preset-unknown")
        refusal.hint = f"hint: {policy} exists but could not be read: {jr._bound(str(exc), 200)}"
        raise refusal from exc
    presets = _PRESET_LINE_RE.findall(text)
    if not presets:
        return None, None
    if len(presets) > 1:
        raise Refusal("preset-unknown")
    name = presets[0]
    start = _PRESET_LINE_RE.search(text).start()
    tail = text[start + 1:]
    boundary = re.search(r"^\*\*Preset: |^## ", tail, re.M)
    block = text[start:start + 1 + boundary.start()] if boundary else text[start:]
    return name, block


def _resolve_delivery_target(repo, *, run):
    """Resolves `merge`'s delivery target from the control repo's own Deploy policy (spec
    §5.2 Step 0 / MOA-465 Config & trust model). `--target` asserts this value; it cannot
    override it. Returns `(target_branch, no_preset_block, preset)`.

    MOA-465: a PR preset RESOLVES like any other preset instead of refusing
    `preset-unsupported` — its own `Delivery target:` line is the base for a non-`release/*`
    head. A `release/*` head's base comes from `_resolve_production_target` instead
    (`_required_target_for_branch`, decision 5) — this function never reads that line itself.

    Zero preset lines is the COMMON case, not an error: `jax-os`'s own AGENTS.md and 22 other
    repos on this machine predate the preset system, and `jaxflow merge` must work on them —
    they fall back to `_default_branch()`. Two or more lines means the template's "delete the
    presets you did not choose" instruction was not followed. Any name outside the known set
    is refused — an unrecognized preset has no known delivery semantics, and guessing one is
    exactly the class of invention this contract forbids; the refusal hint lists the five
    valid names."""
    name, block = _preset_block(repo)
    if name is None:
        return jr._default_branch(repo, run), True, None
    if name not in _LOCAL_PRESETS + _PR_PRESETS:
        raise jc._refuse("preset-unknown", f"hint: unknown preset `{jr._bound(name, 40)}`; valid presets: "
                    + ", ".join(_LOCAL_PRESETS + _PR_PRESETS))
    hit = _DELIVERY_TARGET_RE.search(block)
    if not hit:
        raise Refusal("preset-unknown")
    return hit.group(1), False, name


def _resolve_production_target(repo):
    """MOA-465 decision 5/Config & trust model: the `Production target:` line, next to
    `Delivery target:` in the SAME release-preset block. Refuses `production-target-unconfigured`
    for any preset outside the release set (so always for `single-branch-pr`, even if the line
    is present), a release block missing the line, or no preset block at all — all three mean
    the same thing this refusal names: nothing configured."""
    name, block = _preset_block(repo)
    if name not in _RELEASE_PRESETS:
        raise Refusal("production-target-unconfigured")
    hit = _PRODUCTION_TARGET_RE.search(block)
    if not hit:
        raise Refusal("production-target-unconfigured")
    return hit.group(1)


def _required_target_for_branch(repo, branch, *, run):
    """MOA-465 decision 5: the Delivery target for a non-`release/*` head, the Production
    target for a `release/*` head — shared by `pr open` (Task 6) and `merge`'s PR path
    (Task 7) so the mutual-exclusion rule can never diverge between the two commands. Returns
    `(required_target, preset)`."""
    target, _no_preset_block, preset = _resolve_delivery_target(repo, run=run)
    if branch.startswith("release/"):
        return _resolve_production_target(repo), preset
    return target, preset


_GH_PR_URL_RE = re.compile(r"^https://github\.com/[^/\s]+/[^/\s]+/pull/([0-9]+)\s*$")

# cold review 75e934eacdca F5: anchored to the two supported remote forms exactly --
# `git@github.com:owner/repo(.git)` or `https://github.com/owner/repo(.git)` -- so a
# lookalike host (`evilgithub.com`, which contains "github.com" as a bare substring) can
# never match. The original `github\.com[:/]` pattern had no start anchor and no host
# boundary, so `.search()` would find "github.com" inside "evilgithub.com" too.
_GH_REMOTE_RE = re.compile(r"^(?:git@github\.com:|https://github\.com/)([^/\s]+)/([^/\s]+?)(?:\.git)?/?$")


def _github_repo_slug(run, repo):
    """owner/repo parsed from origin's remote URL (SSH or HTTPS) -- gh pr create/view/list/
    merge always pass --repo explicitly (decision 5), never letting gh infer it from cwd.
    Spec names no dedicated refusal for a remote that isn't even a GitHub URL, including a
    lookalike host such as evilgithub.com; this folds into github-unreachable, the closest
    existing 'could not identify the GitHub side of this operation' code."""
    remote = jc._git_read(run, repo, ["git", "remote", "get-url", "origin"])
    m = remote and _GH_REMOTE_RE.search(remote)
    if not m:
        raise Refusal("github-unreachable")
    return f"{m.group(1)}/{m.group(2)}"


_GH_PR_VIEW_FIELDS = "number,url,state,headRefOid,headRefName,baseRefName,mergeable,mergeCommit"


def _gh_pr_view(run, repo, repo_slug, number):
    """gh pr view <n> --json ... parsed. Any transport/parse failure is github-unreachable
    (decision 6: network failure means unknown, never 'PR absent')."""
    result = run(["gh", "pr", "view", str(number), "--repo", repo_slug, "--json", _GH_PR_VIEW_FIELDS], cwd=repo)
    if result.returncode != 0:
        raise Refusal("github-unreachable")
    try:
        return json.loads(result.stdout)
    except (ValueError, TypeError):
        raise Refusal("github-unreachable")


def _resolve_pr_number(run, repo, repo_slug, branch, base):
    """gh pr list --head/--base --state all fallback (spec Ledger & card reconciliation): zero
    matches means no PR exists yet, exactly one is adopted, more than one refuses
    pr-ambiguous. Only called when the hub ledger has no record at all."""
    listed = run(["gh", "pr", "list", "--repo", repo_slug, "--head", branch, "--base", base,
                  "--state", "all", "--json", "number"], cwd=repo)
    if listed.returncode != 0:
        raise Refusal("github-unreachable")
    try:
        matches = json.loads(listed.stdout or "[]")
    except (ValueError, TypeError):
        raise Refusal("github-unreachable")
    if len(matches) > 1:
        raise Refusal("pr-ambiguous")
    return matches[0]["number"] if matches else None


def _find_recorded_pr(project, branch, *, db_path=None):
    """Latest pr-opened payload recorded for (project, branch), any sha -- the hub event log
    IS the durable PR record (spec Ledger & card, no separate table). None means jaxflow has
    never recorded a PR for this branch (a fresh pr open, or a lookup this process never
    reaches because the ledger DB itself is unavailable -- same fail-open-to-'unrecorded'
    shape as _refuse_nonterminal_builder, :1270)."""
    db_path = db_path or jr.DB_PATH
    try:
        con = jc._open_ro(db_path)
    except sqlite3.Error:
        return None
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT payload FROM workflow_events WHERE type = 'pr-opened' AND project = ? "
            "ORDER BY ts DESC, id DESC",
            (project,),
        ).fetchall()
    finally:
        con.close()
    for row in rows:
        payload = json.loads(row["payload"])
        if payload.get("branch") == branch:
            return payload
    return None


def _find_latest_release_pr(project, repo_slug, *, db_path=None):
    """Latest pr-opened payload with kind == 'release' for this project AND repo_slug, ANY
    branch (decision 9's own reuse rule -- release names change by date, so this cannot key
    on one branch the way _find_recorded_pr does). Filtering on repo_slug, not just the
    project slug (cold review e83ed7ea5eeb F2), keeps a same-named repository under a
    different owner -- or a stale record from before a remote rename -- from donating its PR
    number. None means jaxflow has never recorded one for this repo."""
    db_path = db_path or jr.DB_PATH
    try:
        con = jc._open_ro(db_path)
    except sqlite3.Error:
        return None
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT payload FROM workflow_events WHERE type = 'pr-opened' AND project = ? "
            "ORDER BY ts DESC, id DESC",
            (project,),
        ).fetchall()
    finally:
        con.close()
    for row in rows:
        payload = json.loads(row["payload"])
        if payload.get("kind") == "release" and payload.get("repo") == repo_slug:
            return payload
    return None


def _status_tracked(run, cwd):
    """The tracked-only status probe (porcelain, untracked files excluded: an untracked
    artifact never blocks anything). Returns the raw `run(...)` result; every caller keeps
    its own `returncode`/`stdout` reading (a failed probe is NOT a clean tree)."""
    return run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd)


def _abort_merge(run, repo):
    """Aborts an in-progress merge and reports what actually survived. `git merge --abort`
    is NOT a guaranteed restore (verified 2026-09-07): when the checks modified a tracked
    file that was part of the merge it exits 128 and leaves an `AM` merge state, and when
    they modified an untouched tracked file it exits 0 and leaves that modification. So
    this returns a bounded description of the leftover state -- appended to the refusal's
    own hint -- or `""` when the checkout is genuinely clean. jaxflow NEVER runs
    `git reset --hard` or any other destructive recovery on its own: that is the tech
    lead's decision (spec §5.3, amended 2026-09-07). The abort's own exit code is part of
    the report (round-3 F11): a non-zero abort and a clean one that merely left a file
    modified are different situations for whoever has to fix the checkout."""
    aborted = run(["git", "merge", "--abort"], cwd=repo)
    notes = []
    if aborted.returncode != 0:
        notes.append(f"`git merge --abort` exited {aborted.returncode}")
    # `-q --verify MERGE_HEAD` exits 1 for "no such ref" and 128 for a repository error.
    # Reporting "clean" because the probe itself failed is the one thing this function
    # must not do -- its whole purpose is telling the tech lead what survived (round-5 F9).
    merge_head = run(["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"], cwd=repo)
    if merge_head.returncode == 0:
        notes.append("MERGE_HEAD still present (the checkout is mid-merge)")
    elif merge_head.returncode != 1:
        notes.append("MERGE_HEAD could not be checked")
    dirty = _status_tracked(run, repo)
    if dirty.returncode != 0:
        notes.append("tracked changes could not be checked")
    elif dirty.stdout.strip():
        notes.append(f"tracked changes remain: {jr._bound(dirty.stdout.strip(), 200)}")
    return f" -- abort did not fully clean up: {'; '.join(notes)}" if notes else ""


def _open_pr(repo, project, caller, branch, sha, target, title, *, body_file, run, post, env,
             now, allowlist_root, release_snapshot_sha=None):
    """The pr-open engine (spec Commands > pr open, decisions 1, 5, 8; Ledger & card
    reconciliation). Shared by `cmd_pr_open` and, in Part 2, `cmd_release`'s internal call
    (which supplies `release_snapshot_sha`; `pr open`'s own CLI never does — see Global
    Constraints)."""
    repo_slug = _github_repo_slug(run, repo)
    kind = "release" if branch.startswith("release/") else "feature"

    recorded = _find_recorded_pr(project, branch)
    number = recorded["pr_number"] if recorded is not None else _resolve_pr_number(
        run, repo, repo_slug, branch, target)

    if number is None:
        pushed = run(["git", "push", "origin", f"{sha}:refs/heads/{branch}"], cwd=repo)
        if pushed.returncode != 0:
            raise jc._refuse("push-failed", f"hint: {jr._bound((pushed.stderr or pushed.stdout).strip(), 200)}")
        create_argv = ["gh", "pr", "create", "--repo", repo_slug, "--head", branch,
                        "--base", target, "--title", title]
        if body_file:
            create_argv += ["--body-file", body_file]
        created = run(create_argv, cwd=repo)
        if created.returncode != 0:
            # Spec names no dedicated code for a failed `gh pr create` -- github-unreachable
            # is the closest existing "a GitHub operation failed" bucket (Global Constraints).
            raise jc._refuse("github-unreachable", f"hint: gh pr create failed: {jr._bound((created.stderr or created.stdout).strip(), 200)}")
        m = _GH_PR_URL_RE.match((created.stdout or "").strip())
        if not m:
            raise jc._refuse("github-unreachable", f"hint: gh pr create printed no PR URL: {jr._bound(created.stdout.strip(), 200)}")
        number, pr_url = int(m.group(1)), created.stdout.strip()
    else:
        view = _gh_pr_view(run, repo, repo_slug, number)
        pr_url = view["url"]
        # cold review 75e934eacdca F4: a recorded/list-matched PR whose base no longer matches
        # the approved target must refuse before any push or ledger event -- same
        # pr-identity-mismatch code _cmd_merge_pr uses for this exact shape of mismatch.
        if view["baseRefName"] != target:
            raise jc._refuse("pr-identity-mismatch", f"hint: PR #{number}'s base is {view['baseRefName']!r}, not the approved {target!r}")
        # cold review 24597072c8ac F2: same identity guard as the base check above, for the
        # head branch -- a recorded/resolved PR number whose actual head branch isn't `branch`
        # must refuse before any push or ledger event too.
        if view["headRefName"] != branch:
            raise jc._refuse("pr-identity-mismatch", f"hint: PR #{number}'s head branch is {view['headRefName']!r}, not the approved {branch!r}")
        if view["state"] in ("CLOSED", "MERGED"):
            raise jc._refuse("pr-closed", f"hint: PR #{number} is {view['state'].lower()}; jaxflow never reopens one")
        if view["headRefOid"] != sha:
            if not jc._is_strict_descendant(run, repo, base=view["headRefOid"], head=sha):
                raise jc._refuse("pr-remote-diverged", f"hint: PR #{number}'s remote head {view['headRefOid'][:12]} is not "
                            f"an ancestor of the approved {sha[:12]}; force-push is never used")
            pushed = run(["git", "push", "origin", f"{sha}:refs/heads/{branch}"], cwd=repo)
            if pushed.returncode != 0:
                raise jc._refuse("push-failed", f"hint: {jr._bound((pushed.stderr or pushed.stdout).strip(), 200)}")
            print(f"fast-forwarded PR #{number} to {sha}; a NEW approval is required for this sha")
        else:
            # Spec recovery table: "PR already open" is a true no-op -- no push, no gh
            # mutation, and (this plan's own fix, replayed against
            # test_pr_open_resumes_at_the_same_sha_with_no_mutation, which passes
            # post=_never_post) no re-post of an identical pr-opened event either.
            print(f"PR #{number} already open at {sha}; nothing to do")
            return {"number": number, "url": pr_url, "repo_slug": repo_slug}

    payload = {"repo": repo_slug, "branch": branch, "sha": sha, "base": target,
               "pr_number": number, "pr_url": pr_url, "kind": kind}
    if release_snapshot_sha:
        payload["snapshot_sha"] = release_snapshot_sha
    event = {"project": project, "role": "lead", "type": "pr-opened", "source": "deterministic",
             "emitter": "wrapper", "payload": payload}
    pane = env.get("TMUX_PANE")
    if pane and re.fullmatch(r"%[0-9]{1,10}", pane):
        event["pane"] = pane
    try:
        post(event)
    except Exception as exc:
        refusal = jc._hub_refusal(exc)
        if _hub_retryable(exc):
            print(f"PR #{number} ({pr_url}) is open on GitHub; re-run the same jaxflow pr open to resume")
            refusal.hint = f"hint: {jr._bound(str(exc), 200)}"
        raise refusal from exc

    jc._update_status_md(
        {"repo": str(repo), "kind": "pr-open", "runtime": None if caller == "jaxos" else caller,
         "branch": branch, "dispatch_start": jc._iso8601(now()),
         "worker_summary": f"PR #{number} opened: {pr_url}", "worker_outcome": None},
        run=run, allowlist_root=allowlist_root,
    )
    print(f"PR #{number}: {pr_url}")
    return {"number": number, "url": pr_url, "repo_slug": repo_slug}


def cmd_release(args, *, run=jr.run_command, post=jc._post_event, env=None, now=None,
                 allowlist_root=ALLOWLIST_ROOT_DEFAULT):
    """`jaxflow release` (spec Commands > release, decision 9). Cuts a fresh, named snapshot
    of the Delivery target and opens its PR through `_open_pr` -- the SAME engine `pr open`
    uses, in-process, never a second `jaxflow pr open` shell-out. Reuses an existing OPEN
    release PR untouched (never re-fetches/advances its snapshot); an interrupted attempt
    reconciles via `_open_pr`'s own gh-list fallback rather than duplicating a PR."""
    env = os.environ if env is None else env
    now = now or (lambda: datetime.now().astimezone())
    cwd = Path.cwd().resolve()
    repo = jc._require_toplevel(run, cwd, strict=True)
    settings = general_settings.read_settings()
    if not (settings.get("ok") and settings["data"]["integrations"]["github"]):
        raise Refusal("github-integration-disabled")
    # cold review e83ed7ea5eeb F1: validated BEFORE any remote read or ref write below, not
    # only as the silent status.md skip buried at the end of `_open_pr` -- single-user VPS
    # trust model makes an actual escape unrealistic, but checking before mutating is simply
    # the correct order.
    if not _contained(repo, allowlist_root):
        raise Refusal("path-outside-allowlist")
    # Release presets only (`dual-branch-pr`), same ordering as `pr open`: checked before any
    # ledger read, fetch or ref write. Production target is read after, at `_resolve_production_target`.
    if _resolve_delivery_target(repo, run=run)[2] not in _RELEASE_PRESETS:
        raise Refusal("preset-no-release")
    caller = jc.resolve_caller(env, args.from_caller)
    project = jc.slugify_project(repo.name)
    repo_slug = _github_repo_slug(run, repo)
    production_target = _resolve_production_target(repo)

    latest = _find_latest_release_pr(project, repo_slug)
    if latest is not None and latest.get("branch", "").startswith("release/"):
        view = _gh_pr_view(run, repo, repo_slug, latest["pr_number"])
        # cold review e83ed7ea5eeb F2: the ledger record and the LIVE PR must agree on base
        # and head before reuse -- a same-numbered PR can belong to a different release, and
        # a recorded snapshot_sha the live head has moved past is no longer current. Either
        # mismatch falls through to the normal fetch/reconcile/create path below, never a
        # silent reuse.
        if (view["state"] == "OPEN" and view["baseRefName"] == production_target
                and view["headRefOid"] == latest.get("snapshot_sha")):
            print(f"reusing PR #{latest['pr_number']} ({latest['pr_url']}) -- "
                  f"snapshot {latest.get('snapshot_sha')}")
            return {"number": latest["pr_number"], "url": latest["pr_url"],
                    "snapshot_sha": latest.get("snapshot_sha"), "branch": latest["branch"]}

    delivery_target, _no_preset_block, _preset = _resolve_delivery_target(repo, run=run)
    fetched = run(["git", "fetch", "origin", delivery_target], cwd=repo)
    if fetched.returncode != 0:
        raise jc._refuse("github-unreachable", f"hint: git fetch origin {delivery_target} failed: {jr._bound((fetched.stderr or fetched.stdout).strip(), 200)}")
    snapshot_sha = jc._git_read(run, repo, ["git", "rev-parse", "--verify", f"origin/{delivery_target}^{{commit}}"],
                              shape=r"[0-9a-f]{40}")
    if snapshot_sha is None:
        raise jc._refuse("github-unreachable", f"hint: origin/{delivery_target} does not resolve to a commit after fetch")

    date = now().strftime("%Y-%m-%d")
    branch = f"release/{date}-staging-promotion"
    suffix = 1
    def _release_branch_taken(candidate):
        # F2: an interrupted run can leave a LOCAL ref with no remote counterpart yet --
        # check both, never just origin's. A ref found at the SAME snapshot_sha is THIS
        # release's own interrupted attempt, not a real collision -- decision 9 reconciles it
        # under its ORIGINAL name via _open_pr's own lookup below, never bumps to a new name.
        for ref in (f"refs/heads/{candidate}", f"refs/remotes/origin/{candidate}"):
            found = jc._git_read(run, repo, ["git", "rev-parse", "--verify", ref],
                               shape=r"[0-9a-f]{40}")
            if found is not None and found != snapshot_sha:
                return True
        return False
    while _release_branch_taken(branch):
        suffix += 1
        branch = f"release/{date}-staging-promotion-{suffix}"

    updated = run(["git", "update-ref", f"refs/heads/{branch}", snapshot_sha], cwd=repo)
    if updated.returncode != 0:
        raise jc._refuse("merge-failed", f"hint: could not create local ref {branch}: {jr._bound((updated.stderr or updated.stdout).strip(), 200)}")

    result = _open_pr(repo, project, caller, branch, snapshot_sha, production_target,
                       f"Release: {delivery_target} promotion {date}", body_file=None, run=run,
                       post=post, env=env, now=now, allowlist_root=allowlist_root,
                       release_snapshot_sha=snapshot_sha)
    result["snapshot_sha"] = snapshot_sha
    result["branch"] = branch
    return result


def cmd_pr_open(args, *, run=jr.run_command, post=jc._post_event, env=None, now=None,
                 allowlist_root=ALLOWLIST_ROOT_DEFAULT):
    """`jaxflow pr open` (spec Commands > pr open). No Rafa approval gate of its own
    (decision 1) -- validation mirrors `cmd_merge`'s own sha/branch/target shape checks
    exactly, since both commands pin the same kind of approval."""
    env = os.environ if env is None else env
    now = now or (lambda: datetime.now().astimezone())
    cwd = Path.cwd().resolve()
    repo = jc._require_toplevel(run, cwd, strict=True)
    caller = jc.resolve_caller(env, args.from_caller)
    project = jc.slugify_project(repo.name)
    settings = general_settings.read_settings()
    if not (settings.get("ok") and settings["data"]["integrations"]["github"]):
        raise Refusal("github-integration-disabled")

    if not re.fullmatch(r"[0-9a-f]{40}", args.sha or ""):
        raise jc._refuse("sha-mismatch", "hint: --sha must be the full 40-character commit sha from the approval")
    if not (args.branch and not any(c.isspace() for c in args.branch)
            and 1 <= jc._utf16_len(args.branch) <= 512):
        raise jc._refuse("branch-invalid", "hint: --branch must be a non-empty ref of at most 512 UTF-16 code units "
                    "with no whitespace")
    approved_target = args.target
    if not (isinstance(approved_target, str) and not approved_target.startswith("-")
            and not any(c.isspace() for c in approved_target)
            and 1 <= jc._utf16_len(approved_target) <= 512):
        raise jc._refuse("target-invalid", "hint: --target must be a literal branch name of 1-512 UTF-16 code units "
                    "with no whitespace or leading dash")
    fmt = run(["git", "check-ref-format", f"refs/heads/{approved_target}"], cwd=repo)
    if fmt.returncode != 0:
        raise jc._refuse("target-invalid", f"hint: --target is not a valid branch name: {jr._bound(approved_target, 120)}")

    head = run(["git", "rev-parse", "--verify", f"{args.branch}^{{commit}}"], cwd=repo)
    if head.returncode != 0 or head.stdout.strip() != args.sha:
        raise jc._refuse("sha-mismatch", f"hint: {args.branch}@{head.stdout.strip() or '?'} is not the approved {args.sha}")

    # cold review 9f7f7290c510 F2: the preset-set check comes FIRST. `_required_target_for_branch`
    # resolves the Production target for a release/* head and would raise
    # production-target-unconfigured before this refusal if it ran earlier. A local-preset or
    # no-preset repo must refuse before any push or gh call, never silently deliver.
    preset = _resolve_delivery_target(repo, run=run)[2]
    if preset not in _PR_PRESETS:
        raise jc._refuse("preset-not-pr", "hint: jaxflow pr open only works on a repo whose AGENTS.md Preset is a PR "
                    "preset (`single-branch-pr` or `dual-branch-pr`)")
    required_target, _preset = _required_target_for_branch(repo, args.branch, run=run)
    if approved_target != required_target:
        raise jc._refuse("target-mismatch", f"hint: approved target {jr._bound(approved_target, 120)} differs from the "
                    f"configured {jr._bound(required_target, 120)}")

    return _open_pr(repo, project, caller, args.branch, args.sha, approved_target, args.title,
                     body_file=args.body_file, run=run, post=post, env=env, now=now,
                     allowlist_root=allowlist_root)


def _cleanup_merged_worktree(run, repo, worktree, branch, allowlist_root):
    """Best-effort cleanup after a delivered merge: copy the run reports out of the build
    worktree, then remove it and delete the branch. Fails closed on a failed report copy
    (the worktree is kept, spec §4 guard 7) and never turns a completed merge into a
    refusal: every failure is a printed note."""
    if not (_contained(worktree, allowlist_root) and worktree.is_dir()
            and jc._is_registered_worktree(run, repo, worktree, branch)):
        return
    copy_result = jc._copy_run_reports(worktree, repo)
    if copy_result["failed"]:
        print(f"worktree {worktree} kept (report copy failed)")
        return
    removed = run(["git", "worktree", "remove", str(worktree)], cwd=repo)
    if removed.returncode == 0:
        deleted = run(["git", "branch", "-d", branch], cwd=repo)
        if deleted.returncode != 0:
            print(f"branch {branch} kept ({jr._bound((deleted.stderr or deleted.stdout).strip(), 200)})")
    else:
        print(f"worktree {worktree} kept ({jr._bound(removed.stderr.strip(), 200)})")


def _post_merge_audit(post, env, project, payload, *, retry_msg, warn_bad_pane):
    """Post the `merge-approved` audit event. The delivery is already durable, so a failed
    POST becomes the hub refusal; when the failure is retryable `retry_msg` is printed and
    the refusal carries the hint (resume advice). `pane` is present-or-absent, never null
    (the `caller_pane` discipline): a merge run outside tmux has no pane, which the
    validator's `wrapper` carve-out allows. `warn_bad_pane` (the local merge path) also
    says so when TMUX_PANE is set but unusable -- the validator checks `%<1-10 digits>`,
    and a junk value must not strand a merge that is already committed (round-6 F14); the
    PR path omits the pane silently, as before."""
    event = {"project": project, "role": "lead", "type": "merge-approved",
             "source": "deterministic", "emitter": "wrapper", "payload": payload}
    pane = env.get("TMUX_PANE")
    if pane and re.fullmatch(r"%[0-9]{1,10}", pane):
        event["pane"] = pane
    elif pane and warn_bad_pane:
        print(f"TMUX_PANE={jr._bound(pane, 60)} is not a pane target; reporting without it")
    try:
        post(event)
    except Exception as exc:
        refusal = jc._hub_refusal(exc)
        if _hub_retryable(exc):
            print(retry_msg)
            refusal.hint = f"hint: {jr._bound(str(exc), 200)}"
        raise refusal from exc


def _record_merge_status(run, allowlist_root, *, repo, caller, target, dispatch_start,
                         outcome, checks_audit, checks_cmd, target_note=None):
    """status.md + the closing stdout lines of a delivered merge; returns `OK`. `branch` is
    passed to `_update_status_md` explicitly rather than left to `_current_branch()`
    (part-1 cold review): a RESUME never switches to the target, so the current checkout is
    whatever the tech lead was standing on -- or nothing, from a detached HEAD.
    `target_note` is the local path's `target: ... (no preset block)` line. Order: status
    update -> note -> outcome -> checks line."""
    jc._update_status_md(
        {"repo": str(repo), "kind": "merge", "runtime": None if caller == "jaxos" else caller,
         "branch": target, "dispatch_start": dispatch_start, "worker_summary": outcome,
         "worker_outcome": None},
        run=run, allowlist_root=allowlist_root,
    )
    if target_note:
        print(target_note)
    print(outcome)
    line = _merge_checks_line(checks_audit, checks_cmd)
    if line:
        print(line)
    return jc.OK


def _validate_merge_inputs(args, *, run, repo):
    """Shape-validates every merge input BEFORE any git state is touched: full 40-hex sha,
    `--branch`, ONE canonical `--phase` (UTF-16 bounded, no newline, no U+FEFF), `--target`
    (literal branch name + a read-only `git check-ref-format`). Precedence between several
    invalid inputs is the order below. Returns `(phase, approved_target)`."""
    # An abbreviated sha is an INCOMPLETE approval per merge-contract.md, not a lookup to
    # widen -- refused before any git state is touched.
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha or ""):
        raise jc._refuse("sha-mismatch", "hint: --sha must be the full 40-character commit sha from the approval")

    # `--branch` reaches the commit subject AND the audit payload, where the validator
    # bounds it at `LIMITS.target` (512). `git check-ref-format` happily accepts a
    # 2048-character ref, so an oversized-but-valid branch would commit and only then fail
    # its POST -- the same stranded-merge shape the phase bound prevents, one field over
    # (part-2 cold review round 4 / round-5 F11). Bounded here, ahead of every mutation.
    # UTF-16 code units, not Python characters -- the same unit the validator measures and
    # the same mistake as the phase bound, one field over (round-6 F13). A branch may
    # legally carry astral characters, and `len()` would count each as one where
    # `String.length` counts two.
    if not (args.branch and not any(c.isspace() for c in args.branch)
            and 1 <= jc._utf16_len(args.branch) <= 512):   # -1 means unencodable, not short
        raise jc._refuse("branch-invalid", "hint: --branch must be a non-empty ref of at most 512 UTF-16 code "
                    "units (emoji count as two) with no whitespace")

    # ONE canonical phase title, used byte-for-byte by the commit subject, the audit
    # payload and the resume comparison (round-4 F15/F16). Two shapes had to be refused
    # here rather than accommodated later:
    #   * a title containing a newline -- `git commit -m` accepts it, but `%s` FLATTENS it
    #     into the subject, so the resume equality check would never match again and a
    #     post-commit retry would strand the merge it was meant to finish;
    #   * a title over the event validator's `LIMITS.summary` bound -- the audit row would
    #     silently record a truncated phase while the commit carried the full one.
    # The length is measured in UTF-16 CODE UNITS, not Python characters (part-1 cold
    # review): the validator is TypeScript and `String.length` counts units, so 200 emoji
    # are 200 to Python and 400 to the validator. Bounding by `len()` would let that title
    # commit and then fail its POST, and the documented retry would fail identically --
    # a stranded merge. Refusing costs the tech lead one retype.
    phase = (args.phase or "").strip()
    utf16_len = jc._utf16_len(phase)
    if utf16_len < 0:
        # POSIX argv is decoded with `surrogateescape`, so an undecodable byte reaches here
        # as a lone surrogate that cannot be encoded at all. It has no honest length and no
        # business in a commit subject; refuse instead of crashing past the refusal contract
        # (part-1 cold review round 2).
        phase = ""
    # `\ufeff` is the one character the VALIDATOR calls whitespace and Python's `strip()`
    # does not (part-2 cold review round 4, measured over every BMP code point). The other
    # five differences run the safe way -- Python strips them, so this gate refuses first
    # and the validator never sees them. This one runs the dangerous way, so refuse it
    # here rather than relaxing the validator: only producer-accepts/validator-rejects
    # commits and then fails audit.
    if (not phase or utf16_len > 200 or phase != args.phase
            or any(c in phase for c in "\r\n\t\ufeff")):
        raise jc._refuse("phase-invalid", "hint: --phase must be a single line of at most 200 UTF-16 code units "
                    "(emoji count as two) with no leading or trailing whitespace")

    # `--target` is the approved destination, asserted against policy — never an override
    # and never defaulted from current policy (MOA-458). Shape first, then a read-only
    # `git check-ref-format`; do not trim, normalize, or resolve. SHA/branch/phase above
    # keep their precedence when several inputs are invalid.
    approved_target = getattr(args, "target", None)
    if not (isinstance(approved_target, str) and not approved_target.startswith("-")
            and not any(c.isspace() for c in approved_target)
            and 1 <= jc._utf16_len(approved_target) <= 512):
        raise jc._refuse("target-invalid", "hint: --target must be a literal branch name of 1–512 UTF-16 code "
                    "units with no whitespace or leading dash")
    fmt = run(["git", "check-ref-format", f"refs/heads/{approved_target}"], cwd=repo)
    if fmt.returncode != 0:
        raise jc._refuse("target-invalid", f"hint: --target is not a valid branch name: "
                    f"{jr._bound(approved_target, 120)}")
    return phase, approved_target


def _merge_resume_verify(args, *, run, repo, target, phase):
    """Proves a RESUME is really the approved delivery: the target's merge commit carries
    exactly the subject jaxflow itself writes for this branch/phase (the WHOLE subject, not
    a substring: `--phase` is free text), and the source branch is either gone (exit 1 of
    `show-ref`) or still the approved sha. Returns `(merge_sha, branch_present)`."""
    branch_present = True
    # A resume skips the whole switch/merge/checks/commit sequence, so it never gets
    # the source-branch identity for free -- and it later DELETES that branch. Earn it
    # here (cold review F2): the caller-named branch must still resolve to the approved
    # sha, or resolve to nothing at all (a previous resume already cleaned it up).
    # The merge commit lives on TARGET, and a resume never switches to it -- reading
    # HEAD here would report whatever branch the caller happens to be standing on
    # (found while fixing round-2 F4).
    merge_sha = jc._git_read(run, repo, ["git", "rev-parse", f"{target}^{{commit}}"],
                          shape=r"[0-9a-f]{40}")
    if merge_sha is None:
        raise jc._refuse("sha-mismatch", f"hint: resume refused -- {target} does not resolve to a commit")

    # Resolving to the approved sha does NOT prove this is the branch the approval
    # named: any alias pointing at the same commit resolves identically, and the
    # cleanup below would then delete the alias and its worktree while the delivered
    # branch survived (round-2 F4, reproduced on real git). The merge commit jaxflow
    # itself wrote carries the branch name in its subject; that is the binding. A
    # merge jaxflow did not write is refused rather than resumed.
    # The WHOLE subject, not a substring of it (round-3 F7): `--phase` is free text,
    # so a real merge of `other` titled `Release (merge feat/x)` writes
    # `feat: Release (merge feat/x) (merge other)` -- a substring test would accept a
    # resume for feat/x and delete it. Equality against the subject jaxflow itself
    # writes cannot be forged from the phase title alone.
    subject = jc._git_read(run, repo, ["git", "log", "-1", "--format=%s", merge_sha])
    expected_subject = f"feat: {phase} (merge {args.branch})"
    if subject != expected_subject:
        raise jc._refuse("sha-mismatch", f"hint: resume refused -- {target}'s merge commit {merge_sha} does "
                    f"not record {args.branch} "
                    f"(subject: {jr._bound(subject or '<unreadable>', 120)}; "
                    f"expected: {jr._bound(expected_subject, 120)})")

    # "The branch is gone" and "the ref read failed" are different facts, and
    # `rev-parse --verify` returns 128 for both (round-5 F6). `show-ref --verify`
    # separates them on real git: exit 1 is "no such ref", exit 128 is a repository
    # error. Only exit 1 may skip cleanup; anything else refuses rather than posting
    # an audit for a delivery whose source branch it could not read.
    present = run(["git", "show-ref", "--verify", "--quiet",
                   f"refs/heads/{args.branch}"], cwd=repo)
    if present.returncode == 1:
        branch_present = False
    elif present.returncode != 0:
        raise jc._refuse("sha-mismatch", f"hint: resume refused -- could not read {args.branch}: "
                    f"{jr._bound((present.stderr or present.stdout).strip(), 200)}")
    if branch_present:
        src = jc._git_read(run, repo,
                        ["git", "rev-parse", "--verify", f"{args.branch}^{{commit}}"],
                        shape=r"[0-9a-f]{40}")
        if src != args.sha:
            raise jc._refuse("sha-mismatch", f"hint: resume refused -- {args.branch} now points at "
                        f"{src or '<unreadable>'}, not the approved {args.sha}")
    print(f"resuming: {target} already carries the merge of {args.sha}")
    return merge_sha, branch_present


def _merge_prepare_target(args, *, run, repo, target):
    """Fresh (non-resume) delivery preflight: the approved branch still points at the approved
    sha, the CONTROL repo's tracked tree is clean, and the target is checked out. Returns
    `target_tip` (the target HEAD captured BEFORE `git merge` runs -- never a later HEAD
    re-read, AC 8)."""
    # NO `git switch <branch>` here (round-4 F12, reproduced on real git). `build`
    # leaves the approved branch checked out in its own linked worktree, and git
    # refuses to check the same branch out twice:
    #     fatal: 'feat/x' is already used by worktree at '<builder worktree>'
    # Every real delivery would refuse. Nothing in this verb needs that checkout
    # anyway: `git merge` takes a COMMIT, and the tree that has to be clean is the
    # CONTROL repo's, wherever its HEAD happens to be. A ref read answers the only
    # question the switch was ever asked -- does <branch> point at the approved sha?
    head = run(["git", "rev-parse", "--verify", f"{args.branch}^{{commit}}"], cwd=repo)
    if head.returncode != 0 or head.stdout.strip() != args.sha:
        raise jc._refuse("sha-mismatch", f"hint: {args.branch}@{head.stdout.strip() or '?'} is not the approved "
                    f"{args.sha}")

    # Tracked-only: an untracked build artifact never blocks a delivery (§2.10 #1's
    # allow_untracked rule, implemented here because `merge` does not call preflight()).
    status = _status_tracked(run, repo)
    if status.returncode != 0 or status.stdout.strip():
        # A failed probe is NOT a clean tree (round-4 F4): empty stdout from a git
        # that errored would otherwise read as "nothing dirty" and let the merge
        # proceed on a tree whose state was never established.
        raise jc._refuse(
            "dirty-tracked-tree",
            "hint: git status failed, so the tree could not be checked: "
            f"{jr._bound((status.stderr or status.stdout).strip(), 200)}"
            if status.returncode != 0 else None)

    switched = run(["git", "switch", target], cwd=repo)
    if switched.returncode != 0:
        raise jc._refuse("target-mismatch", f"hint: git switch {target} failed: "
                    f"{jr._bound((switched.stderr or switched.stdout).strip(), 200)}")
    on_target = jc._git_read(run, repo, ["git", "branch", "--show-current"])
    if on_target != target:
        raise jc._refuse("target-mismatch", f"hint: expected to be on {target}, got {on_target or '?'}")
    # New (spec §4.5, 77f30f829248 F4): captured HERE, before `git merge` runs --
    # never a later HEAD re-read (AC 8). `None` on a resume path is never read,
    # since `fast_forward` below is only computed inside `if not resuming:`.
    target_tip = jc._git_read(run, repo, ["git", "rev-parse", "HEAD"],
                           shape=r"[0-9a-f]{40}")
    return target_tip


def _merge_stage_check_commit(args, *, run, repo, project, worktree, target, phase,
                              allowlist_root, target_tip):
    """Stage the merge (`--no-ff --no-commit`), fingerprint the index, reuse or run the
    checks, prove the checks did not change WHAT GETS DELIVERED, commit. Returns
    `(merge_sha, checks_audit)`. The two `git write-tree` calls stay DIRECT `run` calls
    (Decision 8): `_git_read` discards the underlying git failure text, and the F9 test
    expects it in the refusal hint."""
    checks_cmd = args.checks
    merged = run(["git", "merge", "--no-ff", "--no-commit", args.sha], cwd=repo)
    if merged.returncode != 0:
        raise jc._refuse("merge-failed", f"hint: {jr._bound(merged.stderr.strip(), 200)}{_abort_merge(run, repo)}")

    # The exact merge result, as an object id, BEFORE the checks touch anything.
    # `git write-tree` is what `git commit` itself runs; it succeeds here because a
    # merge that left conflicts already refused above.
    # A tree object id, not just an exit code (round-5 F3): these two values are
    # compared to each other, so two malformed-but-equal outputs would report "the
    # checks staged nothing" -- the one thing `git diff --quiet` cannot see.
    # Read directly rather than through `_git_read` (deviation from the plan's literal
    # text, see report): the plan's own F9 test expects the underlying git failure
    # text (e.g. "unable to write new index file") IN the refusal hint, which
    # `_git_read` -- by design -- discards on any failure. The shape check itself is
    # unchanged (round-5 F3: a tree object id, not just an exit code).
    tree_before_probe = run(["git", "write-tree"], cwd=repo)
    tree_before = tree_before_probe.stdout.strip()
    if tree_before_probe.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", tree_before):
        # An index that cannot be fingerprinted is a broken merge, not a dirty check
        # (round-3 F9). Refuse BEFORE spending a checks run on it.
        detail = jr._bound((tree_before_probe.stderr or tree_before_probe.stdout).strip(), 200)
        raise jc._refuse("merge-failed", f"hint: could not fingerprint the merged index: {detail}"
                      + _abort_merge(run, repo))

    # MOA-510 D4: `fast_forward` keeps its meaning ("the target has no commits the
    # branch lacks": the merged tree is byte-identical to the branch head) but is no
    # longer a skip by itself -- the head must also be the one the latest diff review
    # already tested. A moved target never reaches the lookup; an EQUAL tip stays
    # outside `fast_forward` (`_is_strict_descendant` needs `base != head`).
    fast_forward = target_tip is not None and jc._is_strict_descendant(
        run, repo, base=target_tip, head=args.sha)
    if fast_forward:
        reused, _reason, review_id = _merge_checks_reuse(
            run, repo, project, worktree, args.branch, args.sha, checks_cmd,
            allowlist_root=allowlist_root, recheck=args.recheck)
    else:
        reused, review_id = False, None
    checks_audit = _checks_audit(reused, review_id, args.sha)
    if not reused:
        checked = run(["/bin/bash", "-lc", checks_cmd], cwd=repo)
        if checked.returncode != 0:
            raise jc._refuse("checks-failed",
                          f"hint: {checks_cmd} exit {checked.returncode}{_abort_merge(run, repo)}")

        # The checks must not have changed WHAT GETS DELIVERED: a checks command that
        # rewrites a lockfile or a snapshot would otherwise ride along inside the merge.
        # Two comparisons are needed (cold review round 2, F5, reproduced on real git):
        # `git diff --quiet` sees only worktree-vs-index, so a check that edits a tracked
        # file and then runs `git add` passes it while the index -- and therefore the
        # commit -- has already changed. Untracked artifacts the checks created are their
        # own output and are left alone.
        # Same deviation as tree_before, same reason: keep the underlying git failure
        # text available for the hint instead of losing it inside `_git_read`.
        tree_after_probe = run(["git", "write-tree"], cwd=repo)
        tree_after = tree_after_probe.stdout.strip()
        if tree_after_probe.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", tree_after):
            detail = jr._bound((tree_after_probe.stderr or tree_after_probe.stdout).strip(), 200)
            raise jc._refuse("checks-dirtied-tree",
                          f"hint: the index could not be fingerprinted after the checks: {detail}"
                          + _abort_merge(run, repo))
        staged_changed = tree_after != tree_before
        if staged_changed or run(["git", "diff", "--quiet"], cwd=repo).returncode != 0:
            raise jc._refuse("checks-dirtied-tree",
                          "hint: the checks command modified a tracked file "
                          + ("(staged)" if staged_changed else "(unstaged)")
                          + _abort_merge(run, repo))

    # A commit hook can refuse -- this very repository has one (`.githooks/commit-msg`).
    # An unchecked failure here would audit and push a merge that was never committed
    # (cold review F3).
    committed = run(["git", "commit", "-m", f"feat: {phase} (merge {args.branch})"],
                    cwd=repo)
    if committed.returncode != 0:
        raise jc._refuse("merge-failed", f"hint: commit refused: "
                    f"{jr._bound((committed.stderr or committed.stdout).strip(), 200)}"
                    + _abort_merge(run, repo))
    merge_sha = jc._git_read(run, repo, ["git", "rev-parse", "HEAD"],
                          shape=r"[0-9a-f]{40}")
    return merge_sha, checks_audit


def _merge_push(run, repo, target):
    """Push the delivered target to `origin` when there is one. Exit 2 of `git remote
    get-url` is "No such remote" and NOTHING else is (verified on real git, round-5 F4):
    any other non-zero result refuses `push-failed` -- the commit is already durable and the
    documented resume finishes the job. Returns whether a push happened."""
    pushed = False
    # Exit 2 is "No such remote" and NOTHING else is (verified on real git, round-5 F4).
    # Treating every non-zero result as "no remote" would let a broken repository, a
    # permission error or a git that failed to start report a complete local delivery and
    # skip the push entirely. The commit is already durable here, so the refusal is
    # `push-failed` and the documented resume finishes the job.
    remote = run(["git", "remote", "get-url", "origin"], cwd=repo)
    if remote.returncode not in (0, 2):
        raise jc._refuse("push-failed", f"hint: could not probe origin: "
                    f"{jr._bound((remote.stderr or remote.stdout).strip(), 200)}")
    if remote.returncode == 0:
        push = run(["git", "push", "origin", target], cwd=repo)
        if push.returncode != 0:
            raise jc._refuse("push-failed", f"hint: {jr._bound(push.stderr.strip(), 200)}")
        pushed = True
    return pushed


def _pr_already_merged(view, number, sha):
    """Recovery path: the PR is already MERGED (a previous run merged it but recording
    failed). cold review e441aa1770e3 F2: a PR merged with extra commits pushed outside the
    approval flow must never be recorded as an approval of a DIFFERENT sha -- same
    `pr-head-moved` code the open-PR branch uses. Returns GitHub's own merge commit sha."""
    if view["headRefOid"] != sha:
        raise jc._refuse(
            "pr-head-moved",
            f"hint: PR #{number} merged at head {view['headRefOid'][:12]}, not the approved {sha[:12]}")
    merge_sha = (view.get("mergeCommit") or {}).get("oid")
    if not merge_sha:
        raise jc._refuse(
            "github-unreachable",
            f"hint: PR #{number} is merged but its merge commit sha could not be read")
    return merge_sha


def _pr_run_checks(args, *, run, repo, project, allowlist_root):
    """Run (or reuse) `--checks` for an open PR head. Feature heads run in their registered
    build worktree (decision 7); a `release/*` head has no such worktree, so a detached
    temporary checkout of the approved sha is created and torn down in `finally`, success
    or failure. Returns the `checks` audit dict."""
    is_release = args.branch.startswith("release/")
    if not is_release:
        checks_dir = jc._branch_worktree_path(allowlist_root, project, args.branch)
        if not (_contained(checks_dir, allowlist_root) and checks_dir.is_dir()
                and jc._is_registered_worktree(run, repo, checks_dir, args.branch)):
            raise jc._refuse("checks-failed",
                          f"hint: no registered build worktree for {args.branch} at {checks_dir}")
    else:
        checks_dir = (allowlist_root / f"{project}-release-checks-{args.sha[:12]}").resolve()
        added = run(["git", "worktree", "add", "--detach", str(checks_dir), args.sha], cwd=repo)
        if added.returncode != 0:
            raise jc._refuse(
                "merge-failed",
                f"hint: could not create the release checks checkout: {jr._bound((added.stderr or added.stdout).strip(), 200)}")

    try:
        head = run(["git", "rev-parse", "HEAD"], cwd=checks_dir)
        if head.returncode != 0 or head.stdout.strip() != args.sha:
            raise jc._refuse("sha-mismatch", f"hint: {checks_dir} is not at the approved {args.sha}")
        status = _status_tracked(run, checks_dir)
        if status.returncode != 0 or status.stdout.strip():
            raise jc._refuse(
                "dirty-tracked-tree",
                f"hint: git status failed: {jr._bound((status.stderr or status.stdout).strip(), 200)}"
                if status.returncode != 0 else None)
        # MOA-510 D7: AFTER the HEAD and clean-tracked-tree guards above; reuse only adds
        # the review/evidence/command conditions and full-porcelain cleanliness.
        if is_release:
            reused, review_id = False, None
        else:
            reused, _reason, review_id = _merge_checks_reuse(
                run, repo, project, checks_dir, args.branch, args.sha, args.checks,
                allowlist_root=allowlist_root, recheck=args.recheck)
        checks_audit = _checks_audit(reused, review_id, args.sha)
        if not reused:
            checked = run(["/bin/bash", "-lc", args.checks], cwd=checks_dir)
            if checked.returncode != 0:
                raise jc._refuse("checks-failed", f"hint: {args.checks} exit {checked.returncode}")
            status_after = _status_tracked(run, checks_dir)
            if status_after.returncode != 0 or status_after.stdout.strip():
                raise jc._refuse("checks-dirtied-tree", "hint: the checks command modified a tracked file")
        return checks_audit
    finally:
        if is_release:
            # cold review 75e934eacdca F2: no --force (Global Constraints: jaxflow never
            # forces a git/gh mutation). A normal removal can fail (e.g. the checks command
            # left an untracked file); that is non-fatal to the delivery result -- the
            # worktree is kept and its path reported, never force-discarded.
            removed = run(["git", "worktree", "remove", str(checks_dir)], cwd=repo)
            if removed.returncode != 0:
                print(f"release checks worktree {checks_dir} kept "
                      f"({jr._bound((removed.stderr or removed.stdout).strip(), 200)})")


def _revalidate_pr_identity(v, number, sha, target, branch):
    """cold review 75e934eacdca F3 (partial accept): `--match-head-commit` pins the head at
    merge time, but nothing pins the base or the open state -- a PR retargeted or closed
    during the mergeability wait must still be caught. Same refusal codes as the initial
    read, but the check ORDER here is its own (open state, head oid, base, head branch)
    and must NOT be unified with the initial sequence (Decision 9)."""
    if v["state"] != "OPEN":
        raise jc._refuse(
            "pr-closed",
            f"hint: PR #{number} is no longer open ({v['state'].lower()}); jaxflow never reopens or re-merges one")
    if v["headRefOid"] != sha:
        raise jc._refuse(
            "pr-head-moved",
            f"hint: PR #{number}'s head is {v['headRefOid'][:12]}, not the approved {sha[:12]}")
    if v["baseRefName"] != target:
        raise jc._refuse(
            "pr-identity-mismatch",
            f"hint: PR #{number}'s base is {v['baseRefName']!r}, not the approved {target!r}")
    # cold review 24597072c8ac F2: every re-read must re-verify the head branch too,
    # same as the initial read.
    if v["headRefName"] != branch:
        raise jc._refuse(
            "pr-identity-mismatch",
            f"hint: PR #{number}'s head branch is {v['headRefName']!r}, not the approved {branch!r}")


def _pr_await_mergeable(view, *, run, repo, repo_slug, number, sha, target, branch):
    """Wait (up to 3 x 2 s) for GitHub to compute `mergeable`, re-reading and
    re-validating the PR identity after EVERY wait; refuse `mergeability-unknown` if it
    never settles; then re-validate once more immediately before the merge call, even when
    the loop never ran. Returns the latest view. `time.sleep` is looked up on the `time`
    module at call time (23 tests patch `jaxflow.time.sleep`); `_gh_pr_view` likewise."""
    mergeable = view.get("mergeable")
    attempts = 0
    while mergeable == "UNKNOWN" and attempts < 3:
        time.sleep(2)
        view = _gh_pr_view(run, repo, repo_slug, number)
        _revalidate_pr_identity(view, number, sha, target, branch)
        mergeable = view.get("mergeable")
        attempts += 1
    if mergeable == "UNKNOWN":
        raise Refusal("mergeability-unknown")
    # Immediately before the merge call too, even when the loop above never ran (mergeable
    # was never UNKNOWN) -- same reasoning as inside the loop.
    _revalidate_pr_identity(view, number, sha, target, branch)
    return view


def _pr_merge_and_readback(args, *, run, repo, repo_slug, number, phase, target):
    """`gh pr merge` pinned with `--match-head-commit`, then a FRESH `gh pr view` that must
    read back as MERGED at the approved head/base (cold review 24597072c8ac F1: exit 0 does
    not guarantee GitHub merged -- a merge queue or auto-merge can accept asynchronously;
    that shape gets its own `merge-not-completed` and nothing is recorded as merged).
    Returns `(view, merge_sha)`. Do not skip the post-merge read: it is what makes
    `merge-not-completed` possible."""
    merged = run(["gh", "pr", "merge", str(number), "--repo", repo_slug, "--merge",
                  "--match-head-commit", args.sha, "--subject",
                  f"feat: {phase} (merge {args.branch})"], cwd=repo)
    if merged.returncode != 0:
        stderr_text = merged.stderr or merged.stdout or ""
        code = "merge-queue-required" if "merge queue" in stderr_text.lower() else "github-merge-refused"
        checks_out = run(["gh", "pr", "checks", str(number), "--repo", repo_slug], cwd=repo)
        raise jc._refuse(code, f"hint: {jr._bound(stderr_text.strip(), 300)}\nchecks: "
                      f"{jr._bound((checks_out.stdout or checks_out.stderr or '').strip(), 300)}")
    view = _gh_pr_view(run, repo, repo_slug, number)
    # cold review 24597072c8ac F1 (downgraded to LOW -- no project here uses a merge queue
    # -- relabelled): `gh pr merge` exiting 0 does not guarantee GitHub actually merged the
    # PR (a merge queue or auto-merge can accept the request asynchronously). That specific
    # shape -- success exit, still not MERGED -- gets its own refusal so the lead knows
    # GitHub accepted the request without merging and to check the PR directly; nothing is
    # recorded as merged either way.
    if view["state"] != "MERGED":
        raise jc._refuse(
            "merge-not-completed",
            f"hint: gh pr merge exited 0 but PR #{number} is still "
            f"{view['state'].lower()}, not merged -- GitHub may have accepted the "
            f"request without merging it yet (e.g. a merge queue or auto-merge); "
            f"check the PR directly. Nothing was recorded as merged.")
    if view["headRefOid"] != args.sha or view["baseRefName"] != target:
        raise jc._refuse(
            "github-unreachable",
            f"hint: gh pr merge reported success but PR #{number} does not read back as merged")
    merge_sha = (view.get("mergeCommit") or {}).get("oid")
    if not merge_sha:
        raise jc._refuse(
            "github-unreachable",
            f"hint: PR #{number} merged but no merge commit sha was reported")
    return view, merge_sha


def _sync_local_target(run, repo, target):
    """Best-effort local sync (decision 10): fetch + fast-forward the local target only when
    clean and direct; a divergence or a dirty tree is reported, never reset or force."""
    run(["git", "fetch", "origin", target], cwd=repo)
    switched = run(["git", "switch", target], cwd=repo)
    if switched.returncode == 0:
        status = _status_tracked(run, repo)
        if status.returncode == 0 and not status.stdout.strip():
            ff = run(["git", "merge", "--ff-only", f"origin/{target}"], cwd=repo)
            if ff.returncode != 0:
                print(f"local {target} not fast-forwarded (diverged from origin/{target}); left as-is")
        else:
            print(f"local {target} not synced (tracked tree not clean)")


def _cmd_merge_pr(args, *, repo, project, caller, target, phase, run, post, env, now,
                   allowlist_root):
    """The PR-preset merge sequence (spec Commands > merge (PR path), decisions 3-7, 10).
    No local git merge at all -- delivery happens on GitHub via `gh pr merge`. Feature heads
    run --checks in their registered build worktree (decision 7); a release/* head has no
    such worktree, so this creates a detached temporary checkout of the approved sha and
    tears it down after, success or failure."""
    repo_slug = _github_repo_slug(run, repo)
    # Ledger & card reconciliation: the recorded PR number if the hub has one for this branch,
    # otherwise a `gh pr list` fallback -- zero matches means no PR exists at all (pr-not-found,
    # unlike `pr open` which would proceed to create one); more than one refuses pr-ambiguous
    # (already handled inside _resolve_pr_number, Task 5).
    recorded = _find_recorded_pr(project, args.branch)
    number = recorded["pr_number"] if recorded is not None else _resolve_pr_number(
        run, repo, repo_slug, args.branch, target)
    if number is None:
        raise jc._refuse("pr-not-found", f"hint: no recorded PR for {args.branch} -- run jaxflow pr open first")

    view = _gh_pr_view(run, repo, repo_slug, number)
    if view["baseRefName"] != target:
        raise jc._refuse("pr-identity-mismatch", f"hint: PR #{number}'s base is {view['baseRefName']!r}, not the approved {target!r}")
    # cold review 24597072c8ac F2: base was already checked; the head BRANCH must be verified
    # too, not just its sha (below) -- a recorded/resolved PR number pointing at the wrong
    # branch is a distinct identity mismatch, unconditionally, before MERGED vs. not is decided.
    if view["headRefName"] != args.branch:
        raise jc._refuse("pr-identity-mismatch", f"hint: PR #{number}'s head branch is {view['headRefName']!r}, not the approved {args.branch!r}")

    merge_sha = None
    if view["state"] == "MERGED":
        # Recovery table: already merged, local recording failed -- skip checks/merge, just
        # record + sync + cleanup below, using GitHub's own merge commit sha.
        merge_sha = _pr_already_merged(view, number, args.sha)
        checks_audit = {"mode": "resumed"}
    else:
        if view["state"] == "CLOSED":
            raise jc._refuse("pr-closed", f"hint: PR #{number} is closed and unmerged; jaxflow never reopens one")
        if view["headRefOid"] != args.sha:
            raise jc._refuse("pr-head-moved", f"hint: PR #{number}'s head is {view['headRefOid'][:12]}, not the approved {args.sha[:12]}")

        checks_audit = _pr_run_checks(
            args, run=run, repo=repo, project=project, allowlist_root=allowlist_root)

        view = _pr_await_mergeable(
            view, run=run, repo=repo, repo_slug=repo_slug, number=number,
            sha=args.sha, target=target, branch=args.branch)

        view, merge_sha = _pr_merge_and_readback(
            args, run=run, repo=repo, repo_slug=repo_slug, number=number, phase=phase,
            target=target)

    _post_merge_audit(
        post, env, project,
        {"phase": phase, "branch": args.branch, "sha": args.sha, "target": target,
         "approved_by": "rafa", "merge_sha": merge_sha, "pr_number": number,
         "pr_url": view["url"], "checks": checks_audit},
        retry_msg=f"PR #{number} merged as {merge_sha}; re-run the same jaxflow merge to resume",
        warn_bad_pane=False)

    _sync_local_target(run, repo, target)

    if not args.branch.startswith("release/"):
        _cleanup_merged_worktree(
            run, repo, jc._branch_worktree_path(allowlist_root, project, args.branch),
            args.branch, allowlist_root)

    return _record_merge_status(
        run, allowlist_root, repo=repo, caller=caller, target=target,
        dispatch_start=jc._iso8601(now()),
        outcome=f"merged {merge_sha} via PR #{number} ({view['url']})",
        checks_audit=checks_audit, checks_cmd=args.checks)


def cmd_merge(args, *, run=jr.run_command, post=jc._post_event, env=None, now=None,
              allowlist_root=ALLOWLIST_ROOT_DEFAULT):
    """Delivers an approved branch (spec §5). Synchronous and foreground by design: Rafa's
    approval already named this phase/branch/sha, and the tech lead is waiting right here
    for a pass/fail — there is nothing to poll later, so `merge` allocates no run id, posts
    no `run-started`/`run-finished`, and owns no tmux session. Its only ledger row is the
    `merge-approved` audit event in §5.3."""
    env = os.environ if env is None else env
    now = now or (lambda: datetime.now().astimezone())
    started_at = jc._iso8601(now())

    cwd = Path.cwd().resolve()
    repo = jc._require_toplevel(run, cwd, strict=True)
    caller = jc.resolve_caller(env, args.from_caller)
    project = jc.slugify_project(repo.name)

    phase, approved_target = _validate_merge_inputs(args, run=run, repo=repo)

    target, no_preset_block, preset = _resolve_delivery_target(repo, run=run)
    # Same bound as `--branch`, and for the same reason: the target is audit payload too,
    # and it can come from a hand-written `AGENTS.md` line or from `_default_branch`'s
    # symbolic-ref read, neither of which validates a shape (round-5 F8/F11). A policy
    # that does not name a usable target is exactly what `preset-unknown` already means.
    if not (target and not any(c.isspace() for c in target)
            and 1 <= jc._utf16_len(target) <= 512):
        raise jc._refuse("preset-unknown", f"hint: the delivery target is not a usable ref: "
                    f"{jr._bound(target or '<empty>', 120)}")

    if preset in _PR_PRESETS:
        settings = general_settings.read_settings()
        if not (settings.get("ok") and settings["data"]["integrations"]["github"]):
            raise Refusal("github-integration-disabled")
        required_target, _preset = _required_target_for_branch(repo, args.branch, run=run)
        if approved_target != required_target:
            raise jc._refuse("target-mismatch", f"hint: approved target {jr._bound(approved_target, 120)} differs from "
                        f"the configured {jr._bound(required_target, 120)}")
        if args.branch.startswith("release/"):
            return _cmd_merge_pr(args, repo=repo, project=project, caller=caller,
                                  target=required_target, phase=phase, run=run, post=post,
                                  env=env, now=now, allowlist_root=allowlist_root)
        worktree = jc._branch_worktree_path(allowlist_root, project, args.branch)
        with jresume.worktree_claim(repo, worktree):
            return _cmd_merge_pr(args, repo=repo, project=project, caller=caller,
                                       target=required_target, phase=phase, run=run, post=post,
                                       env=env, now=now, allowlist_root=allowlist_root)

    if approved_target != target:
        raise jc._refuse("target-mismatch", f"hint: approved target {jr._bound(approved_target, 120)} "
                    f"differs from policy target {jr._bound(target, 120)}")

    worktree = jc._branch_worktree_path(allowlist_root, project, args.branch)
    with jresume.worktree_claim(repo, worktree):
        _refuse_nonterminal_builder(project, repo, args.branch)
        # Resume check (§5.2). `--verify` rather than a bare rev-parse: the bare form exits
        # non-zero but ECHOES the unresolved argument on stdout for a non-merge commit, so the
        # returncode is the only reliable signal there; `--verify` gives an empty stdout too.
        # Both the returncode AND an exact sha match are required, so a target whose HEAD is
        # some unrelated merge is never mistaken for this delivery.
        probe = run(["git", "rev-parse", "--verify", f"{target}^2"], cwd=repo)
        resuming = probe.returncode == 0 and probe.stdout.strip() == args.sha

        merge_sha = None
        branch_present = True
        if resuming:
            merge_sha, branch_present = _merge_resume_verify(
                args, run=run, repo=repo, target=target, phase=phase)
        else:
            target_tip = _merge_prepare_target(args, run=run, repo=repo, target=target)

        checks_cmd = args.checks
        checks_audit = {"mode": "resumed"}   # D5: a resume neither runs nor reuses the checks
        if not resuming:
            merge_sha, checks_audit = _merge_stage_check_commit(
                args, run=run, repo=repo, project=project, worktree=worktree, target=target,
                phase=phase, allowlist_root=allowlist_root, target_tip=target_tip)

        # One guard for BOTH paths (round-4 F5). `merge-approved` requires a full 40-hex sha
        # and the validator rejects anything else, so an unchecked probe would turn a durable
        # commit into an un-auditable one. Nothing is aborted here: on the normal path the
        # commit already landed, and re-running the same `jaxflow merge` resumes from it.
        if merge_sha is None:
            raise jc._refuse("merge-failed", "hint: the merge commit exists but its sha could not be read; "
                        "re-run the same jaxflow merge to resume")

        _post_merge_audit(
            post, env, project,
            {"phase": phase, "branch": args.branch, "sha": args.sha,   # already <= 200
             "target": target, "approved_by": "rafa", "merge_sha": merge_sha,
             "checks": checks_audit},
            retry_msg=f"merge commit {merge_sha} kept; re-run the same jaxflow merge to resume",
            warn_bad_pane=True)

        pushed = _merge_push(run, repo, target)

        # Everything below is best-effort: the delivery is durable and pushed, and a failure
        # here must never turn a completed merge into a refusal.
        if branch_present:
            _cleanup_merged_worktree(run, repo, worktree, args.branch, allowlist_root)

        outcome = (f"merged {merge_sha} pushed origin/{target}" if pushed
                   else f"merged {merge_sha} — no remote delivery configured")
        return _record_merge_status(
            run, allowlist_root, repo=repo, caller=caller, target=target,
            dispatch_start=started_at, outcome=outcome, checks_audit=checks_audit,
            checks_cmd=checks_cmd,
            target_note=f"target: {target} (no preset block)" if no_preset_block else None)
