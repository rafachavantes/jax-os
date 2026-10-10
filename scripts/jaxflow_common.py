"""jaxflow common module -- P2 split of scripts/jaxflow.py (lean spec, Decision 18)."""
from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import stat
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
import general_settings
import jaxflow_run as jr
import jaxflow_env as jenv
from jax_init import Refusal, _contained, _is_secret_path, canonicalize_target
from jaxflow_hook import UUID_RE


OK = 0

REFUSED = 2

SCRIPT_PATH = Path(__file__).resolve().with_name("jaxflow.py")

# 4 MiB is far above any real run report and far below anything that could be used to
# fill the control repo (spec §5.3 report-copy bound).
REPORT_COPY_MAX = 4 << 20

# ~/.jax-os/callbacks/<caller_session>/<run_id>.{json,claimed,line} -- mirrors
# jaxflow_run.DB_PATH's own module-constant-plus-patch.object test-isolation pattern.
# Never read HOME to relocate this; tests patch the constant itself.
CALLBACKS_ROOT = jenv.jaxos_home() / "callbacks"

# D9: the one-time hook install target. Read-only from this file's perspective.
CLAUDE_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

# Drift-checked against src/lib/workflow.ts LIMITS. Do not invent a value.
HUB_CAPS = {
    "verify": 500, "build": 500, "phase": 64, "target": 512,
    "callerSession": 128, "callerPane": 128, "model": 384, "effort": 64,
    "session": 80, "repo": 200,
}

_CAP_LABELS = {"callerSession": "caller-session", "callerPane": "caller-pane"}

_STARTED_TO_CAP = {"caller_session": "callerSession", "caller_pane": "callerPane"}

_LONG_CMD_HINT = (
    "hint: move long commands into a package.json script and pass the script (e.g. pnpm run verify)"
)

_REFUSAL_MAP = {
    "detached HEAD": "detached-head",
    "builder cannot run on default branch": "builder-on-default-branch",
    "runtime not allowed": "runtime-not-allowed",
}

_SLUG_RUN_RE = re.compile(r"[^a-z0-9]+")

# The exact shape `uuid.uuid4().hex[:12]` always produces (fixes Part 3 diff-review F1,
# shared by both worker kinds since Part 2 diff-review F1) -- every worker validates its
# manifest's `run_id` against this BEFORE using it to construct any path, since an
# absolute-looking value would otherwise make `safe_run_paths`'s own
# `reports_root / f"{run_id}.md"` escape the reports root entirely (joining a Path with an
# absolute string discards the left side).
_RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")

_CALLER_SESSION_VAR = {
    "claude": "CLAUDE_CODE_SESSION_ID",
    "codex": "CODEX_THREAD_ID",
    # Phase 2 (Mission Control B, spec §8): the dashboard is a third caller with no interactive
    # session -- the route sets this to the constant "jaxos"; the value only has to be non-empty.
    "jaxos": "JAXOS_CALLER_SESSION",
}


def _map_refusal(message: str) -> str:
    return _REFUSAL_MAP.get(message, message)


def _caps_from_started(payload):
    return {_STARTED_TO_CAP.get(key, key): value for key, value in payload.items()}


def _check_hub_caps(fields):
    for name, value in fields.items():
        cap = HUB_CAPS.get(name)
        if cap is None or value is None or type(value) is not str:
            continue
        n = _utf16_len(value)
        if n > cap:
            raise _refuse(f"{_CAP_LABELS.get(name, name)}-too-long ({n} > {cap})",
                          _LONG_CMD_HINT if name in ("verify", "build") else None)


def _dispatch_run(*, run, post, repo, project, phase, kind, role, run_id, session, manifest,
                   manifest_dir, started_payload, cleanup_paths):
    """Shared dispatch tail for `build` and (Part 2) `review --diff`: write the manifest,
    POST run-started, launch the tmux worker, and -- on ANY failure, including an exception
    from `run()` itself (fixes cold review F7) -- clean up and refuse. `dispatch_review`
    (slice a) keeps its own inline copy of this same sequence untouched, to avoid touching
    already-shipped, already-tested code for a ~25-line extraction."""
    _check_hub_caps(_caps_from_started(started_payload))
    manifest_path = manifest_dir / "manifest.json"
    manifest_dir.mkdir(parents=True, exist_ok=False)
    _write_manifest(manifest_path, manifest)

    _warn_if_claude_callback_hook_missing(manifest)

    started_event = {
        "run_id": run_id, "project": project, "role": role, "type": "run-started",
        "source": "deterministic", "emitter": "wrapper", "payload": started_payload,
    }
    try:
        post(started_event)
    except Exception as exc:
        shutil.rmtree(manifest_dir, ignore_errors=True)
        for p in cleanup_paths:
            p.unlink(missing_ok=True)
        raise _hub_refusal(exc) from exc

    cmd = " ".join(
        shlex.quote(part) for part in (
            "env", "HONCHO_ENABLED=false", "JAXFLOW_ONESHOT=1",
            sys.executable, str(SCRIPT_PATH), "--run-worker", str(manifest_path),
        )
    )
    # A launcher EXCEPTION (tmux missing, fork failure, ...) is treated exactly like a
    # non-zero exit (fixes cold review F7) -- both leave the run-started row with no
    # session, and both need the same cancelled-row-then-refuse sequence.
    try:
        result = run(["tmux", "new-session", "-d", "-s", session, "-c", str(repo), cmd])
        launch_failed = result.returncode != 0
    except Exception:
        launch_failed = True
    if launch_failed:
        cancelled_payload = {
            "phase": phase, "exit_code": None, "contract_status": "cancelled",
            "report_path": None, "summary": "tmux new-session failed at dispatch",
        }
        if role == "builder":
            cancelled_payload["head_sha"] = None
        cancelled_event = {
            "run_id": run_id, "project": project, "role": role, "type": "run-finished",
            "source": "deterministic", "emitter": "wrapper", "payload": cancelled_payload,
        }
        try:
            post(cancelled_event)
        except Exception:
            pass
        shutil.rmtree(manifest_dir, ignore_errors=True)
        for p in cleanup_paths:
            p.unlink(missing_ok=True)
        raise Refusal("tmux-failed")
    _write_callback_pointer(manifest)
    return run_id


def _cleanup_worktree(worktree, branch, *, run, repo):
    """Undoes a SUCCESSFUL `git worktree add -b` (fixes cold review round 2 F1): a raw
    `shutil.rmtree` on the reserved directory leaves Git's own `.git/worktrees/<id>` admin
    entry behind, which then makes the branch look "in use" and can make a plain
    `git branch -D` refuse. `git worktree remove --force` clears the admin entry first;
    `git worktree prune` and a best-effort `shutil.rmtree` cover the (rare) case the
    directory somehow survives; `git branch -D` runs last, once the worktree is fully gone.
    Never call this for a FAILED `git worktree add -b` -- a failed add never registers a
    worktree admin entry, so there is nothing to remove (see `dispatch_build`'s own
    add-failure branch, which never calls this helper)."""
    run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo)
    run(["git", "worktree", "prune"], cwd=repo)
    shutil.rmtree(worktree, ignore_errors=True)
    run(["git", "branch", "-D", branch], cwd=repo)


def slugify_project(basename: str) -> str:
    # Capped at 40, not PROJECT_RE's 64 (fixes branch review F2): jax-<slug>-<kind>-<run_id>
    # must stay under the ledger's 80-char `session` bound -- 4 + 40 + 1 + 5("build") + 1
    # + 12 = 63 < 80.
    collapsed = _SLUG_RUN_RE.sub("-", basename.lower()).strip("-")
    return collapsed[:40].rstrip("-") or "repo"


def resolve_caller(env, from_flag):
    if from_flag:
        return from_flag
    has_claude = bool(env.get("CLAUDECODE"))
    has_codex = bool(env.get("CODEX_THREAD_ID"))
    if has_claude and not has_codex:
        return "claude"
    if has_codex and not has_claude:
        return "codex"
    raise Refusal("caller-unknown")


def _require_caller_session(env, caller):
    # Acceptance smoke 2026-09-06: an unset/empty caller session used to sail through as
    # `None`/"" and only fail once it reached the hub's `run-started` validation, where it
    # surfaced as the unrelated-looking `hub-unreachable`. Refusing here, before the
    # manifest directory exists and before any POST, gives the real reason instead.
    var = _CALLER_SESSION_VAR[caller]
    session = env.get(var)
    if not session:
        raise _refuse("caller-session-missing", f"hint: set {var}")
    return session


def _no_callback(args, caller):
    """Phase 2 spec §8 / Decision 6: a dashboard-dispatched run has no interactive session to
    call back into -- `no_callback` is forced for caller "jaxos" regardless of the flag, so
    neither the codex-queue nor the claude file-drop branch is ever reached for it."""
    return bool(getattr(args, "no_callback", False)) or caller == "jaxos"


def _enabled_agents():
    """MOA-504 D8: the integrations.agents switches, read live. Fail CLOSED: a malformed or
    unreadable settings.json refuses review and build, never falls back to defaults."""
    settings = general_settings.read_settings()
    if not settings.get("ok"):
        raise _refuse(settings.get("error") or "settings-unreadable", "hint: fix settings.json or save /settings")
    return settings["data"]["integrations"]["agents"]


def _reviewer_runtime_line(runtime, fallback):
    return f"reviewer_runtime: {runtime} (fallback: {fallback})" if fallback else ""


def _iso8601(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _branch_worktree_path(allowlist_root, project, branch, *, resolve=True):
    """`<allowlist_root>/<project>-<branch with '/' as '-'>` -- the build worktree `build`
    reserves for a branch. `resolve=False` for the two callers that must NOT canonicalize
    (`dispatch_build` reserves a path that does not exist yet; `cmd_result` compares the
    string form before it resolves anything)."""
    path = allowlist_root / f"{project}-{branch.replace('/', '-')}"
    return path.resolve() if resolve else path


def _manifest_dir(repo: Path, run_id: str) -> Path:
    return repo / ".local" / "runs" / run_id


def _spool_path(control_repo, run_id):
    """`<control-repo>/.local/runs/<run_id>/finished.json` (spec D3) -- a sibling of
    manifest.json/child.log, in the SAME directory every worker already writes to."""
    return _manifest_dir(control_repo, run_id) / "finished.json"


def _write_spool_at(path, event):
    """tmp + os.replace (spec D3) -- the shared write primitive. `_write_spool` (below)
    is the normal-completion/signal-path caller, which always has a trustworthy
    `control_repo`; MOA-474 cold review F1's pre-launch refusal fix (Task 3 Step 2.4)
    calls this directly for the two refusal functions that do NOT (identity
    validation is exactly what just failed there), deriving the run directory from
    `manifest_path` instead -- one write implementation, not two."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(event), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _write_spool(control_repo, run_id, event):
    """tmp + os.replace (spec D3), written BEFORE the terminal POST -- the reaper's
    next tick or `jaxflow cancel` can re-POST this exact event if delivery failed."""
    return _write_spool_at(_spool_path(control_repo, run_id), event)


def _delete_spool(control_repo, run_id):
    _spool_path(control_repo, run_id).unlink(missing_ok=True)


_CHAIN_MAX_DEPTH = 32


def _safe_run_subpath(repo, subdir, run_id, suffix=""):
    """Resolves `<repo>/.local/<subdir>/<run_id><suffix>`, refusing (None) a `run_id`
    that isn't canonical 12-hex (`_RUN_ID_RE`) or whose resolved path escapes
    `<repo>/.local/<subdir>` -- run_id can come from disk, so a valid-shaped id behind
    an escaping symlink still refuses (F2)."""
    if not _RUN_ID_RE.match(run_id):
        return None
    # ponytail: base dir is trusted; containment guards the run id only. A symlinked
    # `.local/runs`/`.local/reports` is the operator's own control repo, not untrusted
    # input (cr 99ec95af61a2 F2, rejected by the tech lead).
    try:
        base = (repo / ".local" / subdir).resolve()
        candidate = (base / f"{run_id}{suffix}").resolve()
    except (OSError, RuntimeError):  # symlink loop -> RuntimeError (cr e2766def7714 F1)
        return None
    if not _contained(candidate, base):
        return None
    return candidate


def _load_manifest(run_dir):
    """The decoded `manifest.json` of a run directory, whatever JSON value it holds.
    Raises exactly what the inline reads raised (`OSError`, `ValueError`/`JSONDecodeError`,
    `RecursionError`): each caller keeps its own `try/except`, its non-dict handling and
    its own refusal code. `run_worker`'s direct parse (it must PROPAGATE read errors) and
    `_read_prior_manifest` (no-follow open) deliberately do not use it."""
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _load_diff_review_node(con, repo, run_id):
    """Everything the item 9 chain walk needs for one diff-review run: its own manifest
    (base_sha/head_sha/since_review_run_id), its ledger target/verdict, and its linked
    builder run's own target (lineage). None on a failed `_safe_run_subpath` check, a
    missing/unreadable manifest, a manifest whose own `kind` isn't `diff`, no `kind: diff`
    run-started row, or no builder run."""
    manifest_dir = _safe_run_subpath(repo, "runs", run_id)
    if manifest_dir is None:
        return None
    try:
        manifest = _load_manifest(manifest_dir)
    except (OSError, ValueError):
        return None
    if manifest.get("kind") != "diff":
        # F1 (diff review 4884f63bdd16): the manifest's OWN kind, not just the ledger
        # row's, must say `diff` -- a mismatch is treated exactly like a missing/
        # unreadable manifest.
        return None
    started = con.execute(
        "SELECT payload FROM workflow_events WHERE run_id = ? AND role = 'reviewer' "
        "AND type = 'run-started' LIMIT 1", (run_id,),
    ).fetchone()
    if not started:
        return None
    started_payload = json.loads(started["payload"])
    if started_payload.get("kind") != "diff":
        return None
    finished = con.execute(
        "SELECT payload FROM workflow_events WHERE run_id = ? AND role = 'reviewer' "
        "AND type = 'run-finished' LIMIT 1", (run_id,),
    ).fetchone()
    finished_payload = json.loads(finished["payload"]) if finished else None
    builder_run_id = manifest.get("builder_run_id")
    builder_target = None
    if builder_run_id:
        b_started = con.execute(
            "SELECT payload FROM workflow_events WHERE run_id = ? AND role = 'builder' "
            "AND type = 'run-started' LIMIT 1", (builder_run_id,),
        ).fetchone()
        if b_started:
            builder_target = json.loads(b_started["payload"]).get("target")
    return {
        "run_id": run_id, "target": started_payload.get("target"), "builder_target": builder_target,
        "base_sha": manifest.get("base_sha"), "head_sha": manifest.get("head_sha"),
        "since_review_run_id": manifest.get("since_review_run_id"),
        "contract_status": finished_payload.get("contract_status") if finished_payload else None,
        "verdict": finished_payload.get("verdict") if finished_payload else None,
    }


def _walk_review_chain(load_node, start_run_id):
    """Follows `since_review_run_id` links BACKWARD from `start_run_id` to the chain's
    root. Returns `(chain, None)` root-first on success, else `(None, (kind, run_id))`
    with kind in `"missing"`/`"cycle"`/`"depth"` -- bounded to `_CHAIN_MAX_DEPTH`."""
    chain = []
    seen = set()
    current = start_run_id
    while True:
        if current in seen:
            return None, ("cycle", current)
        if len(chain) >= _CHAIN_MAX_DEPTH:
            return None, ("depth", current)
        seen.add(current)
        node = load_node(current)
        if node is None:
            return None, ("missing", current)
        chain.append(node)
        nxt = node["since_review_run_id"]
        if not nxt:
            break
        current = nxt
    chain.reverse()
    return chain, None


def _read_manifest_field(repo, run_id, field):
    try:
        data = _load_manifest(_manifest_dir(repo, run_id))
    except (OSError, ValueError, RecursionError):
        return None
    if type(data) is not dict:
        return None
    return data.get(field)


def _write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    os.chmod(path, 0o644)


def _write_callback_file(caller_session, run_id, suffix, content, *, context):
    """Atomically writes `<CALLBACKS_ROOT>/<caller_session>/<run_id><suffix>` (temp file
    + os.replace) after both values pass the path-safety gate (`fullmatch`, never
    `.match`, which accepts a trailing newline). Shared by the dispatch pointer write
    (`.json`) and the completion-line write (`.line`). Never raises: a gate failure or an
    OSError prints ONE bounded stderr warning naming `context` (the Codex branch's
    "<context> failed: <run_id> <reason>; use jaxflow status/result" wording) and
    returns None, so neither dispatch nor `_send_callback` is ever blocked by it. At the
    gate-failure print `run_id` is still unvalidated (it may carry newlines or control
    characters), so it is rendered with `ascii(run_id)[:80]`: escaped, one line, bounded.
    The write-error print uses it as-is because it has already passed the gate."""
    if not (
        isinstance(caller_session, str) and UUID_RE.fullmatch(caller_session)
        and isinstance(run_id, str) and _RUN_ID_RE.fullmatch(run_id)
    ):
        print(f"{context} failed: {ascii(run_id)[:80]} path-unsafe; use jaxflow status/result", file=sys.stderr)
        return None
    try:
        session_dir = CALLBACKS_ROOT / caller_session
        session_dir.mkdir(parents=True, exist_ok=True)
        dest = session_dir / f"{run_id}{suffix}"
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, dest)
        return dest
    except OSError:
        print(f"{context} failed: {run_id} write-error; use jaxflow status/result", file=sys.stderr)
        return None


def _write_callback_pointer(manifest):
    """D4/§5.1: for a Claude caller with callbacks enabled, atomically registers the
    pending pointer `{run_id, kind}` this run's eventual `_send_callback` call will look
    for. Called only once the worker's tmux session has launched successfully: a refusal
    or launch failure before that point never reaches this call, so there is nothing to
    clean up on a hub-unreachable/tmux-failed/launcher-exception path. No race with a
    fast-finishing run: `_send_callback` writes `.line` independently of this pointer and
    the hook delivers a `.line` already present at claim time. Returns the written Path, or
    None (Codex caller, `--no-callback`, or a caller_session/run_id that fails the
    path-safety gate -- either way dispatch still proceeds; `_write_callback_file`
    already prints the one bounded warning)."""
    if manifest.get("caller") != "claude" or manifest.get("no_callback"):
        return None
    return _write_callback_file(
        manifest.get("caller_session"), manifest.get("run_id"), ".json",
        json.dumps({"run_id": manifest.get("run_id"), "kind": manifest.get("kind")}),
        context="callback pointer write",
    )


def _claude_callback_hook_installed():
    """Structural check, not a substring search. True only if
    CLAUDE_SETTINGS_PATH parses as JSON and some `hooks.PostToolUse[]` entry has
    `matcher == "Bash"` with a `hooks[]` entry that has `type == "command"`,
    `asyncRewake is True`, and a `command` ending in `jaxflow_hook.py claude-callback`.
    Any read/parse failure (missing file, malformed JSON, unexpected shape) counts as
    missing -- best-effort, never raises."""
    try:
        data = json.loads(CLAUDE_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    try:
        entries = data["hooks"]["PostToolUse"]
    except (KeyError, TypeError):
        return False
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("matcher") != "Bash":
            continue
        hooks = entry.get("hooks")
        if not isinstance(hooks, list):
            continue
        for h in hooks:
            if not isinstance(h, dict):
                continue
            if (h.get("type") == "command" and h.get("asyncRewake") is True
                    and isinstance(h.get("command"), str)
                    and h["command"].endswith("jaxflow_hook.py claude-callback")):
                return True
    return False


def _warn_if_claude_callback_hook_missing(manifest):
    """D9: a missing `claude-callback` PostToolUse hook in CLAUDE_SETTINGS_PATH prints
    one best-effort stderr warning naming the run and pointing at `jaxflow
    status/result` -- never blocks dispatch, never raises. Called right after the
    manifest is written, independent of the launch outcome and of whether
    `_write_callback_pointer` above will later succeed (a malformed caller_session still
    deserves the missing-hook warning too -- they are two unrelated facts)."""
    if manifest.get("caller") != "claude" or manifest.get("no_callback"):
        return
    if not _claude_callback_hook_installed():
        print(
            f"jaxflow: no claude-callback hook installed in {CLAUDE_SETTINGS_PATH} -- "
            f"run {manifest.get('run_id')} will not wake this session on finish; use "
            f"jaxflow status/result {manifest.get('run_id')}, or ask the tech lead to "
            "install the hook",
            file=sys.stderr,
        )


_LAUNCH_KEYS = (
    "profile_name", "settings_revision", "model", "runtime_model", "effort", "credential",
)


def _post(url, payload, *, opener=urllib.request.urlopen, timeout=5):
    """jaxflow's own tiny HTTP helper. Returns (status_code, body_dict) for any real HTTP
    response, error statuses included -- callers that need to distinguish 409 from every
    other failure (cmd_cancel) read the status directly instead of losing it to an
    exception. Raises only on a transport error (connection refused, timeout, DNS, ...);
    never rely on the retired jr.post_event here, since that helper's non-2xx/2xx collapse
    into one exception is exactly what loses the status code."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with opener(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            decoded = json.loads(exc.read().decode("utf-8"))
        except (ValueError, OSError):
            decoded = {}
        return exc.code, decoded


class HubRejected(RuntimeError):
    def __init__(self, status, error=None):
        self.status = status
        self.error = error
        super().__init__(f"{status} {error}" if isinstance(error, str) else str(status))


def _refuse(code, hint=None):
    """Build the `Refusal` for `code` with its optional `.hint` (the second stderr line
    `main` prints). Callers write `raise _refuse(...)`, so the exception type, `.code`,
    `.hint` and the traceback frame stay what the inline `exc = Refusal(code);
    exc.hint = ...; raise exc` form produced. `.hint` is set only when given (never None),
    exactly like the inline form where a refusal without a hint has no attribute at all.
    `Refusal` itself stays in `jax_init.py`."""
    exc = Refusal(code)
    if hint is not None:
        exc.hint = hint
    return exc


def _hub_refusal(exc):
    if isinstance(exc, HubRejected):
        return Refusal(jr._bound(f"hub-rejected: {exc}", 200))
    return Refusal("hub-unreachable")


def _post_event(event, *, url=jr.EVENTS_URL, opener=urllib.request.urlopen):
    """The default `post` for dispatch/worker: raises on anything outside 2xx, matching
    the observable contract the retired jr.post_event already had, but built on jaxflow's own _post
    so there is one HTTP code path in this file, not two.

    MOA-474 D3: a 409 is treated as delivered, not a failure -- it means another
    writer (worker vs reaper, or worker vs cmd_cancel) already posted this run's
    terminal row first (the DB's first-run-finished-wins claim,
    src/server/db/workflows.ts:117-122). A worker/reaper race is an accepted edge
    (spec §Trust model), not an error to surface as "event delivery FAILED"."""
    status, decoded = _post(url, event, opener=opener)
    if status == 409:
        # Explicit marker: callers must not confuse this with a 200 {ok:false},
        # which raises below and therefore keeps the spool.
        return {"ok": False, "claimed": True}
    if not (200 <= status < 300) or not isinstance(decoded, dict) or decoded.get("ok") is not True:
        error = decoded.get("error") if isinstance(decoded, dict) else None
        raise HubRejected(status, error if isinstance(error, str) else None)
    return decoded


def _latest_builder_attempt(con, project, repo, branch):
    """"Latest" is by event TIMESTAMP, `id` breaking an exact tie (F6, round 4) -- the
    same rule `worktrees.ts`'s `collectRetainedWorktrees` uses on the TypeScript side,
    so GC and the dashboard's retained-worktree count never disagree on which attempt
    is the latest for a backfilled or out-of-order write."""
    rows = con.execute(
        "SELECT run_id, payload FROM workflow_events "
        "WHERE project = ? AND role = 'builder' AND type = 'run-started' "
        "ORDER BY ts DESC, id DESC",
        (project,),
    ).fetchall()
    repo_r = Path(repo).resolve()
    for row in rows:
        payload = json.loads(row["payload"])
        try:
            if Path(payload.get("repo", "")).resolve() != repo_r:
                continue
        except OSError:
            continue
        if payload.get("target") == branch:
            return row["run_id"]
    return None


def _plan_path_defect(plan_path, allowlist_root):
    """Why a (resolved) plan path is unusable, or None. The check order is load-bearing and
    shared by every caller: missing -> secret FIRST (a symlink planted inside the worktree
    that resolves to a control-repo `.env` must say `secret-detected` even though it is
    "contained") -> outside the allowlist / not a file / unreadable. Callers map the result
    to their own refusal: `resume-ineligible`, `secret-detected: <path>`,
    `plan is not a readable file`, `unknown-run` or `path-outside-allowlist`."""
    if plan_path is None:
        return "path-outside-allowlist"
    if _is_secret_path(plan_path):
        return "secret-detected"
    if (not _contained(plan_path, allowlist_root) or not plan_path.is_file()
            or not os.access(plan_path, os.R_OK)):
        return "path-outside-allowlist"
    return None


def _builder_run_rows(con, run_id):
    """The builder `run-started` row (`project`, `payload`) and `run-finished` row
    (`payload`) of a run, as `(started, finished)`; either may be None. `role = 'builder'`
    stays in BOTH queries (diff-review F3: a reviewer's own `run-started` must never pair
    with an unrelated builder `run-finished` of the same id). `con.row_factory` must be
    `sqlite3.Row`; the caller owns the connection."""
    started = con.execute(
        "SELECT project, payload FROM workflow_events WHERE run_id = ? AND role = 'builder' "
        "AND type = 'run-started' LIMIT 1",
        (run_id,),
    ).fetchone()
    finished = con.execute(
        "SELECT payload FROM workflow_events WHERE run_id = ? AND role = 'builder' "
        "AND type = 'run-finished' LIMIT 1",
        (run_id,),
    ).fetchone()
    return started, finished


def _decide_verify_reuse(*, head_matches, porcelain_clean, commands_match, frames, parse_error):
    """Pure decision for MOA-471 item 8. Returns `(should_reuse: bool, reason: str)`.
    `frames`/`parse_error` come from `jr.parse_tests_frames` (the caller passes
    `parse_error="tests-file-missing"` when no evidence file existed at all)."""
    if not head_matches:
        return False, "head-sha-changed"
    if not porcelain_clean:
        return False, "worktree-dirty"
    if not commands_match:
        return False, "verify-command-changed"
    if parse_error is not None:
        return False, parse_error
    if not frames:
        return False, "tests-file-missing"
    if any(f["exit_code"] != 0 for f in frames):
        return False, "prior-verify-failed"
    return True, "all-conditions-met"


_SPEC_LINE_RE = re.compile(r"^\**Spec:\**\s*`?([^`\n]+)`?", re.I | re.M)

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.M)


def _split_spec_fragment(raw):
    """Splits a **Spec:** line's raw value at the LAST '#' into (path, fragment) (spec
    item 2). A malformed fragment -- empty, or containing '..' -- is treated as no
    fragment at all: the ENTIRE raw string (hash included) is returned as the path,
    exactly as if no '#' had ever been recognized, since a real file name could
    legitimately contain either."""
    if "#" not in raw:
        return raw, None
    path_part, _, frag_part = raw.rpartition("#")
    fragment = frag_part.strip()
    if not fragment or ".." in fragment:
        return raw, None
    return path_part, fragment


def _find_heading(text, fragment):
    """True when `text` has a Markdown heading line (# through ######) whose trimmed
    text equals `fragment`, case-sensitive (spec item 2)."""
    for m in _HEADING_RE.finditer(text):
        if m.group(2).strip() == fragment:
            return True
    return False


def _resolve_handoff_spec_path(plan_text, plan_dest, worktree, allowlist_root):
    """Resolves the plan header's own **Spec:** file to its ORIGINAL, validated, readable
    file (MOA-467 -- no copy is made anywhere; the path itself is what the handoff names
    and the reviewer/builder reads). When the header has NO **Spec:** line at all, falls
    back to the plan itself for BOTH paths.spec and paths.plan -- orchestrator-
    handoff.md requires paths.spec non-empty (`none` is not in its allowed-none list),
    and `build` has no other spec input of its own (round-1 cold review F2; flagged as an
    open item in this plan's header). Returns `(spec_dest, refusal_code)` -- exactly one
    is not None. A DECLARED path that is outside the allowlist, secret, or not a real
    readable file returns a refusal code instead of silently substituting the plan for it
    (fixes cold review round 2 F5) -- only the absence of a **Spec:** line falls back
    that way. Symlink escapes are closed by canonicalize_target's own resolve-then-
    contain, which is exactly why the raw declared string must never be trusted as-is.
    `worktree` is retained only for call-site signature stability; the original doc
    location is what governs now.

    MOA-471 item 2: the declared value may carry a `#<heading>` fragment, split off by
    `_split_spec_fragment`. When present, the resolved file must also contain a
    Markdown heading whose text matches (`_find_heading`) -- a miss returns the
    `spec-fragment-not-found` refusal code. On success `spec_dest` becomes the STRING
    `f"{candidate}#{fragment}"` instead of the bare `Path` -- every caller only ever
    formats it (`f"...{spec_dest}..."`) or re-derives it a second time for a str-vs-str
    comparison (see the diff-review worker's own fix, `scripts/jaxflow_review.py`, `dispatch_diff_review`'s worker), never
    compares it directly against a `Path`. No fragment: byte-for-byte the old
    behavior -- `candidate` returned as a `Path`, unchanged."""
    m = _SPEC_LINE_RE.search(plan_text)
    if not m:
        return plan_dest, None
    raw = m.group(1).strip()
    path_str, fragment = _split_spec_fragment(raw)
    try:
        candidate = canonicalize_target(Path(path_str), allowlist_root)
    except Refusal as exc:
        return None, exc.code
    if _is_secret_path(candidate):
        return None, f"secret-detected: {candidate}"
    if not candidate.is_file() or not os.access(candidate, os.R_OK):
        return None, "spec-reference-invalid"
    if fragment is None:
        return candidate, None
    try:
        doc_text = candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None, "spec-reference-invalid"
    if not _find_heading(doc_text, fragment):
        return None, "spec-fragment-not-found"
    return f"{candidate}#{fragment}", None


def _control_owner(repo, run, allowlist_root):
    """The canonical control checkout behind `repo` and whether that relationship is
    resolved -- the ONE git relationship probe shared by read grants (`_control_repo_of`)
    and status ownership (`_update_status_md`), so the two can never diverge (MOA-467
    acceptance fixes). `(repo, True)` means `repo` IS the owner: a directory `.git`
    (ordinary main checkout), no git metadata at all (status.md writes are best-effort
    and never required a repo), or a main checkout whose gitdir equals its common dir --
    a `--separate-git-dir` MAIN included, whose `.git` is a FILE. A linked worktree
    (gitdir != common) resolves to the MAIN entry of git's OWN worktree list, when that
    entry is not `repo`, carries git metadata and sits inside the allowlist -- never a
    `.git`-name guess or a sibling-directory scan. Every unresolved probe returns
    `(repo, False)`: a status write SKIPS instead of touching the worktree, while read
    grants fall back to `repo` -- a failed probe must never invent a relationship."""
    if (repo / ".git").is_dir():
        return repo, True
    if not (repo / ".git").exists():
        return repo, True
    try:
        probe = run(["git", "rev-parse", "--path-format=absolute",
                     "--git-dir", "--git-common-dir"], cwd=repo)
    except Exception:
        return repo, False
    lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if probe.returncode != 0 or len(lines) < 2:
        return repo, False
    gitdir, common = Path(lines[0]).resolve(), Path(lines[1]).resolve()
    if gitdir == common:
        # The checkout's gitdir IS the common dir: a main repo (a `--separate-git-dir`
        # layout included) that owns its own card.
        return repo, True
    try:
        listed = run(["git", "worktree", "list", "--porcelain"], cwd=repo)
    except Exception:
        return repo, False
    if listed.returncode != 0:
        return repo, False
    for line in listed.stdout.splitlines():
        if line.startswith("worktree "):
            candidate = Path(line[len("worktree "):].strip()).resolve()
            break
    else:
        return repo, False
    if candidate == repo:
        return repo, False
    if not (candidate / ".git").exists() or not _contained(candidate, allowlist_root):
        return repo, False
    return candidate, True


def _open_ro(db_path):
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


_STATUS_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.S)

_STATUS_NOW_RE = re.compile(r"(^##[ \t]+Now[ \t]*\r?\n)((?:.*\r?\n?)*?)(?=\r?\n##[ \t]|\Z)", re.M)

# spec §4.5's stage-mapping table.
# MOA-465: pr-open lands on "review" -- the PR exists and awaits the SAME merge approval a
# diff review's own approve/approve-with-changes gate awaits (spec Ledger & card).
_STAGE_BY_KIND = {"spec": "spec", "plan": "spec", "build": "build", "diff": "review",
                   "merge": "ship", "pr-open": "review"}


def _parse_status_frontmatter(text):
    m = _STATUS_FRONTMATTER_RE.match(text)
    if not m:
        return None, None, None
    fields = {}
    for line in m.group(1).splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            fields[key.strip()] = value.strip()
    # `frontmatter_text` is the FULL raw block, both `---` delimiters and their own exact
    # EOLs included -- `_splice_status_frontmatter` rewrites owned lines inside it byte-for-
    # byte, so nothing here throws away the original's order, spacing, or line endings.
    return fields, text[:m.end()], text[m.end():]


def _splice_status_frontmatter(frontmatter_text, updates, remove_keys=()):
    """Rewrites ONLY the given owned key lines inside a full frontmatter block, preserving
    every other line byte-for-byte (fixes cold review round 2 F7 -- rebuilding every line
    with a fixed field order/spacing/EOL silently destroyed unmanaged content like
    `project`/`flag`/`tmux`, and normalized CRLF to LF). `updates` maps key -> new value for
    keys to set or add; `remove_keys` lists keys to drop entirely (only ever `"gate"`, when
    no longer gated). A key in `updates` with no existing line is inserted right before the
    closing `---` (always the last line here, by construction of the caller's regex), using
    a plain `"\\n"` -- new content, not preservation."""
    lines = frontmatter_text.splitlines(keepends=True)
    seen = set()
    out = []
    for line in lines:
        stripped = line.rstrip("\r\n")
        if stripped == "---" or ": " not in stripped:
            out.append(line)
            continue
        key = stripped.split(": ", 1)[0].strip()
        if key in remove_keys:
            continue  # drop this line entirely
        if key in updates:
            eol = line[len(stripped):]  # preserves THIS line's own "\n" or "\r\n"
            out.append(f"{key}: {updates[key]}{eol}")
            seen.add(key)
            continue
        out.append(line)
    missing = [f"{key}: {value}\n" for key, value in updates.items() if key not in seen]
    if missing:
        out[-1:-1] = missing
    return "".join(out)


def _replace_now_section(body, new_paragraph):
    m = _STATUS_NOW_RE.search(body)
    if not m:
        sep = "" if body == "" or body.endswith("\n\n") else ("\n" if body.endswith("\n") else "\n\n")
        return f"{body}{sep}## Now\n{new_paragraph}\n"
    return body[: m.start(2)] + f"{new_paragraph}\n" + body[m.end(2):]


def _current_branch(run, repo):
    result = run(["git", "branch", "--show-current"], cwd=repo)
    return result.stdout.strip() if result.returncode == 0 else ""


def _update_status_md(manifest, *, run, allowlist_root):
    """Rewrites <repo>/.jax-os/status.md as a run by-product (spec §4.5). Best-effort and
    silent on EVERY failure mode -- missing file, invalid UTF-8, unparseable frontmatter, a
    malformed or non-comparable timestamp, a stale-write race, an injected run() error, an
    OSError on write -- a status.md problem never fails the run itself. Also refuses
    silently (fixes F2, HIGH, NEW) when the manifest's own `repo` does not canonicalize
    under the allowlist root -- a forged builder/diff/doc-review manifest must never be
    able to redirect this write to an arbitrary status.md outside the allowlist; this one
    check covers all three call sites in `run_worker`.
    MOA-467 Task 4 (acceptance fixes): the CONTROL repo owns project status, derived
    through the shared `_control_owner` relationship (the same one the read grants use);
    when that lookup cannot establish the owner (a linked worktree whose git probe
    failed), the write is SKIPPED -- it must never fall back to writing the worktree's
    own card."""
    repo = Path(manifest["repo"]).resolve()
    if not _contained(repo, allowlist_root):
        print("status.md: skipped (repo outside allowlist)", file=sys.stdout)
        return
    owner, resolved = _control_owner(repo, run, allowlist_root)
    if not resolved:
        print("status.md: skipped (owner lookup unresolved)", file=sys.stdout)
        return
    repo = owner
    path = repo / ".jax-os" / "status.md"
    try:
        # `newline=""` disables universal-newline translation on read (execution rule
        # fixture/impl deviation: the plan's own literal `read_text(encoding="utf-8")` call
        # -- with no `newline` kwarg -- silently translates every `\r\n` to `\n` on read,
        # which contradicts spec §4.5's byte-for-byte preservation rule and defeats
        # `_STATUS_FRONTMATTER_RE`/`_splice_status_frontmatter`'s own `\r?` handling, both
        # already written to expect a CRLF-preserving read).
        text = path.read_text(encoding="utf-8", newline="")
    except (OSError, UnicodeError):
        # fixes cold review round 2 F6: invalid UTF-8 raises UnicodeDecodeError (a
        # UnicodeError/ValueError, never an OSError) -- the original except-clause let it
        # escape this function entirely instead of being treated as unreadable.
        return

    try:
        fields, frontmatter_text, body = _parse_status_frontmatter(text)
        if fields is None:
            print("status.md: unparseable frontmatter, skipping", file=sys.stdout)
            return

        dispatch_start = manifest.get("dispatch_start", "")
        existing_updated = fields.get("updated", "")
        stale = False
        if existing_updated and dispatch_start:
            try:
                # Parsed and compared as real instants (a plain string compare is wrong
                # across differing UTC offsets). A naive/aware mismatch raises TypeError,
                # not ValueError -- fixes cold review round 2 F6, second half: EITHER
                # failure means staleness cannot be PROVEN, so `stale` stays False and the
                # write proceeds, rather than the whole function skipping via the outer
                # `except Exception` below.
                stale = datetime.fromisoformat(existing_updated) >= datetime.fromisoformat(dispatch_start)
            except (ValueError, TypeError):
                stale = False
        if stale:
            print("status.md: skipped stale write (a newer run already updated it)", file=sys.stdout)
            return

        kind = manifest["kind"]
        updates = {"stage": _STAGE_BY_KIND[kind]}
        # Phase 2 Decision 23: a merge dispatched from the dashboard passes runtime=None -- the
        # `builder:` field keeps whatever the last real run wrote, never the literal "jaxos".
        if manifest.get("runtime"):
            updates["builder"] = manifest["runtime"]
        if kind in ("build", "diff", "merge", "pr-open"):
            # A `diff` manifest (Part 2, Task 5) has no "branch" key of its own -- `target`
            # IS the branch there (this plan's target=branch decision, see header). Falling
            # back to `_current_branch(repo)` for a diff review would show the CONTROL
            # repo's own current branch instead of the worktree branch under review.
            updates["branch"] = manifest.get("branch") or manifest["target"]
        else:
            updates["branch"] = _current_branch(run, repo)
        updates["updated"] = _iso8601(datetime.now().astimezone())

        outcome = manifest.get("worker_outcome")
        gate = None
        if kind == "diff" and outcome in ("approve", "approve-with-changes"):
            gate = "awaiting-approval"
        # A spec review's own gate rule (§4.5): zero HIGH findings (== verdict != "reject",
        # since a HIGH finding forces `reject` per reviewer-contract.md's verdict rule) AND
        # the reviewed spec's own header opts in -- `manifest["spec_gate_required"]` is set
        # in Task 3's edit to run_worker's doc-review body, where the worker already has
        # the target file's text in hand.
        elif kind == "spec" and outcome != "reject" and manifest.get("spec_gate_required"):
            gate = "awaiting-approval"
        elif kind == "pr-open":
            # MOA-465: opening a PR always awaits Rafa's merge approval next -- unlike diff's
            # outcome-conditional gate, there is no "pr-open but not ready" state.
            gate = "awaiting-approval"
        remove_keys = ()
        if gate:
            updates["gate"] = gate
        else:
            remove_keys = ("gate",)

        summary = manifest.get("worker_summary", "")
        new_body = _replace_now_section(body, summary)
        # `body` (from `_parse_status_frontmatter`) already carries the ORIGINAL file's own
        # blank line between the closing `---` and `## Now` -- concatenating a SECOND blank
        # line here would accumulate one extra blank line on every rewrite.
        new_frontmatter = _splice_status_frontmatter(frontmatter_text, updates, remove_keys)
        path.write_text(new_frontmatter + new_body, encoding="utf-8")
    except Exception as exc:
        print(f"status.md: skipped ({exc})", file=sys.stdout)


def _utf16_len(text):
    """Length in UTF-16 code units -- what the TypeScript validator's `String.length`
    counts, and NOT what Python's `len()` counts (an emoji is 1 to Python, 2 there).
    Returns -1 for a value that cannot be encoded at all, which is how a lone surrogate
    from `surrogateescape` argv decoding arrives. Every string that ends up in the
    `merge-approved` payload is measured with this, so a value can never pass the
    producer and then fail the validator on length alone."""
    try:
        return len(text.encode("utf-16-le")) // 2
    except UnicodeEncodeError:
        return -1


def _git_read(run, repo, argv, *, shape=None):
    """Read a VALUE from git. Returns the stripped stdout, or `None` when the command
    failed, printed nothing, or printed something that is not the shape the caller needs.

    This exists because of the round-5 call-by-call table: eight call sites each decided
    on their own whether to look at `returncode`, and five had not. Git writes to stdout
    before it fails, so a failed read whose output happens to look right was reaching a
    commit subject, an audit payload and an index fingerprint -- three different bugs with
    one cause. One helper decides what "a git read succeeded" means, so the next call site
    cannot forget.

    Note what it is NOT for: a probe whose non-zero exit is the ANSWER (`rev-parse
    --verify <target>^2` meaning "not a resume", `show-ref` meaning "no such branch",
    `remote get-url` exiting 2 for "no origin") still reads its own `returncode`, because
    there the exit code carries information a value-reader would throw away."""
    result = run(argv, cwd=repo)
    out = result.stdout.strip()
    if result.returncode != 0 or not out:
        return None
    if shape and not re.fullmatch(shape, out):
        return None
    return out


def _require_toplevel(run, cwd, *, strict=False):
    """The resolved git toplevel of `cwd`, or the toplevel refusal. Two forms,
    both preserved from the inline copies. `strict=False` (dispatchers, gc): only a failed
    `rev-parse` refuses and the stdout is taken as-is. `strict=True` (pr/merge verbs, round-5
    F8): the value becomes the repo root every later call is scoped to, so it must be a
    non-empty absolute path to a directory -- an empty or relative one would silently
    resolve against the process cwd. `_git_read` is looked up at call time."""
    argv = ["git", "rev-parse", "--show-toplevel"]
    if strict:
        top = _git_read(run, cwd, argv)
        valid = top is not None and top.startswith("/") and Path(top).is_dir()
    else:
        result = run(argv, cwd=cwd)
        top = result.stdout.strip()
        valid = result.returncode == 0
    if not valid:
        raise Refusal("not-a-git-toplevel")
    return Path(top).resolve()


def _is_strict_descendant(run, repo, base, head):
    """True iff `head` descends from `base` AND `head != base` (spec §4.2, pinned once:
    both call sites -- this fallback's commit-beyond-base check, and `cmd_merge`'s
    fast-forward check in Task 5 -- name their own `base`/`head` explicitly, never
    "ancestor"/"descendant" alone, after an earlier draft reversed them (77f30f829248
    F1). NOT a call site for `_validate_since_chain` (`:329-350`), which keeps its own
    separate ancestor-or-equal check -- out of scope (spec §5). F4: `base`/`head` can
    come from a hand-edited (or corrupted) manifest.json -- never trusted to even be a
    string -- so the type is checked with `isinstance` BEFORE `re.match` ever sees it,
    the same guard `scripts/jaxflow_run.py:359`'s `reviewer_head_matches` already uses
    for the identical risk. Never raises; a malformed value simply resolves False."""
    if not (isinstance(base, str) and isinstance(head, str)):
        return False
    if base == head or not (jr.SHA_RE.match(base) and jr.SHA_RE.match(head)):
        return False
    ancestor = run(["git", "merge-base", "--is-ancestor", base, head], cwd=repo)
    return ancestor.returncode == 0


def _is_registered_worktree(run, repo, worktree, branch):
    """True only when git itself reports `worktree` as the checkout of `branch` (cold
    review F2). Deriving the path from the project slug and the branch name is a guess;
    removing a directory on the strength of a guess is not acceptable for a command that
    also deletes a branch."""
    listed = run(["git", "worktree", "list", "--porcelain"], cwd=repo)
    if listed.returncode != 0:
        return False
    current = None
    for line in listed.stdout.splitlines():
        if line.startswith("worktree "):
            current = line[len("worktree "):].strip()
        elif line.startswith("branch ") and current is not None:
            ref = line[len("branch "):].strip()
            if ref == f"refs/heads/{branch}" and Path(current).resolve() == worktree:
                return True
    return False


def _copy_run_reports(worktree, repo, run_id=None):
    """Extracted from `cmd_merge`'s inline copy loop (spec §4 guard 7, round 3 G6):
    copies every `<12-hex>.md`/`.tests.txt` report from `<worktree>/.local/reports/`
    into `<repo>/.local/reports/` (security properties UNCHANGED from the original).
    Returns {"failed": bool, "existed": bool}: `existed` = `run_id`'s own `<run_id>.md`
    was present (only tracked when `run_id` is given). `failed` (F1, plan review round 1)
    is set by every way a matched report can fail to be preserved -- an invalid/oversized/
    changed-under-read report or any `OSError`, not only a hard copy error -- so a silent
    skip anywhere below is never mistaken for success. A genuinely MISSING reports
    directory is the one exception (nothing to preserve); any other directory-open error
    (permission denied, wrong type...) fails closed, since reports might exist there.
    F1 (round 2, HIGH): a `FileExistsError` on the DESTINATION now re-verifies it is a
    regular file with the exact same bytes as the source before calling it preserved;
    anything else (dir/FIFO/symlink/different bytes) fails closed too."""
    failed = False
    existed = False
    with contextlib.ExitStack() as stack:
        def _open_dir(name, parent=None, *, create=False):
            if create:
                try:
                    os.mkdir(name, 0o700, dir_fd=parent)
                except FileExistsError:
                    pass
            fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            stack.callback(os.close, fd)
            return fd

        names = None
        try:
            src_fd = _open_dir(str(worktree))
            dest_fd = _open_dir(str(repo))
            for part in (".local", "reports"):
                src_fd = _open_dir(part, src_fd)
                dest_fd = _open_dir(part, dest_fd, create=True)
            names = sorted(os.listdir(src_fd))
        except FileNotFoundError:
            pass  # F1: no reports directory at all -- nothing to preserve, not a failure.
        except OSError as exc:
            failed = True  # F1: exists but could not be opened/listed -- fail closed.
            print(f"reports copy skipped ({jr._bound(str(exc), 200)})")

        for name in names or []:
            if not re.fullmatch(r"[0-9a-f]{12}(\.md|\.tests\.txt)", name):
                print(f"report {name} skipped (not a run report)")
                continue
            if run_id is not None and name == f"{run_id}.md":
                existed = True
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=src_fd)
                try:
                    st = os.fstat(fd)
                    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                        print(f"report {name} skipped (not a single-linked regular file)")
                        failed = True  # F1: invalid report shape, never silently dropped.
                        existed = False  # F1: a non-regular file is not a preserved report.
                        continue
                    if st.st_size > REPORT_COPY_MAX:
                        print(f"report {name} skipped ({st.st_size} bytes, over the {REPORT_COPY_MAX} limit)")
                        failed = True  # F1: oversized report, never silently dropped.
                        existed = False  # F1: an oversized file is not a preserved report.
                        continue
                    body = bytearray()
                    while len(body) <= REPORT_COPY_MAX:
                        chunk = os.read(fd, min(1 << 16, REPORT_COPY_MAX + 1 - len(body)))
                        if not chunk:
                            break
                        body += chunk
                    if len(body) != st.st_size:
                        print(f"report {name} skipped (changed while being read)")
                        failed = True  # F1: unverifiable bytes, never silently dropped.
                        continue
                finally:
                    os.close(fd)
                out = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dest_fd)
                try:
                    written = 0
                    while written < len(body):
                        written += os.write(out, body[written:])
                except BaseException:
                    os.close(out)
                    out = None
                    with contextlib.suppress(OSError):
                        os.unlink(name, dir_fd=dest_fd)
                    raise
                finally:
                    if out is not None:
                        os.close(out)
            except FileExistsError:
                # F1 (round 2, HIGH): verify same shape+bytes before calling it "preserved".
                try:
                    existing_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dest_fd)
                    try:
                        est = os.fstat(existing_fd)
                        if not stat.S_ISREG(est.st_mode) or est.st_nlink != 1:
                            print(f"report {name} skipped (existing destination is not a regular file)")
                            failed = True
                        else:
                            existing_body = bytearray()
                            while len(existing_body) <= REPORT_COPY_MAX:
                                chunk = os.read(existing_fd, min(1 << 16, REPORT_COPY_MAX + 1 - len(existing_body)))
                                if not chunk:
                                    break
                                existing_body += chunk
                            if bytes(existing_body) != bytes(body):
                                print(f"report {name} skipped (existing destination differs from source)")
                                failed = True
                            else:
                                print(f"report {name} already present, kept")
                    finally:
                        os.close(existing_fd)
                except OSError as exc:
                    print(f"report {name} skipped (existing destination unreadable: {jr._bound(str(exc), 200)})")
                    failed = True
            except OSError as exc:
                print(f"report {name} skipped ({jr._bound(str(exc), 200)})")
                failed = True
    return {"failed": failed, "existed": existed}
