#!/usr/bin/env python3
"""jaxflow -- single dispatch CLI (slice a: review --spec|--plan, status, result, cancel)."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import selectors
import shlex
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import general_settings
import jaxflow_run as jr
import jaxflow_env as jenv
import jaxflow_resume as jresume
import jaxflow_settings as jset
import jev_client
from jax_init import ALLOWLIST_ROOT_DEFAULT, Refusal, _contained, _is_secret_path, canonicalize_target
from jaxflow_hook import redact, UUID_RE
import jaxflow_common as jc
import jaxflow_workerkit as jk
import jaxflow_merge as jm
import jaxflow_review as jv
import jaxflow_build as jb
# P2-REEXPORT-BEGIN (transition; the final shim keeps exactly these names)
from jaxflow_common import (  # noqa: F401
    CALLBACKS_ROOT,
    _RUN_ID_RE,
    _delete_spool,
    _is_strict_descendant,
    _manifest_dir,
    _open_ro,
    _post_event,
    _spool_path,
    _write_spool,
)
from jaxflow_workerkit import (  # noqa: F401
    _interrupted_payload,
    _persist_resume_checkpoint,
    _send_callback,
)
from jaxflow_merge import (  # noqa: F401
    cmd_merge,
    cmd_pr_open,
    cmd_release,
)
from jaxflow_review import (  # noqa: F401
    dispatch_diff_review,
    dispatch_review,
    parse_threat_model,
)
from jaxflow_build import (  # noqa: F401
    dispatch_build,
)
# P2-REEXPORT-END


REVIEWER_DEFAULTS = {k: v for k, v in jset.REVIEWER_DEFAULTS.items() if k in ("claude", "codex")}


def _chain_block(run_id, *, db_path):
    """Returns the `chain:` block text for a `kind: diff` run, `None` for any other run
    kind. Root-first, one line per node, `[since <predecessor>]` on non-root lines. A
    DISPLAY path -- never raises. Known shapes (missing `repo`/`base_sha`/`head_sha`, a
    broken/cyclic chain) render their specific `chain: broken at <run_id> (<reason>)`;
    the outer guard (cr 99ec95af61a2 F3) catches everything else -- malformed ledger
    JSON, a non-string `since_review_run_id`/SHA -- as `chain: broken (malformed)`."""
    # ponytail: one guard instead of per-field validation; status/result must never raise.
    try:
        con = jc._open_ro(db_path)
        con.row_factory = sqlite3.Row
        try:
            started = con.execute(
                "SELECT role, payload FROM workflow_events WHERE run_id = ? "
                "AND type = 'run-started' LIMIT 1", (run_id,),
            ).fetchone()
            if not started:
                return None
            started_payload = json.loads(started["payload"])
            if started["role"] != "reviewer" or started_payload.get("kind") != "diff":
                return None
            raw_repo = started_payload.get("repo")
            if not isinstance(raw_repo, str) or not raw_repo:
                return f"chain: broken at {run_id} (no-repo)"
            repo = Path(raw_repo)
            chain, error = jc._walk_review_chain(
                lambda rid: jc._load_diff_review_node(con, repo, rid), run_id,
            )
        finally:
            con.close()
        if error is not None:
            kind, bad_run_id = error
            if kind == "cycle":
                return f"chain: cycle at {bad_run_id}"
            return f"chain: broken at {bad_run_id} ({kind})"
        lines = ["chain:"]
        for i, node in enumerate(chain):
            if not node["base_sha"] or not node["head_sha"]:
                return f"chain: broken at {node['run_id']} (missing-sha)"
            verdict = node["verdict"] or "running"
            suffix = f" [since {node['since_review_run_id']}]" if i > 0 else ""
            lines.append(f"  {node['run_id']} {node['base_sha'][:12]}..{node['head_sha'][:12]} {verdict}{suffix}")
        return "\n".join(lines)
    except Exception:
        return "chain: broken (malformed)"


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


def _valid_launch_selection(record):
    if type(record) is not dict or set(record) != set(jc._LAUNCH_KEYS):
        return None
    if record["profile_name"] not in ("default", "fallback"):
        return None
    revision = record["settings_revision"]
    if type(revision) is not str or not jset.REVISION_RE.fullmatch(revision):
        return None
    for key in ("model", "runtime_model"):
        if type(record[key]) is not str or not record[key]:
            return None
    effort = record["effort"]
    if effort is not None and (type(effort) is not str or not jset.EFFORT_RE.fullmatch(effort)):
        return None
    cred = record["credential"]
    if type(cred) is not dict or cred.get("kind") not in ("native", "env"):
        return None
    return record


def _checkpoint_hint(payload, run_id):
    if type(payload) is not dict or not jc._RUN_ID_RE.match(str(run_id)):
        return ""
    if payload.get("runtime") != "opencode-builder":
        return ""
    repo = payload.get("repo")
    if type(repo) is not str:
        return ""
    path = Path(repo) / ".local" / "runs" / run_id / "resume-checkpoint.json"
    return f"checkpoint: {jresume.checkpoint_status(path)}"


def _managed_history_lines(payload, run_id):
    if type(payload) is not dict or payload.get("runtime") != "opencode-builder":
        return ""
    lines = [
        f"preview model: {payload.get('model')}",
        f"preview effort: {payload.get('effort')}",
    ]
    repo = payload.get("repo")
    if type(repo) is str:
        record = _valid_launch_selection(
            jc._read_manifest_field(Path(repo), run_id, "launch_selection"),
        )
        if record is not None:
            effort = record["effort"]
            lines.append(f"resolved selection model: {record['model']}")
            lines.append(f"resolved selection effort: {effort if effort is not None else 'n/a'}")
    return "\n".join(lines)


def _hub_rejected_code(status, decoded):
    error = decoded.get("error") if isinstance(decoded, dict) else None
    detail = f"{status} {error}" if isinstance(error, str) else str(status)
    return jr._bound(f"hub-rejected: {detail}", 200)


MISSION_BASE_URL = f"{jenv.api_base_url()}/api/mission"


def _get(url, *, opener=urllib.request.urlopen, timeout=5):
    """GET counterpart to `_post` -- same (status, decoded_body) return shape, raising only on
    a transport error. The one mission subcommand that reads instead of posting (`show`) uses
    this instead of `_post`."""
    req = urllib.request.Request(url, method="GET")
    try:
        with opener(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            decoded = json.loads(exc.read().decode("utf-8"))
        except (ValueError, OSError):
            decoded = {}
        return exc.code, decoded


_DOCTOR_SKILL_DIRS = {
    "claude": Path.home() / ".claude" / "skills",
    "codex": Path.home() / ".codex" / "skills",
    "opencode": Path.home() / ".agents" / "skills",
}

_DOCTOR_HOOK_FILES = {
    "claude": Path.home() / ".claude" / "settings.json",
    "codex": Path.home() / ".codex" / "hooks.json",
}


def _doctor_wrapper(name, which, repo_root):
    found = which(name)
    if not found:
        return False, f"install bin/{name} on PATH: ln -s {repo_root}/bin/{name} ~/.local/bin/{name}"
    try:
        Path(found).resolve().relative_to(repo_root)
    except ValueError:
        return False, f"{name} on PATH resolves outside this checkout — re-symlink bin/{name}"
    return True, None


def _doctor_skill(harness, repo_root, skill_dirs):
    base = skill_dirs[harness]
    skills_root = (repo_root / "workflow" / "skills").resolve()
    for name in ("jaxflow", "jax-init"):
        candidate = base / name
        if candidate.exists():
            try:
                if candidate.resolve() == (skills_root / name).resolve():
                    return True, None
            except OSError:
                pass
    return False, f"no jaxflow/jax-init skill dir under {base} resolves into {skills_root} — dashboard-only installs may run none"


_JAXFLOW_HOOK_RE = re.compile(r"(\S*scripts/jaxflow_hook\.py)")


def _doctor_hook(harness, repo_root, hook_files):
    path = hook_files[harness]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, f"{path} missing or unreadable"
    cmd = ""
    for entry in data.get("hooks", {}).get("Stop", []):
        for h in entry.get("hooks", []):
            cmd = h.get("command", "")
            if cmd:
                break
        if cmd:
            break
    if "jaxflow-hook" in cmd:
        return True, None
    m = _JAXFLOW_HOOK_RE.search(cmd)
    if m:
        try:
            Path(m.group(1)).resolve().relative_to(repo_root)
            return True, None
        except (OSError, ValueError):
            pass
    return False, f"no Stop hook in {path} resolves to jaxflow-hook or this checkout"


def cmd_doctor(*, env=None, which=None, repo_root=None, read_settings=None, get=None,
               skill_dirs=None, hook_files=None, env_status=None, home=None):
    """Read-only health check (spec MOA-502 Decision 8): one line per check, `ok`/`missing`
    plus a fix hint on `missing`; never writes, never calls gh/git, never reaches the network
    beyond one GET to the local jax-os API. Exit is non-zero iff a REQUIRED check is missing --
    an integration-gated check (cli:gh) and an enabled agent's cli:<a> are required only when that switch is on.
    `skill_dirs`/`hook_files`/`env_status` default to the real per-harness home locations and
    `jev_client.env_file_status` — tests always inject fixtures so no test ever reads Rafa's
    real home directory or env file (plan review df958c45b481, F2)."""
    env = os.environ if env is None else env
    which = which or (lambda name, path=None: shutil.which(name, path=path or env.get("PATH")))
    repo_root = repo_root or jc.SCRIPT_PATH.resolve().parents[1]
    read_settings = read_settings or general_settings.read_settings
    get = get or _get
    skill_dirs = skill_dirs if skill_dirs is not None else _DOCTOR_SKILL_DIRS
    hook_files = hook_files if hook_files is not None else _DOCTOR_HOOK_FILES
    env_status = env_status or jev_client.env_file_status
    home = Path(home) if home is not None else Path.home()

    lines = []
    required_ok = True
    check_total = 0
    check_passed = 0

    def check(name, ok, required, hint=None):
        nonlocal required_ok, check_total, check_passed
        lines.append(f"{name} ok" if ok else f"{name} missing" + (f" — {hint}" if hint else ""))
        check_total += 1
        if ok:
            check_passed += 1
        if required and not ok:
            required_ok = False

    settings = read_settings()
    agents_cfg = settings["data"]["integrations"]["agents"] if settings.get("ok") else None

    def agent_on(name):
        return agents_cfg is None or agents_cfg[name]  # unreadable settings: check all, as before

    def has_cli(name):
        if which(name):
            return True
        if name != "opencode":
            return False
        installed = home / ".opencode" / "bin" / "opencode"  # the official installer's location
        return installed.is_file() and os.access(installed, os.X_OK)

    for name in ("jaxflow", "jaxflow-hook", "jax-init"):
        ok, hint = _doctor_wrapper(name, which, repo_root)
        check(f"wrapper:{name}", ok, True, hint)

    for harness in ("claude", "codex", "opencode"):
        if not agent_on(harness):
            continue
        ok, hint = _doctor_skill(harness, repo_root, skill_dirs)
        check(f"skill:{harness}", ok, False, hint)

    for harness in ("claude", "codex"):
        if not agent_on(harness):
            continue
        ok, hint = _doctor_hook(harness, repo_root, hook_files)
        check(f"hook:{harness}", ok, False, hint)

    for name in ("tmux", "git", "rg"):
        check(f"cli:{name}", bool(which(name)), True, f"install {name}")
    check("cli:python3.11+", sys.version_info >= (3, 11), True,
          f"running under {sys.version_info.major}.{sys.version_info.minor}, need 3.11+")

    github_on = bool(settings.get("ok") and settings["data"]["integrations"]["github"])
    check("cli:gh", bool(which("gh")), github_on,
          "install gh or turn integrations.github off in /settings")

    for name in ("claude", "codex", "opencode"):
        if agent_on(name):
            check(f"cli:{name}", has_cli(name), agents_cfg is not None,
                  f"install {name} or turn it off in /settings -> General")

    if agents_cfg is not None:
        any_on = any(agents_cfg.values())
        check("agents:any", any_on, True,
              "jaxflow will not run: it needs at least one agent configured (turn one on in /settings -> General)")
        if any_on:
            check("agents:reviewer", agents_cfg["claude"] or agents_cfg["codex"], True,
                  "jaxflow review will not run: it needs Claude Code or Codex, builds still work "
                  "(turn one on in /settings -> General)")

    check("settings:readable", bool(settings.get("ok")), True,
          settings.get("error") or "settings.json unreadable")

    env_state = env_status()
    check("env:present", env_state["state"] != "missing", True,
          f"create $JAXOS_HOME/.env (state: {env_state['state']})")
    check("env:mode", env_state["state"] == "ok", True,
          f"chmod 0600 $JAXOS_HOME/.env (state: {env_state['state']})")

    try:
        get(f"{jenv.api_base_url()}/api/health")
        api_ok = True
    except Exception:
        api_ok = False
    check("api:reachable", api_ok, True, "start jaxos (systemctl --user start jaxos)")

    if env.get("CODEX_HOME"):
        lines.append("harness-home: custom CODEX_HOME set — doctor validates default locations only")

    lines.append(f"{check_passed}/{check_total} checks passed")
    return lines, (0 if required_ok else 1)


def _mission_post(url, payload, *, post=jc._post):
    """Spec §8 Atomicity/§7: the route is the single validator -- a non-2xx or {ok:false} body's
    own `error` string IS the refusal code, re-raised verbatim, no per-subcommand
    special-casing. Only a genuine transport failure (connection refused, timeout, DNS) or a
    body with no usable `error` string collapses to the generic `mission-hub-unreachable`."""
    try:
        status, decoded = post(url, payload)
    except Exception as exc:
        raise Refusal("mission-hub-unreachable") from exc
    if not (200 <= status < 300) or not isinstance(decoded, dict) or decoded.get("ok") is not True:
        error = decoded.get("error") if isinstance(decoded, dict) else None
        if isinstance(error, str) and error:
            raise Refusal(error)
        raise Refusal(f"mission-hub-rejected: {status}")
    return decoded.get("data")


def cmd_mission_start(args, *, post=jc._post):
    if not args.milestones:
        raise Refusal("malformed milestone")
    _mission_post(MISSION_BASE_URL, {"name": args.name, "goal": args.goal, "milestones": args.milestones}, post=post)


def cmd_mission_status(args, *, post=jc._post):
    _mission_post(f"{MISSION_BASE_URL}/status", {"status_line": args.text}, post=post)


def cmd_mission_mark(args, *, post=jc._post):
    if args.state not in ("done", "in-progress"):
        raise Refusal("malformed state")
    _mission_post(f"{MISSION_BASE_URL}/milestone", {"milestone": args.milestone, "state": args.state}, post=post)


def cmd_mission_finish(outcome, *, post=jc._post):
    _mission_post(f"{MISSION_BASE_URL}/finish", {"outcome": outcome}, post=post)


def _format_mission_show(data):
    if data is None:
        return "no active mission"
    lines = [f"{data['name']}: {data['goal']}", f"status: {data['statusLine']}"]
    marks = {"done": "x", "in-progress": "~", "pending": " "}
    for m in data["milestones"]:
        lines.append(f"  [{marks[m['state']]}] {m['title']}")
    return "\n".join(lines)


def cmd_mission_show(*, get=_get):
    try:
        status, decoded = get(f"{MISSION_BASE_URL}/current")
    except Exception as exc:
        raise Refusal("mission-hub-unreachable") from exc
    if not (200 <= status < 300) or not isinstance(decoded, dict) or decoded.get("ok") is not True:
        raise Refusal("mission-hub-unreachable")
    return _format_mission_show(decoded.get("data"))


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


def _build_diff_handoff(manifest, *, run, repo, worktree, spec_path, plan_path, tests_path,
                        base_sha, head_sha):
    """The reviewer handoff text of a `--diff` review (grammar `parse_reviewer_handoff`
    parses: one `diff:` line, one `paths:` line, one indented `  test-output:` line).
    Runs `git diff base..head`; a failure RAISES RuntimeError (never a placeholder
    review). Returns the handoff string."""
    diff_result = run(["git", "diff", f"{base_sha}..{head_sha}"], cwd=worktree)
    if diff_result.returncode != 0:
        # fixes cold review F6: a diff-generation failure must not silently substitute
        # placeholder text and dispatch a review anyway -- the reviewer would be judging
        # a diff that never actually happened. An uncaught crash here ends the tmux
        # session with no run-finished event, surfacing as `dead` via `jaxflow status
        # <run_id>` (spec §2.5's existing status vocabulary) -- an honest signal over a
        # corrupted review.
        raise RuntimeError(f"git diff {base_sha}..{head_sha} failed: {diff_result.stderr}")
    diff_text = diff_result.stdout

    # Grammar `parse_reviewer_handoff` parses: exactly one `diff:` line, exactly one
    # `paths:` line, and exactly one indented `  test-output:` line in the block right
    # after it -- everything else in this text (the framing, the `spec:`/`plan:` lines
    # below, the actual diff, --focus) is free-form and never scanned by that parser
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
        # ponytail: whole diff embedded inline, no size cap -- revisit if a --diff review
        # ever times out on a huge diff. This is also the resolution to the Claude
        # --tools Read,Glob,Grep containment (spec §6 I2): neither reviewer runtime needs
        # to run `git diff` itself for a --diff review. Doc reviews name the file and
        # read it from disk instead of embedding its contents.
        f"{diff_text}\n"
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
        manifest, run=run, repo=repo, worktree=worktree, spec_path=spec_path,
        plan_path=plan_path, tests_path=tests_path, base_sha=base_sha, head_sha=head_sha,
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


def cmd_status(run_id, *, run=jr.run_command, db_path=None):
    db_path = db_path or jr.DB_PATH
    con = jc._open_ro(db_path)
    con.row_factory = sqlite3.Row
    try:
        started = con.execute(
            "SELECT role, payload FROM workflow_events WHERE run_id = ? AND type = 'run-started' LIMIT 1", (run_id,),
        ).fetchone()
        if not started:
            return "unknown"
        started_payload = json.loads(started["payload"])
        finished = con.execute(
            "SELECT payload FROM workflow_events WHERE run_id = ? AND type = 'run-finished' LIMIT 1", (run_id,),
        ).fetchone()
        if finished:
            payload = json.loads(finished["payload"])
            status = payload["contract_status"]
            stage = payload.get("stage")
            diagnostic = payload.get("diagnostic")
            # MOA-474 §12.2: branches checked in order, first match wins.
            if status == "interrupted":
                # branch 1 -- checked BEFORE branch 2 on purpose: an interrupted
                # builder row also carries result: failure, which would otherwise
                # match branch 2 first and never say "interrupted" (spec §12.2).
                text = f"finished(interrupted) · {stage} — {diagnostic}"
            elif status == "cancelled":
                text = "finished(cancelled)"
            else:
                result = payload.get("result")
                verdict = payload.get("verdict")
                if result is not None or verdict is not None:
                    outcome = result if started["role"] == "builder" else verdict
                elif started["role"] == "reviewer":
                    outcome = "no verdict"
                else:
                    outcome = None
                if outcome is not None:
                    paren = outcome
                    if status in ("missing", "invalid"):
                        paren = f"{paren}, report {status}"
                    text = f"finished({paren})"
                    if stage is not None:
                        text = f"{text} · {stage} — {diagnostic}"
                else:
                    # branch 4 (legacy, §Backward compatibility): a builder row with
                    # neither result nor verdict.
                    reason = payload.get("reason")
                    if reason:
                        text = f"finished(failed: {reason}, exit {payload['exit_code']})"
                    else:
                        text = "finished(no result)"
        else:
            session = started_payload["session"]
            probe = run(["tmux", "has-session", "-t", session])
            text = "running" if probe.returncode == 0 else "dead"
    finally:
        con.close()
    history = _managed_history_lines(started_payload, run_id)
    if history:
        text = f"{text}\n{history}"
    hint = _checkpoint_hint(started_payload, run_id)
    if hint:
        text = f"{text}\n{hint}"
    chain = _chain_block(run_id, db_path=db_path)
    return f"{chain}\n{text}" if chain else text


def cmd_result(run_id, *, allowlist_root=ALLOWLIST_ROOT_DEFAULT, db_path=None):
    db_path = db_path or jr.DB_PATH
    con = jc._open_ro(db_path)
    con.row_factory = sqlite3.Row
    try:
        started = con.execute(
            "SELECT project, role, payload FROM workflow_events WHERE run_id = ? AND type = 'run-started' LIMIT 1", (run_id,),
        ).fetchone()
        finished = con.execute(
            "SELECT payload FROM workflow_events WHERE run_id = ? AND type = 'run-finished' LIMIT 1", (run_id,),
        ).fetchone()
    finally:
        con.close()
    if not started or not finished:
        raise Refusal("unknown-run")
    finished_payload = json.loads(finished["payload"])
    started_payload = json.loads(started["payload"])
    runtime = started_payload.get("runtime")
    if started["role"] == "reviewer" and runtime:
        fallback_line = jc._reviewer_runtime_line(runtime, jset.reviewer_fallback(started_payload.get("caller"), runtime))
        if fallback_line:
            print(fallback_line, flush=True)
    history = _managed_history_lines(started_payload, run_id)
    hint = _checkpoint_hint(started_payload, run_id)
    extras = "\n".join(part for part in (history, hint) if part)
    if finished_payload["contract_status"] == "cancelled":
        text = f"cancelled — {finished_payload['summary']}"
        return f"{extras}\n{text}" if extras else text
    if finished_payload["contract_status"] == "interrupted":
        outcome = finished_payload.get("result") if started["role"] == "builder" else "no verdict"
        text = (
            f"interrupted — {outcome} · {finished_payload['stage']} — "
            f"{finished_payload['diagnostic']}"
        )
        return f"{extras}\n{text}" if extras else text
    # MOA-470 §4.5, final review 1f119d5d1420 F1: PRINTED immediately (flushed), not
    # folded into the return value -- a `missing` row's checks below still raise
    # `Refusal("report-missing")`, which would discard a local variable never
    # returned. Sourced ENTIRELY from the loaded row, never re-opens child.log.
    # `started_payload["repo"]` is always the CONTROL repo for both roles (unlike
    # `root` below, builder-specific and worktree-shaped).
    reason = finished_payload.get("reason")
    if reason is not None:
        log_path = jr.child_log_path(Path(started_payload["repo"]), run_id)
        lines = [f"exit_code: {finished_payload['exit_code']}\n"]
        outcome = finished_payload.get("result") or finished_payload.get("verdict")
        if outcome is not None:
            lines.append(f"outcome: {outcome}\n")
        if finished_payload.get("stage") is not None:
            lines.append(f"stage: {finished_payload['stage']}\n")
            lines.append(f"diagnostic: {finished_payload['diagnostic']}\n")
        lines.append(f"reason: {reason}\n")
        lines.append(f"tail:\n{finished_payload['tail']}\n")
        # MOA-495 2.1: the log slices Jev found most relevant to the failure, in place
        # of the tech lead having to open the raw child.log themselves. Absent whenever
        # Jev didn't run or failed (today's behaviour: nothing extra).
        log_context = finished_payload.get("log_context")
        if log_context:
            lines.append("log context:\n" + "\n---\n".join(log_context) + "\n")
        lines.append(f"child.log: {log_path}\n\n")
        print("".join(lines), end="", flush=True)
    # MOA-495 2.2: the review-round tally, printed the same way (immediately, before the
    # report body below) -- reviewer-only, only on a finished-ok row.
    tally = finished_payload.get("tally")
    if tally is not None:
        print(f"{tally}\n", flush=True)
    if started["role"] == "builder":
        root = jc._branch_worktree_path(allowlist_root, started["project"], started_payload["target"], resolve=False)
    else:
        root = Path(started_payload["repo"])
    expected = root / ".local" / "reports" / f"{run_id}.md"
    # Byte-for-byte string compare BEFORE any resolving or reading (fixes cold review F8)
    # -- a DB value like "<repo>/.local/reports/../reports/<run_id>.md" resolves to the
    # same file but is not the string this run's own dispatch would have written, so it
    # is refused here rather than accepted by a looser resolved-path comparison.
    if finished_payload["report_path"] != str(expected):
        raise Refusal("report-path-mismatch")
    resolved = expected.resolve()
    if not _contained(resolved, allowlist_root):
        raise Refusal("path-outside-allowlist")
    # MOA-456. Containment is checked FIRST -- existence is never reported for a path
    # outside the allowlist. A distinct code rather than widening `report-path-mismatch`:
    # a mismatch means the ledger disagrees with the dispatch convention, while this means
    # the file is gone and re-running is pointless. `git worktree remove` deleting the
    # worktree's gitignored `.local/` is the way that actually happens here.
    if not resolved.is_file():
        raise Refusal("report-missing")
    chain = _chain_block(run_id, db_path=db_path) if started["role"] == "reviewer" else None
    body = f"{resolved}\n{resolved.read_text(encoding='utf-8')}"
    if extras:
        body = f"{extras}\n{body}"
    return f"{chain}\n{body}" if chain else body


def cmd_cancel(run_id, *, run, post=jc._post, now, db_path=None, wait_s=15.0, poll_interval_s=1.0):
    db_path = db_path or jr.DB_PATH
    con = jc._open_ro(db_path)
    con.row_factory = sqlite3.Row
    try:
        started = con.execute(
            "SELECT project, role, payload FROM workflow_events WHERE run_id = ? AND type = 'run-started' LIMIT 1",
            (run_id,),
        ).fetchone()
        if not started:
            raise Refusal("unknown-run")
        finished = con.execute(
            "SELECT 1 FROM workflow_events WHERE run_id = ? AND type = 'run-finished' LIMIT 1", (run_id,),
        ).fetchone()
        if finished:
            raise Refusal("already-finished")
        payload = json.loads(started["payload"])
        if started["role"] == "builder":
            prior_repo, prior_branch = payload.get("repo"), payload.get("target")
            if prior_repo and prior_branch:
                latest = jc._latest_builder_attempt(con, started["project"], prior_repo, prior_branch)
                if latest is not None and latest != run_id:
                    raise Refusal("resume-ineligible")
    finally:
        con.close()
    run(["tmux", "kill-session", "-t", payload["session"]])
    # MOA-474 §10.4: `tmux kill-session` delivers SIGHUP to the worker, whose own
    # signal path (§10.2, A1 Task 4) now finalizes with real evidence, HEAD, and a
    # checkpoint. Wait up to `wait_s`, polling the DB, before falling back to the
    # plain cancelled row this function used to post unconditionally.
    deadline_polls = max(1, int(wait_s / poll_interval_s))
    for _ in range(deadline_polls):
        con2 = jc._open_ro(db_path)
        con2.row_factory = sqlite3.Row
        try:
            row = con2.execute(
                "SELECT payload FROM workflow_events WHERE run_id = ? AND type = 'run-finished' LIMIT 1",
                (run_id,),
            ).fetchone()
        finally:
            con2.close()
        if row:
            won_payload = json.loads(row["payload"])
            outcome = won_payload.get("result") if started["role"] == "builder" else "no verdict"
            return (
                # F2 (cold review c209417f6061): a normal success payload carries no
                # stage/diagnostic -- never index them.
                f"finalized by worker — {outcome} · {won_payload.get('stage') or '-'} — "
                f"{won_payload.get('diagnostic') or '-'}"
            )
        time.sleep(poll_interval_s)
    # No row landed within the deadline -- check for the worker's own spool (its
    # finalize succeeded, only its POST failed). ponytail: `payload["repo"]` isn't on
    # every started row (a doc review's is; a builder's `repo` field IS a control repo
    # per §Evidence's own worktree/control-repo row) -- reviewer rows always carry it.
    repo_str = payload.get("repo")
    if isinstance(repo_str, str):
        spool_path = jc._spool_path(Path(repo_str), run_id)
        if spool_path.exists():
            try:
                spooled_event = json.loads(spool_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                spooled_event = None
            if spooled_event is not None:
                try:
                    status, decoded = post(jr.EVENTS_URL, spooled_event)
                except Exception as exc:
                    raise Refusal("hub-unreachable") from exc
                delivered = status == 409 or (
                    200 <= status < 300 and isinstance(decoded, dict) and decoded.get("ok") is True
                )
                if not delivered:
                    # Diff review e55f14eba5be F1: the worker's real evidence exists and the
                    # hub just refused it -- never paper over it with a plain `cancelled` row
                    # (that row would win the terminal claim and make the reaper's later
                    # redelivery 409 and delete the spool). Keep the spool; the reaper retries.
                    raise Refusal(_hub_rejected_code(status, decoded))
                spool_path.unlink(missing_ok=True)
                spooled_payload = spooled_event["payload"]
                outcome = spooled_payload.get("result") if started["role"] == "builder" else "no verdict"
                return (
                    f"finalized from spool — {outcome} · {spooled_payload.get('stage') or '-'} — "
                    f"{spooled_payload.get('diagnostic') or '-'}"
                )
    cancel_payload = {
        "phase": payload["phase"], "exit_code": None, "contract_status": "cancelled",
        "report_path": None, "summary": f"cancelled by {payload['caller']} at {jc._iso8601(now())}",
    }
    if started["role"] == "builder":
        cancel_payload["head_sha"] = None
    body = {
        "run_id": run_id, "project": started["project"], "role": started["role"],
        "type": "run-finished", "source": "deterministic", "emitter": "wrapper", "payload": cancel_payload,
    }
    # Fixes cold review F2: `post` here is `_post`-shaped (returns the raw status, never
    # raises except on a transport error), so a `409` (the server's terminal claim, §2.4)
    # is told apart from every other failure instead of collapsing both into the same
    # generic exception the retired jr.post_event-based call could not distinguish.
    try:
        status, decoded = post(jr.EVENTS_URL, body)
    except Exception as exc:
        raise Refusal("hub-unreachable") from exc
    if status == 409:
        raise Refusal("already-finished")
    # Fixes branch review F5: a 2xx status is only a success when the decoded body also
    # claims ok -- the route deliberately returns 200 {ok: false} for a non-claim
    # insert failure, which must not be read as "cancelled".
    if not (200 <= status < 300) or not isinstance(decoded, dict) or decoded.get("ok") is not True:
        raise Refusal(_hub_rejected_code(status, decoded))
    return "cancelled"


def _parse_worktree_porcelain(output):
    """One dict per `git worktree list --porcelain` entry: {"path", "branch"
    (bare name, None if detached), "detached", "locked"} -- `locked` covers both
    a bare `locked` line and `locked "<reason>"` (spec §4 guard 9)."""
    entries = []
    current = None
    for line in output.splitlines():
        if line.startswith("worktree "):
            current = {"path": line[len("worktree "):].strip(), "branch": None,
                       "detached": False, "locked": False}
            entries.append(current)
        elif current is None:
            continue
        elif line.startswith("branch "):
            ref = line[len("branch "):].strip()
            current["branch"] = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref
        elif line == "detached":
            current["detached"] = True
        elif line == "locked" or line.startswith("locked "):
            current["locked"] = True
    return entries


def _classify_gc_state(finished_payload):
    """Narrows `cmd_status`'s branching (jaxflow.py:4106-4136) to gc's own categories. None means `open` (no run-finished row yet)."""
    if finished_payload is None:
        return "open"
    status = finished_payload.get("contract_status")
    if status == "interrupted":
        return "interrupted"
    if status == "cancelled":
        return "cancelled"
    if status not in ("ok", "missing", "invalid"):
        return "unknown"  # F4 defensive catch-all: an unrecognized contract_status
    if finished_payload.get("result") == "success":
        return "success"
    return "other-terminal"  # failure / blocked / legacy (neither result nor reason)


def _gc_worktree_is_reserved(repo, run_id, candidate_path):
    """F2 (round 4): ownership requires the EXACT worktree path jaxflow reserved for
    `run_id` -- its own manifest's `worktree` field (`<repo>/.local/runs/<run_id>/
    manifest.json`, written by `dispatch_build`, `jaxflow.py:1494`) -- to equal
    `candidate_path`. Matching only `(project, repo, branch)` would let a worktree
    manually recreated on the same branch, at a DIFFERENT path, inherit another run's
    ownership and get GC'd on the strength of that guess. No manifest, an unreadable
    one, or a mismatched path are all "not reserved here"."""
    recorded = jc._read_manifest_field(repo, run_id, "worktree")
    if not isinstance(recorded, str):
        return False
    try:
        return Path(recorded).resolve() == candidate_path
    except OSError:
        return False


def _latest_builder_run_for_gc(con, project, repo, branch, candidate_path):
    """Like `_latest_builder_attempt` (jaxflow.py:994) plus the matching run-finished
    row, gated by ownership of `candidate_path` (F2, round 4: see `_gc_worktree_is_
    reserved`). (None, None, None) means no builder ever ran on `branch` at exactly
    this path (`unowned`)."""
    run_id = jc._latest_builder_attempt(con, project, repo, branch)
    if run_id is None or not _gc_worktree_is_reserved(repo, run_id, candidate_path):
        return None, None, None
    row = con.execute(
        "SELECT ts, payload FROM workflow_events WHERE run_id = ? AND type = 'run-finished' LIMIT 1",
        (run_id,),
    ).fetchone()
    if row is None:
        return run_id, None, None
    return run_id, json.loads(row["payload"]), row["ts"]


def _gc_precheck(entry, *, control_repo, allowlist_root, latest_build_run_id,
                   finished_payload, finished_ts, now, older_than_days):
    """The §4/§6 guards that never need a dirty-tree probe: control-repo, detached,
    locked, allowlist, ownership, open/unknown state, and the age floor (F1, round 4).
    Returns the same {"action": "skip", "state": ...} shape `_gc_decide` would, or
    `None` when the candidate is still eligible and the caller must probe `git status`
    next. The caller MUST call this before any `is_dir`/`git status` probe, so an
    ineligible candidate never triggers one."""
    path = Path(entry["path"]).resolve()
    if path == control_repo:
        return {"action": "skip", "state": "control-repo"}
    if entry["detached"]:
        return {"action": "skip", "state": "detached"}
    if entry["locked"]:
        return {"action": "skip", "state": "locked"}
    if not _contained(path, allowlist_root):
        return {"action": "skip", "state": "out-of-allowlist"}
    if latest_build_run_id is None:
        return {"action": "skip", "state": "unowned"}
    state = _classify_gc_state(finished_payload)
    if state in ("open", "unknown"):
        return {"action": "skip", "state": state}
    age_days = (now - datetime.fromisoformat(finished_ts)).days
    if age_days < older_than_days:
        return {"action": "skip", "state": state}
    return None


def _gc_decide(entry, *, control_repo, allowlist_root, latest_build_run_id,
                finished_payload, finished_ts, now, older_than_days, dirty, force):
    """Pure §4/§6 decision for ONE `_parse_worktree_porcelain` entry. `dirty` is the
    caller's `git status --porcelain` result (failed probe = dirty, fail-safe like
    `cmd_merge`'s `dirty-tracked-tree`) -- the caller is expected to have already run
    `_gc_precheck` and only probed for `dirty` when it returned `None` (F1). Returns
    {"action": "skip"|"remove", "state": str}; `state` doubles as the `gc-removed`
    event's `reason` (spec G7)."""
    pre = _gc_precheck(entry, control_repo=control_repo, allowlist_root=allowlist_root,
                        latest_build_run_id=latest_build_run_id, finished_payload=finished_payload,
                        finished_ts=finished_ts, now=now, older_than_days=older_than_days)
    if pre is not None:
        return pre
    state = _classify_gc_state(finished_payload)
    if dirty and not force:
        return {"action": "skip", "state": state}
    return {"action": "remove", "state": state}


def _redeliver_gc_spools(control_repo, *, post):
    """Retries any leftover `gc-removed` spool under `<control_repo>/.local/runs/`
    (round 3 F6: at-least-once), deleting it on success. Identified by its own
    `type` field -- a pending run-finished spool (the reaper's/`cmd_cancel`'s job)
    is left untouched. `run_id` already has a run-finished row, so neither ever
    revisits it."""
    runs_dir = control_repo / ".local" / "runs"
    if not runs_dir.is_dir():
        return
    for run_dir in sorted(runs_dir.iterdir()):
        spool = run_dir / "finished.json"
        if not spool.is_file():
            continue
        try:
            event = json.loads(spool.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(event, dict) or event.get("type") != "gc-removed":
            continue
        try:
            post(event)
        except Exception:
            continue  # still down; the next `gc` invocation retries again
        spool.unlink(missing_ok=True)


def cmd_gc(args, *, run=jr.run_command, post=jc._post_event, now=None, db_path=None,
           allowlist_root=ALLOWLIST_ROOT_DEFAULT):
    if args.dry_run and args.yes:
        raise Refusal("gc-both-modes")
    m = re.fullmatch(r"([0-9]+)d", args.older_than)
    if not m:
        raise Refusal("gc-duration-invalid")
    older_than_days = int(m.group(1))
    now_val = now() if now is not None else datetime.now().astimezone()
    db_path = db_path or jr.DB_PATH
    cwd = Path.cwd().resolve()
    repo = jc._require_toplevel(run, cwd)
    project = jc.slugify_project(repo.name)

    if args.yes:
        _redeliver_gc_spools(repo, post=post)

    listed = run(["git", "worktree", "list", "--porcelain"], cwd=repo)
    entries = _parse_worktree_porcelain(listed.stdout) if listed.returncode == 0 else []

    con = jc._open_ro(db_path)
    con.row_factory = sqlite3.Row
    try:
        candidates = []
        for entry in entries:
            branch = entry["branch"]
            candidate_path = Path(entry["path"]).resolve()
            latest_build_run_id = finished_payload = finished_ts = None
            if branch is not None:
                latest_build_run_id, finished_payload, finished_ts = _latest_builder_run_for_gc(
                    con, project, repo, branch, candidate_path)
            # F1 (round 4): every skip guard (allowlist, detached, locked, ownership,
            # open/unknown state, age floor) runs BEFORE the `is_dir`/`git status` probe
            # below -- an ineligible candidate never triggers it.
            pre = _gc_precheck(
                entry, control_repo=repo, allowlist_root=allowlist_root, latest_build_run_id=latest_build_run_id,
                finished_payload=finished_payload, finished_ts=finished_ts, now=now_val,
                older_than_days=older_than_days)
            if pre is not None:
                candidates.append((entry, latest_build_run_id, finished_payload, finished_ts, pre))
                continue
            path = Path(entry["path"]).resolve()
            dirty = True
            if path.is_dir():
                probe = run(["git", "status", "--porcelain"], cwd=path)
                dirty = probe.returncode != 0 or bool(probe.stdout.strip())
            decision = _gc_decide(
                entry, control_repo=repo, allowlist_root=allowlist_root, latest_build_run_id=latest_build_run_id,
                finished_payload=finished_payload, finished_ts=finished_ts, now=now_val,
                older_than_days=older_than_days, dirty=dirty, force=args.force)
            candidates.append((entry, latest_build_run_id, finished_payload, finished_ts, decision))
    finally:
        con.close()

    for entry, run_id, finished_payload, finished_ts, decision in candidates:
        line = f"{entry['path']} ({entry['branch'] or 'detached'}): {decision['state']} -> {decision['action']}"
        if decision["action"] == "remove":
            # F3 (round 4): every removable candidate's line names its age and warns
            # that removal discards `--resume` eligibility -- true of dry-run (where
            # this is the whole point) and of `--yes` (printed before the attempt).
            age_days = (now_val - datetime.fromisoformat(finished_ts)).days
            line += f" (age {age_days}d, removal discards --resume eligibility)"
        print(line)
        if decision["action"] != "remove" or not args.yes:
            continue
        worktree = Path(entry["path"]).resolve()
        # F2 (plan review round 1): re-probe now, the listing-time one may be stale. A
        # probe failure is never "clean," and --force bypasses only genuine dirt (guard 5/10).
        try:
            reprobe = run(["git", "status", "--porcelain"], cwd=worktree)
        except Exception as exc:
            print(f"{worktree} kept (probe-failed: {jr._bound(str(exc), 200)})")
            continue
        if reprobe.returncode != 0:
            print(f"{worktree} kept (probe-failed)")
            continue
        if reprobe.stdout.strip() and not args.force:
            print(f"{worktree} kept (dirty)")
            continue
        copy_result = jc._copy_run_reports(worktree, repo, run_id)
        contract_status = (finished_payload or {}).get("contract_status")
        if copy_result["failed"] or (not copy_result["existed"] and contract_status != "missing"):
            print(f"{worktree} kept (report not safely preserved)")
            continue
        # (spec §4 guard 5/9) re-list right before a forced removal, bail if no longer
        # registered or now locked -- guards 2/9, which `--force` never bypasses.
        relisted = run(["git", "worktree", "list", "--porcelain"], cwd=repo)
        still_registered = relisted.returncode == 0 and any(
            e["path"] == entry["path"] and e["branch"] == entry["branch"] and not e["locked"] for e in _parse_worktree_porcelain(relisted.stdout))
        if args.force and not still_registered:
            print(f"{worktree} kept (no longer registered or now locked)")
            continue
        # `age_days` was already computed above for the dry-run/`--yes` line (F3).
        # gc-removed is NOT runScoped (workflow.ts EVENT_MATRIX) -- run_id lives in the
        # payload, never as a top-level event key (that's reserved for run-started/-finished).
        event = {"project": project, "role": "lead", "type": "gc-removed",
                 "source": "deterministic", "emitter": "wrapper",
                 "payload": {"run_id": run_id, "branch": entry["branch"], "worktree": str(worktree),
                             "reason": decision["state"], "age_days": age_days}}
        jc._write_spool(repo, run_id, event)
        remove_cmd = ["git", "worktree", "remove", str(worktree)] + (["--force"] if args.force else [])
        removed = run(remove_cmd, cwd=repo)
        if removed.returncode != 0:
            jc._delete_spool(repo, run_id)
            print(f"{worktree} kept ({jr._bound(removed.stderr.strip(), 200)})")
            continue
        deleted = run(["git", "branch", "-d", entry["branch"]], cwd=repo)
        if deleted.returncode != 0:
            print(f"branch {entry['branch']} kept ({jr._bound((deleted.stderr or deleted.stdout).strip(), 200)})")
        try:
            post(event)
        except Exception as exc:
            print(f"gc-removed event for {run_id} kept for redelivery ({jr._bound(str(exc), 200)})")
            continue
        jc._delete_spool(repo, run_id)


def _nonblank(value):
    """argparse `type` for a command that must actually run something."""
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty or whitespace-only")
    return value


_FROM_HELP_DISPATCH = (
    "Caller identity. Required when caller detection is ambiguous (CLAUDECODE and "
    "CODEX_THREAD_ID both set, or neither set) -- jaxflow otherwise cannot tell which "
    "tech lead is calling. Also needs the matching session-id env var set "
    "(CLAUDE_CODE_SESSION_ID for claude, CODEX_THREAD_ID for codex); missing or empty "
    "refuses caller-session-missing."
)

_FROM_HELP_MERGE = (
    "Caller identity. Required when caller detection is ambiguous (CLAUDECODE and "
    "CODEX_THREAD_ID both set, or neither set) -- jaxflow otherwise cannot tell which "
    "tech lead is calling."
)


def _normalize_phase(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _anchor_builds_for_prefix(con, prefix_norm, project=None):
    """Every `run-started` build row whose normalized phase CONTAINS `prefix_norm`
    (spec L1) -- a short prefix is naturally a substring of a real build's phase."""
    sql = "SELECT id, run_id, project, ts, payload FROM workflow_events WHERE type = 'run-started' AND role = 'builder'"
    params = ()
    if project is not None:
        sql += " AND project = ?"
        params = (project,)
    sql += " ORDER BY ts, id"
    result = []
    for row in con.execute(sql, params).fetchall():
        payload = json.loads(row["payload"])
        if payload.get("kind") != "build":
            continue
        if prefix_norm in _normalize_phase(payload.get("phase", "")):
            result.append({"run_id": row["run_id"], "ts": row["ts"], "payload": payload})
    return result


def _reviewer_rows(con, kind):
    """All run-started rows for `kind` ('spec'/'plan'/'diff'), role=reviewer, parsed
    and ordered by ts -- the shared fetch behind every loop-metric join below."""
    rows = con.execute("SELECT run_id, ts, payload FROM workflow_events WHERE type = 'run-started' AND role = 'reviewer' ORDER BY ts, id").fetchall()
    out = []
    for row in rows:
        payload = json.loads(row["payload"])
        if payload.get("kind") == kind:
            out.append({"run_id": row["run_id"], "ts": row["ts"], "payload": payload})
    return out


def _diff_rows_for_builds(con, anchor_builds):
    """Diff rows linked to any of `anchor_builds` -- by `builder_run_id` when present
    (L2), falling back to a `target` branch match for legacy rows predating that field."""
    build_run_ids = {b["run_id"] for b in anchor_builds}
    build_branches = {b["payload"]["target"] for b in anchor_builds}
    result = []
    for r in _reviewer_rows(con, "diff"):
        linked = r["payload"].get("builder_run_id")
        if linked is not None:
            if linked in build_run_ids:
                result.append(r)
        elif r["payload"].get("target") in build_branches:
            result.append(r)
    return result


def _spec_or_plan_rows_for_build_phases(con, kind, build_phases_norm):
    """spec/plan rows whose own normalized phase is a PREFIX of one of
    `build_phases_norm` (L1's new spec/plan-to-build join)."""
    return [r for r in _reviewer_rows(con, kind)
            if any(bp.startswith(_normalize_phase(r["payload"].get("phase", ""))) for bp in build_phases_norm)]


def _rounds_to_approve(con, diff_rows):
    """§9: count up to and including the first diff run-finished row where
    contract_status == "ok" AND verdict is approve/approve-with-changes (F8)."""
    rounds = 0
    approved_run_id = None
    for row in diff_rows:
        rounds += 1
        finished = con.execute(
            "SELECT payload FROM workflow_events WHERE run_id = ? AND type = 'run-finished' LIMIT 1", (row["run_id"],),
        ).fetchone()
        if finished is None:
            continue
        fp = json.loads(finished["payload"])
        if fp.get("contract_status") == "ok" and fp.get("verdict") in ("approve", "approve-with-changes"):
            approved_run_id = row["run_id"]
            break
    return rounds, approved_run_id


def _wall_time_days(con, anchor_builds, spec_rows, plan_rows):
    """§9: matched merge-approved ts (by branch, L5) minus the earliest matched spec
    review's ts, falling back to plan then the build itself (L4). None = not merged yet.

    F4 (plan review round 1, MEDIUM): the merge match is WINDOWED to the anchored build's
    own dispatch per branch, window = [that branch's earliest anchor-build ts, its next
    build's dispatch ts if any) -- never "the first merge-approved ever seen for this
    branch," which a reused branch would resolve to the wrong (often much older) merge.
    Same tie-break as `getLoopSummary`'s TS mirror (F4).

    F1 (diff review 79cbcce4, MEDIUM): a trailing spec/plan review dispatched AFTER the
    merge (e.g. a re-review) must not push start_ts past merge_ts -- that would render as
    a negative wall time. None instead."""
    if not anchor_builds:
        return None
    branches = {b["payload"]["target"] for b in anchor_builds}
    all_builds = con.execute(
        "SELECT ts, payload FROM workflow_events WHERE type = 'run-started' AND role = 'builder' ORDER BY ts"
    ).fetchall()
    build_ts_by_branch = {}
    for row in all_builds:
        payload = json.loads(row["payload"])
        if payload.get("kind") != "build":
            continue
        build_ts_by_branch.setdefault(payload.get("target"), []).append(row["ts"])
    merges = con.execute("SELECT ts, payload FROM workflow_events WHERE type = 'merge-approved' ORDER BY ts").fetchall()

    merge_ts = None
    for branch in branches:
        window_start = min(b["ts"] for b in anchor_builds if b["payload"]["target"] == branch)
        later = sorted(ts for ts in build_ts_by_branch.get(branch, []) if ts > window_start)
        window_end = later[0] if later else None
        for row in merges:
            if json.loads(row["payload"]).get("branch") != branch:
                continue
            if row["ts"] < window_start:
                continue
            if window_end is not None and row["ts"] >= window_end:
                continue
            if merge_ts is None or row["ts"] < merge_ts:
                merge_ts = row["ts"]
            break  # merges is ts-ordered ascending -- first in-window match is the earliest
    if merge_ts is None:
        return None
    start_rows = spec_rows or plan_rows or anchor_builds
    start_ts = min(r["ts"] for r in start_rows)
    if start_ts > merge_ts:
        return None  # F1: trailing review after the merge -- never a negative wall time
    return (datetime.fromisoformat(merge_ts) - datetime.fromisoformat(start_ts)).total_seconds() / 86400


def _direct_phase_match(con, kind, prefix_norm):
    """L1 fallback: a spec/plan row with no anchor build yet, matched directly by
    its own normalized phase containing `prefix_norm`."""
    return [r for r in _reviewer_rows(con, kind) if prefix_norm in _normalize_phase(r["payload"].get("phase", ""))]


def cmd_loop(prefix, *, db_path=None):
    prefix_norm = _normalize_phase(prefix)
    if len(prefix_norm) < 4:
        raise Refusal("loop-prefix-too-short")
    db_path = db_path or jr.DB_PATH
    con = jc._open_ro(db_path)
    con.row_factory = sqlite3.Row
    try:
        anchors = _anchor_builds_for_prefix(con, prefix_norm)
        if anchors:
            build_phases_norm = [_normalize_phase(b["payload"]["phase"]) for b in anchors]
            spec_rows = _spec_or_plan_rows_for_build_phases(con, "spec", build_phases_norm)
            plan_rows = _spec_or_plan_rows_for_build_phases(con, "plan", build_phases_norm)
        else:
            # L1 fallback: no anchor build yet -- match spec/plan rows directly by prefix.
            spec_rows = _direct_phase_match(con, "spec", prefix_norm)
            plan_rows = _direct_phase_match(con, "plan", prefix_norm)
        diff_rows = _diff_rows_for_builds(con, anchors)
        rounds, approved_run_id = _rounds_to_approve(con, diff_rows)
        wall_days = _wall_time_days(con, anchors, spec_rows, plan_rows)
        resumed = sum(1 for b in anchors if b["payload"].get("resumes_run_id"))
    finally:
        con.close()
    build_part = f"build {len(anchors)}" + (f" (incl. {resumed} resume)" if resumed else "")
    header = f"{prefix} — spec {len(spec_rows)} · plan {len(plan_rows)} · {build_part} · diff {len(diff_rows)}"
    if diff_rows:
        rounds_line = (f"rounds to approve: {rounds} (round {rounds} approved, {approved_run_id})"
                       if approved_run_id else f"{rounds} rounds, not approved yet")
    else:
        rounds_line = "rounds to approve: n/a (no diff review dispatched yet)"
    wall_line = "wall time: not merged yet" if wall_days is None else f"wall time: {wall_days:.0f} days"
    return f"{header}\n{rounds_line}\n{wall_line}"


_RUN_ID_HELP = "Run id printed by 'jaxflow review' or 'jaxflow build'."

_OLDER_THAN_HELP = "Age threshold, e.g. 7d (default). Only whole days are supported."


def _add_from(parser, help_text):
    """The `--from` caller-identity option shared by review, build, pr open, release and merge
    (the help text differs: dispatch verbs also need the session variable)."""
    parser.add_argument(
        "--from", dest="from_caller", choices=("claude", "codex", "jaxos"), help=help_text,
    )


def _add_review_parser(sub):
    review = sub.add_parser(
        "review",
        help="Dispatch a cold review of a spec, a plan, or a finished build's diff.",
        description="Dispatch a cold review in a background tmux session and print its "
                     "run_id. The reviewer uses the opposite runtime (the other reviewer runtime when that agent is off in /settings) with saved UI model "
                     "and effort defaults. Exactly one of --spec/--plan/--diff selects "
                     "what gets reviewed.",
    )
    target = review.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--spec",
        help="Path to the spec file to review. Exactly one of --spec/--plan/--diff is "
             "required.",
    )
    target.add_argument(
        "--plan",
        help="Path to the plan file to review. Exactly one of --spec/--plan/--diff is "
             "required.",
    )
    target.add_argument(
        "--diff",
        help="RUN ID of a finished 'jaxflow build' whose diff to review (from the build's "
             "recorded base to HEAD) -- NOT a path and NOT a branch name. Exactly one of --spec/--plan/--diff is required.",
    )
    review.add_argument(
        "--focus",
        help="Extra instructions appended to the reviewer's prompt (what to look at first).",
    )
    review.add_argument(
        "--model",
        help="Override the saved UI reviewer model for this call only.",
    )
    review.add_argument(
        "--effort",
        help="Override the saved UI reviewer effort for this call only.",
    )
    review.add_argument(
        "--phase",
        help="Run identifier token. Defaults to the target file's stem for --spec/--plan, "
             "or to the build run's own phase for --diff.",
    )
    _add_from(review, _FROM_HELP_DISPATCH)
    review.add_argument(
        "--no-callback", action="store_true",
        help="Do not send the '[JAXFLOW] ... finished' line back to the caller's tmux pane "
             "when the run completes.",
    )
    since_full = review.add_mutually_exclusive_group()
    since_full.add_argument(
        "--since", metavar="PRIOR_REVIEW_RUN_ID",
        help="Correction round: review only the delta since a prior terminal diff "
             "review's head_sha, instead of merge-base..HEAD. Also bypasses the "
             "terminal-verdict re-review guard (MOA-471 items 7/9). Mutually "
             "exclusive with --full.",
    )
    since_full.add_argument(
        "--full", metavar="REASON",
        help="Bypass the terminal-verdict re-review guard and run a full review "
             "anyway. Requires a non-blank reason, recorded verbatim in the manifest "
             "(MOA-471 item 7). Mutually exclusive with --since.",
    )
    review.add_argument(
        "--reverify", action="store_true",
        help="For --diff: force a full verify+build re-run, ignoring the reuse check "
             "(MOA-471 item 8).",
    )
    review.set_defaults(func=_cli_review)


def _add_build_parser(sub):
    build = sub.add_parser(
        "build",
        help="Reserve a worktree and dispatch a builder to implement a plan.",
        description="Reserve a fresh worktree for a NEW branch and dispatch the managed "
                     "opencode-builder in the background using profiles saved via /tools "
                     "Agents. Uses saved Default unless --fallback; --resume reuses the "
                     "recorded worktree and profile name. The plan -- and the "
                     "spec it declares -- are read from their ORIGINAL locations in the "
                     "validated read scope; no copies are made. Prints a run_id.",
    )
    build.add_argument(
        "--plan",
        help="Path to the ORIGINAL plan file the builder implements. Never copied: the "
             "builder reads it in place, along with its declared spec. Required unless "
             "--resume.",
    )
    build.add_argument(
        "--phase",
        help="Short label for this run. Recorded in the run ledger and validated against "
             "the builder report's own front matter. It names NEITHER the worktree (that "
             "comes from --branch) NOR the tmux session (jax-<project>-build-<run_id>). "
             "Required unless --resume.",
    )
    build.add_argument(
        "--branch",
        help="Name of the NEW branch to create for this build. Refuses branch-exists if "
             "the branch or its worktree path already exists. Required unless --resume.",
    )
    build.add_argument(
        "--whitelist",
        help="Comma-separated list of paths the builder may touch. Required unless --resume.",
    )
    build.add_argument(
        "--verify", type=_nonblank,
        help="ONE shell command string (a pipeline or && chain counts as one) that jaxflow "
             "itself re-runs after the builder exits, IN ADDITION to whatever the builder "
             "ran on its own; a failure marks the run's result as failure. This is the TEST "
             "command -- pass the build command to --build rather than chaining it here, or "
             "a failing test hides whether the tree still builds. Required unless --resume.",
    )
    build.add_argument(
        "--resume",
        help="Retry a failed/blocked managed builder by reusing its recorded worktree. "
             "Supplies plan/phase/whitelist/verify/build/base from the prior attempt; "
             "fresh-build flags are refused.",
    )
    build.add_argument(
        "--build", type=_nonblank,
        help="OPTIONAL second shell command string, the BUILD command, run after --verify "
             "and always run even when --verify failed. Both land in the builder handoff as "
             "commands.test/commands.build and get their own frame in the evidence file. "
             "Omit it and the handoff's commands.build is honestly `none`.",
    )
    build.add_argument(
        "--base",
        help="ref the new branch starts from (default: the repo's default branch). Use it "
             "to STACK work: a part-2 plan builds on part 1's branch with "
             "--base feat/thing-p1 --branch feat/thing-p2. Resolved to a commit SHA before "
             "anything is created; an unknown ref refuses base-invalid.",
    )
    build.add_argument(
        "--builder",
        help=argparse.SUPPRESS,
    )
    build.add_argument(
        "--model", help=argparse.SUPPRESS,
    )
    build.add_argument(
        "--effort",
        help=argparse.SUPPRESS,
    )
    build.add_argument(
        "--fallback", action="store_true",
        help="Use the saved fallback builder profile instead of default.",
    )
    _add_from(build, _FROM_HELP_DISPATCH)
    build.add_argument(
        "--no-callback", action="store_true",
        help="Do not send the '[JAXFLOW] build ... finished' line back to the caller's "
             "tmux pane when the run completes.",
    )
    build.set_defaults(func=_cli_build)


def _add_pr_parser(sub):
    pr = sub.add_parser(
        "pr",
        help="Open a GitHub PR for a PR-preset delivery.",
        description="Open (or resume) the GitHub PR a PR-preset delivery merges through: "
                     "push <branch> and open its PR into the Delivery target (or, for a "
                     "release/* head, the Production target). Run it once the diff review "
                     "passes; no Rafa approval is needed to open a PR -- the merge approval "
                     "is the only gate.",
    )
    pr_sub = pr.add_subparsers(dest="pr_command", required=True)
    popen_ = pr_sub.add_parser("open", help="Open or resume a PR for <branch>.")
    popen_.add_argument("branch", help="Branch to open a PR for. Must resolve to --sha.")
    popen_.add_argument("--sha", required=True,
                         help="Full 40-hex head SHA the PR must point at.")
    popen_.add_argument("--target", required=True,
                         help="Base branch. Asserted against the Delivery target (a "
                              "non-release/* head) or the Production target (a release/* "
                              "head) -- refuses target-mismatch otherwise.")
    popen_.add_argument("--title", required=True, type=_nonblank,
                         help="PR title, passed to gh pr create --title verbatim.")
    popen_.add_argument("--body-file", help="Optional path passed to gh pr create --body-file.")
    _add_from(popen_, _FROM_HELP_MERGE)
    popen_.set_defaults(func=_cli_pr_open)


def _add_release_parser(sub):
    release = sub.add_parser(
        "release",
        help="Cut and open the staging -> production promotion PR (dual-branch-pr only).",
        description="Snapshots the Delivery target, names release/<date>-staging-promotion, "
                     "and opens its PR into the Production target, reusing an open release PR. A SEPARATE "
                     "Rafa approval then runs jaxflow merge on the printed branch/sha.",
    )
    _add_from(release, _FROM_HELP_MERGE)
    release.set_defaults(func=_cli_release)


def _add_merge_parser(sub):
    merge = sub.add_parser(
        "merge",
        help="Merge a branch Rafa has explicitly approved by SHA.",
        description="Merge an approved branch in the foreground. Run only after Rafa's "
                     "explicit approval naming the branch, the full 40-hex head SHA, and "
                     "the destination (--target); there is no run_id and no status/result "
                     "for a merge. Local presets merge --no-ff here and push the target. A "
                     "PR preset (single-branch-pr, dual-branch-pr) merges the branch's GitHub PR instead (gh pr merge "
                     "--merge --match-head-commit): the PR's head branch, head SHA and base "
                     "must match the approval, GitHub's own branch protection must allow the "
                     "merge, and --checks run on the approved SHA first, unless an up-to-date "
                     "diff review already proved the same command on this exact SHA "
                     "(--recheck forces the run).",
    )
    merge.add_argument(
        "branch",
        help="Name of the branch to merge. Must resolve to the SHA passed via --sha.",
    )
    merge.add_argument(
        "--sha", required=True,
        help="Full 40-hex head SHA Rafa approved. Refuses sha-mismatch if it is not full "
             "40-hex or HEAD does not match it.",
    )
    merge.add_argument(
        "--phase", required=True,
        help="Becomes the merge commit's subject line. Refuses phase-invalid if empty, "
             "untrimmed, over 200 UTF-16 code units, or containing a tab, newline, BOM, "
             "or unpaired surrogate.",
    )
    # `required=True` only demands the FLAG. `--checks ""` reaches `/bin/bash -lc ""`,
    # which exits 0, so the merge would commit and audit with the verification gate
    # never having run anything (whole-branch review, F1). Refused at the CLI boundary
    # like every other argument shape here, before any git state is touched.
    merge.add_argument(
        "--checks", required=True, type=_nonblank,
        help="Shell command run after the merge, before push (PR presets: before the PR "
             "merge, on the approved SHA), unless an up-to-date diff review already proved the same command on this exact SHA. Still required and must not be blank -- an "
             "empty string would exit 0 without checking anything, and is refused here "
             "at the CLI boundary before any git state is touched.",
    )
    merge.add_argument(
        "--recheck", action="store_true",
        help="Force the --checks run even when an up-to-date diff review already proved the "
             "same command on this exact SHA (MOA-510). No effect when the delivery is "
             "already merged and only needs recording.",
    )
    merge.add_argument(
        "--target", required=True,
        help="Approved destination branch. Asserted against the Deploy policy target "
             "(PR presets: the Delivery target for a feature branch, the Production target "
             "for a release/* branch); cannot override it. Refuses target-invalid if "
             "empty, malformed, or not a literal branch name, and target-mismatch if it "
             "differs from policy.",
    )
    _add_from(merge, _FROM_HELP_MERGE)
    merge.set_defaults(func=_cli_merge)


def _add_run_query_parsers(sub):
    status = sub.add_parser(
        "status",
        help="Print a run's current state.",
        description="Print a run's current state: unknown, running, dead, or "
                     "finished(ok|failed|cancelled).",
    )
    status.add_argument("run_id", help=_RUN_ID_HELP)
    status.set_defaults(func=_cli_status)
    result = sub.add_parser(
        "result",
        help="Print a finished run's report.",
        description="Print a finished run's report path and content. A cancelled run "
                     "prints 'cancelled \u2014 <summary>' instead (no report). Refuses if the "
                     "run has not finished.",
    )
    result.add_argument("run_id", help=_RUN_ID_HELP)
    result.set_defaults(func=_cli_result)
    cancel = sub.add_parser(
        "cancel",
        help="Kill a running run.",
        description="Kill a run's tmux session and post a cancelled terminal row to the "
                     "ledger.",
    )
    cancel.add_argument("run_id", help=_RUN_ID_HELP)
    cancel.set_defaults(func=_cli_cancel)
    sub.add_parser("doctor", help="Read-only health check: what is and isn't wired up.").set_defaults(func=_cli_doctor)


def _add_gc_loop_parsers(sub):
    gc = sub.add_parser(
        "gc",
        help="Remove stale jaxflow-reserved worktrees.",
        description="Remove worktrees whose latest builder run finished (any terminal "
                     "state) at least --older-than ago and whose tree is clean. Dry-run "
                     "by default; --yes executes; --force also removes a dirty tree.",
    )
    gc.add_argument("--older-than", default="7d", help=_OLDER_THAN_HELP)
    gc.add_argument("--dry-run", action="store_true")
    gc.add_argument("--yes", action="store_true")
    gc.add_argument("--force", action="store_true")
    gc.set_defaults(func=_cli_gc)

    loop = sub.add_parser(
        "loop",
        help="Print round/wall-time metrics for a phase.",
        description="Groups spec/plan/build/diff/merge runs by a normalized phase "
                     "prefix and reports counts, rounds to approve, and wall time. "
                     "No git repo required.",
    )
    loop.add_argument("prefix", help="Normalized to at least 4 chars; e.g. moa-474.")
    loop.set_defaults(func=_cli_loop)


def _add_mission_parser(sub):
    mission = sub.add_parser(
        "mission",
        help="Track Rafa's own multi-phase objective (opt-in, cross-project).",
        description="Start, update, and finish the ONE active mission -- a name, goal, "
                     "milestone checklist, and free-text status line, entirely separate "
                     "from jaxflow's own per-run tracking. Never invoked automatically; "
                     "only /jaxflow-mission or Rafa's own explicit request drives it. No "
                     "--from, no git-repo requirement, no caller-identity check -- same "
                     "carve-out status/result/cancel already have.",
    )
    mission_sub = mission.add_subparsers(dest="mission_command", required=True)
    mstart = mission_sub.add_parser("start", help="Start the one active mission.")
    mstart.add_argument("--name", required=True)
    mstart.add_argument("--goal", required=True)
    mstart.add_argument("--milestone", dest="milestones", action="append", default=[],
                         help="Repeatable; at least one is required.")
    mstatus = mission_sub.add_parser("status", help="Update the active mission's free-text status line.")
    mstatus.add_argument("text")
    mmark = mission_sub.add_parser("mark", help="Set one milestone's state.")
    mmark.add_argument("milestone", help="1-based index or exact milestone title.")
    mmark.add_argument("state", help="done or in-progress.")
    mission_sub.add_parser("done", help="Finish the active mission as done.")
    mission_sub.add_parser("cancel", help="Finish the active mission as cancelled.")
    mission_sub.add_parser("show", help="Print the active mission, or 'no active mission'.")
    mission.set_defaults(func=_cli_mission)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="jaxflow",
        description="Single dispatch path for jaxflow reviews, builds, and merges. Never "
                     "call the reviewer/builder runtime (codex, claude, opencode) directly.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for add in (_add_review_parser, _add_build_parser, _add_pr_parser, _add_release_parser,
                _add_merge_parser, _add_run_query_parsers, _add_gc_loop_parsers,
                _add_mission_parser):
        add(sub)
    return parser.parse_args(argv)


def _cli_review(args, deps):
    dispatch = jv.dispatch_diff_review if args.diff else jv.dispatch_review
    run_id = dispatch(args, run=deps.run, post=deps.post, env=deps.env, now=deps.now,
                      allowlist_root=deps.allowlist_root)
    print(run_id)
    return jc.OK


def _cli_build(args, deps):
    run_id = jb.dispatch_build(args, run=deps.run, post=deps.post, env=deps.env, now=deps.now,
                            allowlist_root=deps.allowlist_root)
    print(run_id)
    return jc.OK


def _cli_pr_open(args, deps):
    result = jm.cmd_pr_open(args, run=deps.run, post=deps.post, env=deps.env, now=deps.now,
                         allowlist_root=deps.allowlist_root)
    print(f"{result['url']}")
    return jc.OK


def _cli_release(args, deps):
    result = jm.cmd_release(args, run=deps.run, post=deps.post, env=deps.env, now=deps.now,
                         allowlist_root=deps.allowlist_root)
    print(f"{result['url']}")
    print(f"snapshot: {result['snapshot_sha']}")
    print(f"next: jaxflow merge {result['branch']} --sha {result['snapshot_sha']} "
          f"--phase \"<title>\" --checks \"<cmd>\" --target <Production target>")
    return jc.OK


def _cli_merge(args, deps):
    return jm.cmd_merge(args, run=deps.run, post=deps.post, env=deps.env, now=deps.now,
                     allowlist_root=deps.allowlist_root)


def _cli_gc(args, deps):
    cmd_gc(args, run=deps.run, post=deps.post, now=deps.now, allowlist_root=deps.allowlist_root)
    return jc.OK


def _cli_loop(args, deps):
    print(cmd_loop(args.prefix))
    return jc.OK


def _cli_status(args, deps):
    print(cmd_status(args.run_id, run=deps.run))
    return jc.OK


def _cli_result(args, deps):
    sys.stdout.write(cmd_result(args.run_id, allowlist_root=deps.allowlist_root))
    return jc.OK


def _cli_cancel(args, deps):
    # cmd_cancel does NOT take main()'s own `post` (that one is `_post_event`-shaped
    # for dispatch/worker); it uses its own `_post`-shaped default so it can see the
    # raw HTTP status (fixes cold review F2).
    print(cmd_cancel(args.run_id, run=deps.run, now=deps.now))
    return jc.OK


def _cli_doctor(args, deps):
    lines, code = cmd_doctor()
    for line in lines:
        print(line)
    return code


def _cli_mission(args, deps):
    # Mission uses the module's own `_get`/`_post` (HTTP-status shaped), never main()'s `post`.
    if args.mission_command == "show":
        print(cmd_mission_show(get=_get))
    elif args.mission_command == "start":
        cmd_mission_start(args, post=jc._post)
    elif args.mission_command == "status":
        cmd_mission_status(args, post=jc._post)
    elif args.mission_command == "mark":
        cmd_mission_mark(args, post=jc._post)
    elif args.mission_command == "done":
        cmd_mission_finish("done", post=jc._post)
    elif args.mission_command == "cancel":
        cmd_mission_finish("cancelled", post=jc._post)
    return jc.OK


def main(argv=None, *, run=jr.run_command, post=jc._post_event, env=None, now=None,
          allowlist_root=ALLOWLIST_ROOT_DEFAULT, popen=subprocess.Popen):
    argv = sys.argv[1:] if argv is None else list(argv)
    env = os.environ if env is None else env
    now = (lambda: datetime.now().astimezone()) if now is None else now
    if argv and argv[0] == "--run-worker":
        if len(argv) != 2 or not Path(argv[1]).is_absolute():
            print("run-worker requires one absolute manifest path", file=sys.stderr)
            return jc.REFUSED
        return run_worker(argv[1], run=run, post=post, popen=popen, env=env, allowlist_root=allowlist_root)
    args = parse_args(argv)
    deps = SimpleNamespace(run=run, post=post, env=env, now=now, allowlist_root=allowlist_root)
    try:
        return args.func(args, deps)
    except Refusal as exc:
        print(exc.code, file=sys.stderr)
        hint = getattr(exc, "hint", None)
        if hint:
            print(hint, file=sys.stderr)
        return jc.REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
