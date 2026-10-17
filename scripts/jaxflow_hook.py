#!/usr/bin/env python3
"""jaxflow-hook — tmux-gated Claude/Codex lead-session event translator (D1)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import jaxflow_env as jenv

EVENTS_URL = f"{jenv.api_base_url()}/api/workflow/events"
SESSIONS_URL = f"{jenv.api_base_url()}/api/workflow/sessions"
MODES = {
    "claude-pretool", "claude-posttool", "claude-notification", "claude-stop", "claude-userprompt",
    "codex-stop", "codex-permission", "codex-userprompt", "claude-callback",
    "claude-toolused", "codex-toolused", "claude-subagentstart", "codex-subagentstart",
    "claude-subagentstop", "codex-subagentstop",
}
CAPSULE_RE = re.compile(r"^\[JAXFLOW: (done|needs_input|blocked|waiting ~?\d{1,3}[mh])\]")
DURATION_RE = re.compile(r"waiting ~?(\d{1,3})([mh])")
# Merge question contract (spec 2026-10-09 section 4): the ONE canonical question. Case-insensitive on
# the verb only; backticks around branch and target are required; the optional "(`<sha>`)" is
# tolerated and ignored (merge_head_sha always comes from git); the "?" must close the same line. Branch and target are capped at 200 chars (= ingress bound).
MERGE_QUESTION_RE = re.compile(
    r"(?i:posso mergear|may i merge) `([A-Za-z0-9._/-]{1,200})`(?: \(`([0-9a-f]{7,40})`\))? (?:em|into) `([A-Za-z0-9._/-]{1,200})`[^\n?]*\?"
)
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")

MESSAGE_TAIL_MAX = 2000  # mirrors LIMITS.messageTail in src/lib/workflow.ts

# Redaction, extended (§4.3): four patterns kept verbatim from classify_stop.py, five new.
SECRET_RE = re.compile(
    r"(?i)\b((?:api[_-]?key|token|password|secret|[a-z][a-z0-9_-]*(?:api[_-]?key|token|password|secret)[a-z0-9_-]*))\s*([:=])\s*([^\s,;]+)"
)
BEARER_RE = re.compile(r"(?i)\b(authorization\s*:\s*bearer\s+)([^\s,;]+)")
COOKIE_RE = re.compile(r"(?i)\b((?:cookie\s*:\s*)?(?:session|sessionid|sid)\s*=\s*)([^;\s,]+)")
RAW_KEY_RE = re.compile(r"\b(?:AIza[A-Za-z0-9_-]{35}|sk-[A-Za-z0-9_-]{20,})\b")
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")
CRED_URL_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/@:]+:[^\s/@]+@")
PROVIDER_TOKEN_RE = re.compile(r"\b(?:ghp_|gho_|ghu_|ghs_|github_pat_|xox[baprs]-|AKIA[0-9A-Z]{16}|npm_)[A-Za-z0-9_-]{10,}\b")
GENERIC_KEY_ASSIGN_RE = re.compile(r"(?i)\b([a-z][a-z0-9_-]*[_-](?:key|secret|credential))\s*([:=])\s*([^\s,;]{8,})")


def redact(text):
    text = PRIVATE_KEY_RE.sub("[REDACTED]", str(text))  # multi-line first — before line-oriented patterns
    text = SECRET_RE.sub(r"\1\2[REDACTED]", text)
    text = BEARER_RE.sub(r"\1[REDACTED]", text)
    text = COOKIE_RE.sub(r"\1[REDACTED]", text)
    text = RAW_KEY_RE.sub("[REDACTED]", text)
    text = JWT_RE.sub("[REDACTED]", text)
    text = CRED_URL_RE.sub("[REDACTED@]", text)
    text = PROVIDER_TOKEN_RE.sub("[REDACTED]", text)
    return GENERIC_KEY_ASSIGN_RE.sub(r"\1\2[REDACTED]", text)


def run_command(argv, cwd=None):
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)


def _truthy(value):
    """None (absent) is falsy in an `if` — the caller's fail-open default. A present value is
    truthy unless it matches the falsy set, case-insensitively (same convention as HONCHO_ENABLED)."""
    if value is None:
        return None
    return str(value).strip().lower() not in ("0", "false", "no", "off")


CODEX_MODES = {
    "codex-stop", "codex-permission", "codex-userprompt",
    "codex-toolused", "codex-subagentstart", "codex-subagentstop",
}
# Canonical session UUID: lowercase, hyphenated. Codex emits UUIDv7 here; the server-side
# validator (workflow-events.ts UUID_RE) enforces the same shape.
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# D3/D4/D5/D6/D7: the Claude-callback file drop this hook mode polls. A SEPARATE
# constant from jaxflow.CALLBACKS_ROOT, same value -- jaxflow.py already imports
# `redact`/`UUID_RE` FROM this module, so importing jaxflow.CALLBACKS_ROOT back here
# would be circular.
CALLBACKS_ROOT = jenv.jaxos_home() / "callbacks"
# Phase 3 (spec §6, Decision 4): per-identity throttle markers for tool-used — one file per
# (identity, 10s bucket), claimed with O_CREAT|O_EXCL so the claim and the check are ONE syscall.
# Same ~/.jax-os/ local-state convention as CALLBACKS_ROOT; tests monkeypatch it.
THROTTLE_ROOT = jenv.jaxos_home() / "hook-throttle"
THROTTLE_BUCKET_S = 10  # matches the dashboard's useMission(..., 10_000) poll
THROTTLE_SWEEP_BUCKETS = 6  # markers more than 60s old are removed opportunistically
TOOLUSED_MODES = {"claude-toolused", "codex-toolused"}
TOOL_NAME_MAX = 64  # mirrors LIMITS.tool in src/lib/workflow.ts
AGENT_ID_MAX = 128  # mirrors LIMITS.agentId
AGENT_TYPE_MAX = 64  # mirrors LIMITS.agentType
# Mirrors jaxflow._RUN_ID_RE's value, same reason (no shared import).
RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_DISPATCH_KINDS = {"build", "diff", "spec", "plan"}
POLL_INTERVAL_S = 5  # ponytail: fixed cadence, no backoff/jitter -- D7
WATCH_DEADLINE_S = 14280  # ponytail: the hook's own 14400s asyncRewake timeout minus a 120s margin -- D7


def _git_owner(cwd, run):
    """Canonical repository root (the MAIN checkout) for cwd, or None outside a repository.

    Mirrors jaxflow._control_owner (MOA-467): an ordinary main checkout and a
    `--separate-git-dir` main checkout both have gitdir == common dir and resolve to their own
    toplevel; a linked worktree (gitdir != common) resolves to the MAIN entry of `git worktree
    list` — never a `.git`-parent name guess or the worktree's own directory name. Every failed
    probe returns None; the caller's non-repository fallback then applies.
    """
    if not isinstance(cwd, str) or not cwd:
        return None
    try:
        probe = run(["git", "rev-parse", "--path-format=absolute", "--git-dir", "--git-common-dir"], cwd=cwd)
    except OSError:
        return None
    if probe is None or probe.returncode != 0:
        return None
    lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    gitdir, common = os.path.normpath(lines[0]), os.path.normpath(lines[1])
    if gitdir == common:
        try:
            top = run(["git", "rev-parse", "--show-toplevel"], cwd=cwd)
        except OSError:
            return None
        if top is None or top.returncode != 0 or not top.stdout.strip():
            return None
        return top.stdout.strip()
    try:
        listed = run(["git", "worktree", "list", "--porcelain"], cwd=cwd)
    except OSError:
        return None
    if listed is None or listed.returncode != 0:
        return None
    for line in listed.stdout.splitlines():
        if line.startswith("worktree "):
            return line[len("worktree "):].strip() or None
    return None


def _project_from_cwd(cwd, run):
    """Project identity = the basename of the session's REPOSITORY ROOT (spec B §7.1c).

    Uses the canonical owner resolution above (MOA-467) so a builder in a linked worktree or a
    `--separate-git-dir` checkout still attributes to the main repo, never to the worktree's own
    directory name. Falls back to the basename of cwd outside any repository — today's behaviour,
    unchanged.
    """
    if not isinstance(cwd, str) or not cwd:
        return None
    owner = _git_owner(cwd, run)
    if owner:
        return os.path.basename(os.path.normpath(owner)) or None
    base = os.path.basename(os.path.normpath(cwd))
    return base or None


def _role_for(session):
    return "lead" if session and session.endswith("-lead") else "adhoc"


def _resolve_identity(pane, run):
    """Best-effort: one `tmux display-message` call returns session name + incarnation token
    (§4.1/§4.2.2, Finding 3). Returns (session, tmux_incarnation), either half None on failure."""
    result = run(["tmux", "display-message", "-p", "-t", pane, "#{session_name} #{pid} #{start_time}"])
    if result.returncode != 0:
        return None, None
    parts = result.stdout.strip().split()
    if len(parts) != 3:
        return None, None
    session, pid, start_time = parts
    return session, f"{pid}:{start_time}"


def classify_turn_stopped(message):
    """Returns (payload, source). The hook decides ONLY the capsule tag; every other rung of
    the ladder is the server's (spec §3.2). An untagged turn ships its redacted tail as data.

    Redaction runs ONCE, here, on the WHOLE message before anything is sliced. Slicing first
    would let a secret straddling the boundary lose the head that makes it matchable and keep
    the tail that makes it a secret."""
    if not isinstance(message, str) or not message:
        return {}, "behavioral"
    stripped = redact(message.strip())
    if not stripped:
        # Whitespace-only, or a message that was nothing but a redacted secret. An empty
        # message_tail would be REFUSED by optStr at ingress and the whole stop event would
        # vanish; an absent one is the shape the route settles as `abandoned`.
        return {}, "behavioral"
    m = CAPSULE_RE.match(stripped[:64])
    if not m:
        return {"message_tail": stripped[-MESSAGE_TAIL_MAX:]}, "behavioral"
    tag = m.group(1)
    if tag.startswith("waiting"):
        d = DURATION_RE.match(tag)
        n, unit = int(d.group(1)), d.group(2)
        bound = 480 if unit == "m" else 8
        if n < 1 or n > bound:
            # out-of-bounds duration: fully malformed, treated exactly like no tag at all
            return {"message_tail": stripped[-MESSAGE_TAIL_MAX:]}, "behavioral"
        minutes = n if unit == "m" else n * 60
        payload = {"capsule_status": "waiting", "capsule_minutes": minutes, "capsule_rule": "tag"}
    else:
        payload = {"capsule_status": tag, "capsule_rule": "tag"}
    rest = stripped[m.end():].lstrip()
    excerpt = rest.split("\n", 1)[0][:200]
    if excerpt:
        payload["excerpt"] = excerpt
    return payload, "deterministic"


def find_merge_question(text):
    """(branch, target) of the LAST canonical merge question in `text`, else None."""
    if not isinstance(text, str):
        return None
    matches = list(MERGE_QUESTION_RE.finditer(text))
    return (matches[-1].group(1), matches[-1].group(3)) if matches else None


def _content(entry):
    message = entry.get("message")
    return message.get("content") if isinstance(message, dict) else None


def _is_prompt(entry):
    """A real user prompt: not meta, not a subagent line, not a bare tool_result answer."""
    if entry.get("type") != "user" or entry.get("isMeta") or entry.get("isSidechain"):
        return False
    content = _content(entry)
    if isinstance(content, str):
        return True
    return isinstance(content, list) and any(isinstance(b, dict) and b.get("type") != "tool_result" for b in content)


def transcript_turn_text(path):
    """Assistant text blocks after the last user prompt of a Claude transcript JSONL, joined
    by newlines; None when the file is unreadable, holds ANY malformed line, or holds no such text. ponytail: reads the
    whole file; tail-read it only if Stop latency is ever measured as a problem."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return None
    texts = []
    for line in lines:
        # ANY malformed line abandons the transcript (-> caller falls back to last_assistant_message):
        # skipping it could drop a newer user prompt and resurrect a PREVIOUS turn's question.
        try:
            entry = json.loads(line)
        except ValueError:
            return None
        if not isinstance(entry, dict):
            return None
        if _is_prompt(entry):
            texts = []
        elif entry.get("type") == "assistant" and not entry.get("isSidechain"):
            content = _content(entry)
            if isinstance(content, list):
                texts += [b["text"] for b in content
                          if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)]
    return "\n".join(texts) or None


def turn_text(mode, payload):
    """The text scanned for the merge question: Claude = the whole turn from the transcript PLUS
    last_assistant_message appended (the Stop hook may fire before the final block is flushed to the
    transcript; the last match wins, so the duplicate is harmless); Codex = last_assistant_message
    only (one final message per turn)."""
    last = payload.get("last_assistant_message")
    last = last if isinstance(last, str) else ""
    if mode == "claude-stop":
        path = payload.get("transcript_path")
        if isinstance(path, str) and path:
            text = transcript_turn_text(path)
            if text:
                return text + "\n" + last
    return last or None


def merge_head_sha(branch, cwd, run=run_command):
    """Branch head observed at Stop time (audit only); None on any failure."""
    if branch.startswith("-"):
        return None
    try:
        result = run(["git", "rev-parse", "--verify", f"{branch}^{{commit}}"], cwd=cwd)
    except Exception:
        return None
    out = result.stdout.strip() if result.returncode == 0 else ""
    return out if GIT_SHA_RE.fullmatch(out) else None


def stop_payload(mode, payload, run=run_command):
    """classify_turn_stopped plus the merge-question contract. A matching turn is ALWAYS the one
    deterministic needs_input shape (any capsule tag in the turn is dropped: the question wins);
    every other non-empty shape just gains merge_ask: 0."""
    event_payload, source = classify_turn_stopped(payload.get("last_assistant_message"))
    found = find_merge_question(redact(turn_text(mode, payload) or ""))
    if found:
        branch, target = found
        return {
            "capsule_status": "needs_input", "capsule_rule": "merge-question", "capsule_attempts": 0,
            "merge_ask": 1, "merge_branch": branch, "merge_target": target,
            "merge_head_sha": merge_head_sha(branch, payload.get("cwd"), run),
        }, "deterministic"
    if event_payload:
        event_payload = {**event_payload, "merge_ask": 0}
    return event_payload, source


def event_from_input(mode, project, pane, session, payload, run=run_command):
    if not isinstance(payload, dict):
        return None
    role = _role_for(session)
    base = {"project": project, "role": role}
    if pane:
        # Optional pane metadata is not authority for Codex correlation (§3); a native Codex
        # event carries no pane at all and must not fabricate one.
        base["pane"] = pane
    harness_session = payload.get("session_id")
    if isinstance(harness_session, str) and 0 < len(harness_session) <= 128:
        base["harness_session"] = harness_session
    if mode == "claude-pretool":
        if payload.get("tool_name") != "AskUserQuestion":
            return None
        tool_use_id = payload.get("tool_use_id")
        tool_input = payload.get("tool_input")
        questions = tool_input.get("questions") if isinstance(tool_input, dict) else None
        if not isinstance(tool_use_id, str) or not tool_use_id or not isinstance(questions, list):
            return None
        return {**base, "type": "question", "source": "deterministic", "emitter": "claude-pretool",
                "payload": {"tool_use_id": tool_use_id, "questions": questions}}
    if mode == "claude-posttool":
        if payload.get("tool_name") != "AskUserQuestion":
            return None
        tool_use_id = payload.get("tool_use_id")
        if not isinstance(tool_use_id, str) or not tool_use_id:
            return None
        return {**base, "type": "question-resolved", "source": "deterministic", "emitter": "claude-posttool",
                "payload": {"tool_use_id": tool_use_id}}
    if mode == "claude-notification":
        if payload.get("notification_type") != "agent_needs_input":
            return None
        return {**base, "type": "attention-needed", "source": "behavioral", "emitter": "claude-notification",
                "payload": {"reason": "agent_needs_input"}}
    if mode == "claude-stop":
        event_payload, source = stop_payload(mode, payload, run)
        return {**base, "type": "turn-stopped", "source": source, "emitter": "claude-stop", "payload": event_payload}
    if mode == "codex-stop":
        event_payload, source = stop_payload(mode, payload, run)
        return {**base, "type": "turn-stopped", "source": source, "emitter": "codex-stop", "payload": event_payload}
    if mode == "codex-permission":
        return {**base, "type": "attention-needed", "source": "deterministic", "emitter": "codex-permission",
                "payload": {"reason": "permission_request"}}
    if mode == "claude-userprompt":
        return {**base, "type": "turn-started", "source": "deterministic", "emitter": "claude-userprompt", "payload": {}}
    if mode == "codex-userprompt":
        return {**base, "type": "turn-started", "source": "deterministic", "emitter": "codex-userprompt", "payload": {}}
    if mode in TOOLUSED_MODES:
        # Decision 2: the tool NAME only — never tool_input, tool_response, or a path.
        tool = payload.get("tool_name")
        if not isinstance(tool, str) or not tool:
            return None
        return {**base, "type": "tool-used", "source": "deterministic", "emitter": mode,
                "payload": {"tool": tool[:TOOL_NAME_MAX]}}
    if mode in ("claude-subagentstart", "codex-subagentstart"):
        agent_id, agent_type = payload.get("agent_id"), payload.get("agent_type")
        if not isinstance(agent_id, str) or not agent_id or not isinstance(agent_type, str) or not agent_type:
            return None
        return {**base, "type": "subagent-started", "source": "deterministic", "emitter": mode,
                "payload": {"agent_id": agent_id[:AGENT_ID_MAX], "agent_type": agent_type[:AGENT_TYPE_MAX]}}
    if mode in ("claude-subagentstop", "codex-subagentstop"):
        agent_id = payload.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id:
            return None
        return {**base, "type": "subagent-stopped", "source": "deterministic", "emitter": mode,
                "payload": {"agent_id": agent_id[:AGENT_ID_MAX]}}
    return None


def claim_tool_bucket(identity, now=None, root=None):
    """Decision 4 / round 2 F2: at most one tool-used post per identity per 10s bucket. Returns True
    when this call won the bucket (marker created), False when another invocation already had it
    (FileExistsError) -- the same O_CREAT|O_EXCL idiom jaxflow's _open_child_log uses, so there is
    no read-then-write window. Markers more than THROTTLE_SWEEP_BUCKETS buckets old are removed
    best-effort; a delete racing another process's sweep just finds the file already gone."""
    root = THROTTLE_ROOT if root is None else root
    bucket = int((time.time() if now is None else now) // THROTTLE_BUCKET_S)
    key = re.sub(r"[^A-Za-z0-9]", "_", str(identity))[:80]
    root.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(root / f"{key}-{bucket}"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    os.close(fd)
    for entry in root.iterdir():
        stem, _, suffix = entry.name.rpartition("-")
        if stem and suffix.isdigit() and int(suffix) < bucket - THROTTLE_SWEEP_BUCKETS:
            try:
                entry.unlink()
            except OSError:
                pass
    return True


def post_json(url, payload, timeout=5):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError("hook post failed")
        decoded = json.loads(resp.read().decode("utf-8"))
    if not isinstance(decoded, dict) or decoded.get("ok") is not True:
        err = decoded.get("error") if isinstance(decoded, dict) else None
        if isinstance(err, str) and err:
            raise RuntimeError(err)
        raise RuntimeError("hook post failed")
    return decoded


def _claude_callback(stdin):
    """D3/D5/D6/D7: claims at most one pending callback pointer for this invocation's
    own session (from stdin's `session_id`), then blocks in-process polling for its
    `.line` file. Exits 2 with the delivered `[JAXFLOW] ...` line on stderr (the
    asyncRewake wake), or 0 silently whenever there is nothing safe/pending to act on."""
    try:
        raw = stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        return 0
    session_id = payload.get("session_id") if isinstance(payload, dict) else None
    if not isinstance(session_id, str) or not UUID_RE.fullmatch(session_id):
        return 0
    session_dir = CALLBACKS_ROOT / session_id
    if not session_dir.is_dir():
        return 0
    # ponytail: the lexicographically-first pending pointer wins when more than one is
    # queued -- D5's own ceiling is "claimed one per subsequent Bash call," not a
    # fairness guarantee across several pending pointers in the same session.
    pending = sorted(session_dir.glob("*.json"))
    if not pending:
        return 0
    claimed_path = _claim_pointer(pending[0])
    if claimed_path is None:
        return 0  # lost the race to another invocation
    parsed = _validate_claimed_pointer(claimed_path)
    if parsed is None:
        claimed_path.unlink(missing_ok=True)
        print(f"jaxflow-hook: dropped malformed callback pointer {claimed_path}", file=sys.stderr)
        return 0
    run_id, kind = parsed
    return _watch_for_line(session_dir, run_id, kind, claimed_path)


def _claim_pointer(pointer_path):
    """Atomic rename `<run_id>.json` -> `<run_id>.claimed` (D5): stops two invocations
    racing the same pending pointer. Returns the claimed Path, or None when the rename
    lost the race (the file was already gone)."""
    claimed_path = pointer_path.with_suffix(".claimed")
    try:
        pointer_path.rename(claimed_path)
    except OSError:
        return None
    return claimed_path


def _validate_claimed_pointer(claimed_path):
    """D5: structural-only validation of an already-claimed pointer's own content --
    valid JSON, an object whose key set is EXACTLY {run_id, kind}, both string
    values, run_id fullmatching RUN_ID_RE and equal to the filename stem, kind one of
    the four dispatch kinds. Returns (run_id, kind) or None on any failure."""
    try:
        data = json.loads(claimed_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or set(data.keys()) != {"run_id", "kind"}:
        return None
    run_id, kind = data.get("run_id"), data.get("kind")
    if not isinstance(run_id, str) or not isinstance(kind, str):
        return None
    if not RUN_ID_RE.fullmatch(run_id) or run_id != claimed_path.stem:
        return None
    if kind not in _DISPATCH_KINDS:
        return None
    return run_id, kind


def _watch_for_line(session_dir, run_id, kind, claimed_path):
    """D7: polls for `<run_id>.line` every POLL_INTERVAL_S seconds (a plain module
    constant, patched to a tiny value by tests that need a fast loop, rather than
    mocking `time` itself). A `.line` already present at claim time is delivered on the
    FIRST check, before any sleep -- no poll needed. Self-deadlines at WATCH_DEADLINE_S
    seconds (the hook's own asyncRewake timeout minus a safety margin)."""
    line_path = session_dir / f"{run_id}.line"
    deadline = time.monotonic() + WATCH_DEADLINE_S
    while True:
        if line_path.is_file():
            return _deliver_line(line_path, claimed_path)
        if time.monotonic() >= deadline:
            claimed_path.unlink(missing_ok=True)
            print(
                f"[JAXFLOW] {kind} {run_id} callback watcher expired — run may still be "
                f"going — use jaxflow status/result {run_id}",
                file=sys.stderr,
            )
            return 2
        time.sleep(POLL_INTERVAL_S)


def _deliver_line(line_path, claimed_path):
    """D7/§5.2: reads at most 4096 bytes, first line only, then deletes both the
    `.claimed` and `.line` files before printing -- so a crash between delete and print
    never leaves the pair around to be redelivered."""
    try:
        with line_path.open("rb") as fh:
            raw = fh.read(4096)
    except OSError:
        raw = b""
    text = raw.decode("utf-8", errors="replace").splitlines()[0] if raw.strip() else ""
    claimed_path.unlink(missing_ok=True)
    line_path.unlink(missing_ok=True)
    print(text, file=sys.stderr)
    return 2


CLASSIFY_KICK = ["systemctl", "--user", "start", "--no-block", "jaxos-workflow-poll.service"]


def _kick_classifier(run=subprocess.run):
    """A text stop is classified (rung 7) by the workflow poll; start it now instead of
    waiting up to 2 min for its timer, so the card's buttons show within seconds. Best
    effort: the timer stays the safety net. Tests set JAXFLOW_NO_CLASSIFY_KICK."""
    if os.environ.get("JAXFLOW_NO_CLASSIFY_KICK"):
        return
    try:
        run(CLASSIFY_KICK, capture_output=True, timeout=5, check=False)
    except Exception:
        pass


def _should_arm_answer_watcher(payload):
    """D3: arm on a reason to expect an eventual answer -- message_tail (untagged prose, the
    rung-7 candidate) or a needs_input/blocked self-tag (rungs 1/3) or a merge-question (merge_ask 1). waiting/done or an empty
    tail (classify_turn_stopped's own {} return) arms nothing."""
    if payload.get("merge_ask") == 1:
        return True
    if "message_tail" in payload:
        return True
    return payload.get("capsule_status") in ("needs_input", "blocked")


def _armed_event_id(path):
    """`answer-<id>.armed` -> id; a malformed name counts as -1 (never supersedes)."""
    try:
        return int(path.name[len("answer-"):-len(".armed")])
    except ValueError:
        return -1


def _watch_for_answer(session_dir, harness_session, event_id):
    """D3: the Stop hook's own answer-watch loop -- polls `answer-<event_id>.answer` every
    POLL_INTERVAL_S seconds, deadlined at WATCH_DEADLINE_S, mirroring `_watch_for_line`/
    `_deliver_line` but with its own filenames (`.armed`/`.answer`, never `.json`/`.claimed`/
    `.line`) and no claim/rename step -- one Stop fires at a time, no concurrent race like
    PostToolUse(Bash). `harness_session` is kept for call-site readability, unused here."""
    del harness_session
    armed_path = session_dir / f"answer-{event_id}.armed"
    answer_path = session_dir / f"answer-{event_id}.answer"
    deadline = time.monotonic() + WATCH_DEADLINE_S
    while True:
        # Supersession: a later Stop armed a newer marker in this same session directory --
        # checked BEFORE reading the answer (plan review F2), so a stale answer is never
        # delivered into a later turn; this watcher's own answer (if any arrived anyway) is
        # discarded unread -- it will never come through this watcher (a newer one superseded
        # it). Silent exit, unlike the dispatch callback's own expiry wake: a wake here costs a
        # paid turn and carries no information.
        # Only a HIGHER event id supersedes (plan review round 2): watcher 6 must ignore marker 5.
        if any(_armed_event_id(p) > event_id for p in session_dir.glob("answer-*.armed")):
            armed_path.unlink(missing_ok=True)
            answer_path.unlink(missing_ok=True)
            return 0
        if answer_path.is_file():
            try:
                with answer_path.open("rb") as fh:
                    raw = fh.read(4096)
            except OSError:
                raw = b""
            text = raw.decode("utf-8", errors="replace").strip()
            armed_path.unlink(missing_ok=True)
            answer_path.unlink(missing_ok=True)
            print(text, file=sys.stderr)
            return 2
        if time.monotonic() >= deadline:
            armed_path.unlink(missing_ok=True)
            return 0
        time.sleep(POLL_INTERVAL_S)


def main(argv=None, stdin=None, run=run_command, post=post_json):
    argv = sys.argv[1:] if argv is None else list(argv)
    stdin = sys.stdin if stdin is None else stdin
    if _truthy(os.environ.get("JAXFLOW_ONESHOT")):
        return 0  # §4.8: checked first, before any other gate — silent, no stderr line
    if not argv or argv[0] not in MODES:
        print("jaxflow-hook: unknown mode", file=sys.stderr)
        return 0
    mode = argv[0]
    if mode == "claude-callback":
        # D3: never requires a git project from cwd or TMUX -- reads only stdin's own
        # session_id, so a sealed run or a cwd with no jaxflow-relevant project never
        # reaches (or needs) the shared branch below.
        return _claude_callback(stdin)
    try:
        raw = stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        project = _project_from_cwd(payload.get("cwd"), run)
        if not project:
            return 0
        if mode in CODEX_MODES:
            # §3: native Codex UserPromptSubmit/Stop/PermissionRequest hooks may fire with no TMUX.
            # The validated session UUID is the identity; a missing/invalid one is a bounded
            # diagnostic and NO invented event (never fall back to cwd or a pane).
            session_id = payload.get("session_id")
            if not isinstance(session_id, str) or not UUID_RE.match(session_id):
                print("jaxflow-hook: codex session_id missing or not a canonical uuid", file=sys.stderr)
                return 0
            event = event_from_input(mode, project, None, None, payload, run=run)
            if event is None:
                return 0
            if mode in TOOLUSED_MODES and not claim_tool_bucket(session_id):
                return 0  # lost this bucket's claim — skip silently (subagent modes never throttle)
            post(EVENTS_URL, event)
            return 0
        # Claude keeps its current pane requirement: no pane environment, no event.
        pane = os.environ.get("TMUX_PANE")
        if not os.environ.get("TMUX") or not pane:
            return 0
        session, tmux_incarnation = _resolve_identity(pane, run)
        event = event_from_input(mode, project, pane, session, payload, run=run)
        if event is None:
            return 0
        if mode in TOOLUSED_MODES and not claim_tool_bucket(pane):
            return 0  # before the session registration too: a skipped tool-use re-registers nothing
        if tmux_incarnation:
            event["tmux_incarnation"] = tmux_incarnation
        if session and mode in ("claude-userprompt", "claude-stop"):
            # Best-effort: session registration is a convenience registry (Spec B's watchtower),
            # never the correctness signal. A failure here must not drop the event POST below,
            # which is (Finding 3).
            # P3 (lean spec D15): registered once per turn boundary only; pane and tmux
            # incarnation are stable within a turn, no state kept. Tool/notification/subagent
            # events never re-register.
            try:
                post(SESSIONS_URL, {
                    "project": project, "session": session, "pane": pane,
                    "role": event["role"], "tmux_incarnation": tmux_incarnation,
                })
            except Exception:
                pass
        response = post(EVENTS_URL, event)
        if mode in ("claude-stop", "codex-stop") and "message_tail" in event["payload"]:
            _kick_classifier()
        if mode == "claude-stop":
            # D3: the arm decision is LOCAL, off the payload event_from_input already classified
            # (event["payload"] IS classify_turn_stopped's own result -- no second call, no new
            # route field). Only arm when post() actually returned a row id to key by.
            harness_session = event.get("harness_session")
            data = response.get("data") if isinstance(response, dict) else None
            event_id = data.get("id") if isinstance(data, dict) else None
            if (
                isinstance(harness_session, str) and UUID_RE.fullmatch(harness_session)
                and isinstance(event_id, int)
                and _should_arm_answer_watcher(event["payload"])
            ):
                session_dir = CALLBACKS_ROOT / harness_session
                session_dir.mkdir(parents=True, exist_ok=True)
                (session_dir / f"answer-{event_id}.armed").write_text("", encoding="utf-8")
                return _watch_for_answer(session_dir, harness_session, event_id)
    except Exception as exc:
        print(f"jaxflow-hook: {_bound_error(exc)}", file=sys.stderr)
    return 0


def _bound_error(exc):
    text = "".join(ch if ord(ch) >= 32 else " " for ch in str(exc))
    return text[:200] or "hook failed"


if __name__ == "__main__":
    raise SystemExit(main())
