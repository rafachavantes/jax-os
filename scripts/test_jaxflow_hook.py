#!/usr/bin/env python3
"""Stdlib-only tests for jaxflow-hook: D1 identity, JAXFLOW_ONESHOT, event mapping."""
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import jaxflow_hook as hook
import jev_client
import workflow_poll as wp


def _run_identity(session="jax-jax-os-lead", pid="234790", start_time="1787586213", ok=True):
    def run(argv, cwd=None):
        if argv[:2] == ["tmux", "display-message"]:
            if not ok:
                return SimpleNamespace(returncode=1, stdout="", stderr="no server")
            return SimpleNamespace(returncode=0, stdout=f"{session} {pid} {start_time}\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return run


_SESSION_A = "01234567-89ab-4cde-8f01-23456789abcd"
_SESSION_B = "fedcba98-7654-4321-8fed-cba987654321"
_RUN_A = "aaaaaaaaaaaa"
_RUN_B = "bbbbbbbbbbbb"


def _write_pointer(root, session, run_id, kind="build"):
    session_dir = Path(root) / ".jax-os" / "callbacks" / session
    session_dir.mkdir(parents=True, exist_ok=True)
    pointer = session_dir / f"{run_id}.json"
    pointer.write_text(json.dumps({"run_id": run_id, "kind": kind}), encoding="utf-8")
    return session_dir, pointer


def _stdin(session_id):
    import io
    return io.StringIO(json.dumps({"session_id": session_id}))


def test_tmux_absent_gate_no_http_calls():
    posts = []
    def post(url, payload, timeout=5):
        posts.append((url, payload))
        return {"ok": True}
    with patch.dict(os.environ, {}, clear=True):
        assert hook.main(["claude-stop"], stdin=io.StringIO("{}"), post=post) == 0
    assert posts == []
    with patch.dict(os.environ, {"TMUX": "1"}, clear=True):  # TMUX set, TMUX_PANE absent
        assert hook.main(["claude-stop"], stdin=io.StringIO("{}"), post=post) == 0
    assert posts == []


def test_jaxflow_oneshot_truthy_falsy_matrix_including_absent():
    posts = []
    def post(url, payload, timeout=5):
        posts.append((url, payload))
        return {"ok": True}
    payload = json.dumps({"cwd": "/home/rafa/repos/jax-os", "last_assistant_message": "done"})
    env = {"TMUX": "1", "TMUX_PANE": "%9"}
    for falsy in ("0", "false", "False", "no", "off", "OFF"):
        posts.clear()
        with patch.dict(os.environ, {**env, "JAXFLOW_ONESHOT": falsy}):
            hook.main(["claude-stop"], stdin=io.StringIO(payload), run=_run_identity(ok=False), post=post)
        assert posts, f"falsy value {falsy!r} must NOT suppress"
    for truthy in ("1", "true", "yes", "on"):
        posts.clear()
        with patch.dict(os.environ, {**env, "JAXFLOW_ONESHOT": truthy}):
            hook.main(["claude-stop"], stdin=io.StringIO(payload), run=_run_identity(ok=False), post=post)
        assert posts == [], f"truthy value {truthy!r} must suppress"
    # absent: fail-open, proceeds exactly like an ordinary lead turn (Ruling B, §12 item 12)
    posts.clear()
    with patch.dict(os.environ, env, clear=True):
        hook.main(["claude-stop"], stdin=io.StringIO(payload), run=_run_identity(ok=False), post=post)
    assert len(posts) == 1
    assert posts[0][0] == hook.EVENTS_URL


def test_project_is_basename_of_cwd_no_session_name_pattern():
    posts = []
    def post(url, payload, timeout=5):
        posts.append((url, payload))
        return {"ok": True}
    payload = json.dumps({"cwd": "/home/rafa/repos/Acme.AI", "last_assistant_message": "done"})
    with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
        hook.main(["claude-stop"], stdin=io.StringIO(payload), run=_run_identity(session="probe-session"), post=post)
    events = [p for url, p in posts if url == hook.EVENTS_URL]
    assert events[0]["project"] == "Acme.AI"
    assert events[0]["pane"] == "%9"


def test_role_derived_from_session_suffix_and_registers_session():
    posts = []
    def post(url, payload, timeout=5):
        posts.append((url, payload))
        return {"ok": True}
    payload = json.dumps({"cwd": "/home/rafa/repos/jax-os", "last_assistant_message": "done"})
    with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9", "JAXFLOW_NO_CLASSIFY_KICK": "1"}, clear=True):
        hook.main(["claude-stop"], stdin=io.StringIO(payload), run=_run_identity(session="jax-jax-os-lead"), post=post)
    assert [url for url, _ in posts] == [hook.SESSIONS_URL, hook.EVENTS_URL]
    assert posts[0][1] == {"project": "jax-os", "session": "jax-jax-os-lead", "pane": "%9", "role": "lead", "tmux_incarnation": "234790:1787586213"}
    assert posts[1][1]["role"] == "lead"
    assert posts[1][1]["type"] == "turn-stopped"


def test_session_registration_failure_does_not_drop_the_event_post_finding_3():
    posts = []
    def post(url, payload, timeout=5):
        if url == hook.SESSIONS_URL:
            raise RuntimeError("session registry unavailable")
        posts.append((url, payload))
        return {"ok": True}
    payload = json.dumps({"cwd": "/home/rafa/repos/jax-os", "notification_type": "agent_needs_input"})
    with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
        code = hook.main(["claude-notification"], stdin=io.StringIO(payload), run=_run_identity(session="jax-jax-os-lead"), post=post)
    assert code == 0
    assert [url for url, _ in posts] == [hook.EVENTS_URL]  # session POST failed silently; event POST still fired
    assert posts[0][1]["type"] == "attention-needed"


def test_adhoc_role_for_a_non_lead_session_name(is_adhoc_event=True):
    posts = []
    def post(url, payload, timeout=5):
        posts.append((url, payload))
        return {"ok": True}
    payload = json.dumps({"cwd": "/home/rafa/repos/jax-os", "notification_type": "agent_needs_input"})
    with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
        hook.main(["claude-notification"], stdin=io.StringIO(payload), run=_run_identity(session="probe-scratch"), post=post)
    assert [url for url, _ in posts] == [hook.EVENTS_URL]  # notification no longer registers a session
    assert posts[0][1]["role"] == "adhoc"


def test_failed_identity_lookup_falls_back_to_adhoc_and_skips_session_post_finding_37():
    posts = []
    def post(url, payload, timeout=5):
        posts.append((url, payload))
        return {"ok": True}
    payload = json.dumps({"cwd": "/home/rafa/repos/jax-os", "last_assistant_message": "done"})
    with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
        hook.main(["claude-stop"], stdin=io.StringIO(payload), run=_run_identity(ok=False), post=post)
    assert [url for url, _ in posts] == [hook.EVENTS_URL]  # no session POST — identity lookup failed
    assert posts[0][1]["pane"] == "%9"
    assert posts[0][1]["role"] == "adhoc"
    assert "tmux_incarnation" not in posts[0][1]


def test_claude_question_and_resolved_mapping():
    raw = json.dumps({
        "cwd": "/home/rafa/repos/jax-os",
        "tool_name": "AskUserQuestion", "tool_use_id": "toolu_1",
        "tool_input": {"questions": [{"question": "Merge?", "options": [{"label": "Yes"}]}]},
    })
    posts = []
    def post(url, payload, timeout=5):
        posts.append((url, payload))
        return {"ok": True}
    with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
        hook.main(["claude-pretool"], stdin=io.StringIO(raw), run=_run_identity(), post=post)
    event = posts[-1][1]
    assert event["type"] == "question"
    assert event["payload"] == {"tool_use_id": "toolu_1", "questions": [{"question": "Merge?", "options": [{"label": "Yes"}]}]}
    assert event["source"] == "deterministic"

    posts.clear()
    with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
        hook.main(["claude-posttool"], stdin=io.StringIO(raw), run=_run_identity(), post=post)
    assert posts[-1][1]["type"] == "question-resolved"
    assert posts[-1][1]["payload"] == {"tool_use_id": "toolu_1"}


def test_ignored_hook_inputs_produce_no_post():
    posts = []
    def post(url, payload, timeout=5):
        posts.append((url, payload))
        return {"ok": True}
    with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
        hook.main(["claude-pretool"], stdin=io.StringIO(json.dumps({"cwd": "/x/y", "tool_name": "Bash"})), run=_run_identity(), post=post)
    assert posts == []


def test_codex_permission_is_deterministic_attention_needed_without_tmux():
    posts = []
    def post(url, payload, timeout=5):
        posts.append((url, payload))
        return {"ok": True}
    uuid = "0191f0aa-1111-7000-8000-000000000001"
    raw = json.dumps({"cwd": "/home/rafa/repos/jax-os", "session_id": uuid, "tool_name": "shell", "tool_input": {}})
    with patch.dict(os.environ, {}, clear=True):  # no TMUX/TMUX_PANE at all
        hook.main(["codex-permission"], stdin=io.StringIO(raw), post=post)
    assert [url for url, _ in posts] == [hook.EVENTS_URL]  # native: no session registration
    event = posts[-1][1]
    assert event == {
        "project": "jax-os", "role": "adhoc", "type": "attention-needed",
        "source": "deterministic", "emitter": "codex-permission", "payload": {"reason": "permission_request"},
        "harness_session": uuid,
    }
    assert "pane" not in event and "tmux_incarnation" not in event


def test_codex_native_start_stop_and_permission_emit_once_without_tmux():
    uuid = "0191f0aa-2222-7000-8000-000000000002"
    for mode, payload, event_type, emitter in (
        ("codex-userprompt", {"cwd": "/home/rafa/repos/jax-os", "session_id": uuid, "prompt": "hi"}, "turn-started", "codex-userprompt"),
        ("codex-stop", {"cwd": "/home/rafa/repos/jax-os", "session_id": uuid, "last_assistant_message": "[JAXFLOW: done]"}, "turn-stopped", "codex-stop"),
        ("codex-permission", {"cwd": "/home/rafa/repos/jax-os", "session_id": uuid}, "attention-needed", "codex-permission"),
    ):
        posts = []
        def post(url, p, timeout=5):
            posts.append((url, p))
            return {"ok": True}
        with patch.dict(os.environ, {}, clear=True):
            assert hook.main([mode], stdin=io.StringIO(json.dumps(payload)), post=post) == 0
        assert len(posts) == 1, mode  # emits exactly once
        assert posts[0][0] == hook.EVENTS_URL
        event = posts[0][1]
        assert event["type"] == event_type and event["emitter"] == emitter
        assert event["harness_session"] == uuid
        assert "pane" not in event and "tmux_incarnation" not in event
        assert "hi" not in json.dumps(event)  # the prompt TEXT never leaves the process


def test_codex_native_missing_or_malformed_uuid_is_a_diagnostic_and_no_event():
    for bad in (None, "", "not-a-uuid", "0191F0AA-1111-7000-8000-000000000001", "0191f0aa-1111-7000-8000-00000000000"):
        posts = []
        def post(url, p, timeout=5):
            posts.append((url, p))
            return {"ok": True}
        payload = {"cwd": "/home/rafa/repos/jax-os"}
        if bad is not None:
            payload["session_id"] = bad
        with patch.dict(os.environ, {}, clear=True):
            assert hook.main(["codex-stop"], stdin=io.StringIO(json.dumps(payload)), post=post) == 0
        assert posts == [], bad


def test_claude_still_requires_its_pane_environment():
    posts = []
    def post(url, payload, timeout=5):
        posts.append((url, payload))
        return {"ok": True}
    raw = json.dumps({"cwd": "/home/rafa/repos/jax-os", "last_assistant_message": "done"})
    with patch.dict(os.environ, {}, clear=True):  # Claude has no native path
        hook.main(["claude-stop"], stdin=io.StringIO(raw), post=post)
    assert posts == []
    with patch.dict(os.environ, {"TMUX": "1"}, clear=True):  # TMUX but no pane
        hook.main(["claude-stop"], stdin=io.StringIO(raw), post=post)
    assert posts == []


def test_codex_oneshot_suppression_still_checked_first():
    posts = []
    def post(url, payload, timeout=5):
        posts.append((url, payload))
        return {"ok": True}
    payload = json.dumps({"cwd": "/home/rafa/repos/jax-os", "session_id": "0191f0aa-3333-7000-8000-000000000003"})
    with patch.dict(os.environ, {"JAXFLOW_ONESHOT": "1"}, clear=True):
        hook.main(["codex-stop"], stdin=io.StringIO(payload), post=post)
    assert posts == []


def test_userprompt_modes_map_to_local_turn_started_never_forwarding_prompt_text():
    # Claude rides the tmux pane; Codex is native (no tmux) and keyed by session UUID.
    for mode, emitter, env, extra in (
        ("claude-userprompt", "claude-userprompt", {"TMUX": "1", "TMUX_PANE": "%9"}, {}),
        ("codex-userprompt", "codex-userprompt", {}, {"session_id": "0191f0aa-4444-7000-8000-000000000004"}),
    ):
        posts = []
        def post(url, payload, timeout=5):
            posts.append((url, payload))
            return {"ok": True}
        raw = json.dumps({"cwd": "/home/rafa/repos/jax-os", "prompt": "should never leave the process", **extra})
        with patch.dict(os.environ, env, clear=True):
            hook.main([mode], stdin=io.StringIO(raw), run=_run_identity(), post=post)
        event = posts[-1][1]
        assert event["type"] == "turn-started"
        assert event["emitter"] == emitter
        assert event["source"] == "deterministic"
        assert event["payload"] == {}
        # plan fixture bug: `"prompt" not in json.dumps(event)` can never pass — the emitter
        # names themselves ("claude-userprompt"/"codex-userprompt") contain the substring
        # "prompt". Spec §4.7's actual requirement is that the PROMPT TEXT never leaves the
        # process (event["payload"] == {} above already proves that); check the literal value
        # from the raw hook input is absent from the serialized event instead.
        assert "should never leave the process" not in json.dumps(event)


def test_codex_stop_and_claude_stop_route_through_the_same_classifier():
    # Finding 7: invoke BOTH modes with independent assertions — a mutation that removed the
    # claude-stop branch entirely must fail this test, not just codex-stop's.
    for mode, emitter, env, extra in (
        ("codex-stop", "codex-stop", {}, {"session_id": "0191f0aa-5555-7000-8000-000000000005"}),
        ("claude-stop", "claude-stop", {"TMUX": "1", "TMUX_PANE": "%9"}, {}),
    ):
        posts = []
        def post(url, payload, timeout=5):
            posts.append((url, payload))
            return {"ok": True}
        raw = json.dumps({"cwd": "/home/rafa/repos/jax-os", "last_assistant_message": "[JAXFLOW: done]", **extra})
        with patch.dict(os.environ, env, clear=True):
            hook.main([mode], stdin=io.StringIO(raw), run=_run_identity(), post=post)
        event = posts[-1][1]
        assert event["type"] == "turn-stopped"
        assert event["emitter"] == emitter
        assert event["source"] == "deterministic"
        assert event["payload"] == {"capsule_status": "done", "capsule_rule": "tag", "merge_ask": 0}


def _fake_urlopen(status_text, calls):
    def urlopen(req, timeout=10):
        calls.append(req)
        body = json.dumps({"choices": [{"message": {"content": json.dumps({"status": status_text, "reason": "ok"})}}]})
        return _FakeClassifierResp(body.encode())
    return urlopen


class _FakeClassifierResp:
    def __init__(self, body):
        self._body = body
    def read(self):
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


def test_capsule_match_never_calls_the_llm_no_match_does():
    calls = []
    with patch("urllib.request.urlopen", _fake_urlopen("blocked", calls)):
        payload, source = hook.classify_turn_stopped("[JAXFLOW: done]")
        assert (payload, source) == ({"capsule_status": "done", "capsule_rule": "tag"}, "deterministic")
        assert calls == []
        payload2, source2 = hook.classify_turn_stopped("no tag here at all")
    assert source2 == "behavioral"
    assert payload2 == {"message_tail": "no tag here at all"}
    assert calls == []


def test_duration_bounds_accept_and_reject():
    for text, expected in (
        ("[JAXFLOW: waiting ~1m] ok", 1), ("[JAXFLOW: waiting ~480m] ok", 480),
        ("[JAXFLOW: waiting ~1h] ok", 60), ("[JAXFLOW: waiting ~8h] ok", 480),
    ):
        payload, source = hook.classify_turn_stopped(text)
        assert source == "deterministic"
        assert payload["capsule_minutes"] == expected
    for text in ("[JAXFLOW: waiting ~0m] ok", "[JAXFLOW: waiting ~481m] ok", "[JAXFLOW: waiting ~9h] ok"):
        payload, source = hook.classify_turn_stopped(text)
        assert source == "behavioral"  # out-of-bounds duration: fully malformed, not needs_input
        assert "message_tail" in payload


def test_case_sensitivity_and_no_tilde_variant():
    payload, source = hook.classify_turn_stopped("[jaxflow: done]")
    assert source == "behavioral"  # lowercase tag never matches — falls to fallback
    payload2, source2 = hook.classify_turn_stopped("[JAXFLOW: waiting 5m] no tilde")
    assert (payload2["capsule_status"], payload2["capsule_minutes"], source2) == ("waiting", 5, "deterministic")


def test_64_char_scan_window_boundary():
    prefix = "x" * 65
    payload, source = hook.classify_turn_stopped(f"{prefix}[JAXFLOW: done]")
    assert source == "behavioral"  # tag starts past character 64 — never scanned


def test_leading_whitespace_excerpt_invariant_at_two_offsets_finding_41():
    for spaces in (1, 4):
        text = f"{' ' * spaces}[JAXFLOW: done] status text"
        payload, source = hook.classify_turn_stopped(text)
        assert source == "deterministic"
        assert payload["capsule_status"] == "done"
        assert payload["excerpt"] == "status text"  # never a substring starting inside the tag


def test_tag_matched_with_nothing_after_it_has_no_excerpt_key():
    payload, _ = hook.classify_turn_stopped("[JAXFLOW: done]")
    assert "excerpt" not in payload


def test_redaction_covers_all_nine_patterns():
    # plan fixture bug: the plan's Google-API-key literal was 34 chars after "AIza" — one short
    # of RAW_KEY_RE's exact `{35}`, which is correctly ported verbatim from classify_stop.py —
    # so the regex never matched, redact() left it untouched, and the leak assertion below
    # failed even though the (correct) redaction code worked. Fixed to a correctly-sized 35-char
    # fake key, matching classify_stop.py's own test fixture convention ("AIza" + 35 chars).
    sample = (
        "api_key=sk-abcdefghijklmnopqrstuvwx01 "
        "Authorization: Bearer abcXYZ123token "
        "session=deadbeefcafefeed "
        "AIzaSy123456789012345678901234567890123 "
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sflKxwRJSMeKKF2QT4fwpM "
        "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAK\n-----END RSA PRIVATE KEY-----\n"
        "postgres://user:hunter2@db.example.com/prod "
        "ghp_abcdefghij0123456789 "
        "private_key=abcdefgh12345678"
    )
    redacted = hook.redact(sample)
    for leak in (
        "sk-abcdefghijklmnopqrstuvwx01", "abcXYZ123token", "deadbeefcafefeed",
        "AIzaSy123456789012345678901234567890123", "eyJhbGciOiJIUzI1NiJ9",
        "MIIBOgIBAAJBAK", "hunter2", "ghp_abcdefghij0123456789", "abcdefgh12345678",
    ):
        assert leak not in redacted, leak


def test_credential_is_read_from_one_secure_env_file_only(monkeypatch):
    """ONE source since 2026-09-07 (BWS is the single origin of shared secrets). The insecure-
    mode case must degrade to "" rather than fall through to some other credential file: a
    second source could only ever disagree with BWS. No test may touch a real credential, so
    every assertion here runs against a temporary file.

    MOA-498: `_classifier_api_key` now delegates to `jev_client.api_key`, whose precedence
    is process env > LOCAL_ENV_PATH > ENV_PATH. Patch jev_client's own constants (not
    workflow_poll's deleted ENV_PATH) and neuter the repo's real `.env.local` + process env
    so this test can only ever see the temporary file. The surrounding-quote strip matters:
    `hermes-env-sync` writes some values quoted."""
    import tempfile
    from pathlib import Path
    monkeypatch.delenv("OPENROUTER_API_KEY_BUILDER", raising=False)
    monkeypatch.setattr(jev_client, "LOCAL_ENV_PATH", Path("/nonexistent/jaxflow-hook-test/.env.local"))
    with tempfile.TemporaryDirectory() as d:
        env_path = Path(d) / ".env"
        env_path.write_text(
            "OTHER_KEY=not-this-one\nOPENROUTER_API_KEY_BUILDER=env-key-value\n", encoding="utf-8")
        os.chmod(env_path, 0o600)
        with patch.object(jev_client, "ENV_PATH", env_path):
            assert wp._classifier_api_key() == "env-key-value"

        env_path.write_text('OPENROUTER_API_KEY_BUILDER="quoted-value"\n', encoding="utf-8")
        os.chmod(env_path, 0o600)
        with patch.object(jev_client, "ENV_PATH", env_path):
            assert wp._classifier_api_key() == "quoted-value"

        os.chmod(env_path, 0o644)  # insecure mode — must degrade to empty, never fall through
        with patch.object(jev_client, "ENV_PATH", env_path):
            assert wp._classifier_api_key() == ""


class _FakeResp:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_post_json_surfaces_server_error_without_request_payload():
    # ponytail: post_json is untouched by the D1 rewrite (only its callers' shapes changed) —
    # kept from the pre-B1 suite rather than dropped, since it is the only coverage of the
    # no-secret-leak contract on a real RuntimeError message.
    payload = {"project": "jax-os", "secret": "do-not-leak", "body": "AskUserQuestion"}

    def urlopen(req, timeout=5):
        return _FakeResp(200, json.dumps({"ok": False, "error": "unknown option field"}).encode())

    with patch("urllib.request.urlopen", urlopen):
        try:
            hook.post_json(hook.EVENTS_URL, payload)
        except RuntimeError as exc:
            text = str(exc)
        else:
            raise AssertionError("expected RuntimeError")
    assert "unknown option field" in text
    assert "do-not-leak" not in text
    assert "AskUserQuestion" not in text
    assert "secret" not in text


def test_post_json_keeps_generic_fallback_when_error_malformed():
    payload = {"project": "jax-os", "secret": "do-not-leak"}
    for body in (
        {"ok": False},
        {"ok": False, "error": 12},
        {"ok": False, "error": ""},
        {"ok": False, "error": None},
        ["not", "an", "object"],
        {"ok": "true"},
    ):
        def urlopen(req, timeout=5, raw=body):
            return _FakeResp(200, json.dumps(raw).encode())

        with patch("urllib.request.urlopen", urlopen):
            try:
                hook.post_json(hook.EVENTS_URL, payload)
            except RuntimeError as exc:
                text = str(exc)
            else:
                raise AssertionError(f"expected RuntimeError for {body!r}")
        assert text == "hook post failed", body
        assert "do-not-leak" not in text


def _run_git(common_dir, ok=True):
    """Stub runner for the fallback probes: answers the git calls a linked-worktree
    resolution makes, nothing else."""
    def run(argv, cwd=None):
        if argv[:3] == ["git", "rev-parse", "--path-format=absolute"]:
            if not ok:
                return SimpleNamespace(returncode=128, stdout="", stderr="not a git repository")
            return SimpleNamespace(returncode=0, stdout=f"{common_dir}\n{common_dir}\n", stderr="")
        if argv[:2] == ["git", "worktree", "list"]:
            return SimpleNamespace(returncode=0, stdout=f"worktree {common_dir}\n", stderr="")
        if argv[:3] == ["git", "rev-parse", "--show-toplevel"]:
            return SimpleNamespace(returncode=0, stdout=f"{common_dir}\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return run


def _real_repo(parent, name, separate_git_dir=None):
    """Creates a real repository (main checkout) at parent/name and returns its path."""
    repo = os.path.join(parent, name)
    os.makedirs(repo, exist_ok=True)
    cmd = ["git", "init", "-q", "-b", "main"]
    if separate_git_dir is not None:
        cmd += ["--separate-git-dir", separate_git_dir]
    subprocess.run(cmd, cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "hook@test.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "hook test"], cwd=repo, check=True)
    with open(os.path.join(repo, "file.txt"), "w", encoding="utf-8") as f:
        f.write("x")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def test_project_from_repo_root_not_subdir():
    # A session opened in ~/repos/jax-os/docs must attribute to jax-os, not docs.
    with tempfile.TemporaryDirectory() as d:
        repo = _real_repo(d, "jax-os")
        os.makedirs(os.path.join(repo, "docs"))
        assert hook._project_from_cwd(os.path.join(repo, "docs"), hook.run_command) == "jax-os"


def test_project_from_worktree_uses_common_dir():
    # Inside a worktree, the owner resolves to the MAIN repo, not the worktree's dir name.
    with tempfile.TemporaryDirectory() as d:
        repo = _real_repo(d, "jax-os")
        wt = os.path.join(d, "wt-feat-x")
        subprocess.run(["git", "worktree", "add", "-q", "-b", "feat/x", wt], cwd=repo, check=True)
        os.makedirs(os.path.join(wt, "sub"))
        assert hook._project_from_cwd(wt, hook.run_command) == "jax-os"
        assert hook._project_from_cwd(os.path.join(wt, "sub"), hook.run_command) == "jax-os"


def test_project_from_separate_git_dir_main():
    # A `--separate-git-dir` MAIN checkout (gitdir == common) owns its own card.
    with tempfile.TemporaryDirectory() as d:
        gitdir = os.path.join(d, "gitdir")
        os.makedirs(gitdir)
        repo = _real_repo(d, "jax-os", separate_git_dir=gitdir)
        assert hook._project_from_cwd(repo, hook.run_command) == "jax-os"


def test_project_from_relative_linked_worktree():
    # A linked worktree added with a RELATIVE path must still resolve to the main repo.
    with tempfile.TemporaryDirectory() as d:
        repo = _real_repo(d, "jax-os")
        subprocess.run(["git", "worktree", "add", "-q", "-b", "feat/x", "wt-rel"], cwd=repo, check=True)
        assert hook._project_from_cwd(os.path.join(repo, "wt-rel"), hook.run_command) == "jax-os"


def test_project_outside_repo_falls_back_to_basename():
    run = _run_git("", ok=False)
    assert hook._project_from_cwd("/home/rafa/Documents", run) == "Documents"


def test_project_from_cwd_rejects_empty():
    run = _run_git("/home/rafa/repos/jax-os/.git")
    assert hook._project_from_cwd(None, run) is None
    assert hook._project_from_cwd("", run) is None


def test_project_git_output_whitespace_is_stripped():
    # A git runner that pads its output must still resolve cleanly (the probe strips lines).
    run = _run_git("  /home/rafa/repos/jax-os  ")
    assert hook._project_from_cwd("/home/rafa/repos/jax-os", run) == "jax-os"


def test_project_from_cwd_survives_git_oserror_falls_back_to_basename():
    # F1: a runner that raises OSError (missing git binary, invalid cwd) must not escape
    # _project_from_cwd — it should fall through to the basename(normpath(cwd)) fallback.
    def run(argv, cwd=None):
        raise FileNotFoundError("git")
    assert hook._project_from_cwd("/home/rafa/repos/jax-os", run) == "jax-os"


def test_hook_main_still_posts_event_when_git_raises_oserror():
    # F1: the failure must not escape to main()'s outer handler and skip the event POST.
    posts = []
    def post(url, payload, timeout=5):
        posts.append((url, payload))
        return {"ok": True}
    def run(argv, cwd=None):
        if argv[:2] == ["tmux", "display-message"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="no server")
        raise FileNotFoundError("git")
    payload = json.dumps({"cwd": "/home/rafa/repos/jax-os", "last_assistant_message": "done"})
    with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
        code = hook.main(["claude-stop"], stdin=io.StringIO(payload), run=run, post=post)
    assert code == 0
    events = [p for url, p in posts if url == hook.EVENTS_URL]
    assert events, "event POST must still fire when git raises OSError"
    assert events[0]["project"] == "jax-os"


def test_redaction_precedes_slicing_so_a_straddling_secret_cannot_survive():
    # The secret must STRADDLE the cut, or the test passes against a slice-then-redact
    # implementation and proves nothing. 520 + 43 + 1980 = 2543, so the 2000-char tail begins at
    # index 543 — 23 characters INTO a secret that spans [520, 563). The assertion is not
    # decoration: the first version of this fixture put the cut exactly at the secret's end,
    # which left the secret wholly outside the tail and the test unable to fail.
    secret = "sk-" + "A" * 40
    prefix, suffix = "x" * 519 + " ", "y" * 1980
    message = prefix + secret + suffix
    cut = len(message) - 2000
    assert len(prefix) < cut < len(prefix) + len(secret), f"fixture must straddle the cut, got {cut}"
    payload, source = hook.classify_turn_stopped(message)
    assert source == "behavioral"
    tail = payload["message_tail"]
    assert "sk-" not in tail
    assert "A" * 20 not in tail          # no surviving fragment, not merely a [REDACTED] present
    assert len(tail) <= 2000


def test_every_status_the_tag_regex_accepts_is_a_valid_capsule_status():
    # CAPSULE_RE and CAPSULE_STATUSES (src/lib/workflow.ts) must agree. If the regex ever gains a
    # word the enum lacks, ingress refuses every event carrying it and the tag path dies
    # silently: the hook keeps matching, the server keeps rejecting, and nothing says so.
    # Asserted behaviourally rather than by parsing the pattern — a pattern parser would trip
    # over \d and [mh] and assert something other than what it claims.
    VALID = {"done", "needs_input", "blocked", "waiting"}   # CAPSULE_STATUSES minus "unknown"
    for tag in ("done", "needs_input", "blocked", "waiting ~30m", "waiting ~2h"):
        payload, source = hook.classify_turn_stopped(f"[JAXFLOW: {tag}] shipped")
        assert source == "deterministic", tag
        assert payload["capsule_status"] in VALID, tag
    # `unknown` is in the enum but must never be reachable through the tag path — a producer
    # claiming it is refused at ingress (Task 2), so the regex must not match it either.
    _, source = hook.classify_turn_stopped("[JAXFLOW: unknown] shipped")
    assert source == "behavioral"


def test_tagged_turn_marks_the_rung_so_the_server_short_circuits():
    # Without capsule_rule the server cannot tell "the hook decided" from "the hook found
    # nothing", and rung 1 of the ladder could never fire.
    payload, _ = hook.classify_turn_stopped("[JAXFLOW: blocked] waiting on review")
    assert payload["capsule_rule"] == "tag"
    payload2, _ = hook.classify_turn_stopped("[JAXFLOW: waiting ~30m] long build")
    assert payload2 == {"capsule_status": "waiting", "capsule_minutes": 30, "capsule_rule": "tag",
                        "excerpt": "long build"}


def test_untagged_turn_sends_the_tail_and_no_status():
    payload, source = hook.classify_turn_stopped("just some prose with no tag")
    assert source == "behavioral"
    assert payload == {"message_tail": "just some prose with no tag"}
    assert "capsule_status" not in payload
    assert "excerpt" not in payload


def test_tagged_turn_sends_excerpt_and_no_tail_and_the_excerpt_is_redacted():
    payload, source = hook.classify_turn_stopped("[JAXFLOW: done] api_key=abcdefgh12345678 shipped")
    assert source == "deterministic"
    assert payload["capsule_status"] == "done"
    assert "message_tail" not in payload
    assert "abcdefgh12345678" not in payload["excerpt"]
    assert "[REDACTED]" in payload["excerpt"]


def test_multiline_private_key_is_caught_across_the_tail_boundary():
    """Diff-review F3: the first fixture put the whole key INSIDE the 2000-char window, so a
    slice-before-redact regression passed it. Only a key the cut runs through proves anything —
    and a multi-line pattern is the case no line- or window-oriented rule can ever recover."""
    key = "-----BEGIN RSA PRIVATE KEY-----\n" + "k" * 100 + "\n-----END RSA PRIVATE KEY-----"
    prefix, suffix = "z" * 500, "y" * 1900
    message = prefix + key + suffix
    cut = len(message) - 2000
    assert len(prefix) < cut < len(prefix) + len(key), f"fixture must straddle the cut, got {cut}"
    payload, _ = hook.classify_turn_stopped(message)
    assert "BEGIN RSA PRIVATE KEY" not in payload["message_tail"]
    assert "kkkkkkkkkk" not in payload["message_tail"]


def test_empty_and_whitespace_only_messages_yield_an_absent_tail_not_an_empty_one():
    # An empty string here would be refused by ingress (optStr rejects "") and the stop event
    # would never be recorded at all.
    for message in ("", "   \n\n  ", "\t"):
        payload, source = hook.classify_turn_stopped(message)
        assert source == "behavioral"
        assert payload == {}, message


def test_the_hook_never_calls_the_classifier_any_more():
    calls = []
    with patch("urllib.request.urlopen", _fake_urlopen("done", calls)):
        hook.classify_turn_stopped("no tag here at all")
    assert calls == []


def test_session_id_is_forwarded_bounded_and_omitted_when_absent():
    posts = []
    def post(url, payload, timeout=5):
        posts.append((url, payload)); return {"ok": True}
    def run_hook(body):
        with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
            hook.main(["claude-stop"], stdin=io.StringIO(json.dumps(body)),
                      run=_run_identity(session="jax-jax-os-lead"), post=post)
    base = {"cwd": "/home/rafa/repos/jax-os", "last_assistant_message": "done here"}
    run_hook({**base, "session_id": "sess-abc"})
    assert posts[-1][1]["harness_session"] == "sess-abc"
    posts.clear(); run_hook(base)
    assert "harness_session" not in posts[-1][1]
    posts.clear(); run_hook({**base, "session_id": "x" * 129})
    assert "harness_session" not in posts[-1][1] and posts[-1][1]["type"] == "turn-stopped"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all tests passed")


def test_build_prompt_redacts_before_slicing():
    """Diff review F1: _build_prompt sliced first, so a secret straddling the 2000-char cut
    lost the head that makes it matchable and kept the tail that makes it a secret."""
    import workflow_poll as wp
    secret = "sk-" + "A" * 40
    prefix, suffix = "x" * 520, "y" * 1980
    message = prefix + " " + secret + suffix
    cut = len(message) - 2000
    assert len(prefix) < cut < len(prefix) + len(secret), f"fixture must straddle the cut, got {cut}"
    prompt = wp._build_prompt(message)
    assert "sk-" not in prompt
    assert "A" * 20 not in prompt


def test_claude_callback_oneshot_exits_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")
    _write_pointer(tmp_path, _SESSION_A, _RUN_A)
    with patch.dict(os.environ, {"JAXFLOW_ONESHOT": "1"}, clear=True):
        code = hook.main(["claude-callback"], stdin=_stdin(_SESSION_A))
    assert code == 0
    # untouched -- the pointer is still pending, not claimed
    assert (hook.CALLBACKS_ROOT / _SESSION_A / f"{_RUN_A}.json").exists()


def test_claude_callback_noncanonical_session_id_exits_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")
    for bad_session in ("not-a-uuid", "", _SESSION_A + "\n", _SESSION_A.upper()):
        _write_pointer(tmp_path, _SESSION_A, _RUN_A)
        with patch.dict(os.environ, {}, clear=True):
            code = hook.main(["claude-callback"], stdin=_stdin(bad_session))
        assert code == 0
        assert (hook.CALLBACKS_ROOT / _SESSION_A / f"{_RUN_A}.json").exists()


def test_claude_callback_no_dir_exits_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")
    with patch.dict(os.environ, {}, clear=True):
        code = hook.main(["claude-callback"], stdin=_stdin(_SESSION_A))
    assert code == 0


def test_claude_callback_no_pending_exits_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")
    session_dir = hook.CALLBACKS_ROOT / _SESSION_A
    session_dir.mkdir(parents=True)
    (session_dir / f"{_RUN_A}.claimed").write_text("orphan", encoding="utf-8")
    (session_dir / f"{_RUN_A}.line").write_text("orphan", encoding="utf-8")
    with patch.dict(os.environ, {}, clear=True):
        code = hook.main(["claude-callback"], stdin=_stdin(_SESSION_A))
    assert code == 0
    # orphans are ignored, never touched
    assert (session_dir / f"{_RUN_A}.claimed").exists()
    assert (session_dir / f"{_RUN_A}.line").exists()


def test_claude_callback_line_present_at_claim_delivers_immediately(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")
    session_dir, _ = _write_pointer(tmp_path, _SESSION_A, _RUN_A, kind="diff")
    (session_dir / f"{_RUN_A}.line").write_text(
        "[JAXFLOW] diff aaaaaaaaaaaa finished — approve — done — /r.md\n", encoding="utf-8",
    )
    with patch.object(hook.time, "sleep", side_effect=AssertionError("must not sleep")):
        with patch.dict(os.environ, {}, clear=True):
            code = hook.main(["claude-callback"], stdin=_stdin(_SESSION_A))
    assert code == 2
    assert capsys.readouterr().err.strip() == (
        "[JAXFLOW] diff aaaaaaaaaaaa finished — approve — done — /r.md"
    )
    assert not (session_dir / f"{_RUN_A}.claimed").exists()
    assert not (session_dir / f"{_RUN_A}.line").exists()
    assert not (session_dir / f"{_RUN_A}.json").exists()


def test_claude_callback_valid_pointer_claims_and_delivers(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")
    monkeypatch.setattr(hook, "POLL_INTERVAL_S", 0.01)
    session_dir, _ = _write_pointer(tmp_path, _SESSION_A, _RUN_A, kind="spec")

    calls = {"n": 0}
    real_sleep = hook.time.sleep

    def fake_sleep(seconds):
        calls["n"] += 1
        if calls["n"] == 2:
            (session_dir / f"{_RUN_A}.line").write_text(
                "[JAXFLOW] spec aaaaaaaaaaaa finished — approve — done — /r.md\n",
                encoding="utf-8",
            )
        real_sleep(0)  # yield without a real delay

    with patch.object(hook.time, "sleep", side_effect=fake_sleep):
        with patch.dict(os.environ, {}, clear=True):
            code = hook.main(["claude-callback"], stdin=_stdin(_SESSION_A))
    assert code == 2
    assert capsys.readouterr().err.strip() == (
        "[JAXFLOW] spec aaaaaaaaaaaa finished — approve — done — /r.md"
    )
    assert calls["n"] >= 2
    assert not (session_dir / f"{_RUN_A}.claimed").exists()
    assert not (session_dir / f"{_RUN_A}.line").exists()


def test_claude_callback_malformed_pointer_drops_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")
    session_dir = hook.CALLBACKS_ROOT / _SESSION_A
    session_dir.mkdir(parents=True)
    (session_dir / f"{_RUN_A}.json").write_text("{not json", encoding="utf-8")
    with patch.dict(os.environ, {}, clear=True):
        code = hook.main(["claude-callback"], stdin=_stdin(_SESSION_A))
    assert code == 0
    assert not (session_dir / f"{_RUN_A}.json").exists()
    assert not (session_dir / f"{_RUN_A}.claimed").exists()


@pytest.mark.parametrize("bad_body", [
    '{"run_id": "../x", "kind": "build"}',
    '{"run_id": "AAAAAAAAAAAA", "kind": "build"}',
    '{"run_id": "aaaaaaaaaaaa", "kind": "not-a-kind"}',
    '{"run_id": "bbbbbbbbbbbb", "kind": "build"}',  # stem/run_id mismatch
])
def test_claude_callback_structural_invalid_drops_claim(tmp_path, monkeypatch, bad_body):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")
    session_dir = hook.CALLBACKS_ROOT / _SESSION_A
    session_dir.mkdir(parents=True)
    (session_dir / f"{_RUN_A}.json").write_text(bad_body, encoding="utf-8")
    with patch.dict(os.environ, {}, clear=True):
        code = hook.main(["claude-callback"], stdin=_stdin(_SESSION_A))
    assert code == 0
    assert not (session_dir / f"{_RUN_A}.json").exists()
    assert not (session_dir / f"{_RUN_A}.claimed").exists()


@pytest.mark.parametrize("bad_body", [
    '{"run_id": "aaaaaaaaaaaa", "kind": "build", "extra": 1}',
    '{"run_id": "aaaaaaaaaaaa"}',
    "[1, 2, 3]",
    '{"run_id": 111111111111, "kind": "build"}',
])
def test_claude_callback_pointer_wrong_shape_drops_claim(tmp_path, monkeypatch, bad_body):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")
    session_dir = hook.CALLBACKS_ROOT / _SESSION_A
    session_dir.mkdir(parents=True)
    (session_dir / f"{_RUN_A}.json").write_text(bad_body, encoding="utf-8")
    with patch.dict(os.environ, {}, clear=True):
        code = hook.main(["claude-callback"], stdin=_stdin(_SESSION_A))
    assert code == 0
    assert not (session_dir / f"{_RUN_A}.json").exists()
    assert not (session_dir / f"{_RUN_A}.claimed").exists()


def test_claude_callback_watch_finds_completion_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")
    monkeypatch.setattr(hook, "POLL_INTERVAL_S", 0.01)
    session_dir, _ = _write_pointer(tmp_path, _SESSION_A, _RUN_A, kind="plan")

    def fake_sleep(seconds):
        (session_dir / f"{_RUN_A}.line").write_text(
            "[JAXFLOW] plan aaaaaaaaaaaa finished — reject — nope — /r.md\n", encoding="utf-8",
        )

    with patch.object(hook.time, "sleep", side_effect=fake_sleep):
        with patch.dict(os.environ, {}, clear=True):
            code = hook.main(["claude-callback"], stdin=_stdin(_SESSION_A))
    assert code == 2
    assert "reject" in capsys.readouterr().err


def test_claude_callback_watch_expires_before_deadline(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")
    monkeypatch.setattr(hook, "POLL_INTERVAL_S", 0.001)
    monkeypatch.setattr(hook, "WATCH_DEADLINE_S", 0.01)
    session_dir, _ = _write_pointer(tmp_path, _SESSION_A, _RUN_A, kind="build")
    with patch.dict(os.environ, {}, clear=True):
        code = hook.main(["claude-callback"], stdin=_stdin(_SESSION_A))
    assert code == 2
    assert capsys.readouterr().err.strip() == (
        f"[JAXFLOW] build {_RUN_A} callback watcher expired — run may still be going — "
        f"use jaxflow status/result {_RUN_A}"
    )
    assert not (session_dir / f"{_RUN_A}.claimed").exists()


def test_claude_callback_concurrent_claim_only_one_wins(tmp_path, monkeypatch, capsys):
    """Cold review round 1 F4: two REAL `hook.main(["claude-callback"], ...)`
    invocations race the SAME pending pointer from separate threads, released together
    by a Barrier so neither can start before the other -- a non-atomic check-then-rename
    implementation could let both threads win. The `.line` file is written up front so
    the winner delivers on its very first check (no poll needed), keeping the whole test
    sub-second and non-flaky."""
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")
    monkeypatch.setattr(hook, "POLL_INTERVAL_S", 0.001)
    monkeypatch.setattr(hook, "WATCH_DEADLINE_S", 2)
    session_dir, _ = _write_pointer(tmp_path, _SESSION_A, _RUN_A, kind="build")
    (session_dir / f"{_RUN_A}.line").write_text(
        "[JAXFLOW] build aaaaaaaaaaaa finished — success — done — /r.md\n", encoding="utf-8",
    )
    barrier = threading.Barrier(2)
    results = [None, None]

    def worker(index):
        barrier.wait(timeout=5)
        results[index] = hook.main(["claude-callback"], stdin=_stdin(_SESSION_A))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    with patch.dict(os.environ, {}, clear=True):
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
    assert not any(t.is_alive() for t in threads)

    assert sorted(results) == [0, 2]
    err = capsys.readouterr().err.strip()
    assert err == "[JAXFLOW] build aaaaaaaaaaaa finished — success — done — /r.md"
    assert not (session_dir / f"{_RUN_A}.json").exists()
    assert not (session_dir / f"{_RUN_A}.claimed").exists()
    assert not (session_dir / f"{_RUN_A}.line").exists()


def test_claude_callback_cross_session_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")
    _write_pointer(tmp_path, _SESSION_B, _RUN_B)  # a DIFFERENT session's pointer
    with patch.dict(os.environ, {}, clear=True):
        code = hook.main(["claude-callback"], stdin=_stdin(_SESSION_A))
    assert code == 0
    # session A has no directory at all -- session B's pointer is untouched
    assert (hook.CALLBACKS_ROOT / _SESSION_B / f"{_RUN_B}.json").exists()


def test_claude_callback_sigterm_leaves_claim_orphaned(tmp_path):
    """A real subprocess, HOME redirected to an isolated tmp dir (the only correct way
    to redirect a FRESH process's own Path.home() -- this is not the in-process
    monkeypatch/patch.object case the rest of this plan uses, where changing HOME after
    import would have no effect on an already-computed module constant; here the module
    is imported fresh, inside the child, so HOME at spawn time is what it resolves
    CALLBACKS_ROOT against). Sends SIGTERM once the child has claimed the pointer (proven
    by the .claimed file's own appearance, polled for in this test process with a short
    real sleep loop -- POLL_INTERVAL_S stays at its real 5s default, since a SIGTERM
    interrupts a sleeping process immediately regardless of the interval)."""
    home = tmp_path / "home"
    home.mkdir()
    session_dir = home / ".jax-os" / "callbacks" / _SESSION_A
    session_dir.mkdir(parents=True)
    pointer = session_dir / f"{_RUN_A}.json"
    pointer.write_text(json.dumps({"run_id": _RUN_A, "kind": "build"}), encoding="utf-8")

    hook_path = Path(__file__).resolve().parent / "jaxflow_hook.py"
    env = dict(os.environ)
    env["HOME"] = str(home)
    # JAXOS_HOME IS the state dir itself (jaxflow_env.jaxos_home(): JAXOS_HOME
    # overrides the default ~/.jax-os), while HOME only changes where Path.home()
    # points. The fixture below builds <home>/.jax-os/callbacks, so JAXOS_HOME must
    # point at <home>/.jax-os to keep the child's CALLBACKS_ROOT on that fixture.
    env["JAXOS_HOME"] = str(home / ".jax-os")
    env.pop("JAXFLOW_ONESHOT", None)
    proc = subprocess.Popen(
        [sys.executable, str(hook_path), "claude-callback"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )
    proc.stdin.write(json.dumps({"session_id": _SESSION_A}))
    proc.stdin.close()

    claimed = session_dir / f"{_RUN_A}.claimed"
    deadline = time.monotonic() + 5
    while not claimed.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert claimed.exists(), "subprocess never claimed the pointer within 5s"

    proc.terminate()  # SIGTERM
    out, err = proc.communicate(timeout=5)
    assert proc.returncode is not None
    assert out == ""
    assert claimed.exists()  # left in place, no replay (D7)
    assert not pointer.exists()  # it was renamed at claim time, not recreated


def _tool_post():
    posts = []

    def post(url, payload, timeout=5):
        posts.append((url, payload))
        return {"ok": True}
    return posts, post


def _events(posts):
    return [p for url, p in posts if url == hook.EVENTS_URL]


def test_toolused_modes_post_the_tool_name_only_never_input_response_or_path(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "THROTTLE_ROOT", tmp_path / "throttle")
    for mode, env, extra in (
        ("claude-toolused", {"TMUX": "1", "TMUX_PANE": "%9"}, {}),
        ("codex-toolused", {}, {"session_id": "0191f0aa-6666-7000-8000-000000000006"}),
    ):
        posts, post = _tool_post()
        raw = json.dumps({"cwd": "/home/rafa/repos/jax-os", "tool_name": "Bash",
                          "tool_input": {"command": "cat /home/rafa/.env"}, "tool_response": "SECRET=1", **extra})
        with patch.dict(os.environ, env, clear=True):
            assert hook.main([mode], stdin=io.StringIO(raw), run=_run_identity(), post=post) == 0
        event = _events(posts)[-1]
        assert event["type"] == "tool-used" and event["emitter"] == mode and event["source"] == "deterministic"
        assert event["payload"] == {"tool": "Bash"}
        assert "/home/rafa/.env" not in json.dumps(event) and "SECRET" not in json.dumps(event)


def test_toolused_tool_name_is_bounded_to_64_chars_and_a_missing_name_posts_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "THROTTLE_ROOT", tmp_path / "throttle")
    posts, post = _tool_post()
    with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
        hook.main(["claude-toolused"], stdin=io.StringIO(json.dumps({"cwd": "/x/y", "tool_name": "T" * 70})), run=_run_identity(), post=post)
    assert _events(posts)[-1]["payload"] == {"tool": "T" * 64}
    posts.clear()
    with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
        hook.main(["claude-toolused"], stdin=io.StringIO(json.dumps({"cwd": "/x/y"})), run=_run_identity(), post=post)
    assert posts == []


def test_subagent_modes_post_agent_identity_only_and_are_never_throttled(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "THROTTLE_ROOT", tmp_path / "throttle")
    for start_mode, stop_mode, env, extra in (
        ("claude-subagentstart", "claude-subagentstop", {"TMUX": "1", "TMUX_PANE": "%9"}, {}),
        ("codex-subagentstart", "codex-subagentstop", {}, {"session_id": "0191f0aa-7777-7000-8000-000000000007"}),
    ):
        posts, post = _tool_post()
        raw = json.dumps({"cwd": "/home/rafa/repos/jax-os", "agent_id": "agent-1", "agent_type": "Explore",
                          "prompt": "find the secret", **extra})
        with patch.dict(os.environ, env, clear=True):
            for _ in range(3):  # three starts inside one 10s bucket — none dropped (round 2 F2)
                hook.main([start_mode], stdin=io.StringIO(raw), run=_run_identity(), post=post)
            hook.main([stop_mode], stdin=io.StringIO(raw), run=_run_identity(), post=post)
        events = _events(posts)
        assert [e["type"] for e in events] == ["subagent-started"] * 3 + ["subagent-stopped"]
        assert events[0]["payload"] == {"agent_id": "agent-1", "agent_type": "Explore"}
        assert events[-1]["payload"] == {"agent_id": "agent-1"}
        assert events[0]["emitter"] == start_mode and events[-1]["emitter"] == stop_mode
        assert "find the secret" not in json.dumps(events)
    assert not (tmp_path / "throttle").exists()  # subagent modes never claim a bucket


def test_toolused_throttle_posts_once_per_bucket_per_identity_and_again_in_the_next_bucket(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "THROTTLE_ROOT", tmp_path / "throttle")
    posts, post = _tool_post()
    raw = json.dumps({"cwd": "/home/rafa/repos/jax-os", "tool_name": "Bash"})
    with patch.object(hook.time, "time", return_value=1_000_000.0):
        with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
            hook.main(["claude-toolused"], stdin=io.StringIO(raw), run=_run_identity(), post=post)
            hook.main(["claude-toolused"], stdin=io.StringIO(raw), run=_run_identity(), post=post)  # same bucket: skipped
        with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%10"}, clear=True):  # another identity, same bucket
            hook.main(["claude-toolused"], stdin=io.StringIO(raw), run=_run_identity(), post=post)
    assert [e["payload"] for e in _events(posts)] == [{"tool": "Bash"}, {"tool": "Bash"}]
    assert sorted(f.name for f in (tmp_path / "throttle").iterdir()) == ["_10-100000", "_9-100000"]
    with patch.object(hook.time, "time", return_value=1_000_010.0):  # next bucket
        with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
            hook.main(["claude-toolused"], stdin=io.StringIO(raw), run=_run_identity(), post=post)
    assert len(_events(posts)) == 3


def test_toolused_throttle_claim_is_one_exclusive_create_and_a_lost_race_skips_silently(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hook, "THROTTLE_ROOT", tmp_path / "throttle")
    posts, post = _tool_post()
    raw = json.dumps({"cwd": "/home/rafa/repos/jax-os", "tool_name": "Bash"})
    calls = []

    def fake_open(path, flags, mode=0o777, *args, **kwargs):
        calls.append((path, flags, mode))
        raise FileExistsError(path)
    with patch.object(hook.os, "open", fake_open):
        with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
            assert hook.main(["claude-toolused"], stdin=io.StringIO(raw), run=_run_identity(), post=post) == 0
    assert posts == []  # the loser posts nothing — not even a session registration
    assert capsys.readouterr().err == ""
    assert len(calls) == 1
    path, flags, mode = calls[0]
    assert flags & os.O_CREAT and flags & os.O_EXCL and mode == 0o600  # the claim IS the exclusive create
    assert str(path).startswith(str(tmp_path / "throttle"))


def test_toolused_two_concurrent_claims_on_the_same_bucket_post_exactly_once(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "THROTTLE_ROOT", tmp_path / "throttle")
    posts = []
    lock = threading.Lock()

    def post(url, payload, timeout=5):
        with lock:
            posts.append((url, payload))
        return {"ok": True}
    raw = json.dumps({"cwd": "/home/rafa/repos/jax-os", "tool_name": "Bash"})
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait(timeout=5)
        hook.main(["claude-toolused"], stdin=io.StringIO(raw), run=_run_identity(), post=post)
    with patch.object(hook.time, "time", return_value=2_000_000.0):
        with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)
    assert not any(t.is_alive() for t in threads)
    assert len(_events(posts)) == 1


def test_toolused_sweeps_markers_older_than_six_buckets_and_never_blocks_a_fresh_claim(tmp_path, monkeypatch):
    root = tmp_path / "throttle"
    monkeypatch.setattr(hook, "THROTTLE_ROOT", root)
    root.mkdir()
    (root / "_9-99990").write_text("")   # 10 buckets old — swept
    (root / "_9-99995").write_text("")   # 5 buckets old — kept
    (root / "junk").write_text("")       # no bucket suffix — ignored
    posts, post = _tool_post()
    raw = json.dumps({"cwd": "/home/rafa/repos/jax-os", "tool_name": "Bash"})
    with patch.object(hook.time, "time", return_value=1_000_000.0):
        with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
            hook.main(["claude-toolused"], stdin=io.StringIO(raw), run=_run_identity(), post=post)
    assert len(_events(posts)) == 1
    assert sorted(f.name for f in root.iterdir()) == ["_9-100000", "_9-99995", "junk"]


def test_new_modes_keep_the_existing_identity_gates_and_the_oneshot_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "THROTTLE_ROOT", tmp_path / "throttle")
    posts, post = _tool_post()
    raw = json.dumps({"cwd": "/home/rafa/repos/jax-os", "tool_name": "Bash", "agent_id": "a", "agent_type": "t"})
    for mode in ("claude-toolused", "claude-subagentstart", "claude-subagentstop"):
        with patch.dict(os.environ, {"TMUX": "1"}, clear=True):  # TMUX set, TMUX_PANE absent
            assert hook.main([mode], stdin=io.StringIO(raw), run=_run_identity(), post=post) == 0
    for mode in ("codex-toolused", "codex-subagentstart", "codex-subagentstop"):
        with patch.dict(os.environ, {}, clear=True):  # no session_id at all
            assert hook.main([mode], stdin=io.StringIO(raw), post=post) == 0
    with patch.dict(os.environ, {"JAXFLOW_ONESHOT": "1", "TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
        assert hook.main(["claude-toolused"], stdin=io.StringIO(raw), run=_run_identity(), post=post) == 0
    assert posts == []


def test_should_arm_answer_watcher_matrix():
    assert hook._should_arm_answer_watcher({"message_tail": "posso mergear?"}) is True
    assert hook._should_arm_answer_watcher({"capsule_status": "needs_input", "capsule_rule": "tag"}) is True
    assert hook._should_arm_answer_watcher({"capsule_status": "blocked", "capsule_rule": "tag"}) is True
    assert hook._should_arm_answer_watcher({"capsule_status": "waiting", "capsule_minutes": 30, "capsule_rule": "tag"}) is False
    assert hook._should_arm_answer_watcher({"capsule_status": "done", "capsule_rule": "tag"}) is False
    assert hook._should_arm_answer_watcher({}) is False


def test_claude_stop_arms_and_delivers_an_answer_via_exit_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")
    monkeypatch.setattr(hook, "POLL_INTERVAL_S", 0.01)
    session_id = _SESSION_A

    def post(url, payload, timeout=5):
        if url == hook.EVENTS_URL:
            return {"ok": True, "data": {"id": 42, "ts": "2026-09-24T00:00:00.000Z", "delivery": "local"}}
        return {"ok": True}

    session_dir = tmp_path / ".jax-os" / "callbacks" / session_id
    answer_path = session_dir / "answer-42.answer"

    def fake_sleep(seconds):
        session_dir.mkdir(parents=True, exist_ok=True)
        answer_path.write_text("[Jax OS · Rafa] pode\n", encoding="utf-8")

    payload = json.dumps({"cwd": "/home/rafa/repos/jax-os", "session_id": session_id,
                           "last_assistant_message": "posso mergear feat/x em main?"})
    with patch.object(hook.time, "sleep", side_effect=fake_sleep):
        with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
            code = hook.main(["claude-stop"], stdin=io.StringIO(payload), run=_run_identity(), post=post)
    assert code == 2
    assert capsys.readouterr().err.strip() == "[Jax OS · Rafa] pode"
    assert not (session_dir / "answer-42.armed").exists()
    assert not answer_path.exists()


def test_claude_stop_does_not_arm_on_a_waiting_or_done_self_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")

    def post(url, payload, timeout=5):
        return {"ok": True, "data": {"id": 7, "ts": "t", "delivery": "local"}}

    payload = json.dumps({"cwd": "/home/rafa/repos/jax-os", "session_id": _SESSION_A,
                           "last_assistant_message": "[JAXFLOW: done] shipped"})
    with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
        code = hook.main(["claude-stop"], stdin=io.StringIO(payload), run=_run_identity(), post=post)
    assert code == 0
    assert not (hook.CALLBACKS_ROOT / _SESSION_A).exists()


def test_claude_stop_does_not_arm_on_an_empty_redacted_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")

    def post(url, payload, timeout=5):
        return {"ok": True, "data": {"id": 9, "ts": "t", "delivery": "local"}}

    payload = json.dumps({"cwd": "/home/rafa/repos/jax-os", "session_id": _SESSION_A, "last_assistant_message": "   \n  "})
    with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
        code = hook.main(["claude-stop"], stdin=io.StringIO(payload), run=_run_identity(), post=post)
    assert code == 0
    assert not hook.CALLBACKS_ROOT.exists()


def test_claude_stop_does_not_arm_when_post_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")

    def post(url, payload, timeout=5):
        raise RuntimeError("hook post failed")

    payload = json.dumps({"cwd": "/home/rafa/repos/jax-os", "session_id": _SESSION_A, "last_assistant_message": "posso mergear?"})
    with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
        code = hook.main(["claude-stop"], stdin=io.StringIO(payload), run=_run_identity(), post=post)
    assert code == 0
    assert not hook.CALLBACKS_ROOT.exists()


def test_claude_stop_does_not_arm_on_a_non_canonical_harness_session(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "CALLBACKS_ROOT", tmp_path / ".jax-os" / "callbacks")

    def post(url, payload, timeout=5):
        return {"ok": True, "data": {"id": 11, "ts": "t", "delivery": "local"}}

    payload = json.dumps({"cwd": "/home/rafa/repos/jax-os", "session_id": "not-a-uuid", "last_assistant_message": "posso mergear?"})
    with patch.dict(os.environ, {"TMUX": "1", "TMUX_PANE": "%9"}, clear=True):
        code = hook.main(["claude-stop"], stdin=io.StringIO(payload), run=_run_identity(), post=post)
    assert code == 0
    assert not hook.CALLBACKS_ROOT.exists()


def test_answer_watcher_expires_silently(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hook, "POLL_INTERVAL_S", 0.001)
    monkeypatch.setattr(hook, "WATCH_DEADLINE_S", 0.01)
    session_dir = tmp_path / ".jax-os" / "callbacks" / _SESSION_A
    session_dir.mkdir(parents=True)
    armed = session_dir / "answer-5.armed"
    armed.write_text("", encoding="utf-8")
    code = hook._watch_for_answer(session_dir, _SESSION_A, 5)
    assert code == 0
    assert capsys.readouterr().err == ""
    assert not armed.exists()


def test_answer_watcher_supersession_exits_silently_and_keeps_the_newer_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "POLL_INTERVAL_S", 0.001)
    session_dir = tmp_path / ".jax-os" / "callbacks" / _SESSION_A
    session_dir.mkdir(parents=True)
    old_armed = session_dir / "answer-5.armed"
    old_armed.write_text("", encoding="utf-8")

    def fake_sleep(seconds):
        (session_dir / "answer-6.armed").write_text("", encoding="utf-8")

    with patch.object(hook.time, "sleep", side_effect=fake_sleep):
        code = hook._watch_for_answer(session_dir, _SESSION_A, 5)
    assert code == 0
    assert not old_armed.exists()
    assert (session_dir / "answer-6.armed").exists()


def test_answer_watcher_supersession_with_own_answer_present_discards_it_unread(tmp_path, monkeypatch, capsys):
    # F2: a newer answer-*.armed AND this watcher's own .answer both already on disk when the
    # loop checks -- supersession is checked first, so the own answer is discarded unread: no
    # delivery, both of this watcher's own files (.armed and .answer) are removed, the newer
    # marker survives untouched.
    session_dir = tmp_path / ".jax-os" / "callbacks" / _SESSION_A
    session_dir.mkdir(parents=True)
    own_armed = session_dir / "answer-5.armed"
    own_armed.write_text("", encoding="utf-8")
    own_answer = session_dir / "answer-5.answer"
    own_answer.write_text("[Jax OS · Rafa] pode\n", encoding="utf-8")
    (session_dir / "answer-6.armed").write_text("", encoding="utf-8")

    code = hook._watch_for_answer(session_dir, _SESSION_A, 5)
    assert code == 0
    assert capsys.readouterr().err == ""
    assert not own_armed.exists()
    assert not own_answer.exists()
    assert (session_dir / "answer-6.armed").exists()


def test_answer_watcher_ignores_an_older_marker(tmp_path, monkeypatch, capsys):
    # Plan review round 2: watcher 6 must NOT treat marker 5 as a supersession -- it keeps
    # waiting and delivers its own answer; marker 5 is left for watcher 5 to handle.
    monkeypatch.setattr(hook, "POLL_INTERVAL_S", 0.001)
    session_dir = tmp_path / ".jax-os" / "callbacks" / _SESSION_A
    session_dir.mkdir(parents=True)
    (session_dir / "answer-5.armed").write_text("", encoding="utf-8")
    own_armed = session_dir / "answer-6.armed"
    own_armed.write_text("", encoding="utf-8")

    def fake_sleep(seconds):
        (session_dir / "answer-6.answer").write_text("[Jax OS · Rafa] pode\n", encoding="utf-8")

    with patch.object(hook.time, "sleep", side_effect=fake_sleep):
        code = hook._watch_for_answer(session_dir, _SESSION_A, 6)
    assert code == 2
    assert capsys.readouterr().err.strip() == "[Jax OS · Rafa] pode"
    assert not own_armed.exists()
    assert (session_dir / "answer-5.armed").exists()


def test_text_stop_kicks_the_classifier_and_a_tagged_stop_does_not(monkeypatch):
    kicks = []
    monkeypatch.setattr(hook, "_kick_classifier", lambda: kicks.append(1))
    post = lambda url, payload, timeout=5: {"ok": True}
    env = {"TMUX": "1", "TMUX_PANE": "%9"}
    with patch.dict(os.environ, env, clear=True):
        text = json.dumps({"cwd": "/home/rafa/repos/jax-os", "last_assistant_message": "Posso fazer o merge?"})
        hook.main(["claude-stop"], stdin=io.StringIO(text), run=_run_identity(ok=False), post=post)
        assert kicks == [1]
        hook.main(["claude-stop"], stdin=io.StringIO("{}"), run=_run_identity(ok=False), post=post)
        assert kicks == [1]


def test_kick_classifier_runs_the_poll_unless_disabled(monkeypatch):
    calls = []
    monkeypatch.delenv("JAXFLOW_NO_CLASSIFY_KICK", raising=False)
    hook._kick_classifier(run=lambda argv, **kw: calls.append(argv))
    assert calls == [hook.CLASSIFY_KICK]
    monkeypatch.setenv("JAXFLOW_NO_CLASSIFY_KICK", "1")
    hook._kick_classifier(run=lambda argv, **kw: calls.append(argv))
    assert len(calls) == 1


def test_deliver_line_passes_the_fallback_suffix_through_intact(tmp_path, capsys):
    line, claimed = tmp_path / "r.line", tmp_path / "r.claimed"
    text = "[JAXFLOW] spec aaaabbbbcccc finished — approve — /report.md (fallback: codex off)"
    line.write_text(text + "\n", encoding="utf-8")
    claimed.write_text("", encoding="utf-8")
    assert hook._deliver_line(line, claimed) == 2
    assert capsys.readouterr().err == text + "\n"
    assert not line.exists() and not claimed.exists()


# ---- merge question contract (spec 2026-10-09) ---------------------------------------------

_SHA = "0123456789abcdef0123456789abcdef01234567"
_MQ_PAYLOAD = {
    "capsule_status": "needs_input", "capsule_rule": "merge-question", "capsule_attempts": 0,
    "merge_ask": 1, "merge_branch": "feat/x", "merge_target": "main", "merge_head_sha": _SHA,
}
_MQ = "Posso mergear `feat/x` em `main`?"


def _git_run(sha=_SHA, calls=None):
    def run(argv, cwd=None):
        if calls is not None:
            calls.append((argv, cwd))
        if argv[:2] == ["git", "rev-parse"]:
            if sha is None:
                return SimpleNamespace(returncode=128, stdout="", stderr="bad revision")
            return SimpleNamespace(returncode=0, stdout=sha + "\n", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="")
    return run


def _stop_event(payload, mode="claude-stop", run=None):
    pane, session = ("%1", "jax-p1-lead") if mode == "claude-stop" else (None, None)
    return hook.event_from_input(mode, "p1", pane, session, payload, run=run or _git_run())


def _user(content):
    return {"type": "user", "message": {"role": "user", "content": content}}


def _tool_result():
    return {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}}


def _assistant(*blocks):
    return {"type": "assistant", "message": {"role": "assistant", "content": list(blocks)}}


def _text(t):
    return {"type": "text", "text": t}


def _write_transcript(tmp_path, entries):
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("text,expected", [
    ("Posso mergear `docs/merge-worktree-note` em `main`?", ("docs/merge-worktree-note", "main")),
    ("May I merge `feat/x` into `main`?", ("feat/x", "main")),
    ("Posso mergear `docs/merge-worktree-note` (`320538a`) em `main`?", ("docs/merge-worktree-note", "main")),
    ("Posso já perguntar: posso mergear `a` em `b` assim que o `ci` passar?", ("a", "b")),
])
def test_find_merge_question_positive_rows(text, expected):  # A1
    assert hook.find_merge_question(text) == expected


@pytest.mark.parametrize("text", [
    "pode abrir o PR?",
    "Posso mergear docs/x em main?",
    "Mergeado: `main` = `19d4d7a` via PR #6",
    "Posso mergear feat/x em main?",                        # no backticks
    "Mergear `feat/x` em `main`?",                          # "mergear" without "posso"
    "Posso mergear `feat/x`, a `main` fica como está?",     # branch and target in prose without em|into
    "pode abrir o PR de `x` em `main`?",
])
def test_find_merge_question_negative_rows(text):  # A1
    assert hook.find_merge_question(text) is None


def test_find_merge_question_tolerated_mixed_forms():  # F3: documented grammar, not a bug
    assert hook.find_merge_question("May I merge `x` em `main`?") == ("x", "main")
    assert hook.find_merge_question("Posso mergear `x` into `main` depois do CI?") == ("x", "main")


def test_find_merge_question_identifier_length_boundary_matches_ingress():  # F5
    ok, too_long = "a" * 200, "a" * 201
    assert hook.find_merge_question(f"Posso mergear `{ok}` em `main`?") == (ok, "main")
    assert hook.find_merge_question(f"Posso mergear `feat/x` em `{ok}`?") == ("feat/x", ok)
    assert hook.find_merge_question(f"Posso mergear `{too_long}` em `main`?") is None
    assert hook.find_merge_question(f"Posso mergear `feat/x` em `{too_long}`?") is None
    # the ordinary stop payload is kept (not a merge-question) when the identifier is too long
    event = _stop_event({"cwd": "/r", "last_assistant_message": f"Posso mergear `{too_long}` em `main`?"})
    assert event["payload"] == {"message_tail": f"Posso mergear `{too_long}` em `main`?", "merge_ask": 0}


def test_find_merge_question_last_match_wins():  # A1
    text = "Posso mergear `a` em `b`?\nO branch andou. May I merge `c` into `d`?"
    assert hook.find_merge_question(text) == ("c", "d")


def test_claude_stop_scans_earlier_blocks_of_the_turn_event_26239_shape(tmp_path):  # A2a
    transcript = _write_transcript(tmp_path, [
        _user("old prompt"),
        _assistant(_text("Posso mergear `old/branch` em `main`?")),  # BEFORE the last prompt: must not count
        _user("go on"),
        _assistant(_text(_MQ)),
        _assistant({"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}),
        _tool_result(),
        _assistant(_text("Esperando o CI do PR #7 e a sua resposta sobre o merge de feat/x em main.")),
    ])
    event = _stop_event({"cwd": "/r", "transcript_path": transcript,
                         "last_assistant_message": "Esperando o CI do PR #7 e a sua resposta."})
    assert event["type"] == "turn-stopped" and event["source"] == "deterministic"
    assert event["payload"] == _MQ_PAYLOAD


def test_codex_stop_scans_only_last_assistant_message(tmp_path):  # A2b
    inside = "Resumo.\n\nPosso mergear `feat/x` em `main`?\n\nAguardo o CI."
    assert _stop_event({"cwd": "/r", "last_assistant_message": inside}, mode="codex-stop")["payload"] == _MQ_PAYLOAD
    # A question only in an earlier Codex message is NOT seen (documented limitation); a
    # transcript_path on a Codex payload is never read.
    transcript = _write_transcript(tmp_path, [_user("p"), _assistant(_text(_MQ))])
    event = _stop_event({"cwd": "/r", "transcript_path": transcript, "last_assistant_message": "Aguardo."},
                        mode="codex-stop")
    assert event["payload"] == {"message_tail": "Aguardo.", "merge_ask": 0}


def test_no_match_gets_merge_ask_zero_and_no_merge_keys_and_transcript_fallback(tmp_path):  # A3
    event = _stop_event({"cwd": "/r", "last_assistant_message": "só prosa"})
    assert event["payload"] == {"message_tail": "só prosa", "merge_ask": 0}
    tagged = _stop_event({"cwd": "/r", "last_assistant_message": "[JAXFLOW: done] shipped"})
    assert tagged["payload"] == {"capsule_status": "done", "capsule_rule": "tag", "excerpt": "shipped", "merge_ask": 0}
    assert _stop_event({"cwd": "/r", "last_assistant_message": ""})["payload"] == {}  # empty stays empty
    # Missing transcript -> falls back to last_assistant_message, which here HAS the question.
    missing = str(tmp_path / "does-not-exist.jsonl")
    fallback = _stop_event({"cwd": "/r", "transcript_path": missing, "last_assistant_message": _MQ})
    assert fallback["payload"] == _MQ_PAYLOAD
    # Unreadable content (not JSON) is also a fallback, not a crash.
    junk = tmp_path / "junk.jsonl"
    junk.write_text("not json\n", encoding="utf-8")
    assert _stop_event({"cwd": "/r", "transcript_path": str(junk), "last_assistant_message": _MQ})["payload"] == _MQ_PAYLOAD


def test_transcript_lagging_the_final_block_still_sees_the_question(tmp_path):  # A3 (transcript lag)
    # The Stop hook may fire before the final assistant block is flushed to the transcript:
    # the transcript is valid and non-empty but lacks the closing message that holds the question.
    transcript = _write_transcript(tmp_path, [_user("go"), _assistant(_text("Rodei os testes, tudo verde."))])
    event = _stop_event({"cwd": "/r", "transcript_path": transcript, "last_assistant_message": _MQ})
    assert event["payload"] == _MQ_PAYLOAD
    # And the duplicate (question present in BOTH) is harmless: still one merge-question payload.
    both = _write_transcript(tmp_path, [_user("go"), _assistant(_text(_MQ))])
    assert _stop_event({"cwd": "/r", "transcript_path": both, "last_assistant_message": _MQ})["payload"] == _MQ_PAYLOAD


def test_a_damaged_newer_user_line_never_resurrects_an_old_question(tmp_path):  # F4
    good = [_user("old prompt"), _assistant(_text("Posso mergear `old/branch` em `main`?"))]
    path = tmp_path / "damaged.jsonl"
    path.write_text(
        "\n".join(json.dumps(e) for e in good) + "\n"
        + '{"type":"user","message":{"role":"user","content":"new pro' + "\n"   # truncated newer prompt
        + json.dumps(_assistant(_text("Aguardo."))) + "\n", encoding="utf-8")
    assert hook.transcript_turn_text(str(path)) is None
    payload = {"cwd": "/r", "transcript_path": str(path), "last_assistant_message": "Aguardo."}
    event = _stop_event(payload)
    assert event["payload"] == {"message_tail": "Aguardo.", "merge_ask": 0}
    assert event["payload"] == _stop_event({"cwd": "/r", "last_assistant_message": "Aguardo."})["payload"]


def test_merge_head_sha_comes_from_git_via_run_and_is_null_on_failure():  # A4
    calls = []
    event = _stop_event({"cwd": "/work/tree", "last_assistant_message": "Posso mergear `feat/x` (`deadbee`) em `main`?"},
                        run=_git_run(calls=calls))
    assert event["payload"]["merge_head_sha"] == _SHA  # the sha in the prose is ignored
    assert calls == [(["git", "rev-parse", "--verify", "feat/x^{commit}"], "/work/tree")]
    failed = _stop_event({"cwd": "/r", "last_assistant_message": _MQ}, run=_git_run(sha=None))
    assert failed["payload"]["merge_head_sha"] is None and failed["payload"]["merge_ask"] == 1
    assert _stop_event({"cwd": "/r", "last_assistant_message": "Posso mergear `-x` em `main`?"})["payload"]["merge_head_sha"] is None


def test_should_arm_answer_watcher_true_on_merge_ask_one_for_a_tagged_payload():  # A5
    assert hook._should_arm_answer_watcher({"capsule_status": "done", "capsule_rule": "tag", "merge_ask": 1}) is True
    assert hook._should_arm_answer_watcher({"capsule_status": "done", "capsule_rule": "tag", "merge_ask": 0}) is False
    assert hook._should_arm_answer_watcher(_MQ_PAYLOAD) is True


@pytest.mark.parametrize("prefix", ["", "[JAXFLOW: done] shipped\n", "[JAXFLOW: waiting ~30m] x\n", "[JAXFLOW: needs_input] y\n"])
def test_a_matching_turn_is_the_single_deterministic_shape_and_the_tag_is_dropped(prefix):  # A13
    event = _stop_event({"cwd": "/r", "last_assistant_message": prefix + _MQ})
    assert event["source"] == "deterministic"
    assert event["payload"] == _MQ_PAYLOAD
    for absent in ("message_tail", "capsule_minutes", "excerpt"):
        assert absent not in event["payload"]


def test_hook_output_matches_the_committed_ingress_fixture():  # A14 (the TS tests read this same file)
    fixture = Path(__file__).resolve().parent.parent / "src/app/api/workflow/events/merge-question-hook-output.fixture.json"
    cases = {
        "plain": _MQ,
        "withDoneTag": "[JAXFLOW: done] shipped\n" + _MQ,
        "withWaitingTag": "[JAXFLOW: waiting ~30m] ci\n" + _MQ,
    }
    produced = {k: _stop_event({"cwd": "/r", "last_assistant_message": v}) for k, v in cases.items()}
    assert json.loads(fixture.read_text(encoding="utf-8")) == produced


def _collect_posts(fail_sessions=False):
    posts = []

    def post(url, payload, timeout=5):
        if fail_sessions and url == hook.SESSIONS_URL:
            raise RuntimeError("session registry unavailable")
        posts.append((url, payload))
        return {"ok": True}
    return posts, post


def _drive_claude(mode, payload, post):
    raw = json.dumps({"cwd": "/home/rafa/repos/jax-os", **payload})
    env = {"TMUX": "1", "TMUX_PANE": "%9", "JAXFLOW_NO_CLASSIFY_KICK": "1"}  # no real poller start on stop
    with patch.dict(os.environ, env, clear=True):
        return hook.main([mode], stdin=io.StringIO(raw), run=_run_identity(), post=post)


_ASK = {"tool_name": "AskUserQuestion", "tool_use_id": "toolu_1", "tool_input": {"questions": [{"q": "x"}]}}


@pytest.mark.parametrize("mode", ["claude-pretool", "claude-posttool"])
def test_hook_tool_events_skip_session_registration(mode):
    posts, post = _collect_posts()
    assert _drive_claude(mode, _ASK, post) == 0
    assert [url for url, _ in posts] == [hook.EVENTS_URL]  # event only, never SESSIONS_URL


@pytest.mark.parametrize("mode,payload", [
    ("claude-userprompt", {"prompt": "hi"}),
    ("claude-stop", {"last_assistant_message": "done"}),
])
def test_hook_turn_events_register_then_post(mode, payload):
    posts, post = _collect_posts()
    assert _drive_claude(mode, payload, post) == 0
    assert [url for url, _ in posts] == [hook.SESSIONS_URL, hook.EVENTS_URL]  # sessions first
    assert posts[0][1] == {"project": "jax-os", "session": "jax-jax-os-lead", "pane": "%9",
                           "role": "lead", "tmux_incarnation": "234790:1787586213"}


def test_hook_sessions_failure_still_posts_event():
    posts, post = _collect_posts(fail_sessions=True)
    assert _drive_claude("claude-userprompt", {"prompt": "hi"}, post) == 0
    assert [url for url, _ in posts] == [hook.EVENTS_URL]  # sessions POST raised; event still posted
    assert posts[0][1]["type"] == "turn-started"
