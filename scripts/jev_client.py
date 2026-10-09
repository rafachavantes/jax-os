#!/usr/bin/env python3
"""Tiny shared Jev (TypeSafe System One) HTTP client.

Same request shape as scripts/workflow_poll.py's `_call_jev`/`_classifier_api_key`,
extracted so scripts/jaxflow_run.py (MOA-495 2.1 build-failure reason) and
scripts/jaxflow.py (MOA-495 2.2 review-round tally) don't each grow their own HTTP
client. workflow_poll.py is left untouched -- it already has its own tested
credential/HTTP path and this module has no reason to disturb it.

Every caller must catch exceptions from `call()`/`api_key()` and fall back to its own
default behaviour: this module never decides what a failure means for a caller.
"""
from __future__ import annotations

import json
import os
import stat
import urllib.error
import urllib.request
from pathlib import Path

from jaxflow_env import jaxos_home

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
# MOA-498 D1: the one secret file jax-os reads for its own service secrets — was
# ~/.hermes/.env (Bitwarden-backed, Rafa-only). $JAXOS_HOME/.env is owner-filled, never
# written by jax-os itself.
ENV_PATH = jaxos_home() / ".env"
# MOA-498 F1: mirrors Next's own .env.local precedence (Next loads it into process.env before
# next.config.ts runs) on the Python side, which has no automatic loader of its own. Derived
# from this script's own location, not cwd, so it resolves correctly regardless of where the
# process is invoked from. Read-only, no _check_secure call — a dev convenience file, never
# Rafa's secret store.
LOCAL_ENV_PATH = Path(__file__).resolve().parent.parent / ".env.local"
JEV_ENV_KEY = "TYPESAFE_API"


def _check_secure(path):
    try:
        info = path.lstat()  # lstat, not stat -- a symlinked credentials file is rejected
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o600


def env_file_status():
    """Non-secret status for a settings UI to render — never a value, a pure function of the
    same lstat check api_key() already runs (MOA-498 D1/F4)."""
    try:
        info = ENV_PATH.lstat()
    except OSError:
        return {"state": "missing"}
    if not stat.S_ISREG(info.st_mode):
        return {"state": "not-regular"}
    if info.st_uid != os.getuid():
        return {"state": "wrong-owner"}
    if stat.S_IMODE(info.st_mode) != 0o600:
        return {"state": "bad-mode"}
    return {"state": "ok"}


def _read_kv(path, name):
    """Parses a simple KEY=value file, returns the stripped/unquoted value for `name`, or None
    if absent/unreadable. Shared by api_key's .env.local and $JAXOS_HOME/.env reads (previously
    duplicated inline once — now duplicated zero times)."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                return value.strip().strip('"').strip("'")
    except (OSError, UnicodeError):
        return None
    return None


def api_key(name=JEV_ENV_KEY):
    """Rafa's own file; jax-os only reads it. Precedence (MOA-498 D1/F1): process env, then
    LOCAL_ENV_PATH (<repo>/.env.local, read-only, no secure-file check), then the secure
    $JAXOS_HOME/.env file. Any failure returns "" and the caller degrades to its own fallback
    without a network call."""
    value = os.environ.get(name)
    if value:
        return value
    if LOCAL_ENV_PATH.exists():
        value = _read_kv(LOCAL_ENV_PATH, name)
        if value:
            return value
    if not _check_secure(ENV_PATH):
        return ""
    return _read_kv(ENV_PATH, name) or ""


def call(questions, state, *, api_key_value, timeout):
    """POST one systemone request and return its `answers` dict. Raises on a missing
    key or any transport/shape problem -- callers must catch and fall back."""
    if not api_key_value:
        raise ValueError("jev key unavailable")
    body = json.dumps({"model": JEV_MODEL, "state": state, "questions": questions}).encode("utf-8")
    request = urllib.request.Request(JEV_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key_value}", "Content-Type": "application/json",
        "User-Agent": "jaxflow/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    return parsed["answers"]
