#!/usr/bin/env python3
"""$JAXOS_HOME/settings.json schema, reader (MOA-497 Build A). Mirrors
src/server/settings.ts field-for-field -- same JSON keys, same closed-object rule at every
level, same independent-nested-defaults rule. Stdlib only, plus the local jaxflow_env
resolver."""
from __future__ import annotations

import json
import stat
from pathlib import Path

import jaxflow_env as jenv

LOCALES = {"en-US", "pt-BR"}
ROOT_KEYS = {"ownerName", "locale", "reposRoot", "vaultPath", "monitoredUnits", "integrations"}
MONITORED_KEYS = {"user", "system"}
AGENT_KEYS = ("claude", "codex", "opencode")
# MOA-504 D2: legacy key, accepted on READ only and mapped into `agents`; never written back.
LEGACY_SUBSCRIPTION_KEYS = ("claude", "codex", "opencodeGo")
BOOL_INTEGRATION_KEYS = ("linear", "ttyd", "webhook", "hermesTokens", "classifier", "github", "vault")
INTEGRATION_KEYS = set(BOOL_INTEGRATION_KEYS) | {"agents", "subscriptions"}


class _Absent:
    """Round 1 F3 (top-level) + round 2 F1 (nested): a sentinel distinct from any
    JSON-decoded value, used both for "no file on disk" and for "this key is not present in
    its parent object". JSON `null` decodes to Python `None`, which is a VALUE at every
    level -- `dict.get(key)` cannot tell an absent key from a key explicitly set to `null`,
    so every nested parser below is called with `raw[key] if key in raw else ABSENT`, never
    `raw.get(key)`, and treats a `None` it receives as malformed, not as "use the default"."""

    def __repr__(self):
        return "ABSENT"


ABSENT = _Absent()


class SettingsError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _malformed():
    raise SettingsError("settings-malformed")


def _default_settings():
    return {
        "ownerName": "", "locale": "en-US", "reposRoot": str(Path.home() / "repos"),
        "vaultPath": None,
        "monitoredUnits": {"user": [], "system": []},
        "integrations": {
            "linear": False, "ttyd": False, "webhook": False, "hermesTokens": False,
            "agents": {"claude": False, "codex": False, "opencode": False},
            "classifier": False, "github": False, "vault": False,
        },
    }


def _is_str_list(v):
    return type(v) is list and all(type(x) is str for x in v)


def _parse_monitored_units(v):
    # v is ABSENT (key not present in its parent) or a JSON-decoded value (round 2 F1: an
    # explicit `null` decodes to None here, a VALUE, not ABSENT -- it falls through to the
    # dict check below and is malformed, same as TS's own undefined-only default check).
    if v is ABSENT:
        return {"user": [], "system": []}
    if type(v) is not dict or set(v) - MONITORED_KEYS:
        _malformed()
    user = v.get("user", [])
    system = v.get("system", [])
    if not _is_str_list(user) or not _is_str_list(system):
        _malformed()
    return {"user": user, "system": system}


def _parse_bool_object(v, keys):
    if type(v) is not dict or set(v) - set(keys):
        _malformed()
    out = {}
    for key in keys:
        val = v.get(key, False)
        if type(val) is not bool:
            _malformed()
        out[key] = val
    return out


def _parse_agents(integrations):
    """MOA-504 D2, identical to the TS parseAgents: `agents` present -> strict, `subscriptions`
    ignored; else legacy `subscriptions` mapped (opencodeGo -> opencode); else all off."""
    if "agents" in integrations:
        return _parse_bool_object(integrations["agents"], AGENT_KEYS)
    if "subscriptions" in integrations:
        s = _parse_bool_object(integrations["subscriptions"], LEGACY_SUBSCRIPTION_KEYS)
        return {"claude": s["claude"], "codex": s["codex"], "opencode": s["opencodeGo"]}
    return {"claude": False, "codex": False, "opencode": False}


def _parse_integrations(v):
    if v is ABSENT:
        return _default_settings()["integrations"]
    if type(v) is not dict or set(v) - INTEGRATION_KEYS:
        _malformed()
    out = {}
    for key in BOOL_INTEGRATION_KEYS:
        val = v.get(key, False)
        if type(val) is not bool:
            _malformed()
        out[key] = val
    out["agents"] = _parse_agents(v)
    return out


def parse_settings(raw):
    """raw is a parsed JSON value, or the ABSENT sentinel for "file absent". Returns the
    settings dict, or raises SettingsError -- read_settings wraps both into the {ok, ...}
    shape shared with the TS reader. Round 1 F3: raw=None (JSON `null` on disk) is NOT
    absence -- it is malformed, same as the TS reader. Round 2 F1: the same rule applies one
    level down -- an explicit `null` for monitoredUnits/integrations/integrations.agents/subscriptions
    is malformed, never defaulted; only a genuinely absent key defaults."""
    default = _default_settings()
    if raw is ABSENT:
        return default
    if type(raw) is not dict or set(raw) - ROOT_KEYS:
        _malformed()
    owner_name = raw.get("ownerName", default["ownerName"])
    if type(owner_name) is not str:
        _malformed()
    locale = raw.get("locale", default["locale"])
    if locale not in LOCALES:
        _malformed()
    repos_root = raw.get("reposRoot", default["reposRoot"])
    if type(repos_root) is not str:
        _malformed()
    vault_path = raw.get("vaultPath", default["vaultPath"])
    if vault_path is not None and type(vault_path) is not str:
        _malformed()
    return {
        "ownerName": owner_name, "locale": locale, "reposRoot": repos_root, "vaultPath": vault_path,
        # Round 2 F1: key-presence check, not .get() -- an explicit `null` value must reach
        # the parser as None (malformed), never be conflated with an absent key (defaulted).
        "monitoredUnits": _parse_monitored_units(raw["monitoredUnits"] if "monitoredUnits" in raw else ABSENT),
        "integrations": _parse_integrations(raw["integrations"] if "integrations" in raw else ABSENT),
    }


def read_settings(path=None):
    path = path or (jenv.jaxos_home() / "settings.json")
    # F1: only FileNotFoundError means absent -> defaults. Path.exists()/is_file() swallow
    # every OSError (including ENOTDIR) and report False, which would misreport a real
    # problem (EACCES, ENOTDIR, ...) as "absent". Stat explicitly instead.
    try:
        st = path.stat()
    except FileNotFoundError:
        return {"ok": True, "data": parse_settings(ABSENT)}
    except OSError:
        return {"ok": False, "error": "settings-unreadable"}
    if not stat.S_ISREG(st.st_mode):
        return {"ok": False, "error": "settings-unreadable"}
    try:
        raw_bytes = path.read_bytes()
    except OSError:
        return {"ok": False, "error": "settings-unreadable"}
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {"ok": False, "error": "settings-malformed"}
    try:
        return {"ok": True, "data": parse_settings(parsed)}
    except SettingsError as exc:
        return {"ok": False, "error": exc.code}
