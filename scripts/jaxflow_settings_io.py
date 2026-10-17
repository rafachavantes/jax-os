#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

from jax_init import Refusal
import jaxflow_settings as jset

HELPER_IO_CAP = 16 * 1024 * 1024
EDITOR_TIMEOUT = 10
EDITOR = Path(__file__).resolve().parent / "jaxflow_opencode_edit.mjs"
REPO = Path(__file__).resolve().parent.parent
PROFILES = ("default", "fallback")
ADAPTER_NPM = {
    "xai": "@ai-sdk/xai",
    "openrouter": "@openrouter/ai-sdk-provider",
    "openai-compatible": "@ai-sdk/openai-compatible",
}
NPM_ADAPTER = {npm: name for name, npm in ADAPTER_NPM.items()}
SUPPORTED_NPM = frozenset(ADAPTER_NPM.values()) | {"@ai-sdk/openai", "@ai-sdk/anthropic"}
_ENV_REF = re.compile(r"^\{env:([A-Za-z_][A-Za-z0-9_]*)\}$")
JAXFLOW_ALIASES = frozenset(f"jaxflow-builder-{name}" for name in PROFILES)
PROVIDER_KINDS = frozenset({
    "save-connection", "save-model", "remove-model", "remove-connection", "bind-credential",
})
_FAULT = None


def edit_source(text, edits, *, jsonc):
    if type(text) is not str or type(edits) is not list:
        raise Refusal("native-config-malformed")
    if len(text.encode("utf-8")) > jset.SOURCE_CAP:
        raise Refusal("agent-settings-too-large")
    payload = json.dumps({"source": text, "edits": edits, "jsonc": bool(jsonc)}).encode("utf-8")
    if len(payload) > HELPER_IO_CAP:
        raise Refusal("agent-settings-too-large")
    try:
        proc = subprocess.run(
            ["node", str(EDITOR)],
            input=payload,
            capture_output=True,
            timeout=EDITOR_TIMEOUT,
            cwd=str(REPO),
        )
    except subprocess.TimeoutExpired:
        raise Refusal("native-config-malformed") from None
    if len(proc.stdout) > HELPER_IO_CAP:
        raise Refusal("agent-settings-too-large")
    try:
        parsed = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        raise Refusal("native-config-malformed") from None
    if type(parsed) is not dict:
        raise Refusal("native-config-malformed")
    if parsed.get("ok") is True and type(parsed.get("text")) is str:
        return {"text": parsed["text"], "value": parsed.get("value")}
    code = parsed.get("error")
    raise Refusal(code if type(code) is str else "native-config-malformed")


def _read_source_bytes(path):
    target = Path(path)
    try:
        info = target.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise Refusal("agent-settings-permissions") from exc
    if stat.S_ISLNK(info.st_mode):
        raise Refusal("agent-settings-symlink")
    if not stat.S_ISREG(info.st_mode):
        raise Refusal("agent-settings-not-file")
    try:
        fd = os.open(str(target), os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise Refusal("agent-settings-permissions") from exc
    try:
        raw = os.read(fd, jset.SOURCE_CAP + 1)
    finally:
        os.close(fd)
    if len(raw) > jset.SOURCE_CAP:
        raise Refusal("agent-settings-too-large")
    return raw


def snapshot(paths, native_metadata=None, bindings=None, *, run=None, env=None):
    settings = jset.read_settings(Path(paths["settings"]))
    sources = {name: jset._source_token(Path(paths[name])) for name in jset.SOURCE_FILES}
    _texts, values = _load_sources(paths)
    meta = _resolve_metadata(paths, native_metadata, values, run, env)
    return {
        "editor_revision": {
            "settings": settings["revision"] if settings else "absent",
            "sources": sources,
        },
        "settings": settings,
        "connections": _public_connections(values, meta, settings, env, bindings, paths),
    }


def _blocked_host(host):
    if type(host) is not str or not host:
        return True
    name = host.lower().rstrip(".")
    if name == "localhost" or name.endswith(".localhost") or name == "metadata.google.internal":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


def _require_https_url(url):
    if type(url) is not str or any(ord(ch) < 32 for ch in url):
        raise Refusal("agent-settings-malformed")
    try:
        parts = urlsplit(url)
    except ValueError:
        raise Refusal("agent-settings-malformed") from None
    if (
        parts.scheme != "https"
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or _blocked_host(parts.hostname)
    ):
        raise Refusal("agent-settings-malformed")


def _dto_url(native):
    options = native.get("options") if type(native) is dict else None
    url = options.get("baseURL") if type(options) is dict else None
    if type(url) is not str:
        return None, False
    if any(ord(ch) < 32 for ch in url):
        return None, True
    try:
        parts = urlsplit(url)
    except ValueError:
        return None, True
    if parts.username is not None or parts.password is not None:
        return None, True
    unsafe = parts.scheme != "https" or bool(parts.query) or bool(parts.fragment) or _blocked_host(parts.hostname)
    return url, unsafe


def _public_adapter(npm):
    return NPM_ADAPTER.get(npm, npm) if type(npm) is str else None


def _env_table(env):
    return os.environ if env is None else env


def _env_ref_name(key):
    if type(key) is str:
        matched = _ENV_REF.fullmatch(key)
        return matched.group(1) if matched else None
    if type(key) is dict and type(key.get("env")) is str:
        return key["env"]
    return None


def _api_key_present(native, env):
    options = native.get("options") if type(native) is dict else None
    key = options.get("apiKey") if type(options) is dict else None
    name = _env_ref_name(key)
    if name is not None:
        val = _env_table(env).get(name) if hasattr(_env_table(env), "get") else None
        return type(val) is str and bool(val)
    return type(key) is str and bool(key)


def _default_pure_run(argv, cwd=None, env=None, **kw):
    from jaxflow import _pure_config_run
    if argv[0] == "opencode" and shutil.which("opencode", path=_env_table(env).get("PATH", os.defpath)) is None:
        argv = [str(Path.home() / ".opencode" / "bin" / "opencode"), *argv[1:]]
    return _pure_config_run(argv, cwd=cwd, env=env, **kw)


def _invoke_run(run, argv, cwd, env):
    try:
        return run(argv, cwd=cwd, env=env)
    except TypeError:
        return run(argv, cwd=cwd)


def _try_effective_config(run, env):
    helper = run if run is not None else _default_pure_run
    try:
        return jset.read_effective_opencode_config(run=helper, env=env, repo=REPO)
    except Refusal:
        return {}


def _safe_model_meta(mid, entry):
    if type(entry) is not dict:
        return None
    out = {"id": entry["id"] if type(entry.get("id")) is str else mid}
    api = entry.get("api")
    if type(api) is dict and type(api.get("npm")) is str:
        out["npm"] = api["npm"]
    if type(entry.get("name")) is str:
        out["name"] = entry["name"]
    limit = entry.get("limit")
    if type(limit) is dict:
        ctx, outp = limit.get("context"), limit.get("output")
        if type(ctx) is int and type(outp) is int:
            out["limit"] = {"context": ctx, "output": outp}
    variants = entry.get("variants")
    if type(variants) is dict:
        out["variants"] = variants
    if type(entry.get("reasoning")) is bool:
        out["reasoning"] = entry["reasoning"]
    if type(entry.get("tool_call")) is bool:
        out["tool_call"] = entry["tool_call"]
    elif type(entry.get("capabilities")) is dict and type(entry["capabilities"].get("toolcall")) is bool:
        out["tool_call"] = entry["capabilities"]["toolcall"]
    return out


def _safe_provider_meta(row):
    if type(row) is not dict:
        return {}
    out = {}
    if type(row.get("npm")) is str:
        out["npm"] = row["npm"]
    if type(row.get("name")) is str:
        out["name"] = row["name"]
    models = row.get("models")
    out["models"] = {}
    if type(models) is dict:
        for mid, entry in models.items():
            if type(mid) is str and mid not in JAXFLOW_ALIASES:
                safe = _safe_model_meta(mid, entry)
                if safe:
                    out["models"][mid] = safe
    return out


def _verbose_models(run, cid, env):
    if not jset.CONNECTION_RE.fullmatch(cid):
        return {}
    helper = run if run is not None else _default_pure_run
    try:
        result = _invoke_run(helper, ["opencode", "models", cid, "--verbose", "--pure"], str(REPO), env)
    except Refusal:
        return {}
    if getattr(result, "returncode", 1) != 0:
        return {}
    text = result.stdout
    if type(text) is bytes:
        if len(text) > jset.CONFIG_STDOUT_CAP:
            return {}
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError:
            return {}
    if type(text) is not str or len(text.encode("utf-8")) > jset.CONFIG_STDOUT_CAP:
        return {}
    models = {}
    offset = 0
    decoder = json.JSONDecoder()
    try:
        while (start := text.find("{", offset)) >= 0:
            parsed, offset = decoder.raw_decode(text, start)
            if type(parsed) is not dict or type(parsed.get("id")) is not str:
                return {}
            safe = _safe_model_meta(parsed["id"], parsed)
            if safe:
                models[parsed["id"]] = safe
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}
    return models


def _provider_visibility(values):
    disabled = set()
    enabled = None
    for value in values.values():
        if type(value) is not dict:
            continue
        d = value.get("disabled_providers")
        if type(d) is list:
            disabled.update(x for x in d if type(x) is str)
        e = value.get("enabled_providers")
        if type(e) is list:
            names = {x for x in e if type(x) is str}
            enabled = names if enabled is None else enabled | names
    return disabled, enabled


def _source_cids(values):
    cids = []
    seen = set()
    for name in jset.SOURCE_FILES:
        value = values.get(name)
        if type(value) is not dict:
            continue
        providers = value.get("provider")
        if type(providers) is not dict:
            continue
        for cid in providers:
            if type(cid) is str and cid not in seen and jset.CONNECTION_RE.fullmatch(cid):
                seen.add(cid)
                cids.append(cid)
    return cids


def _source_native(values, cid):
    native = {}
    for name in jset.SOURCE_FILES:
        value = values.get(name)
        if type(value) is not dict:
            continue
        providers = value.get("provider")
        row = providers.get(cid) if type(providers) is dict else None
        if type(row) is dict:
            native = row
    return native


def _read_auth_file(paths):
    path = paths.get("auth") if type(paths) is dict else None
    if path is None:
        return "missing", None
    try:
        raw = _read_source_bytes(Path(path))
    except Refusal:
        return "unreadable", None
    if raw is None:
        return "missing", None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        return "unreadable", None
    if type(parsed) is not dict:
        return "unreadable", None
    return "ok", parsed


def _auth_entry(row):
    if type(row) is not dict:
        return None
    kind = row.get("type")
    if kind == "oauth":
        return {"auth": "oauth", "status": "present"}
    if kind == "api" and type(row.get("key")) is str and row.get("key"):
        return {"auth": "api-key", "status": "present"}
    return None


def _open_code_config_is_extra(paths, env):
    extra = _env_table(env).get("OPENCODE_CONFIG") if hasattr(_env_table(env), "get") else None
    if type(extra) is not str or not extra.strip():
        return False
    try:
        extra_real = os.path.realpath(extra)
    except OSError:
        return True
    known = set()
    for name in jset.SOURCE_FILES:
        try:
            known.add(os.path.realpath(str(paths[name])))
        except OSError:
            pass
    return extra_real not in known


def _foreign_configured(values, effective):
    if type(effective) is not dict:
        return False
    providers = effective.get("provider")
    if type(providers) is not dict:
        return False
    source_ids = set(_source_cids(values))
    for cid, row in providers.items():
        if cid in source_ids or type(row) is not dict:
            continue
        options = row.get("options") if type(row.get("options")) is dict else {}
        if type(options.get("apiKey")) is str or type(options.get("baseURL")) is str:
            return True
    return False


def _credential_record(native, env, auth_file_status, auth_row):
    if _api_key_present(native, env):
        return {"auth": "api-key", "status": "present"}
    options = native.get("options") if type(native) is dict else None
    key = options.get("apiKey") if type(options) is dict else None
    if _env_ref_name(key) is not None:
        return {"auth": "api-key", "status": "missing"}
    if auth_file_status == "unreadable":
        return {"auth": "unknown", "status": "unreadable"}
    parsed = _auth_entry(auth_row)
    return parsed if parsed else {"auth": "none", "status": "missing"}


def _binding_credential(bindings, cid, native, paths, env):
    if type(bindings) is not list:
        return None
    for binding in bindings:
        if type(binding) is not dict or binding.get("provider_id") != cid:
            continue
        if type(binding.get("env_name")) is not str or type(binding.get("secret_id")) is not str:
            continue
        options = native.get("options") if type(native) is dict else None
        key = options.get("apiKey") if type(options) is dict else None
        if _env_ref_name(key) != binding["env_name"]:
            continue
        table = _protected_env_values(paths, env)
        if type(table) is dict and type(table.get(binding["env_name"])) is str and table[binding["env_name"]]:
            return {
                "auth": "api-key",
                "status": "present",
                "credential": {
                    "kind": "env",
                    "env": binding["env_name"],
                },
            }
    return None


def _protected_env_values(paths, env):
    raw = None
    if type(paths) is dict and paths.get("env") is not None:
        try:
            raw = _read_source_bytes(Path(paths["env"]))
        except Refusal:
            raw = None
    if raw is not None and raw != b"":
        try:
            text = raw.decode("utf-8")
        except (UnicodeDecodeError, UnicodeError):
            text = ""
        table = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key in table:
                continue
            quoted = len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'"
            table[key] = value[1:-1] if quoted else value
        return table
    base = _env_table(env)
    if hasattr(base, "get"):
        return {key: str(value) for key, value in base.items() if type(value) is str}
    return {}


def _discover_native_metadata(paths, values, *, run=None, env=None):
    table = _env_table(env)
    effective = _try_effective_config(run, table)
    auth_status, auth_map = _read_auth_file(paths)
    providers = {}
    eff_prov = effective.get("provider") if type(effective) is dict else None
    if type(eff_prov) is dict:
        for cid, row in eff_prov.items():
            if type(cid) is str and jset.CONNECTION_RE.fullmatch(cid) and type(row) is dict:
                providers[cid] = _safe_provider_meta(row)
    disabled, enabled = _provider_visibility(values)
    listed, seen = [], set()
    for cid in _source_cids(values):
        if cid in disabled or (enabled is not None and cid not in enabled):
            continue
        listed.append(cid)
        seen.add(cid)
    if type(auth_map) is dict:
        for cid, row in auth_map.items():
            if type(cid) is not str or cid in seen or not jset.CONNECTION_RE.fullmatch(cid):
                continue
            if _auth_entry(row) is None:
                continue
            if cid in disabled or (enabled is not None and cid not in enabled):
                continue
            listed.append(cid)
            seen.add(cid)
    for cid in listed:
        native = _source_native(values, cid)
        meta = providers.get(cid)
        if type(meta) is not dict:
            meta = _safe_provider_meta(native)
            providers[cid] = meta
        elif type(native.get("npm")) is str and "npm" not in meta:
            meta = {**meta, "npm": native["npm"]}
            providers[cid] = meta
        models = meta.get("models") if type(meta) is dict else None
        if type(models) is not dict or not models:
            parsed = _verbose_models(run, cid, table)
            if parsed:
                providers[cid] = {**meta, "models": parsed}
                npms = {model.get("npm") for model in parsed.values()}
                if "npm" not in meta and len(npms) == 1:
                    npm = next(iter(npms))
                    if npm in SUPPORTED_NPM:
                        providers[cid]["npm"] = npm
    credentials = {}
    auth_rows = auth_map if type(auth_map) is dict else {}
    for cid in listed:
        credentials[cid] = _credential_record(
            _source_native(values, cid), table, auth_status, auth_rows.get(cid),
        )
    extra = _open_code_config_is_extra(paths, table) or _foreign_configured(values, effective)
    return {"providers": providers, "credentials": credentials, "extra_override": extra}


def _resolve_metadata(paths, native_metadata, values, run, env):
    if native_metadata:
        return native_metadata
    return _discover_native_metadata(paths, values, run=run, env=env)


def _connection_health(native, meta_cred, env=None):
    options = native.get("options") if type(native) is dict else None
    key = options.get("apiKey") if type(options) is dict else None
    env_name = _env_ref_name(key)
    if type(meta_cred) is dict:
        status = meta_cred.get("status")
        auth = meta_cred.get("auth")
        auth = auth if auth in ("api-key", "oauth", "none", "unknown") else "unknown"
        if status == "unreadable":
            return auth, "unavailable", None
        if auth == "oauth":
            return "oauth", "oauth-managed", {"kind": "native"}
        if status == "missing" or auth == "none":
            if auth == "api-key" and env_name is not None:
                return "api-key", "missing", {"kind": "env", "env": env_name}
            return "none" if auth == "unknown" else auth, "missing", None
        if status == "present" and auth == "api-key":
            if env_name is not None:
                return "api-key", "registered", {"kind": "env", "env": env_name}
            return "api-key", "registered", {"kind": "native"}
    if _api_key_present(native, env):
        if env_name is not None:
            return "api-key", "registered", {"kind": "env", "env": env_name}
        return "api-key", "registered", {"kind": "native"}
    if env_name is not None:
        # MOA-498: report WHICH env var is missing instead of collapsing to no credential at
        # all — Task 12's dialog needs the bound name to prefill and to check status.
        return "api-key", "missing", {"kind": "env", "env": env_name}
    return "none", "missing", None


def _used_by(settings, connection, model_id=None):
    if settings is None:
        return []
    names = []
    for name in PROFILES:
        row = settings["builders"][name]
        if row["connection"] != connection:
            continue
        if model_id is not None and row["model"] != model_id:
            continue
        names.append(name)
    return names


def _effort_template_of(src):
    if type(src) is not dict:
        return None
    variants = src.get("variants")
    if type(variants) is not dict or not variants:
        return None
    seen = set()
    for payload in variants.values():
        if type(payload) is not dict:
            return None
        has_nested = type(payload.get("reasoning")) is dict and type(payload["reasoning"].get("effort")) is str
        has_flat = type(payload.get("reasoningEffort")) is str
        if has_nested and not has_flat:
            seen.add("reasoning")
        elif has_flat and not has_nested:
            seen.add("reasoning_effort")
        else:
            return None
    return seen.pop() if len(seen) == 1 else None


def _variant_payload(template, effort):
    if template == "reasoning":
        return {"reasoning": {"effort": effort}}
    return {"reasoningEffort": effort}


def _model_choice(model_id, override, catalog, npm):
    src = override if type(override) is dict else catalog if type(catalog) is dict else None
    if type(src) is not dict:
        return None
    variants = src.get("variants")
    efforts = list(variants) if type(variants) is dict else []
    compatible = (npm or (catalog or {}).get("npm")) in SUPPORTED_NPM
    limit = src.get("limit")
    context = limit.get("context") if type(limit) is dict and type(limit.get("context")) is int else None
    return {
        "id": model_id,
        "label": src["name"] if type(src.get("name")) is str else model_id,
        "origin": "override" if type(override) is dict else "catalog",
        "efforts": efforts,
        "no_effort": not efforts,
        "compatible": compatible,
        "reason": None if compatible else "unsupported-transport",
        "context": context,
        "effort_template": _effort_template_of(src),
    }


def _public_connections(values, native_metadata, settings, env=None, bindings=None, paths=None):
    meta_providers = native_metadata.get("providers") if type(native_metadata) is dict else None
    if type(meta_providers) is not dict:
        meta_providers = {}
    creds = native_metadata.get("credentials") if type(native_metadata) is dict else None
    found = {}
    for name in jset.SOURCE_FILES:
        value = values.get(name)
        if type(value) is not dict:
            continue
        providers = value.get("provider")
        if type(providers) is not dict:
            continue
        for cid, native in providers.items():
            if type(cid) is not str or type(native) is not dict:
                continue
            row = found.setdefault(cid, {"native": native, "files": []})
            row["files"].append(name)
            if name == "opencode.jsonc":
                row["native"] = native
    if type(creds) is dict:
        for cid in creds:
            if type(cid) is str and cid not in found:
                found[cid] = {"native": {}, "files": []}
    disabled, enabled = _provider_visibility(values)
    connections = []
    for cid, row in found.items():
        if cid in disabled or (enabled is not None and cid not in enabled):
            continue
        native = row["native"]
        meta = meta_providers.get(cid) if type(meta_providers.get(cid)) is dict else {}
        npm = native.get("npm") if type(native.get("npm")) is str else meta.get("npm")
        base_url, unsafe = _dto_url(native)
        ambiguous = len(row["files"]) > 1
        auth, health, credential = _connection_health(
            native, creds.get(cid) if type(creds) is dict else None, env,
        )
        bound = _binding_credential(bindings, cid, native, paths, env)
        if bound is not None:
            auth, health, credential = bound["auth"], "registered", bound["credential"]
        native_models = native.get("models") if type(native.get("models")) is dict else {}
        meta_models = meta.get("models") if type(meta.get("models")) is dict else {}
        blacklist = native.get("blacklist") if type(native.get("blacklist")) is list else []
        whitelist = native.get("whitelist") if type(native.get("whitelist")) is list else None
        ids = []
        seen = set()
        for mid in list(native_models) + [key for key in meta_models if key not in native_models]:
            if mid in seen or mid in JAXFLOW_ALIASES or mid in blacklist:
                continue
            if type(whitelist) is list and mid not in whitelist:
                continue
            seen.add(mid)
            ids.append(mid)
        models = []
        for mid in ids:
            choice = _model_choice(mid, native_models.get(mid), meta_models.get(mid), npm)
            if choice:
                models.append(choice)
        connections.append({
            "id": cid,
            "label": native["name"] if type(native.get("name")) is str else meta["name"] if type(meta.get("name")) is str else cid,
            "adapter": _public_adapter(npm),
            "base_url": base_url,
            "auth": auth,
            "health": health,
            "credential": credential,
            "editable": not ambiguous and not unsafe and auth != "oauth",
            "reason": "native-config-ambiguous" if ambiguous else ("unsupported-address" if unsafe else ("oauth-managed" if auth == "oauth" else None)),
            "used_by": _used_by(settings, cid),
            "models": models,
        })
    return connections


def _validate_save_connection(intent):
    conn = intent.get("connection")
    if type(conn) is not dict:
        raise Refusal("agent-settings-malformed")
    cid = conn.get("id")
    if type(cid) is not str or not jset.CONNECTION_RE.fullmatch(cid):
        raise Refusal("agent-settings-malformed")
    if conn.get("adapter") not in ADAPTER_NPM:
        raise Refusal("agent-settings-malformed")
    if "label" in conn and conn["label"] is not None and type(conn["label"]) is not str:
        raise Refusal("agent-settings-malformed")
    if conn.get("base_url") is not None:
        _require_https_url(conn["base_url"])


def _validate_save_model(intent):
    cid = intent.get("connection")
    if type(cid) is not str or not jset.CONNECTION_RE.fullmatch(cid):
        raise Refusal("agent-settings-malformed")
    model = intent.get("model")
    if type(model) is not dict:
        raise Refusal("agent-settings-malformed")
    _validate_model_input(model)


def _validate_model_input(model):
    allowed = {"id", "label", "context", "limit", "reasoning", "tool_call", "effort_template", "efforts"}
    for key in model:
        if key not in allowed:
            raise Refusal("agent-settings-malformed")
    mid = model.get("id")
    if type(mid) is not str or not mid or mid in JAXFLOW_ALIASES:
        raise Refusal("agent-settings-malformed")
    if len(mid.encode("utf-8")) > 256 or any(ord(ch) < 32 for ch in mid):
        raise Refusal("agent-settings-malformed")
    if "label" in model and model["label"] is not None:
        label = model["label"]
        if type(label) is not str or len(label.encode("utf-8")) > 256:
            raise Refusal("agent-settings-malformed")
    if "context" in model:
        context = model["context"]
        if type(context) is not int or context <= 0 or "limit" in model:
            raise Refusal("agent-settings-malformed")
    limit = model.get("limit")
    if limit is not None:
        if type(limit) is not dict or set(limit) - {"context", "output"}:
            raise Refusal("agent-settings-malformed")
        for value in limit.values():
            if type(value) is not int or value <= 0:
                raise Refusal("agent-settings-malformed")
    for key in ("reasoning", "tool_call"):
        if key in model and type(model[key]) is not bool:
            raise Refusal("agent-settings-malformed")
    template = model.get("effort_template")
    if template is not None and template not in ("none", "reasoning", "reasoning_effort"):
        raise Refusal("agent-settings-malformed")
    efforts = model.get("efforts")
    if efforts is not None:
        if type(efforts) is not list or any(type(e) is not str for e in efforts):
            raise Refusal("agent-settings-malformed")
        if len(efforts) != len(set(efforts)) or any(not jset.EFFORT_RE.fullmatch(e) for e in efforts):
            raise Refusal("agent-settings-malformed")
    if model.get("reasoning") is False:
        if template not in (None, "none") or (efforts is not None and efforts):
            raise Refusal("agent-settings-malformed")
    if template is not None:
        if type(model.get("reasoning")) is not bool or type(model.get("tool_call")) is not bool:
            raise Refusal("agent-settings-malformed")
        if type(limit) is not dict or any(limit.get(k) is None for k in ("context", "output")):
            raise Refusal("agent-settings-malformed")
        if model["tool_call"] is not True:
            raise Refusal("agent-settings-malformed")
        if template == "none":
            if model["reasoning"] is not False or (efforts is not None and efforts):
                raise Refusal("agent-settings-malformed")
        else:
            if model["reasoning"] is not True or (not efforts or not any(efforts)):
                raise Refusal("agent-settings-malformed")


def _validate_bind_credential(intent):
    cid = intent.get("connection")
    if type(cid) is not str or not jset.CONNECTION_RE.fullmatch(cid):
        raise Refusal("agent-settings-malformed")
    # MOA-498: the ONE shared credential validator now accepts both {"kind":"native"} and
    # {"kind":"env",...} — bind-credential is a general "set this connection's credential
    # binding" intent, not an env-only one, so it delegates instead of re-checking `kind`
    # itself (the earlier gate only ever allowed one of the two valid shapes).
    jset._credential(intent.get("credential"))


def _validate_intent(intent):
    if type(intent) is not dict:
        raise Refusal("agent-settings-malformed")
    kind = intent.get("kind")
    if kind == "save-settings":
        fake = {
            "schema_version": 1,
            "revision": "0" * 32,
            "reviewers": intent.get("reviewers"),
            "builders": intent.get("builders"),
            "source_revisions": {"opencode.json": "absent", "opencode.jsonc": "absent"},
        }
        jset.decode_settings(json.dumps(fake, separators=(",", ":")).encode())
        return
    if kind not in PROVIDER_KINDS:
        raise Refusal("agent-settings-malformed")
    if kind == "save-connection":
        _validate_save_connection(intent)
        return
    if kind == "save-model":
        _validate_save_model(intent)
        return
    if kind == "bind-credential":
        _validate_bind_credential(intent)
        return
    cid = intent.get("connection")
    if type(cid) is not str or not jset.CONNECTION_RE.fullmatch(cid):
        raise Refusal("agent-settings-malformed")
    if kind == "remove-model":
        mid = intent.get("model")
        if type(mid) is not str or not mid or mid in JAXFLOW_ALIASES:
            raise Refusal("agent-settings-malformed")


def _load_sources(paths):
    texts = {}
    values = {}
    for name in jset.SOURCE_FILES:
        raw = _read_source_bytes(Path(paths[name]))
        if raw is None:
            texts[name] = None
            values[name] = None
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise Refusal("native-config-malformed") from exc
        parsed = edit_source(text, [], jsonc=name.endswith("jsonc"))
        texts[name] = text
        values[name] = parsed["value"]
    return texts, values


def _has_connection(value, connection):
    if type(value) is not dict:
        return False
    providers = value.get("provider")
    return type(providers) is dict and connection in providers


def _target_file(values, connection):
    present = [name for name in jset.SOURCE_FILES if _has_connection(values[name], connection)]
    if len(present) > 1:
        raise Refusal("native-config-ambiguous")
    if len(present) == 1:
        return present[0]
    if values["opencode.jsonc"] is not None:
        return "opencode.jsonc"
    return "opencode.json"


def _check_collision(values, target, connection, settings):
    # Alias names are reserved (JAXFLOW_ALIASES). Once settings exist, any jaxflow-builder-*
    # alias was written by this editor, including one a profile left behind when it moved to
    # another connection. Only first adoption treats a pre-existing alias as foreign.
    if settings is not None:
        return
    value = values[target]
    if type(value) is not dict:
        return
    native = (value.get("provider") or {}).get(connection)
    if type(native) is not dict:
        return
    models = native.get("models")
    if type(models) is not dict:
        return
    for profile_name in PROFILES:
        alias = f"jaxflow-builder-{profile_name}"
        if alias in models:
            raise Refusal("agent-alias-collision")


def _catalog_model(native_metadata, values, connection, model_id):
    providers = (native_metadata or {}).get("providers") if type(native_metadata) is dict else None
    meta = providers.get(connection) if type(providers) is dict else None
    npm = meta.get("npm") if type(meta) is dict else None
    models = meta.get("models") if type(meta) is dict else None
    if type(models) is dict and model_id in models:
        return models[model_id], npm or models[model_id].get("npm")
    for value in values.values():
        if type(value) is not dict:
            continue
        native = (value.get("provider") or {}).get(connection)
        if type(native) is not dict:
            continue
        if npm is None:
            npm = native.get("npm")
        src_models = native.get("models")
        if type(src_models) is dict and model_id in src_models:
            return src_models[model_id], npm
    raise Refusal("agent-profile-conflict")


def _clone_alias(profile, catalog_model, npm):
    if type(catalog_model) is not dict or catalog_model.get("id") != profile["model"]:
        raise Refusal("agent-profile-conflict")
    out = {"id": profile["model"]}
    if "name" in catalog_model:
        out["name"] = catalog_model["name"]
    if "limit" in catalog_model:
        out["limit"] = json.loads(json.dumps(catalog_model["limit"]))
    effort = profile["effort"]
    if effort is not None:
        variants = catalog_model.get("variants")
        if type(variants) is not dict or effort not in variants or type(variants[effort]) is not dict:
            raise Refusal("agent-profile-conflict")
        out["variants"] = {effort: json.loads(json.dumps(variants[effort]))}
    if profile["routing"] is not None:
        out["options"] = {"provider": json.loads(json.dumps(profile["routing"]))}
    return out


def _bindings_from_values(values):
    rows = []
    for profile_name in PROFILES:
        alias = f"jaxflow-builder-{profile_name}"
        found = None
        for value in values.values():
            if type(value) is not dict:
                continue
            providers = value.get("provider")
            if type(providers) is not dict:
                continue
            for connection, native in providers.items():
                if type(native) is not dict:
                    continue
                models = native.get("models")
                if type(models) is not dict or alias not in models:
                    continue
                model = models[alias]
                if type(model) is not dict:
                    continue
                routing = None
                options = model.get("options")
                if type(options) is dict and type(options.get("provider")) is dict:
                    routing = options["provider"]
                effort = None
                variants = model.get("variants")
                if type(variants) is dict and len(variants) == 1:
                    effort = next(iter(variants))
                found = {
                    "profile": profile_name,
                    "connection": connection,
                    "alias": alias,
                    "model": model.get("id"),
                    "effort": effort,
                    "routing": routing,
                }
        if found:
            rows.append(found)
    return rows


def _bindings_from_intent(intent):
    rows = []
    for profile_name in PROFILES:
        profile = intent["builders"][profile_name]
        rows.append({
            "profile": profile_name,
            "connection": profile["connection"],
            "alias": f"jaxflow-builder-{profile_name}",
            "model": profile["model"],
            "effort": profile["effort"],
            "routing": profile["routing"],
        })
    return rows


def _refuse_referenced(settings, connection, model=None):
    if settings is None:
        return
    for name in PROFILES:
        row = settings["builders"][name]
        if row["connection"] != connection:
            continue
        if model is None or row["model"] == model:
            raise Refusal("agent-profile-conflict")


def _refuse_oauth(native_metadata, connection):
    creds = native_metadata.get("credentials") if type(native_metadata) is dict else None
    cred = creds.get(connection) if type(creds) is dict else None
    if type(cred) is dict and cred.get("auth") == "oauth":
        raise Refusal("agent-profile-conflict")


def _refuse_discovered_only(native_metadata, values, connection):
    providers = native_metadata.get("providers") if type(native_metadata) is dict else None
    if not (type(providers) is dict and connection in providers):
        return
    in_source = any(_has_connection(values[name], connection) for name in jset.SOURCE_FILES)
    if not in_source:
        raise Refusal("agent-profile-conflict")


def _edit_target(texts, values, target, edits):
    if not edits:
        return []
    base = texts[target] if texts[target] is not None else "{}\n"
    parsed = edit_source(base, edits, jsonc=target.endswith("jsonc"))
    texts[target] = parsed["text"]
    values[target] = parsed["value"]
    return [target]


def _native_at(values, target, connection):
    current = values[target] if type(values[target]) is dict else {}
    native = (current.get("provider") or {}).get(connection) if type(current) is dict else None
    return native if type(native) is dict else {}


def _save_connection(intent, texts, values):
    conn = intent["connection"]
    cid = conn["id"]
    target = _target_file(values, cid)
    edits = [{"path": ["provider", cid, "npm"], "value": ADAPTER_NPM[conn["adapter"]]}]
    if type(conn.get("label")) is str:
        edits.append({"path": ["provider", cid, "name"], "value": conn["label"]})
    if conn.get("base_url"):
        edits.append({"path": ["provider", cid, "options", "baseURL"], "value": conn["base_url"]})
    return _edit_target(texts, values, target, edits)


def _model_entry_from_intent(model, catalog, npm=None):
    template = model.get("effort_template")
    catalog_copy = type(catalog) is dict and template is None
    if catalog_copy:
        out = {"id": catalog["id"] if type(catalog.get("id")) is str else model["id"]}
        for key in ("name", "limit"):
            if key in catalog:
                out[key] = json.loads(json.dumps(catalog[key]))
        if "variants" in catalog:
            out["variants"] = json.loads(json.dumps(catalog["variants"]))
        if type(catalog.get("reasoning")) is bool:
            out["reasoning"] = catalog["reasoning"]
        if type(catalog.get("tool_call")) is bool:
            out["tool_call"] = catalog["tool_call"]
        for key in ("id", "name", "limit", "reasoning", "tool_call"):
            if key in model:
                out[key] = json.loads(json.dumps(model[key])) if key == "limit" else model[key]
        return out
    out = {"id": model["id"]}
    if type(model.get("label")) is str:
        out["name"] = model["label"]
    if type(model.get("limit")) is dict:
        out["limit"] = json.loads(json.dumps(model["limit"]))
    if type(model.get("reasoning")) is bool:
        out["reasoning"] = model["reasoning"]
    if type(model.get("tool_call")) is bool:
        out["tool_call"] = model["tool_call"]
    efforts = model.get("efforts")
    if template in ("reasoning", "reasoning_effort"):
        variants = {}
        for effort in efforts or []:
            if template == "reasoning":
                variants[effort] = {"reasoning": {"effort": effort}}
            else:
                variants[effort] = {"reasoningEffort": effort}
        out["variants"] = variants
    return out


def _narrow_model_edits(cid, mid, model, existing, catalog):
    base = ["provider", cid, "models", mid]
    source = existing if type(existing) is dict else catalog if type(catalog) is dict else None
    edits = []
    if existing is None:
        entry = {"id": catalog["id"] if type(catalog) is dict and type(catalog.get("id")) is str else mid}
        if type(catalog) is dict:
            for key in ("name", "limit", "variants", "reasoning", "tool_call", "options"):
                if key in catalog:
                    entry[key] = json.loads(json.dumps(catalog[key]))
        edits.append({"path": base, "value": entry})
    if type(model.get("label")) is str:
        edits.append({"path": base + ["name"], "value": model["label"]})
    if type(model.get("context")) is int:
        edits.append({"path": base + ["limit", "context"], "value": model["context"]})
        source_limit = source.get("limit") if type(source) is dict and type(source.get("limit")) is dict else None
        if type(source_limit) is not dict or type(source_limit.get("output")) is not int:
            edits.append({"path": base + ["limit", "output"], "value": 0})
    source_variants = source.get("variants") if type(source) is dict and type(source.get("variants")) is dict else {}
    template = _effort_template_of(source)
    final_efforts = list(source_variants)
    if "efforts" in model:
        wanted = list(model["efforts"])
        if template is None:
            if set(wanted) != set(source_variants):
                raise Refusal("agent-profile-conflict")
        else:
            for effort in wanted:
                if effort not in source_variants:
                    edits.append({"path": base + ["variants", effort], "value": _variant_payload(template, effort)})
            for effort in source_variants:
                if effort not in wanted:
                    edits.append({"path": base + ["variants", effort], "remove": True})
            final_efforts = wanted
    return edits, final_efforts


def _save_model(intent, texts, values, native_metadata, settings):
    cid = intent["connection"]
    model = intent["model"]
    mid = model["id"]
    target = _target_file(values, cid)
    native = _native_at(values, target, cid)
    native_models = native.get("models") if type(native.get("models")) is dict else {}
    existing = native_models.get(mid) if type(native_models.get(mid)) is dict else None
    npm = native.get("npm")
    catalog = None
    try:
        catalog, found_npm = _catalog_model(native_metadata, values, cid, mid)
        if npm is None:
            npm = found_npm
    except Refusal:
        catalog = None
    if npm not in SUPPORTED_NPM and npm is not None:
        raise Refusal("agent-profile-conflict")
    narrow = not any(key in model for key in ("limit", "reasoning", "tool_call", "effort_template"))
    if narrow:
        edits, final_efforts = _narrow_model_edits(cid, mid, model, existing, catalog)
    else:
        entry = _model_entry_from_intent(model, catalog, npm)
        edits = [{"path": ["provider", cid, "models", mid], "value": entry}]
        final_efforts = list(entry.get("variants")) if type(entry.get("variants")) is dict else []
    if settings is not None:
        for profile_name in PROFILES:
            row = settings["builders"][profile_name]
            if row["connection"] != cid or row["model"] != mid:
                continue
            if row["effort"] is None:
                continue
            if row["effort"] not in final_efforts:
                raise Refusal("agent-profile-conflict")
    whitelist = native.get("whitelist")
    if type(whitelist) is list and mid not in whitelist:
        edits.append({"path": ["provider", cid, "whitelist"], "value": whitelist + [mid]})
    blacklist = native.get("blacklist")
    if type(blacklist) is list and mid in blacklist:
        edits.append({"path": ["provider", cid, "blacklist"], "value": [item for item in blacklist if item != mid]})
    return _edit_target(texts, values, target, edits)


def _remove_model(intent, texts, values, native_metadata):
    cid = intent["connection"]
    mid = intent["model"]
    target = _target_file(values, cid)
    native = _native_at(values, target, cid)
    models = native.get("models") if type(native.get("models")) is dict else {}
    whitelist = native.get("whitelist") if type(native.get("whitelist")) is list else None
    blacklist = list(native["blacklist"]) if type(native.get("blacklist")) is list else []
    meta = (native_metadata or {}).get("providers") if type(native_metadata) is dict else None
    meta_models = (meta.get(cid) or {}).get("models") if type(meta) is dict and type(meta.get(cid)) is dict else None
    in_models = mid in models
    in_whitelist = type(whitelist) is list and mid in whitelist
    in_catalog = type(meta_models) is dict and mid in meta_models
    edits = []
    if in_models:
        edits.append({"path": ["provider", cid, "models", mid], "remove": True})
    if in_whitelist:
        edits.append({"path": ["provider", cid, "whitelist"], "value": [item for item in whitelist if item != mid]})
    if in_catalog or in_whitelist or not in_models:
        if mid not in blacklist:
            edits.append({"path": ["provider", cid, "blacklist"], "value": blacklist + [mid]})
    return _edit_target(texts, values, target, edits)


def _remove_connection(intent, texts, values, native_metadata):
    cid = intent["connection"]
    target = _target_file(values, cid)
    edits = []
    if _has_connection(values[target], cid):
        edits.append({"path": ["provider", cid], "remove": True})
    meta = native_metadata.get("providers") if type(native_metadata) is dict else None
    creds = native_metadata.get("credentials") if type(native_metadata) is dict else None
    would = (type(meta) is dict and cid in meta) or (type(creds) is dict and cid in creds)
    current = values[target] if type(values[target]) is dict else {}
    disabled = current.get("disabled_providers") if type(current.get("disabled_providers")) is list else []
    if would and cid not in disabled:
        edits.append({"path": ["disabled_providers"], "value": disabled + [cid]})
    return _edit_target(texts, values, target, edits)


def _bind_credential(intent, texts, values):
    cid = intent["connection"]
    credential = intent["credential"]
    target = _target_file(values, cid)
    if credential["kind"] == "native":
        # Revert to native: remove the apiKey override entirely (the same "remove" edit shape
        # _remove_model/_remove_connection already use) rather than writing an empty/placeholder
        # value. A no-op when there was no override to remove — jsonc-parser's modify() with an
        # undefined value on a path that does not exist produces no edit.
        edits = [{"path": ["provider", cid, "options", "apiKey"], "remove": True}]
    else:
        reference = "{env:%s}" % credential["env"]
        edits = [{"path": ["provider", cid, "options", "apiKey"], "value": reference}]
    return _edit_target(texts, values, target, edits)


def _project_provider(intent, paths, native_metadata, settings):
    texts, values = _load_sources(paths)
    before = _bindings_from_values(values)
    kind = intent["kind"]
    connection = intent.get("connection")
    if type(connection) is dict:
        connection = connection.get("id")
    _refuse_oauth(native_metadata, connection)
    if kind == "save-connection":
        _refuse_discovered_only(native_metadata, values, connection)
        affected = _save_connection(intent, texts, values)
    elif kind == "save-model":
        affected = _save_model(intent, texts, values, native_metadata, settings)
    elif kind == "remove-model":
        _refuse_referenced(settings, connection, intent["model"])
        affected = _remove_model(intent, texts, values, native_metadata)
    elif kind == "bind-credential":
        affected = _bind_credential(intent, texts, values)
    else:
        _refuse_referenced(settings, intent["connection"])
        affected = _remove_connection(intent, texts, values, native_metadata)
    order = [t for t in jset.SOURCE_FILES if t in affected]
    return texts, order, {"before": before, "after": _bindings_from_values(values)}


def _project(intent, paths, native_metadata, settings):
    if intent.get("kind") != "save-settings":
        return _project_provider(intent, paths, native_metadata, settings)
    texts, values = _load_sources(paths)
    before = _bindings_from_values(values)
    by_conn = {}
    for profile_name in PROFILES:
        by_conn.setdefault(intent["builders"][profile_name]["connection"], []).append(profile_name)
    affected = []
    for connection, profile_names in by_conn.items():
        target = _target_file(values, connection)
        _check_collision(values, target, connection, settings)
        edits = []
        npm = ((native_metadata.get("providers") or {}).get(connection) or {}).get("npm")
        aliases = []
        current = values[target] if values[target] is not None else {}
        native = ((current.get("provider") or {}).get(connection) if type(current) is dict else None) or {}
        native_npm = native.get("npm") if type(native) is dict else None
        if native_npm is not None and native_npm not in SUPPORTED_NPM:
            raise Refusal("agent-profile-conflict")
        for profile_name in profile_names:
            profile = intent["builders"][profile_name]
            catalog, found_npm = _catalog_model(native_metadata, values, connection, profile["model"])
            if (native_npm or found_npm) not in SUPPORTED_NPM:
                raise Refusal("agent-profile-conflict")
            aliases.append(f"jaxflow-builder-{profile_name}")
            edits.append({
                "path": ["provider", connection, "models", f"jaxflow-builder-{profile_name}"],
                "value": _clone_alias(profile, catalog, npm),
            })
        if npm and (type(native) is not dict or native.get("npm") != npm) and npm in SUPPORTED_NPM and native_npm in (None, npm):
            edits.insert(0, {"path": ["provider", connection, "npm"], "value": npm})
        whitelist = native.get("whitelist") if type(native) is dict else None
        if type(whitelist) is list:
            extra = [alias for alias in aliases if alias not in whitelist]
            if extra:
                edits.append({"path": ["provider", connection, "whitelist"], "value": whitelist + extra})
        base = texts[target] if texts[target] is not None else "{}\n"
        parsed = edit_source(base, edits, jsonc=target.endswith("jsonc"))
        texts[target] = parsed["text"]
        values[target] = parsed["value"]
        affected.append(target)
    order = [name for name in jset.SOURCE_FILES if name in affected]
    return texts, order, {"before": before, "after": _bindings_from_intent(intent)}


def _same_revision(current, expected):
    if type(expected) is not dict:
        raise Refusal("agent-settings-malformed")
    if current.get("settings") != expected.get("settings"):
        return False
    sources = expected.get("sources")
    return type(sources) is dict and sources == current.get("sources")


def preview(intent, expected, paths, native_metadata=None, bindings=None, *, run=None, env=None):
    _texts, values = _load_sources(paths)
    meta = _resolve_metadata(paths, native_metadata, values, run, env)
    current = snapshot(paths, meta, bindings=bindings, run=run, env=env)
    if not _same_revision(current["editor_revision"], expected):
        raise Refusal("agent-settings-source-changed")
    _validate_intent(intent)
    if meta.get("extra_override"):
        raise Refusal("native-config-ambiguous")
    _candidates, affected, aliases = _project(intent, paths, meta, current["settings"])
    return {"affected": affected, "aliases": aliases}


def _hit(name):
    if _FAULT == name:
        raise Refusal("injected-fault")


def _stage_file(parent_fd, name, data, mode):
    tmp = f".{name}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
    jset._write_new(parent_fd, tmp, data, mode)
    return tmp


def _backup_file(parent_fd, name, operation_id, cap):
    bak = f"{name}.bak-{operation_id}"
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode):
        raise Refusal("agent-settings-symlink")
    if not stat.S_ISREG(info.st_mode):
        raise Refusal("agent-settings-not-file")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        raw = os.read(fd, cap + 1)
    finally:
        os.close(fd)
    if len(raw) > cap:
        raise Refusal("agent-settings-too-large")
    jset._write_new(parent_fd, bak, raw, 0o600)
    return bak


def _existing_mode(parent_fd, name, default=0o600):
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return default
    mode = stat.S_IMODE(info.st_mode)
    return mode if mode else default


def _settings_document(intent, operation_id, candidates):
    revision = operation_id.replace("-", "")
    sources = {}
    for name in jset.SOURCE_FILES:
        text = candidates[name]
        sources[name] = "absent" if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()
    doc = {
        "schema_version": 1,
        "revision": revision,
        "reviewers": intent["reviewers"],
        "builders": intent["builders"],
        "source_revisions": sources,
    }
    raw = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    jset.decode_settings(raw)
    return raw, revision, sources


def _effective_from_values(values):
    overlay = values.get("opencode.jsonc")
    if overlay is not None:
        return overlay
    base = values.get("opencode.json")
    return base if base is not None else {}


def publish_candidates(operation_id, paths, candidates, *, affected, expected_sources, expected_settings, settings_spec, run=None, env=None):
    """Shared protected publication sequence (Task 4).

    The caller MUST already hold the configuration claim. `candidates` maps each
    jset.SOURCE_FILES name to its full new text (or None for absent);
    `affected` lists the native files to write. `settings_spec` is None for a
    native-only publish, else {"raw": bytes, "revision": str}. Native files
    publish first and settings last, so a settings-only failure leaves a
    recoverable native result ("activation-pending")."""
    publish_settings = settings_spec is not None
    source_tokens = {
        name: "absent" if candidates.get(name) is None else hashlib.sha256(candidates[name].encode("utf-8")).hexdigest()
        for name in jset.SOURCE_FILES
    }
    _hit("before-stage")
    staged = []
    parent_fds = []
    native_published = False
    settings_published = False
    try:
        for name in affected:
            parent = Path(paths[name]).parent
            leaf = Path(paths[name]).name
            parent_fd = jset._open_directory_nofollow(parent, create=True)
            parent_fds.append(parent_fd)
            jset._require_same_directory(parent_fd, parent)
            data = candidates[name].encode("utf-8")
            mode = _existing_mode(parent_fd, leaf)
            _backup_file(parent_fd, leaf, operation_id, jset.SOURCE_CAP)
            tmp = _stage_file(parent_fd, leaf, data, mode)
            staged.append({"parent_fd": parent_fd, "tmp": tmp, "name": leaf, "kind": "native"})
        if publish_settings:
            settings_parent = Path(paths["settings"]).parent
            settings_leaf = Path(paths["settings"]).name
            settings_fd = jset._open_directory_nofollow(settings_parent, create=True)
            parent_fds.append(settings_fd)
            jset._require_same_directory(settings_fd, settings_parent)
            _backup_file(settings_fd, settings_leaf, operation_id, jset.SETTINGS_CAP)
            settings_tmp = _stage_file(settings_fd, settings_leaf, settings_spec["raw"], 0o600)
            staged.append({"parent_fd": settings_fd, "tmp": settings_tmp, "name": settings_leaf, "kind": "settings"})
        for name in jset.SOURCE_FILES:
            if jset._source_token(Path(paths[name])) != expected_sources[name]:
                raise Refusal("agent-settings-source-changed")
        now_settings = jset.read_settings(Path(paths["settings"]))
        now_rev = now_settings["revision"] if now_settings else "absent"
        if expected_settings is not None and now_rev != expected_settings:
            raise Refusal("agent-settings-source-changed")
        _hit("before-native")
        for item in staged:
            if item["kind"] != "native":
                continue
            os.rename(item["tmp"], item["name"], src_dir_fd=item["parent_fd"], dst_dir_fd=item["parent_fd"])
            os.fsync(item["parent_fd"])
            item["tmp"] = None
            native_published = True
        if publish_settings:
            _hit("before-settings")
        for item in staged:
            if item["kind"] != "settings":
                continue
            os.rename(item["tmp"], item["name"], src_dir_fd=item["parent_fd"], dst_dir_fd=item["parent_fd"])
            os.fsync(item["parent_fd"])
            item["tmp"] = None
            settings_published = True
        _hit("before-readback")
        if not publish_settings:
            return {
                "effect": "published",
                "editor_revision": {"settings": "absent", "sources": source_tokens},
                "settings": None,
            }
        published = jset.read_settings(Path(paths["settings"]))
        if published is None or published["revision"] != settings_spec["revision"]:
            raise Refusal("activation-pending")
        _texts, values = _load_sources(paths)
        effective = _effective_from_values(values)
        jset.resolve_profile(published, "default", effective_config=effective)
        jset.resolve_profile(published, "fallback", effective_config=effective)
        return {
            "effect": "published",
            "editor_revision": {"settings": published["revision"], "sources": source_tokens},
            "settings": published,
        }
    except Refusal:
        if publish_settings and native_published and not settings_published:
            return {"effect": "activation-pending"}
        raise
    finally:
        for item in staged:
            if item["tmp"] is None:
                continue
            try:
                os.unlink(item["tmp"], dir_fd=item["parent_fd"])
            except OSError:
                pass
        for fd in parent_fds:
            os.close(fd)


def apply(intent, expected, operation_id, paths, native_metadata=None, bindings=None, *, run=None, env=None):
    if type(operation_id) is not str or not jset.UUID_RE.fullmatch(operation_id):
        raise Refusal("agent-settings-malformed")
    with jset.configuration_claim(Path(paths["lock"]), create=True):
        _texts, values = _load_sources(paths)
        meta = _resolve_metadata(paths, native_metadata, values, run, env)
        current = snapshot(paths, meta, bindings=bindings, run=run, env=env)
        if not _same_revision(current["editor_revision"], expected):
            raise Refusal("agent-settings-source-changed")
        _validate_intent(intent)
        if meta.get("extra_override"):
            raise Refusal("native-config-ambiguous")
        candidates, affected, _aliases = _project(intent, paths, meta, current["settings"])
        publish_settings = intent.get("kind") == "save-settings" or current["settings"] is not None
        bind_builders = None
        if (
            intent.get("kind") == "bind-credential"
            and current["settings"] is not None
        ):
            bind_builders = {}
            for profile_name in PROFILES:
                row = current["settings"]["builders"][profile_name]
                if row["connection"] != intent["connection"]:
                    bind_builders[profile_name] = row
                    continue
                bind_builders[profile_name] = {**row, "credential": intent["credential"]}
        if publish_settings:
            if intent.get("kind") == "save-settings":
                settings_raw, revision, _source_tokens = _settings_document(intent, operation_id, candidates)
            elif bind_builders is not None:
                settings_raw, revision, _source_tokens = _settings_document({
                    "reviewers": current["settings"]["reviewers"],
                    "builders": bind_builders,
                }, operation_id, candidates)
            else:
                settings_raw, revision, _source_tokens = _settings_document({
                    "reviewers": current["settings"]["reviewers"],
                    "builders": current["settings"]["builders"],
                }, operation_id, candidates)
            settings_spec = {"raw": settings_raw, "revision": revision}
        else:
            settings_spec = None
        return publish_candidates(
            operation_id,
            paths,
            candidates,
            affected=affected,
            expected_sources=expected["sources"],
            expected_settings=expected["settings"],
            settings_spec=settings_spec,
            run=run,
            env=env,
        )


def republish_settings(operation_id, paths, expected, *, run=None, env=None):
    """Settings-only republish for an inventory OpenCode reconciliation (Task 5):
    no native edits, a fresh published revision, and updated source hashes."""
    if type(operation_id) is not str or not jset.UUID_RE.fullmatch(operation_id):
        raise Refusal("agent-settings-malformed")
    with jset.configuration_claim(Path(paths["lock"]), create=True):
        current_settings = jset.read_settings(Path(paths["settings"]))
        if not _same_revision(
            {"settings": current_settings["revision"] if current_settings else "absent",
             "sources": {name: jset._source_token(Path(paths[name])) for name in jset.SOURCE_FILES}},
            expected,
        ):
            raise Refusal("agent-settings-source-changed")
        if current_settings is None:
            raise Refusal("agent-settings-source-changed")
        texts = {name: _read_source_text(paths[name]) for name in jset.SOURCE_FILES}
        settings_raw, revision, _tokens = _settings_document({
            "reviewers": current_settings["reviewers"],
            "builders": current_settings["builders"],
        }, operation_id, texts)
        return publish_candidates(
            operation_id,
            paths,
            texts,
            affected=[],
            expected_sources=expected["sources"],
            expected_settings=expected["settings"],
            settings_spec={"raw": settings_raw, "revision": revision},
            run=run,
            env=env,
        )


def _read_source_text(path):
    raw = _read_source_bytes(path)
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Refusal("native-config-malformed") from exc


def read_credential(binding, paths):
    if type(binding) is not dict or "kind" not in binding:
        raise Refusal("agent-settings-malformed")
    if binding["kind"] == "native":
        return _native_credential(binding, paths)
    jset._credential(binding)
    env = jset.builder_environment({"credential": binding}, {}, env_path=paths["env"])
    return env[binding["env"]]


def _native_credential(binding, paths):
    if type(binding) is not dict or type(binding.get("connection")) is not str:
        raise Refusal("agent-settings-malformed")
    cid = binding["connection"]
    if not jset.CONNECTION_RE.fullmatch(cid):
        raise Refusal("agent-settings-malformed")
    _texts, values = _load_sources(paths)
    native = _source_native(values, cid)
    options = native.get("options") if type(native) is dict else None
    key = options.get("apiKey") if type(options) is dict else None
    name = _env_ref_name(key)
    if name is None:
        if type(key) is str and key:
            return key
        if key is None:
            _status, auth = _read_auth_file(paths)
            row = auth.get(cid) if type(auth) is dict else None
            if _auth_entry(row) == {"auth": "api-key", "status": "present"}:
                return row["key"]
        raise Refusal("agent-settings-malformed")
    table = _protected_env_values(paths, None)
    value = table.get(name) if type(table) is dict else None
    if type(value) is not str or not value:
        raise Refusal("agent-settings-malformed")
    return value


def default_paths():
    data_home = os.environ.get("XDG_DATA_HOME")
    auth_root = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return {
        "settings": jset.SETTINGS_PATH,
        "lock": jset.LOCK_PATH,
        "opencode.json": jset.SOURCE_PATHS["opencode.json"],
        "opencode.jsonc": jset.SOURCE_PATHS["opencode.jsonc"],
        "env": jset.ENV_PATH,
        "auth": auth_root / "opencode" / "auth.json",
    }


def _emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.flush()


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if len(argv) != 2:
        _emit({"ok": False, "error": "agent-settings-malformed"})
        return 0
    op = argv[1]
    raw = sys.stdin.buffer.read(HELPER_IO_CAP + 1)
    if len(raw) > HELPER_IO_CAP:
        _emit({"ok": False, "error": "agent-settings-too-large"})
        return 0
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _emit({"ok": False, "error": "agent-settings-malformed"})
        return 0
    if type(payload) is not dict:
        _emit({"ok": False, "error": "agent-settings-malformed"})
        return 0
    payload.pop("paths", None)
    paths = default_paths()
    bindings = payload.get("bindings")
    try:
        if op == "snapshot":
            data = snapshot(paths, payload.get("native_metadata"), bindings)
        elif op == "preview":
            data = preview(
                payload["intent"], payload["expected"], paths,
                payload.get("native_metadata"), bindings,
            )
        elif op == "apply":
            data = apply(
                payload["intent"],
                payload["expected"],
                payload["operation_id"],
                paths,
                payload.get("native_metadata"),
                bindings,
            )
        elif op == "read-credential":
            data = read_credential(payload["binding"], paths)
        else:
            raise Refusal("agent-settings-malformed")
    except Refusal as exc:
        _emit({"ok": False, "error": exc.code})
        return 0
    except Exception:
        _emit({"ok": False, "error": "agent-settings-malformed"})
        return 0
    _emit({"ok": True, "data": data})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
