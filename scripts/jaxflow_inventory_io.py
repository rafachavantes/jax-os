#!/usr/bin/env python3
"""Finite native inventory helper (spec §7, plan Tasks 3-5).

JSON stdin/stdout only. No shell, no arbitrary path/command/config API: every
target is derived server-side from a fixed managed scope (the collector passes
`roots`; the HTTP route never supplies a path). `snapshot` is read-only and
uses the system interpreter; Codex write actions lazily import the pinned
`tomlkit` (installed in the isolated helper venv, never the system one).
"""
from __future__ import annotations

import hashlib
import ctypes
import errno
import json
import os
import re
import stat
import sys
import uuid
from pathlib import Path

from jax_init import Refusal
import jaxflow_settings as jset
import jaxflow_settings_io as sio
import jaxflow_env as jenv

NATIVE_CAP = 1024 * 1024
INPUT_CAP = 64 * 1024
OUTPUT_CAP = 4 * 1024 * 1024
PROVENANCE_CAP = 32 * 1024
OWNERSHIP_WALK_CAP = 4096
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
LOCAL_PLUGIN_SUFFIXES = (".js", ".ts", ".tsx")
ACTIONS = ("disable", "enable", "remove-registration", "remove-installation")
CLAUDE_OVERRIDE_VALUES = ("on", "name-only", "user-invocable-only", "off")
OPENCODE_POLICY_VALUES = ("allow", "ask", "deny")


def _emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.flush()


def _item_id(executor, kind, registration):
    return hashlib.sha256(f"{executor}\x1f{kind}\x1f{registration}".encode("utf-8")).hexdigest()


def _item(executor, kind, registration, name, **extra):
    item = {
        "id": _item_id(executor, kind, registration),
        "executor": executor,
        "kind": kind,
        "registration": registration,
        "name": name,
        "description": "",
        "path": "",
        "scope": "user",
        "origin": "native",
        "parent": None,
        "canonicalTarget": None,
        "state": "unknown",
        "capabilities": {},
    }
    item.update(extra)
    return item


def _caps(**pairs):
    out = {}
    for action, value in pairs.items():
        if isinstance(value, dict):
            out[action] = value
        elif value is True:
            out[action] = {"available": True, "reason": None}
        elif value is False:
            out[action] = {"available": False, "reason": "unsupported"}
        else:
            out[action] = {"available": False, "reason": value}
    return out


# ---- protected reads ----

def _read_bytes(path):
    path = Path(path)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise Refusal("inventory-permissions") from exc
    if stat.S_ISLNK(info.st_mode):
        raise Refusal("inventory-symlink")
    if not stat.S_ISREG(info.st_mode):
        raise Refusal("inventory-not-file")
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        raw = os.read(fd, NATIVE_CAP + 1)
    finally:
        os.close(fd)
    if len(raw) > NATIVE_CAP:
        raise Refusal("inventory-too-large")
    return raw


def _read_text(path):
    raw = _read_bytes(path)
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Refusal("inventory-format-unsupported") from exc


def _strip_jsonc(text):
    out = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    stripped = "".join(out)
    stripped = re.sub(r",(\s*[}\]])", r"\1", stripped)
    return stripped


def _parse_json(text):
    if text is None:
        return None
    try:
        return json.loads(_strip_jsonc(text))
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise Refusal("inventory-format-unsupported") from exc


def _parse_toml(text):
    if text is None:
        return None
    import tomllib
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise Refusal("inventory-format-unsupported") from exc


def _frontmatter_description(text):
    """Bounded SKILL.md/agent frontmatter description (never executed)."""
    if type(text) is not str:
        return ""
    m = re.match(r"^---\r?\n(.*?)\r?\n---(?:\r?\n|$)", text[:8192], re.DOTALL)
    if not m:
        return ""
    for line in m.group(1).splitlines():
        i = line.find(": ")
        if i > 0 and line[:i].strip() == "description":
            return line[i + 2:].strip()[:512]
    return ""


# ---- roots ----

def _home_roots():
    home = Path.home()
    return {
        "claude_dir": str(home / ".claude"),
        "codex_dir": str(home / ".codex"),
        "opencode_dir": str(home / ".config" / "opencode"),
        "jax_os": str(jenv.jaxos_home()),
    }


_ROOTS_KEYS = ("claude_dir", "codex_dir", "opencode_dir", "jax_os")


def _roots(payload):
    roots = _home_roots()
    supplied = payload.get("roots")
    if supplied is not None:
        if type(supplied) is not dict:
            raise Refusal("inventory-malformed")
        for key in supplied:
            if key not in _ROOTS_KEYS:
                raise Refusal("inventory-malformed")
            value = supplied[key]
            if type(value) is not str or not value.startswith("/"):
                raise Refusal("inventory-malformed")
            roots[key] = value
    return roots


def _settings_paths(roots):
    base = sio.default_paths()
    base["opencode.json"] = Path(roots["opencode_dir"]) / "opencode.json"
    base["opencode.jsonc"] = Path(roots["opencode_dir"]) / "opencode.jsonc"
    base["settings"] = Path(roots["jax_os"]) / "agent-settings.json"
    base["lock"] = Path(roots["jax_os"]) / "agent-settings.lock"
    return base


def _claude_settings_path(roots):
    return Path(roots["claude_dir"]) / "settings.json"


def _codex_config_path(roots):
    return Path(roots["codex_dir"]) / "config.toml"


def _source_bytes(roots, executor):
    """The native source bytes a managed toggle's restore must still match."""
    if executor == "claude":
        return _read_bytes(_claude_settings_path(roots))
    if executor == "codex":
        return _read_bytes(_codex_config_path(roots))
    if executor == "opencode":
        oc = Path(roots["opencode_dir"])
        raw = _read_bytes(oc / "opencode.jsonc")
        return _read_bytes(oc / "opencode.json") if raw is None else raw
    return None


def _managed_disable_digest_from(executor, kind, registration, raw):
    """Digest a checked post-disable candidate (never the pre-disable bytes)."""
    h = hashlib.sha256()
    h.update(b"managed-disable-v1\x00")
    h.update(executor.encode())
    h.update(b"\x00")
    h.update(kind.encode())
    h.update(b"\x00")
    h.update(registration.encode())
    h.update(b"\x00")
    h.update(raw if raw is not None else b"absent")
    return h.hexdigest()


def _managed_disable_digest(roots, executor, kind, registration):
    """Post-disable digest of the relevant native source (restoration binding)."""
    return _managed_disable_digest_from(executor, kind, registration, _source_bytes(roots, executor))


# ---- snapshot per executor ----

def _claude_doc(roots):
    doc = _parse_json(_read_text(_claude_settings_path(roots)))
    return doc if type(doc) is dict else {}


def _claude_plugin_items(doc, path):
    items = []
    enabled = doc.get("enabledPlugins")
    if enabled is not None and type(enabled) is not dict:
        raise Refusal("inventory-format-unsupported")
    for plugin_id, value in (enabled or {}).items():
        if type(value) is not bool:
            raise Refusal("inventory-format-unsupported")
        items.append(_item(
            "claude", "plugin", plugin_id, plugin_id,
            path=str(path), origin="settings",
            state="enabled" if value else "disabled",
            capabilities=_caps(
                disable={"available": value, "reason": None if value else "already-disabled"},
                enable={"available": not value, "reason": None if not value else "already-enabled"},
            ),
        ))
    return items


def _claude_override_items(doc, path):
    items = []
    overrides = doc.get("skillOverrides")
    if overrides is not None and type(overrides) is not dict:
        raise Refusal("inventory-format-unsupported")
    for skill_name, value in (overrides or {}).items():
        if value not in CLAUDE_OVERRIDE_VALUES:
            raise Refusal("inventory-format-unsupported")
        disabled = value == "off"
        items.append(_item(
            "claude", "skill", skill_name, skill_name,
            path=str(path), origin="skill-override",
            state="disabled" if disabled else "restricted",
            capabilities=_caps(
                disable={"available": not disabled, "reason": None if not disabled else "already-disabled"},
                enable={"available": False, "reason": "no-managed-state" if disabled else "already-enabled"},
            ),
        ))
    return items


def _scan_skill_dirs(executor, root, roots, overrides=None):
    """Independent skill installs under <executor>/skills. `overrides` maps a
    registration (canonical SKILL.md path) to its effective override state so
    discovery and policy join into ONE item (no duplicate rows)."""
    items = []
    if not root.is_dir():
        return items
    try:
        children = sorted(root.iterdir(), key=lambda p: p.name)
    except OSError:
        raise Refusal("inventory-permissions") from None
    if len(children) > OWNERSHIP_WALK_CAP:
        raise Refusal("inventory-too-large")
    for child in children:
        if child.name.startswith("."):
            continue
        skill_md = child / "SKILL.md"
        is_link = child.is_symlink()
        if is_link:
            try:
                target = os.readlink(child)
            except OSError:
                target = None
            exists = os.path.exists(child)
            state = "enabled" if exists else "broken"
            caps = {}
            if exists:
                caps["remove-registration"] = {"available": True, "reason": None}
            items.append(_item(
                executor, "skill", str(child), child.name,
                path=str(child), scope="user", origin="registration-link", state=state,
                canonicalTarget=target,
                capabilities=caps,
            ))
            continue
        if not (child.is_dir() and skill_md.is_file()):
            continue
        registration = str(child)
        override = (overrides or {}).get(child.name)  # skillOverrides keys are skill NAMES
        if override == "off":
            state, disabled = "disabled", True
        elif override in ("name-only", "user-invocable-only"):
            state, disabled = "restricted", False
        else:
            state, disabled = "enabled", False
        caps = {
            "disable": {"available": not disabled, "reason": None if not disabled else "already-disabled"},
        }
        if disabled:
            # a managed disable can be inverted; a native one must not offer a fake enable
            caps["enable"] = {"available": False, "reason": "no-managed-state"}
        items.append(_item(
            executor, "skill", registration, child.name,
            path=str(child), scope="user", origin="independent",
            description=_frontmatter_description(_read_text(skill_md) or ""),
            state=state,
            capabilities=_caps(**caps, **{"remove-installation": True}),
        ))
    return items


VERSION_RE = re.compile(r"^\d+(\.\d+){1,3}$")


def _version_key(name):
    return tuple(int(part) for part in name.split(".") if part.isdigit())


def _select_version_dir(dirs):
    """Numeric max when any dir is a dotted numeric version; else the
    lexicographic max excluding "unknown"; else "unknown" if it is the only one."""
    numeric = [d for d in dirs if VERSION_RE.fullmatch(d.name)]
    if numeric:
        return max(numeric, key=lambda d: _version_key(d.name))
    candidates = [d for d in dirs if d.name != "unknown"]
    if candidates:
        return max(candidates, key=lambda d: d.name)
    return dirs[0] if dirs else None


def _claude_cache_index(roots):
    """Claude plugin cache: registration -> (plugin dir, selected version dir)."""
    index = {}
    cache = Path(roots["claude_dir"]) / "plugins" / "cache"
    if not cache.is_dir():
        return index
    try:
        marketplaces = sorted(cache.iterdir(), key=lambda p: p.name)
    except OSError:
        raise Refusal("inventory-permissions") from None
    if len(marketplaces) > OWNERSHIP_WALK_CAP:
        raise Refusal("inventory-too-large")
    for mp in marketplaces:
        if not mp.is_dir() or mp.name.startswith("."):
            continue
        try:
            plugins = sorted(mp.iterdir(), key=lambda p: p.name)
        except OSError:
            raise Refusal("inventory-permissions") from None
        if len(plugins) > OWNERSHIP_WALK_CAP:
            raise Refusal("inventory-too-large")
        for plugin in plugins:
            if not plugin.is_dir() or plugin.name.startswith("."):
                continue
            try:
                versions = sorted(
                    (v for v in plugin.iterdir() if v.is_dir() and not v.name.startswith(".")),
                    key=lambda p: p.name,
                )
            except OSError:
                raise Refusal("inventory-permissions") from None
            if len(versions) > OWNERSHIP_WALK_CAP:
                raise Refusal("inventory-too-large")
            index[f"{plugin.name}@{mp.name}"] = (plugin, _select_version_dir(versions))
    return index


def _claude_bundled_children(roots, version_dir, parent, parent_state):
    """Read-only bundled skills: parent locked to the logical plugin identity."""
    if version_dir is None:
        return []
    items = []
    for skill in _scan_skill_dirs("claude", version_dir / "skills", roots):
        if skill["origin"] != "independent":
            continue
        skill["parent"] = parent
        skill["state"] = "parent-disabled" if parent_state == "disabled" else skill["state"]
        skill["capabilities"] = _caps()  # control lives on the parent, never the child
        items.append(skill)
    return items


def _scan_agent_dir(executor, root):
    """Read-only local subagent definitions (<root>/*.md). Codex has none."""
    items = []
    if not root.is_dir():
        return items
    try:
        children = sorted(root.iterdir(), key=lambda p: p.name)
    except OSError:
        raise Refusal("inventory-permissions") from None
    if len(children) > OWNERSHIP_WALK_CAP:
        raise Refusal("inventory-too-large")
    for child in children:
        if child.name.startswith(".") or not child.name.endswith(".md"):
            continue
        try:
            if not child.is_file():
                continue
        except OSError:
            continue
        items.append(_item(
            executor, "agent", str(child), child.name[:-3] if child.name.endswith(".md") else child.name,
            path=str(child), scope="user", origin="local-agent",
            description=_frontmatter_description(_read_text(child) or ""),
            state="enabled",
            capabilities={},  # read-only: no fabricated controls
        ))
    return items


def _snapshot_claude(roots):
    path = _claude_settings_path(roots)
    doc = _claude_doc(roots)
    items = _claude_plugin_items(doc, path)
    overrides = doc.get("skillOverrides")
    if overrides is not None and type(overrides) is not dict:
        raise Refusal("inventory-format-unsupported")
    items.extend(_claude_override_items(doc, path))
    items.extend(_scan_skill_dirs("claude", Path(roots["claude_dir"]) / "skills", roots, overrides=overrides))
    enabled = doc.get("enabledPlugins") or {}
    # Cache provenance is FOLDED into the configured logical plugin (one stable
    # id); a cache-only plugin stays explicit and read-only.
    for key, (plugin_dir, version_dir) in sorted(_claude_cache_index(roots).items()):
        if key in enabled:
            parent = next((i for i in items if i["origin"] == "settings" and i["registration"] == key), None)
            if parent is None:
                continue
            parent["cachePath"] = str(plugin_dir)
            if version_dir is not None:
                parent["installedVersion"] = version_dir.name
            parent_state = "enabled" if enabled.get(key) is True else "disabled"
        else:
            parent = _item(
                "claude", "plugin", key, Path(key).name if "@" in key else key,
                path=str(plugin_dir), origin="cache", state="unknown", capabilities={},
            )
            items.append(parent)
            parent_state = "disabled"
        items.extend(_claude_bundled_children(roots, version_dir, key, parent_state))
    items.extend(_scan_agent_dir("claude", Path(roots["claude_dir"]) / "agents"))
    return items


def _codex_doc(roots):
    doc = _parse_toml(_read_text(_codex_config_path(roots)))
    return doc if type(doc) is dict else {}


def _codex_skill_entries(doc):
    skills = doc.get("skills")
    return skills.get("config") if type(skills) is dict else None


def _join_codex_dir_skills(items, roots):
    """A Codex skills/ directory joins with its config-path entry into ONE row:
    the directory keeps the installation identity, the entry supplies state."""
    by_path = {
        i["registration"]: i
        for i in items
        if i["kind"] == "skill" and i["origin"] == "config-path"
    }
    joined = []
    consumed = set()
    for item in _scan_skill_dirs("codex", Path(roots["codex_dir"]) / "skills", roots):
        canonical = f"{item['registration']}/SKILL.md"
        entry = by_path.get(canonical) if item["origin"] == "independent" else None
        if entry is not None:
            consumed.add(canonical)
            item["origin"] = "config-path"
            item["canonicalTarget"] = canonical
            item["state"] = entry["state"]
            item["capabilities"] = entry["capabilities"]
        joined.append(item)
    return joined, consumed


def _snapshot_codex(roots):
    path = _codex_config_path(roots)
    doc = _codex_doc(roots)
    items = []
    plugins = doc.get("plugins")
    if plugins is not None and type(plugins) is not dict:
        raise Refusal("inventory-format-unsupported")
    for plugin_id, value in (plugins or {}).items():
        if type(value) is not dict:
            raise Refusal("inventory-format-unsupported")
        enabled = value.get("enabled")
        if enabled is not None and type(enabled) is not bool:
            raise Refusal("inventory-format-unsupported")
        state = "enabled" if enabled is not False else "disabled"
        items.append(_item(
            "codex", "plugin", plugin_id, plugin_id,
            path=str(path), origin="config",
            state=state,
            capabilities=_caps(
                disable={"available": enabled is True, "reason": None if enabled is True else "already-disabled"},
                enable={"available": enabled is False, "reason": None if enabled is False else "already-enabled"},
            ),
        ))
    entries = _codex_skill_entries(doc)
    if entries is not None and type(entries) is not list:
        raise Refusal("inventory-format-unsupported")
    for entry in entries or []:
        if type(entry) is not dict:
            raise Refusal("inventory-format-unsupported")
        path_sel = entry.get("path")
        name_sel = entry.get("name")
        if path_sel is not None and type(path_sel) is not str:
            raise Refusal("inventory-format-unsupported")
        if name_sel is not None and type(name_sel) is not str:
            raise Refusal("inventory-format-unsupported")
        if path_sel is None and name_sel is None:
            continue
        enabled = entry.get("enabled")
        if enabled is not None and type(enabled) is not bool:
            raise Refusal("inventory-format-unsupported")
        if type(path_sel) is str:
            canonical = path_sel
            origin = "config-path"
            # absent enabled = native default (installed/enabled), NOT disabled
            state = "enabled" if enabled is not False else "disabled"
            caps = _caps(
                disable={"available": enabled is not False, "reason": None if enabled is not False else "already-disabled"},
                enable={"available": False, "reason": "no-managed-state"} if enabled is False else
                       {"available": enabled is None, "reason": None if enabled is None else "already-enabled"},
            )
            items.append(_item(
                "codex", "skill", canonical, Path(canonical).name or canonical,
                path=canonical, origin=origin, canonicalTarget=canonical,
                state=state, capabilities=caps,
            ))
        else:
            items.append(_item(
                "codex", "skill", name_sel, name_sel,
                path="", origin="config-name",
                state="enabled" if enabled is not False else "disabled",
                capabilities=_caps(),  # name-selector entries: no path-scoped control
            ))
    joined, consumed = _join_codex_dir_skills(items, roots)
    items = [
        i for i in items
        if not (i["kind"] == "skill" and i["origin"] == "config-path" and i["registration"] in consumed)
    ]
    items.extend(joined)
    items.extend(_scan_agent_dir("codex", Path(roots["codex_dir"]) / "agents"))
    out = {"ok": True, "items": items, "unsupported": ["agent"]}
    return out


def _opencode_doc(roots):
    base_path = Path(roots["opencode_dir"]) / "opencode.json"
    overlay_path = Path(roots["opencode_dir"]) / "opencode.jsonc"
    overlay_text = _read_text(overlay_path)
    base_text = _read_text(base_path)
    items = []
    if overlay_text is not None:
        doc = _parse_json(overlay_text)
        source_path = overlay_path
        present_sources = ["opencode.jsonc"]
    elif base_text is not None:
        doc = _parse_json(base_text)
        source_path = base_path
        present_sources = ["opencode.json"]
    else:
        doc = {}
        source_path = base_path
        present_sources = []
    if type(doc) is not dict:
        raise Refusal("inventory-format-unsupported")
    return doc, source_path, present_sources


def _opencode_skill_policy(doc):
    permission = doc.get("permission")
    if type(permission) is dict:
        return permission.get("skill")
    return None


def _opencode_source_name(roots, item):
    """The OpenCode config file that carries this item's policy (jsonc wins)."""
    name = Path(item["path"]).name if item.get("path") else ""
    if name in ("opencode.json", "opencode.jsonc"):
        return name
    if (Path(roots["opencode_dir"]) / "opencode.jsonc").exists():
        return "opencode.jsonc"
    return "opencode.json"


def _snapshot_opencode(roots):
    doc, source_path, present_sources = _opencode_doc(roots)
    items = []
    plugins = doc.get("plugin")
    if plugins is not None and type(plugins) is not list:
        raise Refusal("inventory-format-unsupported")
    counts = {}
    for entry in plugins or []:
        if type(entry) is not str:
            raise Refusal("inventory-format-unsupported")
        counts[entry] = counts.get(entry, 0) + 1
    for entry, count in counts.items():
        unique = count == 1
        items.append(_item(
            "opencode", "plugin", entry, entry,
            path=str(source_path), origin="config-array",
            state="enabled",
            capabilities=_caps(**{"remove-registration": {
                "available": unique,
                "reason": None if unique else "duplicate-registration",
            }}),
        ))
    skill_policy = _opencode_skill_policy(doc)
    if type(skill_policy) is dict:
        for skill_name, value in skill_policy.items():
            if skill_name == "*":
                continue
            if type(value) is not str or value not in OPENCODE_POLICY_VALUES:
                continue
            state = {"allow": "enabled", "ask": "restricted", "deny": "disabled"}[value]
            items.append(_item(
                "opencode", "skill", skill_name, skill_name,
                path=str(source_path), origin="permission-map",
                state=state,
                capabilities=_caps(
                    disable={"available": value != "deny", "reason": None if value != "deny" else "already-disabled"},
                    enable={"available": False, "reason": "no-managed-state"},
                ),
            ))
    items.extend(_scan_skill_dirs("opencode", Path(roots["opencode_dir"]) / "skills", roots))
    items.extend(_scan_agent_dir("opencode", Path(roots["opencode_dir"]) / "agents"))
    items.extend(_scan_local_plugins(roots, source_path))
    return items, present_sources


def _scan_local_plugins(roots, source_path):
    items = []
    root = Path(roots["opencode_dir"]) / "plugin"
    if not root.is_dir():
        return items
    try:
        children = sorted(root.iterdir(), key=lambda p: p.name)
    except OSError:
        raise Refusal("inventory-permissions") from None
    if len(children) > OWNERSHIP_WALK_CAP:
        raise Refusal("inventory-too-large")
    for child in children:
        if child.is_symlink() or not child.is_file() or child.suffix not in LOCAL_PLUGIN_SUFFIXES:
            continue
        items.append(_item(
            "opencode", "plugin", str(child), child.name,
            path=str(child), scope="user", origin="local-file", state="enabled",
            capabilities=_caps(**{"remove-installation": {"available": True, "reason": None}}),
        ))
    return items


def snapshot(roots):
    executors = {}
    for name, fn in (("claude", _snapshot_claude), ("codex", _snapshot_codex), ("opencode", _snapshot_opencode)):
        try:
            data = fn(roots)
            if isinstance(data, tuple):
                bucket = {"ok": True, "items": data[0]}
                if data[1]:
                    bucket["loadingSources"] = data[1]
            else:
                bucket = data if isinstance(data, dict) else {"ok": True, "items": data}
            executors[name] = bucket
        except Refusal as exc:
            executors[name] = {"ok": False, "error": exc.code, "items": []}
    return {"executors": executors}


# ---- preview / apply ----

def _find_item(roots, item_id):
    if type(item_id) is not str or not HEX64_RE.fullmatch(item_id):
        raise Refusal("inventory-malformed")
    snap = snapshot(roots)
    unavailable = None
    for executor, bucket in snap["executors"].items():
        if not bucket["ok"]:
            # One unrelated failed bucket must never block a healthy bucket's
            # items; remember it and keep scanning the readable ones.
            unavailable = bucket["error"]
            continue
        for item in bucket["items"]:
            if item["id"] == item_id:
                return item, executor
    if unavailable is not None:
        # No readable bucket holds the item and an owner cannot be proven:
        # uncertainty, never authoritative absence.
        raise Refusal("inventory-source-unavailable")
    raise Refusal("inventory-item-unknown")


def _path_fingerprint(path):
    path = Path(path)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return b"absent"
    except OSError as exc:
        raise Refusal("inventory-permissions") from exc
    if stat.S_ISLNK(info.st_mode):
        return b"link:" + os.readlink(path).encode("utf-8")
    if stat.S_ISREG(info.st_mode):
        return _read_bytes(path) or b""
    if stat.S_ISDIR(info.st_mode):
        h = hashlib.sha256()
        h.update(b"dir\x00")
        # identity facts: a replaced directory (new inode/dev) must change the
        # revision even when its child names happen to match.
        h.update(str(info.st_ino).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(info.st_dev).encode("utf-8"))
        h.update(b"\x00")
        try:
            children = sorted(path.iterdir(), key=lambda p: p.name)
        except OSError as exc:
            raise Refusal("inventory-permissions") from exc
        if len(children) > OWNERSHIP_WALK_CAP:
            raise Refusal("inventory-too-large")
        for child in children:
            h.update(child.name.encode("utf-8"))
            h.update(b"\x00")
            try:
                cinfo = child.lstat()
            except OSError:
                h.update(b"gone\x00")
                continue
            h.update(str(cinfo.st_ino).encode("utf-8"))
            h.update(b"\x00")
            h.update(str(cinfo.st_mode).encode("utf-8"))
            h.update(b"\x00")
            h.update(str(cinfo.st_size).encode("utf-8"))
            h.update(b"\x00")
        return h.digest()
    raise Refusal("inventory-not-file")


def _executor_source_path(roots, executor):
    if executor == "claude":
        return _claude_settings_path(roots)
    if executor == "codex":
        return _codex_config_path(roots)
    if executor == "opencode":
        oc = Path(roots["opencode_dir"])
        jsonc = oc / "opencode.jsonc"
        return jsonc if jsonc.exists() else oc / "opencode.json"
    return None


def _revision_with_source(roots, item, source_bytes):
    """Identity digest over every fact the action relies on: the item path AND
    the relevant native source bytes (an independent skill dir carries no
    policy state). `source_bytes=None` means the source is absent/unread."""
    path = Path(item["path"]) if item["path"] else None
    source_path = _executor_source_path(roots, item["executor"])
    if path is not None and source_path is not None and path == source_path:
        raw = source_bytes if source_bytes is not None else b"absent"
    else:
        raw = _path_fingerprint(path) if path else b"absent"
    h = hashlib.sha256()
    h.update(b"inventory-v2\x00")
    h.update(item["executor"].encode())
    h.update(b"\x00")
    h.update(item["kind"].encode())
    h.update(b"\x00")
    h.update(item["registration"].encode())
    h.update(b"\x00")
    h.update(raw)
    h.update(b"\x00")
    h.update(source_bytes if source_bytes is not None else b"absent")
    return h.hexdigest()


def _revision_for(roots, item):
    return _revision_with_source(roots, item, _source_bytes(roots, item["executor"]))


def _claude_edit_text(root_text, doc, item, change, provenance):
    if root_text is None:
        raise Refusal("inventory-format-unsupported")
    if item["kind"] == "plugin" and item["origin"] == "settings":
        plugins = doc.get("enabledPlugins")
        if type(plugins) is not dict or type(plugins.get(item["registration"])) is not bool:
            raise Refusal("inventory-item-unknown")
        target_enabled = change == "enable"
        if plugins[item["registration"]] == target_enabled:
            raise Refusal("inventory-already-settled")
        edits = [{"path": ["enabledPlugins", item["registration"]], "value": target_enabled}]
        return sio.edit_source(root_text, edits, jsonc=False)["text"], {"prior": plugins[item["registration"]]}
    if item["kind"] == "skill" and item["origin"] == "skill-override":
        overrides = doc.get("skillOverrides")
        if type(overrides) is not dict or item["registration"] not in overrides:
            raise Refusal("inventory-item-unknown")
        prior = overrides[item["registration"]]
        if change == "disable":
            new_value = "off"
        elif change == "enable":
            prov = provenance if type(provenance) is dict else {}
            if prov.get("selector") != item["registration"]:
                raise Refusal("inventory-no-managed-state")
            if "prior" not in prov:
                raise Refusal("inventory-no-managed-state")
            prior = prov.get("prior")
            if prov.get("priorPresent") is False or prior is None:
                return sio.edit_source(root_text, [{"path": ["skillOverrides", item["registration"]], "remove": True}], jsonc=False)["text"], {
                    "prior": None, "priorPresent": False,
                }
            if prior not in CLAUDE_OVERRIDE_VALUES:
                raise Refusal("inventory-no-managed-state")
            return sio.edit_source(root_text, [{"path": ["skillOverrides", item["registration"]], "value": prior}], jsonc=False)["text"], {
                "prior": prior, "priorPresent": True,
            }
        else:
            raise Refusal("inventory-unsupported")
        return sio.edit_source(root_text, [{"path": ["skillOverrides", item["registration"]], "value": new_value}], jsonc=False)["text"], {"prior": prior}
    raise Refusal("inventory-unsupported")


def _codex_edit_text(root_text, item, change, provenance):
    import tomlkit
    try:
        doc = tomlkit.parse(root_text)
    except Exception as exc:
        raise Refusal("inventory-format-unsupported") from exc
    if tomlkit.dumps(doc) != root_text:
        raise Refusal("inventory-format-unsupported")
    if item["kind"] == "plugin":
        plugins = doc.get("plugins")
        if plugins is None:
            raise Refusal("inventory-item-unknown")
        entry = plugins.get(item["registration"])
        if entry is None or not isinstance(entry, dict) or not isinstance(entry.get("enabled"), bool):
            raise Refusal("inventory-item-unknown")
        desired = change == "enable"
        prior = entry["enabled"]
        entry["enabled"] = desired
        return tomlkit.dumps(doc), {"prior": prior, "priorPresent": True}
    if item["kind"] == "skill":
        canonical = _codex_skill_path(item)
        skills = doc.get("skills")
        if skills is not None and not isinstance(skills, dict):
            raise Refusal("inventory-format-unsupported")
        config = skills.get("config") if isinstance(skills, dict) else None
        if config is not None and not isinstance(config, list):
            raise Refusal("inventory-format-unsupported")
        target = next((e for e in (config or []) if isinstance(e, dict) and e.get("path") == canonical), None)
        prov = provenance if type(provenance) is dict else {}
        if change == "disable":
            if target is None:
                # Jax creates the exact path entry; restore removes it whole.
                if prov.get("selector") not in (item["registration"], canonical):
                    raise Refusal("inventory-no-managed-state")
                had_root = isinstance(skills, dict)
                had_container = had_root and isinstance(config, list)
                skills_tbl = skills if had_root else tomlkit.table()
                if not had_root:
                    doc["skills"] = skills_tbl
                if not had_container:
                    skills_tbl["config"] = tomlkit.aot()
                entry = tomlkit.table()
                entry["path"] = canonical
                entry["enabled"] = False
                skills_tbl["config"].append(entry)
                return tomlkit.dumps(doc), {
                    "prior": None, "priorPresent": False,
                    "createdEntry": True,
                    "createdContainer": not had_container,
                    "createdRoot": not had_root,
                }
            prior = None if prov.get("createdEntry") is True else target.get("enabled")
            target["enabled"] = False
            return tomlkit.dumps(doc), {
                # the canonical selector is REQUIRED by the restore path
                "selector": canonical, "prior": prior, "priorPresent": prior is not None,
                "createdEntry": False,
            }
        # enable: exact prior-policy restoration from the recorded provenance
        if prov.get("selector") not in (item["registration"], canonical) or "prior" not in prov:
            raise Refusal("inventory-no-managed-state")
        if target is None:
            raise Refusal("inventory-item-unknown")
        if target.get("enabled") is not False:
            raise Refusal("inventory-source-changed")
        prior = prov.get("prior")
        if prov.get("priorPresent") is False or prior is None:
            # Jax created this entry (or the field was absent): remove the field or
            # the whole Jax-created entry, retaining every unrelated field.
            if prov.get("createdEntry") is True:
                skills_tbl = doc.get("skills")
                if isinstance(skills_tbl, dict):
                    entries = skills_tbl.get("config")
                    if isinstance(entries, list):
                        entries.remove(target)
                        if len(entries) == 0 and prov.get("createdContainer") is True:
                            del skills_tbl["config"]
                        if len(skills_tbl) == 0 and prov.get("createdRoot") is True:
                            del doc["skills"]
            else:
                target.remove("enabled") if hasattr(target, "remove") else target.pop("enabled", None)
            return tomlkit.dumps(doc), {"prior": None, "priorPresent": False}
        target["enabled"] = prior
        return tomlkit.dumps(doc), {"prior": prior, "priorPresent": True}
    raise Refusal("inventory-unsupported")


def _codex_skill_path(item):
    """Canonical config-path selector for a Codex skill item."""
    registration = item["registration"]
    if registration.endswith("/SKILL.md"):
        return registration
    return f"{registration}/SKILL.md"


class _ProtectedWriter:
    """Single-file protected publication under an already-held claim."""

    def __init__(self, path, operation_id):
        self.path = Path(path)
        self.operation_id = operation_id

    def write(self, text, expected_raw):
        parent = self.path.parent
        leaf = self.path.name
        parent_fd = jset._open_directory_nofollow(parent, create=True)
        try:
            jset._require_same_directory(parent_fd, parent)
            current = _read_bytes(self.path)
            if current != expected_raw:
                raise Refusal("inventory-source-changed")
            sio._backup_file(parent_fd, leaf, self.operation_id, NATIVE_CAP)
            mode = sio._existing_mode(parent_fd, leaf)
            tmp = sio._stage_file(parent_fd, leaf, text.encode("utf-8"), mode)
            try:
                os.rename(tmp, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                os.fsync(parent_fd)
                tmp = None
            finally:
                if tmp is not None:
                    try:
                        os.unlink(tmp, dir_fd=parent_fd)
                    except OSError:
                        pass
        finally:
            os.close(parent_fd)


def _check_symlink_free_tree(root, cap=None):
    """Complete ownership walk: no symlinked ancestor/descendant, no hard-linked
    file, bounded entries. Refuses inability to prove ownership."""
    root = Path(root)
    limit = OWNERSHIP_WALK_CAP if cap is None else cap
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise Refusal("inventory-permissions") from exc
    if not (stat.S_ISDIR(root_info.st_mode) or stat.S_ISREG(root_info.st_mode)):
        raise Refusal("inventory-not-file")
    seen = 0
    dir_inos = set()
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            info = current.lstat()
        except OSError as exc:
            raise Refusal("inventory-permissions") from exc
        seen += 1
        if seen > limit:
            raise Refusal("inventory-too-large")
        if stat.S_ISLNK(info.st_mode):
            raise Refusal("inventory-symlink")
        if info.st_uid != os.getuid():
            raise Refusal("inventory-permissions")
        if stat.S_ISDIR(info.st_mode):
            key = (info.st_dev, info.st_ino)
            if key in dir_inos:
                raise Refusal("inventory-shared-origin")
            dir_inos.add(key)
            try:
                children = sorted(current.iterdir(), key=lambda p: p.name)
            except OSError as exc:
                raise Refusal("inventory-permissions") from exc
            stack.extend(reversed(children))
        elif stat.S_ISREG(info.st_mode):
            if info.st_nlink > 1:
                raise Refusal("inventory-shared-origin")
        else:
            raise Refusal("inventory-not-file")


def _check_no_known_reference(roots, target):
    """Refuse when another known registration/entry references the target
    (alias symlink, second config entry, plugin-cache origin)."""
    target = Path(target)
    known_dirs = [
        Path(roots["claude_dir"]) / "skills",
        Path(roots["claude_dir"]) / "plugins" / "cache",
        Path(roots["codex_dir"]) / "skills",
        Path(roots["opencode_dir"]) / "skills",
        Path(roots["opencode_dir"]) / "plugin",
    ]
    real = os.path.realpath(target)
    for base in known_dirs:
        if not base.is_dir():
            continue
        try:
            children = sorted(base.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        if len(children) > OWNERSHIP_WALK_CAP:
            raise Refusal("inventory-too-large")
        for child in children:
            if child == target:
                continue
            try:
                if child.is_symlink() and os.path.realpath(child) == real:
                    raise Refusal("inventory-shared-origin")
            except OSError:
                continue


def _check_outside_managed_scope(roots, target):
    """Target must sit inside one of the managed installation roots."""
    target = Path(target).resolve()
    scopes = [
        (Path(roots["claude_dir"]) / "skills").resolve(),
        (Path(roots["codex_dir"]) / "skills").resolve(),
        (Path(roots["opencode_dir"]) / "plugin").resolve(),
    ]
    if not any(target == scope or scope in target.parents for scope in scopes):
        raise Refusal("inventory-outside-scope")


def _ownership_proof(roots, target):
    """Complete read-only ownership/reference proof, shared by preview and the
    quarantine apply so a preview can never describe an effect apply refuses."""
    _check_outside_managed_scope(roots, target)
    _check_no_known_reference(roots, target)
    _check_symlink_free_tree(target)


def _quarantine(roots, item, operation_id, expected_revision):
    target = Path(item["path"])
    try:
        info = target.lstat()
    except OSError as exc:
        raise Refusal("inventory-permissions") from exc
    if stat.S_ISLNK(info.st_mode):
        raise Refusal("inventory-symlink")
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        raise Refusal("inventory-not-file")
    _ownership_proof(roots, target)
    dest_dir = Path(roots["jax_os"]) / "removed-tools" / operation_id
    dest = dest_dir / target.name
    with _apply_inventory_claim(roots):
        # Re-resolve identity/revision and re-run the COMPLETE ownership proof
        # UNDER the claim: a file added at claim entry (e.g. a symlink) or a
        # replaced target must refuse before any move. The pre-claim proof is
        # only preview parity.
        if _revision_for(roots, item) != expected_revision:
            raise Refusal("inventory-source-changed")
        _ownership_proof(roots, target)
        # exact target identity immediately before the descriptor-relative move
        try:
            now = target.lstat()
        except OSError as exc:
            raise Refusal("inventory-permissions") from exc
        if (now.st_ino, now.st_dev, now.st_mode) != (info.st_ino, info.st_dev, info.st_mode):
            raise Refusal("inventory-source-changed")
        parent_fd = jset._open_directory_nofollow(target.parent, create=False)
        try:
            jset._require_same_directory(parent_fd, target.parent)
            removed_parent = jset._open_directory_nofollow(dest_dir.parent, create=True)
            try:
                try:
                    os.mkdir(dest_dir.name, 0o700, dir_fd=removed_parent)
                except FileExistsError as exc:
                    raise Refusal("inventory-destination-exists") from exc
                jset._require_same_directory(removed_parent, dest_dir.parent)
                dest_fd = jset._open_directory_nofollow(dest_dir, create=False)
                try:
                    jset._require_same_directory(dest_fd, dest_dir)
                    jset._require_same_directory(parent_fd, target.parent)
                    libc = ctypes.CDLL(None, use_errno=True)
                    renameat2 = getattr(libc, "renameat2", None)
                    if renameat2 is None:
                        raise Refusal("inventory-atomic-move-unsupported")
                    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
                    renameat2.restype = ctypes.c_int
                    encoded_name = os.fsencode(target.name)
                    if renameat2(parent_fd, encoded_name, dest_fd, encoded_name, 1) != 0:
                        code = ctypes.get_errno()
                        if code == errno.EXDEV:
                            raise Refusal("inventory-cross-device")
                        if code in (errno.EEXIST, errno.ENOTEMPTY):
                            raise Refusal("inventory-destination-exists")
                        raise Refusal("inventory-permissions")
                    os.fsync(dest_fd)
                except OSError as exc:
                    raise Refusal("inventory-permissions") from exc
                finally:
                    os.close(dest_fd)
                os.fsync(parent_fd)
                os.fsync(removed_parent)
            finally:
                os.close(removed_parent)
        finally:
            os.close(parent_fd)
    return str(dest)


def _link_inode(path):
    info = os.lstat(path)
    if not stat.S_ISLNK(info.st_mode):
        raise Refusal("inventory-source-changed")
    return (info.st_ino, info.st_dev)


_CAPABILITY_REFUSALS = {
    "already-disabled": "inventory-already-settled",
    "already-enabled": "inventory-already-settled",
    "no-managed-state": "inventory-no-managed-state",
    "unsupported": "inventory-unsupported",
    "duplicate-registration": "inventory-unsupported",
}


def _require_capability(item, change):
    """Shared finite eligibility: an action the item cannot perform never gets a
    user-visible description."""
    cap = item.get("capabilities", {}).get(change)
    if cap and cap.get("available") is True:
        return
    reason = cap.get("reason") if cap else None
    raise Refusal(_CAPABILITY_REFUSALS.get(reason, "inventory-unsupported"))


def _require_enable(item, roots, provenance):
    cap = item.get("capabilities", {}).get("enable")
    if cap and cap.get("available") is True:
        return
    prov = provenance if type(provenance) is dict else {}
    reason = cap.get("reason") if cap else None
    if item["kind"] != "skill":
        raise Refusal(_CAPABILITY_REFUSALS.get(reason, "inventory-unsupported"))
    if not prov.get("selector") and "container" not in prov:
        raise Refusal(_CAPABILITY_REFUSALS.get(reason, "inventory-no-managed-state"))
    if prov.get("digest") is not None and _managed_disable_digest(roots, item["executor"], "skill", item["registration"]) != prov["digest"]:
        raise Refusal("inventory-source-changed")


def preview(roots, item_id, change, provenance=None):
    if change not in ACTIONS:
        raise Refusal("inventory-unsupported")
    item, _executor = _find_item(roots, item_id)
    revision = _revision_for(roots, item)
    effect = None
    recovery = None
    requires_new_session = True
    restoration = None
    targets = [item["path"]] if item["path"] else []
    if change in ("disable", "enable"):
        if change == "disable":
            _require_capability(item, "disable")
        else:
            _require_enable(item, roots, provenance)
        if item["kind"] == "plugin":
            effect = f"{change} plugin {item['name']}"
            if change == "disable":
                restoration = {"selector": item["registration"], "prior": item["state"] != "disabled", "priorPresent": True}
        else:
            effect = f"{change} skill {item['name']}"
            if change == "disable":
                restoration = _disable_restoration(roots, item)
            else:
                restoration = _enable_restoration(roots, item)
    elif change == "remove-registration":
        if item["origin"] != "registration-link" and not (item["executor"] == "opencode" and item["origin"] == "config-array"):
            raise Refusal("inventory-unsupported")
        _require_capability(item, "remove-registration")
        effect = f"remove registration {item['name']} (files remain)"
        recovery = None
        requires_new_session = False
    elif change == "remove-installation":
        if item["origin"] not in ("independent", "local-file"):
            raise Refusal("inventory-unsupported")
        _ownership_proof(roots, item["path"])
        effect = f"quarantine installation {item['name']} (recoverable)"
        recovery = str(Path(roots["jax_os"]) / "removed-tools" / "<operation>" / Path(item["path"]).name)
        requires_new_session = False
    return {
        "itemId": item_id,
        "change": change,
        "revision": revision,
        "targets": targets,
        "effect": effect,
        "recovery": recovery,
        "requiresNewSession": requires_new_session,
        "restoration": restoration,
    }


def _disable_restoration(roots, item):
    """Capture the exact prior policy + created-container flags for a managed disable."""
    executor, registration = item["executor"], item["registration"]
    if executor == "claude":
        if item["origin"] == "skill-override":
            doc = _claude_doc(roots)
            prior = (doc.get("skillOverrides") or {}).get(registration)
            return {"selector": registration, "prior": prior, "priorPresent": prior is not None}
        if item["origin"] == "independent":
            doc = _claude_doc(roots)
            overrides = doc.get("skillOverrides")
            present = type(overrides) is dict and registration in overrides
            prior = overrides.get(registration) if present else None
            # the disable writes the skill name INTO skillOverrides keyed by the skill NAME
            return {"selector": item["name"], "prior": prior, "priorPresent": present}
        return None
    if executor == "codex":
        if item["origin"] in ("config-path", "independent"):
            canonical = _codex_skill_path(item)
            doc = _codex_doc(roots)
            entries = _codex_skill_entries(doc) or []
            target = next((e for e in entries if type(e) is dict and e.get("path") == canonical), None)
            if target is None:
                skills_tbl = doc.get("skills")
                return {
                    "selector": canonical, "prior": None, "priorPresent": False,
                    "createdEntry": True,
                    "createdContainer": not (type(skills_tbl) is dict and "config" in skills_tbl),
                    "createdRoot": type(doc.get("skills")) is not dict,
                }
            return {"selector": canonical, "prior": target.get("enabled"), "priorPresent": "enabled" in target}
        return None
    if executor == "opencode":
        doc, _path, _sources = _opencode_doc(roots)
        policy = _opencode_skill_policy(doc)
        permission = doc.get("permission")
        name = item["name"]
        if type(policy) is str:
            return {"container": "skill-scalar", "prior": policy}
        if type(permission) is str:
            return {"container": "root-scalar", "prior": permission}
        if type(policy) is dict:
            prior = policy.get(name)
            position = list(policy).index(name) if name in policy else None
            return {"selector": name, "prior": prior, "priorPresent": prior is not None, "position": position}
        return {"selector": name, "prior": None, "priorPresent": False}
    return None


def _enable_restoration(roots, item):
    """Reconstruct the prior policy a managed disable recorded (the stored
    provenance is passed by the route on apply; preview only reflects it)."""
    return None


def _apply_inventory_claim(roots):
    return jset.configuration_claim(Path(roots["jax_os"]) / "inventory.lock", create=True)


def apply(roots, item_id, change, expected_revision, operation_id, provenance=None):
    if change not in ACTIONS:
        raise Refusal("inventory-unsupported")
    if type(operation_id) is not str or not jset.UUID_RE.fullmatch(operation_id):
        raise Refusal("inventory-malformed")
    if type(expected_revision) is not str or not HEX64_RE.fullmatch(expected_revision):
        raise Refusal("inventory-malformed")
    item, executor = _find_item(roots, item_id)
    before = _revision_for(roots, item)
    result = _apply_effect(roots, item, executor, change, expected_revision, operation_id, provenance)
    if type(result) is dict:
        result.setdefault("beforeDigest", before)
        result.setdefault("candidateDigest", _current_revision(roots, item_id))
    return result


def _current_revision(roots, item_id):
    """Revision of the item's actual native state AFTER the effect; 'absent' when
    discovery no longer resolves it (a genuine removal)."""
    try:
        item, _executor = _find_item(roots, item_id)
    except Refusal as exc:
        if exc.code == "inventory-item-unknown":
            return "absent"
        return None  # a failed source read never proves the effect
    return _revision_for(roots, item)


def _current_settings_revision(roots):
    paths = _settings_paths(roots)
    settings = jset.read_settings(Path(paths["settings"]))
    sources = {
        name: jset._source_token(Path(paths[name]))
        for name in jset.SOURCE_FILES
    }
    return {"settings": settings["revision"] if settings else "absent", "sources": sources}


def _published_settings_revision(roots):
    """The settings document's own recorded state: its revision and the native
    source hashes it claims to have published. Readback compares THESE against
    the recorded candidate so a native-only change is not mistaken for applied."""
    settings = jset.read_settings(Path(_settings_paths(roots)["settings"]))
    if settings is None:
        return {"settings": "absent", "sources": {}}
    return {"settings": settings["revision"], "sources": dict(settings["source_revisions"])}


def _candidate_edit(roots, executor, item, change, provenance):
    """The SAME pure candidate-edit logic apply uses; never a second generator."""
    if executor == "claude":
        root_text = _read_text(_claude_settings_path(roots))
        doc = _parse_json(root_text)
        if item["kind"] == "skill" and item["origin"] in ("independent", "skill-override"):
            return _claude_skill_text(root_text, doc, item, change, provenance, roots)
        return _claude_edit_text(root_text, doc, item, change, provenance)
    if executor == "codex":
        raw = _read_bytes(_codex_config_path(roots))
        root_text = raw.decode("utf-8") if raw is not None else ""
        text, prov = _codex_edit_text(root_text, item, change, provenance)
        if change == "disable" and item["kind"] == "skill" and type(prov) is dict:
            prov = {**prov, "digest": _managed_disable_digest_from(
                "codex", "skill", item["registration"], text.encode("utf-8")
            )}
        return text, prov
    if executor == "opencode":
        paths = _settings_paths(roots)
        texts, values = sio._load_sources(paths)
        target_name = _opencode_source_name(roots, item)
        value = values.get(target_name)
        if type(value) is not dict:
            raise Refusal("inventory-source-changed")
        if item["kind"] == "skill":
            return _opencode_skill_edit_text(texts[target_name], value, item, change, provenance)
        if change == "remove-registration" and item["origin"] == "config-array":
            # SAME candidate logic the real apply uses (never a second generator).
            edits = [{"path": ["plugin"], "value": _plugin_array_without(value, item["registration"])}]
            return sio.edit_source(texts[target_name], edits, jsonc=target_name.endswith("jsonc"))["text"], None
    raise Refusal("inventory-unsupported")


def _opencode_candidate_settings(roots, candidate_text, target_name, operation_id):
    """Complete post-native OpenCode settings revision: the candidate source
    hashes the publisher will write, plus the revision that candidate publication
    mints (the operation identity), before any native write."""
    paths = _settings_paths(roots)
    texts, _values = sio._load_sources(paths)
    candidates = dict(texts)
    candidates[target_name] = candidate_text
    sources = {
        name: "absent" if candidates.get(name) is None else hashlib.sha256(candidates[name].encode("utf-8")).hexdigest()
        for name in jset.SOURCE_FILES
    }
    if type(operation_id) is str and jset.UUID_RE.fullmatch(operation_id):
        settings_revision = operation_id.replace("-", "")
    else:
        settings_revision = _current_settings_revision(roots)["settings"]
    return {"settings": settings_revision, "sources": sources}


def prepare(roots, item_id, change, expected_revision, provenance=None, operation_id=None):
    """Finite READ-ONLY preparation: eligibility, the checked before/candidate
    digests and the bounded restoration record, all derived before any native
    write. The browser never receives the protected restoration record."""
    if change not in ACTIONS:
        raise Refusal("inventory-unsupported")
    if type(expected_revision) is not str or not HEX64_RE.fullmatch(expected_revision):
        raise Refusal("inventory-malformed")
    item, executor = _find_item(roots, item_id)
    before = _revision_for(roots, item)
    if before != expected_revision:
        raise Refusal("inventory-source-changed")
    persists = change in ("disable", "enable")
    prov_out = provenance if type(provenance) is dict else None
    if change == "disable":
        _require_capability(item, "disable")
    elif change == "enable":
        _require_enable(item, roots, provenance)
    elif change == "remove-registration":
        if item["origin"] != "registration-link" and not (executor == "opencode" and item["origin"] == "config-array"):
            raise Refusal("inventory-unsupported")
        _require_capability(item, "remove-registration")
        persists = False
    elif change == "remove-installation":
        if item["origin"] not in ("independent", "local-file"):
            raise Refusal("inventory-unsupported")
        _ownership_proof(roots, item["path"])
        persists = False
    # Candidate text comes from the SAME checked candidate-edit logic apply uses:
    # managed toggles always, and an OpenCode plugin removal (which also republishes
    # settings), so the post-native settings revision is known BEFORE any write.
    candidate_text = None
    needs_candidate = persists or (executor == "opencode" and change == "remove-registration")
    if needs_candidate:
        candidate_text, prov = _candidate_edit(roots, executor, item, change, provenance)
        if persists:
            prov_out = prov
    candidate_bytes = candidate_text.encode("utf-8") if candidate_text is not None else None
    candidate_digest = _revision_with_source(roots, item, candidate_bytes) if persists else "absent"
    settings_expected = _current_settings_revision(roots) if executor == "opencode" else None
    settings_candidate = None
    if executor == "opencode" and candidate_text is not None:
        settings_candidate = _opencode_candidate_settings(roots, candidate_text, _opencode_source_name(roots, item), operation_id)
    return {
        "itemId": item_id,
        "change": change,
        "executor": executor,
        "selector": item["registration"],
        "target": item["path"],
        "beforeDigest": before,
        "candidateDigest": candidate_digest,
        "provenance": prov_out,
        "settingsExpected": settings_expected,
        "settingsCandidate": settings_candidate,
    }


def revision(roots, item_id, executor=None):
    """Read-only current revision, WITHOUT preview eligibility: recheck must
    read the recorded native state even after the action has settled. When the
    recorded executor is supplied, absence is proven from ITS healthy source
    alone (never inferred from a different, failed bucket)."""
    if type(item_id) is not str or not HEX64_RE.fullmatch(item_id):
        raise Refusal("inventory-malformed")
    settings = None
    if executor is not None:
        if executor not in ("claude", "codex", "opencode"):
            raise Refusal("inventory-malformed")
        snap = snapshot(roots)
        bucket = snap["executors"][executor]
        if not bucket["ok"]:
            raise Refusal("inventory-source-unavailable")
        if executor == "opencode":
            settings = _published_settings_revision(roots)
        for item in bucket["items"]:
            if item["id"] == item_id:
                return {"revision": _revision_for(roots, item), "settings": settings}
        return {"revision": "absent", "settings": settings}
    try:
        item, _executor = _find_item(roots, item_id)
    except Refusal as exc:
        if exc.code == "inventory-item-unknown":
            return {"revision": "absent", "settings": None}
        raise
    return {"revision": _revision_for(roots, item), "settings": None}


def _apply_effect(roots, item, executor, change, expected_revision, operation_id, provenance):
    if change == "enable" and item["kind"] == "skill":
        # a managed restore REQUIRES recorded provenance: no fabricated Enable
        prov = provenance if type(provenance) is dict else {}
        if not prov.get("selector") and "container" not in prov:
            raise Refusal("inventory-no-managed-state")
        # the post-disable policy digest must still match the recorded disable
        if prov.get("digest") is not None:
            if _managed_disable_digest(roots, item["executor"], "skill", item["registration"]) != prov["digest"]:
                raise Refusal("inventory-source-changed")

    if change == "remove-registration":
        if item["origin"] == "registration-link":
            return _remove_registration_link(roots, item, expected_revision)
        if executor == "opencode" and item["origin"] == "config-array":
            return _opencode_remove_plugin(roots, item, operation_id, expected_revision)
        raise Refusal("inventory-unsupported")

    if change == "remove-installation":
        if item["origin"] not in ("independent", "local-file"):
            raise Refusal("inventory-unsupported")
        # the complete ownership/reference proof runs before any move, is
        # re-checked under the claim, and the captured revision is enforced
        dest = _quarantine(roots, item, operation_id, expected_revision)
        return {"effect": "applied", "recovery": dest, "requiresNewSession": False}

    if executor == "claude":
        return _claude_change(roots, item, change, expected_revision, operation_id, provenance)

    if executor == "codex":
        return _codex_change(roots, item, change, expected_revision, operation_id, provenance)

    if executor == "opencode" and item["kind"] == "skill":
        return _opencode_skill_change(roots, item, change, operation_id, provenance, expected_revision)

    raise Refusal("inventory-unsupported")


def _remove_registration_link(roots, item, expected_revision):
    """Hold the claim, verify the exact link inode/target and the checked parent,
    then unlink THAT link via its parent descriptor."""
    link = Path(item["path"])
    target = item.get("canonicalTarget")
    if _revision_for(roots, item) != expected_revision:
        raise Refusal("inventory-source-changed")
    inode = _link_inode(link)
    if target is not None and os.readlink(link) != target:
        raise Refusal("inventory-source-changed")
    with _apply_inventory_claim(roots):
        if _link_inode(link) != inode:
            raise Refusal("inventory-source-changed")
        parent_fd = jset._open_directory_nofollow(link.parent, create=False)
        try:
            jset._require_same_directory(parent_fd, link.parent)
            os.unlink(link.name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    return {"effect": "applied", "recovery": None, "requiresNewSession": False}


def _claude_change(roots, item, change, expected_revision, operation_id, provenance):
    with _apply_inventory_claim(roots):
        path = _claude_settings_path(roots)
        root_text = _read_text(path)
        doc = _parse_json(root_text)
        # re-resolve identity and revision UNDER the claim
        if _revision_for(roots, item) != expected_revision:
            raise Refusal("inventory-source-changed")
        if item["kind"] == "skill" and item["origin"] in ("independent", "skill-override"):
            text, prov = _claude_skill_text(root_text, doc, item, change, provenance, roots)
            _ProtectedWriter(path, operation_id).write(text, root_text.encode("utf-8") if root_text else b"")
            return {"effect": "applied", "provenance": prov, "requiresNewSession": True}
        text, prov = _claude_edit_text(root_text, doc, item, change, provenance)
        _ProtectedWriter(path, operation_id).write(text, root_text.encode("utf-8") if root_text else b"")
    return {"effect": "applied", "provenance": {"prior": prov["prior"]}, "requiresNewSession": True}


def _claude_skill_text(root_text, doc, item, change, provenance, roots):
    """Claude independent/skill-override skill: skillOverrides keyed by NAME."""
    if root_text is None:
        raise Refusal("inventory-format-unsupported")
    name = item["name"]
    if not root_text.strip():
        raise Refusal("inventory-format-unsupported")
    overrides = doc.get("skillOverrides")
    if change == "disable":
        if type(overrides) is not dict and overrides is not None:
            raise Refusal("inventory-format-unsupported")
        prior = overrides.get(name) if type(overrides) is dict else None
        edits = [{"path": ["skillOverrides", name], "value": "off"}]
        text = sio.edit_source(root_text, edits, jsonc=False)["text"]
        return text, {
            "selector": name, "prior": prior, "priorPresent": prior is not None,
            # digest the CHECKED POST-disable candidate, not the pre-disable bytes
            "digest": _managed_disable_digest_from(
                item["executor"], "skill", item["registration"], text.encode("utf-8")
            ),
        }
    # enable: exact provenance-checked restore
    prov = provenance if type(provenance) is dict else {}
    if prov.get("selector") != name:
        raise Refusal("inventory-no-managed-state")
    if "prior" not in prov:
        raise Refusal("inventory-no-managed-state")
    current = overrides.get(name) if type(overrides) is dict else None
    if current != "off":
        # another writer changed the policy since Jax's managed disable
        raise Refusal("inventory-source-changed")
    prior = prov.get("prior")
    if prov.get("priorPresent") is False or prior is None:
        edits = [{"path": ["skillOverrides", name], "remove": True}]
    else:
        if prior not in CLAUDE_OVERRIDE_VALUES:
            raise Refusal("inventory-no-managed-state")
        edits = [{"path": ["skillOverrides", name], "value": prior}]
    return sio.edit_source(root_text, edits, jsonc=False)["text"], {"prior": prior, "priorPresent": prov.get("priorPresent") is not False}


def _codex_change(roots, item, change, expected_revision, operation_id, provenance):
    with _apply_inventory_claim(roots):
        path = _codex_config_path(roots)
        raw = _read_bytes(path)
        root_text = raw.decode("utf-8") if raw is not None else ""  # absent config: tomlkit round-trips an empty doc
        if _revision_for(roots, item) != expected_revision:
            raise Refusal("inventory-source-changed")
        text, prov = _codex_edit_text(root_text, item, change, provenance)
        if change == "disable" and item["kind"] == "skill" and type(prov) is dict:
            # bind the restore to the CHECKED POST-disable config bytes
            prov = {**prov, "digest": _managed_disable_digest_from(
                "codex", "skill", item["registration"], text.encode("utf-8")
            )}
        _ProtectedWriter(path, operation_id).write(text, raw)
    return {"effect": "applied", "provenance": prov, "requiresNewSession": True}


def _opencode_remove_plugin(roots, item, operation_id, expected_revision):
    paths = _settings_paths(roots)
    with jset.configuration_claim(Path(paths["lock"]), create=True):
        if _revision_for(roots, item) != expected_revision:
            raise Refusal("inventory-source-changed")
        texts, values = sio._load_sources(paths)
        target = item["path"]
        target_name = Path(target).name
        if values.get(target_name) is None:
            raise Refusal("inventory-source-changed")
        edits = [{"path": ["plugin"], "value": _plugin_array_without(values[target_name], item["registration"])}]
        edited = sio.edit_source(texts[target_name], edits, jsonc=target_name.endswith("jsonc"))
        return _publish_opencode_edit(paths, texts, target_name, edited["text"], operation_id)


def _plugin_array_without(value, registration):
    if type(value) is not dict:
        raise Refusal("inventory-format-unsupported")
    plugins = value.get("plugin")
    if type(plugins) is not list or plugins.count(registration) != 1:
        raise Refusal("inventory-source-changed")
    return [entry for entry in plugins if entry != registration]


def _publish_opencode_edit(paths, texts, target_name, new_text, operation_id):
    settings = jset.read_settings(Path(paths["settings"]))
    if settings is None:
        raise Refusal("inventory-source-changed")
    candidates = dict(texts)
    candidates[target_name] = new_text
    settings_raw, revision, _tokens = sio._settings_document(
        {"reviewers": settings["reviewers"], "builders": settings["builders"]}, operation_id, candidates
    )
    expected_sources = {name: jset._source_token(Path(paths[name])) for name in jset.SOURCE_FILES}
    result = sio.publish_candidates(
        operation_id, paths, candidates,
        affected=[target_name],
        expected_sources=expected_sources,
        expected_settings=settings["revision"],
        settings_spec={"raw": settings_raw, "revision": revision},
    )
    return result


def _opencode_skill_change(roots, item, change, operation_id, provenance, expected_revision=None):
    paths = _settings_paths(roots)
    with jset.configuration_claim(Path(paths["lock"]), create=True):
        if expected_revision is not None and _revision_for(roots, item) != expected_revision:
            raise Refusal("inventory-source-changed")
        texts, values = sio._load_sources(paths)
        target_name = _opencode_source_name(roots, item)
        value = values.get(target_name)
        if type(value) is not dict:
            raise Refusal("inventory-source-changed")
        new_text, prov_out = _opencode_skill_edit_text(texts[target_name], value, item, change, provenance)
        edited_text = new_text if isinstance(new_text, str) else new_text
        result = _publish_opencode_edit(paths, texts, target_name, edited_text, operation_id)
        if change == "disable":
            result = {**result, "provenance": prov_out}
        return result


def _opencode_skill_edit_text(source_text, value, item, change, provenance):
    """Exact prior-policy restoration for OpenCode skills: scalar root/skill
    wrapping, last-precedence deny, original order/position, ask/deny/allow/absence."""
    name = item["name"]
    if change == "disable":
        new_text = sio.edit_source(source_text, _opencode_deny_edits(value, name), jsonc=True)["text"]
        return new_text, _disable_restoration_from(value, name)
    prov = provenance if type(provenance) is dict else {}
    if prov.get("selector") != name and "container" not in prov:
        raise Refusal("inventory-no-managed-state")
    edits = _opencode_restore_edits(value, name, prov)
    new_text = sio.edit_source(source_text, edits, jsonc=True)["text"]
    return new_text, {"prior": prov.get("prior")}


def _disable_restoration_from(value, name):
    """Capture the prior OpenCode policy shape from the CURRENT values."""
    permission = value.get("permission")
    if type(permission) is str:
        return {"container": "root-scalar", "prior": permission}
    if type(permission) is not dict:
        return {"selector": name, "prior": None, "priorPresent": False}
    policy = permission.get("skill")
    if type(policy) is str:
        return {"container": "skill-scalar", "prior": policy}
    if type(policy) is dict:
        prior = policy.get(name)
        position = list(policy).index(name) if name in policy else None
        return {"selector": name, "prior": prior, "priorPresent": prior is not None, "position": position}
    return {"selector": name, "prior": None, "priorPresent": False}


def _opencode_deny_edits(value, name):
    """Build the last-precedence deny edits, preserving scalar policies by
    wrapping them into ordered maps with the original value first."""
    permission = value.get("permission")
    if type(permission) is str:
        # scalar root permission wraps into {'*': old, skill: {name: deny}}
        return [
            {"path": ["permission"], "value": {"*": permission, "skill": {name: "deny"}}},
        ]
    if type(permission) is dict:
        policy = permission.get("skill")
        if type(policy) is str:
            # scalar skill policy wraps into {'*': old, name: deny}
            return [{"path": ["permission", "skill"], "value": {"*": policy, name: "deny"}}]
        if policy is None:
            return [{"path": ["permission", "skill"], "value": {name: "deny"}}]
        if type(policy) is dict:
            new_map = dict(policy)
            new_map.pop(name, None)
            new_map[name] = "deny"  # re-added last: last-matching precedence
            return [{"path": ["permission", "skill"], "value": new_map}]
    raise Refusal("inventory-format-unsupported")


def _opencode_restore_edits(value, name, prov):
    """Restore the exact prior policy: original order/position, scalar
    unwrap, exact deny/ask/allow or absence."""
    container = prov.get("container")
    prior = prov.get("prior")
    permission = value.get("permission")
    if container == "root-scalar":
        if type(permission) is not dict or type(permission.get("skill")) is not dict:
            raise Refusal("inventory-source-changed")
        skill_map = permission["skill"]
        # never unwrap a scalar over a concurrently added unrelated permission
        if set(skill_map) != {name} or set(permission) != {"*", "skill"} or permission.get("*") != prior:
            raise Refusal("inventory-source-changed")
        return [{"path": ["permission"], "value": prior if type(prior) is str else "allow"}]
    if container == "skill-scalar":
        if type(permission) is not dict or type(permission.get("skill")) is not dict:
            raise Refusal("inventory-source-changed")
        skill_map = permission["skill"]
        if set(skill_map) != {"*", name} or skill_map.get("*") != prior or skill_map.get(name) != "deny":
            raise Refusal("inventory-source-changed")
        return [{"path": ["permission", "skill"], "value": prior}]
    # map container
    if type(permission) is not dict or type(permission.get("skill")) is not dict:
        raise Refusal("inventory-source-changed")
    skill_map = dict(permission["skill"])
    if skill_map.get(name) != "deny":
        raise Refusal("inventory-source-changed")
    del skill_map[name]
    if not (prov.get("priorPresent") is False or prior is None):
        position = prov.get("position")
        entries = list(skill_map.items())
        if type(position) is int and 0 <= position <= len(entries):
            entries.insert(position, (name, prior))
        else:
            entries.insert(0, (name, prior))
        skill_map = dict(entries)
    return [{"path": ["permission", "skill"], "value": skill_map}]


def reconcile(roots, expected, operation_id):
    if type(expected) is not dict:
        raise Refusal("inventory-malformed")
    paths = _settings_paths(roots)
    # A fresh server artifact UUID per confirmed reconcile attempt: exclusive
    # staging/backup names, never the original attempt's. The inventory
    # operation ID remains only the DB/audit identity.
    artifact_id = str(uuid.uuid4())
    return sio.republish_settings(artifact_id, paths, expected)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    op = argv[1] if len(argv) == 2 else None
    raw = sys.stdin.buffer.read(INPUT_CAP + 1)
    if len(raw) > INPUT_CAP:
        _emit({"ok": False, "error": "inventory-too-large"})
        return 0
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _emit({"ok": False, "error": "inventory-malformed"})
        return 0
    if type(payload) is not dict or op not in ("snapshot", "preview", "prepare", "revision", "apply", "reconcile"):
        _emit({"ok": False, "error": "inventory-malformed"})
        return 0
    try:
        roots = _roots(payload)
        if op == "snapshot":
            data = snapshot(roots)
        elif op == "preview":
            data = preview(roots, payload.get("itemId"), payload.get("change"), payload.get("provenance"))
        elif op == "prepare":
            data = prepare(
                roots,
                payload.get("itemId"),
                payload.get("change"),
                payload.get("expectedRevision"),
                payload.get("provenance"),
                payload.get("operationId"),
            )
        elif op == "revision":
            data = revision(roots, payload.get("itemId"), payload.get("executor"))
        elif op == "apply":
            data = apply(
                roots,
                payload.get("itemId"),
                payload.get("change"),
                payload.get("expectedRevision"),
                payload.get("operationId"),
                payload.get("provenance"),
            )
        else:
            data = reconcile(roots, payload.get("expected"), payload.get("operationId"))
    except Refusal as exc:
        _emit({"ok": False, "error": exc.code})
        return 0
    except Exception:
        _emit({"ok": False, "error": "inventory-helper-failed"})
        return 0
    out = json.dumps({"ok": True, "data": data}, ensure_ascii=False, separators=(",", ":"))
    if len(out.encode("utf-8")) > OUTPUT_CAP:
        _emit({"ok": False, "error": "inventory-too-large"})
        return 0
    _emit({"ok": True, "data": data})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
