#!/usr/bin/env python3
"""Tests for the finite native inventory helper.

Codex writes import tomlkit lazily, so these tests need the isolated helper
venv's site-packages on PYTHONPATH (see the plan's Task 7 command).
"""
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from jax_init import Refusal
import jaxflow_inventory_io as inv

OP = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
OP2 = "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
OP3 = "cccccccc-dddd-4eee-8fff-000000000000"
REPO = Path(__file__).resolve().parent.parent

CLAUDE = """{
  "enabledPlugins": {"alpha@market": true, "beta@market": false},
  "skillOverrides": {"solo": "off", "keep": "on"}
}
"""

CODEX = """# keep this comment
[plugins.alpha]
enabled = true
other = "x"

[[skills.config]]
path = "/tmp/skills/one/SKILL.md"
enabled = true
"""

CODEX_ABSENT_ENABLED = """# keep this comment
[plugins."dotted.plugin.id@mp"]
other = "keep-me"

[[skills.config]]
path = "/tmp/skills/pre/SKILL.md"
note = "unrelated"

[[skills.config]]
name = "named-skill"
enabled = true
"""

OPENCODE = """{
  // keep this comment
  "provider": {
    "fixture": {
      "npm": "@openrouter/ai-sdk-provider",
      "models": {
        "jaxflow-builder-default": {
          "id": "wire-real-model",
          "variants": { "high": { "reasoning": { "effort": "high" } } },
          "options": { "provider": { "sort": "price", "allow_fallbacks": true } }
        },
        "jaxflow-builder-fallback": {
          "id": "wire-real-model",
          "variants": { "high": { "reasoning": { "effort": "high" } } },
          "options": { "provider": { "sort": "price", "allow_fallbacks": false } }
        }
      }
    }
  },
  "plugin": ["pkg-a@1.0.0", "pkg-b@2.0.0"],
  "permission": { "skill": { "other": "allow", "internal-*": "deny" } }
}
"""

OPENCODE_SCALAR = """{
  "permission": { "skill": "ask" }
}
"""

# scalar policies plus the full fixture provider, so the settings republish resolves
OPENCODE_SCALAR = """{
  // keep this comment
  "provider": {
    "fixture": {
      "npm": "@openrouter/ai-sdk-provider",
      "models": {
        "jaxflow-builder-default": {
          "id": "wire-real-model",
          "variants": { "high": { "reasoning": { "effort": "high" } } },
          "options": { "provider": { "sort": "price", "allow_fallbacks": true } }
        },
        "jaxflow-builder-fallback": {
          "id": "wire-real-model",
          "variants": { "high": { "reasoning": { "effort": "high" } } },
          "options": { "provider": { "sort": "price", "allow_fallbacks": false } }
        }
      }
    }
  },
  "permission": { "skill": "ask" }
}
"""

OPENCODE_ROOT_SCALAR = """{
  "provider": {
    "fixture": {
      "npm": "@openrouter/ai-sdk-provider",
      "models": {
        "jaxflow-builder-default": {
          "id": "wire-real-model",
          "variants": { "high": { "reasoning": { "effort": "high" } } },
          "options": { "provider": { "sort": "price", "allow_fallbacks": true } }
        },
        "jaxflow-builder-fallback": {
          "id": "wire-real-model",
          "variants": { "high": { "reasoning": { "effort": "high" } } },
          "options": { "provider": { "sort": "price", "allow_fallbacks": false } }
        }
      }
    }
  },
  "permission": "ask"
}
"""


def roots(tmp_path):
    out = {}
    for key, rel in (
        ("claude_dir", "claude"),
        ("codex_dir", "codex"),
        ("opencode_dir", "config/opencode"),
        ("jax_os", "jax-os"),
    ):
        d = tmp_path / rel
        d.mkdir(parents=True, exist_ok=True)
        out[key] = str(d)
    return out


def settings_fixture():
    return json.loads((REPO / "workflow" / "fixtures" / "agent-settings-v1.json").read_text())


def write(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def seed_claude(r):
    write(Path(r["claude_dir"]) / "settings.json", CLAUDE)


def seed_codex(r):
    write(Path(r["codex_dir"]) / "config.toml", CODEX)


def seed_opencode(r):
    write(Path(r["opencode_dir"]) / "opencode.jsonc", OPENCODE)
    settings = Path(r["jax_os"]) / "agent-settings.json"
    write(settings, json.dumps(settings_fixture()))
    os.chmod(settings, 0o600)


def find(items, registration):
    return next(i for i in items if i["registration"] == registration or i["name"] == registration)


# ---- snapshot: discovery sources, states, capabilities ----


def test_snapshot_reports_stable_ids_states_and_capabilities(tmp_path):
    r = roots(tmp_path)
    seed_claude(r)
    seed_codex(r)
    seed_opencode(r)
    snap = inv.snapshot(r)
    assert snap["executors"]["claude"]["ok"] and snap["executors"]["codex"]["ok"] and snap["executors"]["opencode"]["ok"]
    claude = snap["executors"]["claude"]["items"]
    alpha = find(claude, "alpha@market")
    beta = find(claude, "beta@market")
    assert alpha["state"] == "enabled" and beta["state"] == "disabled"
    assert alpha["id"] == inv._item_id("claude", "plugin", "alpha@market")
    assert len(alpha["id"]) == 64
    assert find(claude, "solo")["state"] == "disabled"
    codex = snap["executors"]["codex"]["items"]
    assert find(codex, "alpha")["state"] == "enabled"
    assert find(codex, "/tmp/skills/one/SKILL.md")["kind"] == "skill"
    opc = snap["executors"]["opencode"]["items"]
    assert find(opc, "pkg-a@1.0.0")["capabilities"]["remove-registration"]["available"] is True
    assert find(opc, "other")["state"] == "enabled"
    # Codex has no local-agent category: metadata, never a fabricated agent row.
    assert snap["executors"]["codex"].get("unsupported") == ["agent"]
    assert not any(i["kind"] == "agent" for i in codex)


def test_snapshot_isolates_a_malformed_source(tmp_path):
    r = roots(tmp_path)
    write(Path(r["claude_dir"]) / "settings.json", "{ not json")
    seed_codex(r)
    snap = inv.snapshot(r)
    assert snap["executors"]["claude"]["ok"] is False
    assert snap["executors"]["codex"]["ok"] is True


def test_snapshot_discovers_local_agent_definitions(tmp_path):
    r = roots(tmp_path)
    seed_claude(r)
    seed_opencode(r)
    agents = Path(r["claude_dir"]) / "agents"
    agents.mkdir()
    write(agents / "reviewer.md", "---\ndescription: reviews PRs\n---\nbody")
    write(agents / "notes.txt", "not an agent")
    oc_agents = Path(r["opencode_dir"]) / "agents"
    oc_agents.mkdir(parents=True)
    write(oc_agents / "docs-reader.md", "---\ndescription: reads docs\n---\n")
    snap = inv.snapshot(r)
    claude_agents = [i for i in snap["executors"]["claude"]["items"] if i["kind"] == "agent"]
    assert [i["registration"] for i in claude_agents] == [str(agents / "reviewer.md")]
    assert claude_agents[0]["description"] == "reviews PRs"
    assert all(not i["capabilities"] for i in claude_agents)  # read-only
    oc = [i for i in snap["executors"]["opencode"]["items"] if i["kind"] == "agent"]
    assert [i["name"] for i in oc] == ["docs-reader"]
    # Codex exposes the unsupported category as metadata, never an item.
    assert not [i for i in snap["executors"]["codex"]["items"] if i["kind"] == "agent"]


def test_snapshot_discovers_independent_skills_with_descriptions(tmp_path):
    r = roots(tmp_path)
    seed_claude(r)
    seed_codex(r)
    seed_opencode(r)
    oc_skill = Path(r["opencode_dir"]) / "skills" / "writer"
    oc_skills_md = oc_skill / "SKILL.md"
    oc_skills_md.parent.mkdir(parents=True)
    oc_skills_md.write_text("---\ndescription: writes docs\n---\n# writer\n", encoding="utf-8")
    snap = inv.snapshot(r)
    oc = snap["executors"]["opencode"]["items"]
    writer = find(oc, str(oc_skill))
    assert writer["kind"] == "skill" and writer["origin"] == "independent"
    assert writer["description"] == "writes docs"
    assert writer["state"] == "enabled"  # no permission entry: default policy
    assert writer["capabilities"]["disable"]["available"] is True
    cl_skills = Path(r["claude_dir"]) / "skills" / "independent"
    (cl_skills / "SKILL.md").parent.mkdir(parents=True)
    (cl_skills / "SKILL.md").write_text("# mine\ndescription: standalone\n", encoding="utf-8")
    snap = inv.snapshot(r)
    claude_writer = find(snap["executors"]["claude"]["items"], str(cl_skills))
    assert claude_writer["state"] == "enabled"
    assert claude_writer["capabilities"]["disable"]["available"] is True
    assert "description" in json_keys(claude_writer)


def json_keys(item):
    return set(item.keys())


def test_claude_skill_state_joins_the_overrides_map(tmp_path):
    r = roots(tmp_path)
    skills = Path(r["claude_dir"]) / "skills"
    (skills / "solo").mkdir(parents=True)
    (skills / "solo" / "SKILL.md").write_text("# solo\n", encoding="utf-8")
    (skills / "keep").mkdir(parents=True)
    (skills / "keep" / "SKILL.md").write_text("# keep\n", encoding="utf-8")
    seed_claude(r)
    snap = inv.snapshot(r)
    items = snap["executors"]["claude"]["items"]
    solo = find(items, str(skills / "solo"))
    assert solo["origin"] == "independent" and solo["state"] == "disabled"
    # disabled by a managed override without provenance: no fabricated Enable
    assert solo["capabilities"]["enable"]["available"] is False
    keep = find(items, str(skills / "keep"))
    assert keep["state"] == "enabled"  # "on" is the default: visible, unrestricted
    assert keep["capabilities"]["disable"]["available"] is True


def test_bundled_plugin_skills_carry_parent_provenance(tmp_path):
    r = roots(tmp_path)
    write(Path(r["claude_dir"]) / "settings.json", CLAUDE.replace('"alpha@market": true', '"alpha@market": true, "demo@mp": true'))
    cache = Path(r["claude_dir"]) / "plugins" / "cache" / "mp" / "demo" / "1.0.0"
    (cache / "skills" / "bundled").mkdir(parents=True)
    (cache / "skills" / "bundled" / "SKILL.md").write_text("# b\n", encoding="utf-8")
    snap = inv.snapshot(r)
    claude = snap["executors"]["claude"]["items"]
    parent = next(i for i in claude if i["registration"] == "demo@mp")
    assert parent["origin"] == "settings" and parent["cachePath"].endswith("demo")
    child = next(i for i in claude if i["kind"] == "skill" and i["parent"] == "demo@mp")
    assert child["name"] == "bundled"
    assert child["state"] == "enabled"
    assert all(not cap["available"] for cap in child["capabilities"].values())
    # disabling the parent plugin flips the child's effective state
    prev = inv.preview(r, parent["id"], "disable")
    inv.apply(r, parent["id"], "disable", prev["revision"], OP)
    snap = inv.snapshot(r)
    child = next(i for i in snap["executors"]["claude"]["items"] if i.get("parent") == "demo@mp")
    assert child["state"] == "parent-disabled"


def test_codex_snapshot_handles_absent_enabled_and_name_selectors(tmp_path):
    r = roots(tmp_path)
    write(Path(r["codex_dir"]) / "config.toml", CODEX_ABSENT_ENABLED)
    snap = inv.snapshot(r)
    codex = snap["executors"]["codex"]["items"]
    plugin = find(codex, "dotted.plugin.id@mp")
    assert plugin["state"] == "enabled"  # absent enabled = native default
    skill = find(codex, "/tmp/skills/pre/SKILL.md")
    assert skill["state"] == "enabled" and skill["capabilities"]["disable"]["available"] is True
    # a name-selector entry is visible but has no path-selector control
    named = find(codex, "named-skill")
    assert named["origin"] == "config-name"
    assert all(not cap["available"] for cap in named["capabilities"].values())


def test_codex_dir_skill_matches_config_entry_without_duplicates(tmp_path):
    r = roots(tmp_path)
    skill_dir = Path(r["codex_dir"]) / "skills" / "one"
    (skill_dir / "SKILL.md").parent.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# one\n", encoding="utf-8")
    write(Path(r["codex_dir"]) / "config.toml", CODEX.replace("/tmp/skills/one/SKILL.md", str(skill_dir / "SKILL.md")))
    items = inv.snapshot(r)["executors"]["codex"]["items"]
    # the config-path entry and the independent dir join into ONE row (config-path wins)
    paths = [i["registration"] for i in items if i["kind"] == "skill"]
    assert len(paths) == 1
    skill_row = next(i for i in items if i["kind"] == "skill")
    assert skill_row["origin"] == "config-path"


# ---- Task 4/5: protected writers preserve unrelated content ----


def test_codex_preview_and_disable_preserve_unrelated_content(tmp_path):
    r = roots(tmp_path)
    seed_codex(r)
    item = find(inv.snapshot(r)["executors"]["codex"]["items"], "alpha")
    prev = inv.preview(r, item["id"], "disable")
    assert prev["revision"] and prev["change"] == "disable"
    result = inv.apply(r, item["id"], "disable", prev["revision"], OP)
    assert result["effect"] == "applied"
    text = (Path(r["codex_dir"]) / "config.toml").read_text()
    assert "# keep this comment" in text
    assert 'other = "x"' in text
    assert "enabled = false" in text
    assert "/tmp/skills/one/SKILL.md" in text
    # enabling again restores true
    item2 = find(inv.snapshot(r)["executors"]["codex"]["items"], "alpha")
    prev2 = inv.preview(r, item2["id"], "enable")
    inv.apply(r, item2["id"], "enable", prev2["revision"], OP2)
    assert "enabled = true" in (Path(r["codex_dir"]) / "config.toml").read_text()


def test_codex_edit_refuses_invalid_and_missing_entries(tmp_path):
    r = roots(tmp_path)
    seed_codex(r)
    item = find(inv.snapshot(r)["executors"]["codex"]["items"], "alpha")
    with pytest.raises(Refusal):
        inv._codex_edit_text("this is not = valid toml [[[", item, "disable", None)
    missing = dict(item)
    missing["registration"] = "nope"
    with pytest.raises(Refusal):
        inv._codex_edit_text(CODEX, missing, "disable", None)


def test_apply_refuses_a_stale_revision(tmp_path):
    r = roots(tmp_path)
    seed_codex(r)
    item = find(inv.snapshot(r)["executors"]["codex"]["items"], "alpha")
    with pytest.raises(Refusal) as exc:
        inv.apply(r, item["id"], "disable", "0" * 64, OP)
    assert exc.value.code == "inventory-source-changed"


def test_claude_plugin_toggle_roundtrip(tmp_path):
    r = roots(tmp_path)
    seed_claude(r)
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], "alpha@market")
    prev = inv.preview(r, item["id"], "disable")
    inv.apply(r, item["id"], "disable", prev["revision"], OP)
    doc = json.loads((Path(r["claude_dir"]) / "settings.json").read_text())
    assert doc["enabledPlugins"]["alpha@market"] is False
    assert doc["skillOverrides"] == {"solo": "off", "keep": "on"}  # unrelated field preserved

    item2 = find(inv.snapshot(r)["executors"]["claude"]["items"], "alpha@market")
    prev2 = inv.preview(r, item2["id"], "enable")
    inv.apply(r, item2["id"], "enable", prev2["revision"], OP2)
    assert json.loads((Path(r["claude_dir"]) / "settings.json").read_text())["enabledPlugins"]["alpha@market"] is True


def test_opencode_plugin_removal_updates_settings_source_hashes(tmp_path):
    r = roots(tmp_path)
    seed_opencode(r)
    item = find(inv.snapshot(r)["executors"]["opencode"]["items"], "pkg-a@1.0.0")
    prev = inv.preview(r, item["id"], "remove-registration")
    result = inv.apply(r, item["id"], "remove-registration", prev["revision"], OP)
    assert result["effect"] == "published"
    text = (Path(r["opencode_dir"]) / "opencode.jsonc").read_text()
    assert "pkg-a@1.0.0" not in text and "pkg-b@2.0.0" in text
    assert "keep this comment" in text
    settings = json.loads((Path(r["jax_os"]) / "agent-settings.json").read_text())
    assert settings["source_revisions"]["opencode.jsonc"] != "absent"


def test_opencode_skill_deny_and_restore_with_provenance(tmp_path):
    r = roots(tmp_path)
    seed_opencode(r)
    item = find(inv.snapshot(r)["executors"]["opencode"]["items"], "other")
    prev = inv.preview(r, item["id"], "disable")
    inv.apply(r, item["id"], "disable", prev["revision"], OP)
    doc = inv._parse_json((Path(r["opencode_dir"]) / "opencode.jsonc").read_text())
    assert doc["permission"]["skill"]["other"] == "deny"

    item2 = find(inv.snapshot(r)["executors"]["opencode"]["items"], "other")
    prov = {"selector": "other", "prior": "allow"}
    prev2 = inv.preview(r, item2["id"], "enable", provenance=prov)
    inv.apply(r, item2["id"], "enable", prev2["revision"], OP2, provenance=prov)
    doc2 = inv._parse_json((Path(r["opencode_dir"]) / "opencode.jsonc").read_text())
    assert doc2["permission"]["skill"]["other"] == "allow"


def test_remove_installation_quarantines_and_refuses_destination(tmp_path):
    r = roots(tmp_path)
    skill_dir = Path(r["claude_dir"]) / "skills" / "mine"
    (skill_dir).mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# mine\n")
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], str(skill_dir))
    prev = inv.preview(r, item["id"], "remove-installation")
    result = inv.apply(r, item["id"], "remove-installation", prev["revision"], OP)
    assert result["effect"] == "applied"
    assert not skill_dir.exists()
    assert Path(result["recovery"]).exists()

    # a second quarantine under the same operation refuses an existing destination
    (skill_dir).mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# mine\n")
    item2 = find(inv.snapshot(r)["executors"]["claude"]["items"], str(skill_dir))
    prev2 = inv.preview(r, item2["id"], "remove-installation")
    with pytest.raises(Refusal):
        inv.apply(r, item2["id"], "remove-installation", prev2["revision"], OP)


def test_quarantine_collision_does_not_replace_sentinel(tmp_path, monkeypatch):
    r = roots(tmp_path)
    skill_dir = Path(r["claude_dir"]) / "skills" / "mine"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# mine\n")
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], str(skill_dir))
    prev = inv.preview(r, item["id"], "remove-installation")
    dest_dir = Path(r["jax_os"]) / "removed-tools" / OP
    sentinel = dest_dir / skill_dir.name
    real_require = inv.jset._require_same_directory
    injected = False

    def inject(fd, path):
        nonlocal injected
        result = real_require(fd, path)
        if not injected and Path(path) == dest_dir:
            injected = True
            dest_dir.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("sentinel\n")
        return result

    monkeypatch.setattr(inv.jset, "_require_same_directory", inject)
    with pytest.raises(Refusal) as exc:
        inv.apply(r, item["id"], "remove-installation", prev["revision"], OP)
    assert exc.value.code == "inventory-destination-exists"
    assert skill_dir.exists() and sentinel.read_text() == "sentinel\n"
    assert stat.S_IMODE(dest_dir.stat().st_mode) == 0o700


# ---- Task 1: exact prior-policy restoration ----


def test_claude_skill_disable_and_exact_restore(tmp_path):
    r = roots(tmp_path)
    write(Path(r["claude_dir"]) / "settings.json", CLAUDE.replace('"solo": "off"', '"solo": "name-only"'))
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], "solo")
    prev = inv.preview(r, item["id"], "disable")
    assert prev["restoration"]["prior"] == "name-only"
    result = inv.apply(r, item["id"], "disable", prev["revision"], OP, provenance=prev["restoration"])
    doc = json.loads((Path(r["claude_dir"]) / "settings.json").read_text())
    assert doc["skillOverrides"]["solo"] == "off"
    # managed cycle: the recorded restoration allows the exact inverse
    item2 = find(inv.snapshot(r)["executors"]["claude"]["items"], "solo")
    prov = {"selector": "solo", "prior": "name-only", "priorPresent": True}
    prev2 = inv.preview(r, item2["id"], "enable", provenance=prov)
    inv.apply(r, item2["id"], "enable", prev2["revision"], OP2, provenance=prov)
    assert json.loads((Path(r["claude_dir"]) / "settings.json").read_text())["skillOverrides"]["solo"] == "name-only"


def test_claude_skill_restore_absence(tmp_path):
    r = roots(tmp_path)
    # solo starts ENABLED so the managed disable is eligible
    write(Path(r["claude_dir"]) / "settings.json", CLAUDE.replace('"solo": "off"', '"solo": "on"'))
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], "solo")
    prev = inv.preview(r, item["id"], "disable")
    inv.apply(r, item["id"], "disable", prev["revision"], OP, provenance=prev["restoration"])
    doc = json.loads((Path(r["claude_dir"]) / "settings.json").read_text())
    assert doc["skillOverrides"]["solo"] == "off"
    item2 = find(inv.snapshot(r)["executors"]["claude"]["items"], "solo")
    prov = {"selector": "solo", "prior": "on", "priorPresent": False}
    prev2 = inv.preview(r, item2["id"], "enable", provenance=prov)
    inv.apply(r, item2["id"], "enable", prev2["revision"], OP2, provenance=prov)
    doc = json.loads((Path(r["claude_dir"]) / "settings.json").read_text())
    assert "solo" not in doc["skillOverrides"]


def test_claude_skill_enable_without_provenance_refuses(tmp_path):
    r = roots(tmp_path)
    seed_claude(r)
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], "solo")
    with pytest.raises(Refusal) as exc:
        inv.apply(r, item["id"], "enable", "0" * 64, OP, provenance=None)
    assert exc.value.code == "inventory-no-managed-state"


def test_codex_skill_restore_field_absence_and_keep_unrelated(tmp_path):
    r = roots(tmp_path)
    write(Path(r["codex_dir"]) / "config.toml", CODEX_ABSENT_ENABLED)
    item = find(inv.snapshot(r)["executors"]["codex"]["items"], "/tmp/skills/pre/SKILL.md")
    prev = inv.preview(r, item["id"], "disable")
    inv.apply(r, item["id"], "disable", prev["revision"], OP, provenance=prev["restoration"])
    text = (Path(r["codex_dir"]) / "config.toml").read_text()
    assert "# keep this comment" in text
    assert 'other = "keep-me"' in text
    assert 'path = "/tmp/skills/pre/SKILL.md"' in text
    assert "enabled = false" in text

    # managed cycle restores field absence, retaining the entry and unrelated fields
    item2 = find(inv.snapshot(r)["executors"]["codex"]["items"], "/tmp/skills/pre/SKILL.md")
    prov = {"selector": item2["registration"], "prior": None, "priorPresent": False, "createdEntry": False}
    prev2 = inv.preview(r, item2["id"], "enable", provenance=prov)
    inv.apply(r, item2["id"], "enable", prev2["revision"], OP2, provenance=prov)
    text = (Path(r["codex_dir"]) / "config.toml").read_text()
    assert "note = { enabled = false }" not in text
    assert "path = \"/tmp/skills/pre/SKILL.md\"" in text
    assert "# keep this comment" in text
    import tomllib
    restored = tomllib.loads(text)["skills"]["config"]
    pre = next(e for e in restored if e.get("path") == "/tmp/skills/pre/SKILL.md")
    assert "enabled" not in pre  # field absence restored, entry + unrelated field retained
    assert pre["note"] == "unrelated"
    assert any(e.get("name") == "named-skill" for e in restored)


def test_codex_skill_enable_refuses_native_disable_without_provenance(tmp_path):
    r = roots(tmp_path)
    write(Path(r["codex_dir"]) / "config.toml", CODEX)
    item = find(inv.snapshot(r)["executors"]["codex"]["items"], "/tmp/skills/one/SKILL.md")
    with pytest.raises(Refusal) as exc:
        inv.apply(r, item["id"], "enable", "0" * 64, OP, provenance=None)
    assert exc.value.code == "inventory-no-managed-state"


def test_codex_created_entry_is_removed_on_restore(tmp_path):
    r = roots(tmp_path)
    skill_dir = Path(r["codex_dir"]) / "skills" / "fresh"
    (skill_dir / "SKILL.md").parent.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# x\n", encoding="utf-8")
    r0 = inv.snapshot(r)
    item = find(r0["executors"]["codex"]["items"], str(skill_dir))
    prev = inv.preview(r, item["id"], "disable")
    assert prev["restoration"]["createdEntry"] is True
    inv.apply(r, item["id"], "disable", prev["revision"], OP, provenance=prev["restoration"])
    text = (Path(r["codex_dir"]) / "config.toml").read_text()
    assert "[[skills.config]]" in text and "enabled = false" in text
    item2 = find(inv.snapshot(r)["executors"]["codex"]["items"], str(skill_dir))
    prov = {"selector": item2["registration"], "prior": None, "priorPresent": False, "createdEntry": True, "createdContainer": True, "createdRoot": True}
    prev2 = inv.preview(r, item2["id"], "enable", provenance=prov)
    inv.apply(r, item2["id"], "enable", prev2["revision"], OP2, provenance=prov)
    text = (Path(r["codex_dir"]) / "config.toml").read_text()
    assert "[[skills.config]]" not in text  # Jax-created entry fully removed
    assert "[skills" not in text            # Jax-created containers removed too


def test_opencode_skill_restore_scalar_and_wrapping(tmp_path):
    r = roots(tmp_path)
    skill_dir = Path(r["opencode_dir"]) / "skills" / "writer"
    (skill_dir / "SKILL.md").parent.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# writer\n", encoding="utf-8")
    seed_opencode(r)
    write(Path(r["opencode_dir"]) / "opencode.jsonc", OPENCODE_SCALAR)
    item = find(inv.snapshot(r)["executors"]["opencode"]["items"], "writer")
    prev = inv.preview(r, item["id"], "disable")
    assert prev["restoration"] == {"container": "skill-scalar", "prior": "ask"}
    inv.apply(r, item["id"], "disable", prev["revision"], OP, provenance=prev["restoration"])
    doc = inv._parse_json((Path(r["opencode_dir"]) / "opencode.jsonc").read_text())
    assert doc["permission"]["skill"] == {"*": "ask", "writer": "deny"}

    item2 = find(inv.snapshot(r)["executors"]["opencode"]["items"], "writer")
    prov = {"container": "skill-scalar", "prior": "ask"}
    prev2 = inv.preview(r, item2["id"], "enable", provenance=prov)
    inv.apply(r, item2["id"], "enable", prev2["revision"], OP2, provenance=prov)
    doc = inv._parse_json((Path(r["opencode_dir"]) / "opencode.jsonc").read_text())
    assert doc["permission"]["skill"] == "ask"  # scalar restored exactly


def test_opencode_skill_restore_absence_and_order(tmp_path):
    r = roots(tmp_path)
    seed_opencode(r)
    item = find(inv.snapshot(r)["executors"]["opencode"]["items"], "other")
    prev = inv.preview(r, item["id"], "disable")
    assert prev["restoration"]["prior"] == "allow"
    inv.apply(r, item["id"], "disable", prev["revision"], OP, provenance=prev["restoration"])
    raw = (Path(r["opencode_dir"]) / "opencode.jsonc").read_text()
    doc = inv._parse_json(raw)
    entries = list(doc["permission"]["skill"].items())
    # the deny lands last (last precedence) and other rules keep their positions
    assert list(doc["permission"]["skill"].items())[-1] == ("other", "deny")
    assert doc["permission"]["skill"]["internal-*"] == "deny"

    item2 = find(inv.snapshot(r)["executors"]["opencode"]["items"], "other")
    prov = {"selector": "other", "prior": "allow", "priorPresent": True}
    prev2 = inv.preview(r, item2["id"], "enable", provenance=prov)
    inv.apply(r, item2["id"], "enable", prev2["revision"], OP2, provenance=prov)
    doc2 = inv._parse_json((Path(r["opencode_dir"]) / "opencode.jsonc").read_text())
    assert list(doc2["permission"]["skill"].items())[0] == ("other", "allow")
    assert doc2["permission"]["skill"]["internal-*"] == "deny"


def test_opencode_skill_disable_wraps_scalar_root_permission(tmp_path):
    r = roots(tmp_path)
    skill_dir = Path(r["opencode_dir"]) / "skills" / "writer"
    (skill_dir / "SKILL.md").parent.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# writer\n", encoding="utf-8")
    seed_opencode(r)
    write(Path(r["opencode_dir"]) / "opencode.jsonc", OPENCODE_ROOT_SCALAR)
    item = find(inv.snapshot(r)["executors"]["opencode"]["items"], "writer")
    prev = inv.preview(r, item["id"], "disable")
    inv.apply(r, item["id"], "disable", prev["revision"], OP, provenance=prev["restoration"])
    doc = inv._parse_json((Path(r["opencode_dir"]) / "opencode.jsonc").read_text())
    assert doc["permission"] == {"*": "ask", "skill": {"writer": "deny"}}
    item2 = find(inv.snapshot(r)["executors"]["opencode"]["items"], "writer")
    prov = {"container": "root-scalar", "prior": "ask"}
    prev2 = inv.preview(r, item2["id"], "enable", provenance=prov)
    inv.apply(r, item2["id"], "enable", prev2["revision"], OP2, provenance=prov)
    doc2 = inv._parse_json((Path(r["opencode_dir"]) / "opencode.jsonc").read_text())
    assert doc2["permission"] == "ask"  # root scalar restored


def test_enable_refuses_when_source_changed_since_disable(tmp_path):
    r = roots(tmp_path)
    # solo starts ENABLED ("on" is the default visible state) so disable is eligible
    write(Path(r["claude_dir"]) / "settings.json", CLAUDE.replace('"solo": "off"', '"solo": "on"'))
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], "solo")
    prev = inv.preview(r, item["id"], "disable")
    inv.apply(r, item["id"], "disable", prev["revision"], OP, provenance=prev["restoration"])
    # another writer touches the file after the managed disable
    path = Path(r["claude_dir"]) / "settings.json"
    doc = json.loads(path.read_text())
    doc["skillOverrides"]["solo"] = "name-only"  # an unrelated writer flipped it back
    path.write_text(json.dumps(doc), encoding="utf-8")
    item2 = find(inv.snapshot(r)["executors"]["claude"]["items"], "solo")
    with pytest.raises(Refusal):
        inv.apply(r, item2["id"], "enable", inv.preview(r, item2["id"], "enable", provenance=prev["restoration"])["revision"], OP2, provenance=prev["restoration"])


# ---- Task 2: protected writes, ownership proof, claim-time recheck ----


def test_apply_rechecks_revision_under_claim(tmp_path, monkeypatch):
    r = roots(tmp_path)
    seed_claude(r)
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], "alpha@market")
    prev = inv.preview(r, item["id"], "disable")
    real_claim = inv._apply_inventory_claim

    def claim_that_mutates(roots_arg):
        # a concurrent writer lands between the preview revision check and the claim
        path = Path(roots_arg["claude_dir"]) / "settings.json"
        doc = json.loads(path.read_text())
        doc["enabledPlugins"]["gamma@market"] = True
        path.write_text(json.dumps(doc), encoding="utf-8")
        return real_claim(roots_arg)

    monkeypatch.setattr(inv, "_apply_inventory_claim", claim_that_mutates)
    with pytest.raises(Refusal) as exc:
        inv.apply(r, item["id"], "disable", prev["revision"], OP)
    assert exc.value.code == "inventory-source-changed"


def test_quarantine_refuses_symlink_descendant_hardlink_and_shared_origin(tmp_path):
    r = roots(tmp_path)
    # symlink descendant inside the skill dir
    bad = Path(r["claude_dir"]) / "skills" / "bad-link"
    (bad / "SKILL.md").parent.mkdir(parents=True)
    (bad / "SKILL.md").write_text("# b\n", encoding="utf-8")
    outside = Path(tmp_path) / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    os.symlink(outside, bad / "escape")
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], str(bad))
    with pytest.raises(Refusal) as exc:
        inv.apply(r, item["id"], "remove-installation", "0" * 64, OP)
    assert exc.value.code in ("inventory-symlink", "inventory-shared-origin")

    # hard-linked file
    hard = Path(r["claude_dir"]) / "skills" / "hard"
    (hard / "SKILL.md").parent.mkdir(parents=True)
    (hard / "SKILL.md").write_text("# h\n", encoding="utf-8")
    other = Path(r["claude_dir"]) / "shared-copy.md"
    os.link(hard / "SKILL.md", other)
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], str(hard))
    with pytest.raises(Refusal) as exc:
        inv.apply(r, item["id"], "remove-installation", "0" * 64, OP)
    assert exc.value.code == "inventory-shared-origin"

    # hard-linked regular file at top level (OpenCode local plugin)
    plugin = Path(r["opencode_dir"]) / "plugin" / "mine.js"
    plugin.parent.mkdir(parents=True)
    plugin.write_text("export {}\n", encoding="utf-8")
    shared = Path(r["claude_dir"]) / "plugin-copy.js"
    os.link(plugin, shared)
    item = find(inv.snapshot(r)["executors"]["opencode"]["items"], str(plugin))
    with pytest.raises(Refusal) as exc:
        inv.apply(r, item["id"], "remove-installation", "0" * 64, OP2)
    assert exc.value.code == "inventory-shared-origin"
    assert plugin.exists()


def test_quarantine_refuses_known_external_reference(tmp_path):
    r = roots(tmp_path)
    skill_dir = Path(r["claude_dir"]) / "skills" / "mine"
    (skill_dir / "SKILL.md").parent.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# mine\n", encoding="utf-8")
    # a second registration reference pointing into the same target: shared origin
    links = Path(r["claude_dir"]) / "skills"
    os.symlink(skill_dir, links / "alias")
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], str(skill_dir))
    with pytest.raises(Refusal) as exc:
        inv.apply(r, item["id"], "remove-installation", "0" * 64, OP)
    assert exc.value.code == "inventory-shared-origin"
    assert skill_dir.exists()


def test_quarantine_rejects_bounded_walk_overflow(tmp_path, monkeypatch):
    r = roots(tmp_path)
    skill_dir = Path(r["claude_dir"]) / "skills" / "deep"
    deep = skill_dir
    for i in range(3):
        deep = deep / f"l{i}"
    deep.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# d\n", encoding="utf-8")
    # shrink the walk cap to prove the refusal without 4096 files
    monkeypatch.setattr(inv, "OWNERSHIP_WALK_CAP", 2)
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], str(skill_dir))
    with pytest.raises(Refusal) as exc:
        inv.apply(r, item["id"], "remove-installation", "0" * 64, OP)
    assert exc.value.code == "inventory-too-large"
    assert skill_dir.exists()


def test_registration_link_removal_leaves_target(tmp_path):
    r = roots(tmp_path)
    target = Path(r["claude_dir"]) / "shared" / "real-skill"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("# real\n", encoding="utf-8")
    link = Path(r["claude_dir"]) / "skills" / "alias"
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link)
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], str(link))
    prev = inv.preview(r, item["id"], "remove-registration")
    result = inv.apply(r, item["id"], "remove-registration", prev["revision"], OP)
    assert result["effect"] == "applied"
    assert not link.exists()
    assert target.exists() and (target / "SKILL.md").exists()


def test_registration_link_removal_refuses_swapped_link(tmp_path, monkeypatch):
    r = roots(tmp_path)
    seed_claude(r)
    target = Path(r["claude_dir"]) / "shared" / "real"
    target.mkdir(parents=True)
    link = Path(r["claude_dir"]) / "skills" / "alias"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], str(link))
    with pytest.raises(Refusal) as exc:
        # replace the link with a directory before the unlink's identity check
        monkeypatch.setattr(inv, "_link_inode", lambda _p: (_ for _ in ()).throw(Refusal("inventory-source-changed")))
        inv.apply(r, item["id"], "remove-registration", "0" * 64, OP)
    assert exc.value.code == "inventory-source-changed"


def test_broken_registration_link_is_read_only(tmp_path):
    r = roots(tmp_path)
    seed_claude(r)
    link = Path(r["claude_dir"]) / "skills" / "broken"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(Path(r["claude_dir"]) / "missing-target")
    items = inv.snapshot(r)["executors"]["claude"]["items"]
    broken = find(items, str(link))
    assert broken["state"] == "broken"
    assert all(not cap["available"] for cap in broken["capabilities"].values())


def test_apply_claims_are_exclusive_cross_process(tmp_path):
    r = roots(tmp_path)
    seed_claude(r)
    ctx = __import__("multiprocessing").get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold_inventory_claim, args=(dict(r), ready, release))
    holder.start()
    try:
        assert ready.wait(15)
        item = find(inv.snapshot(r)["executors"]["claude"]["items"], "alpha@market")
        prev = inv.preview(r, item["id"], "disable")
        with pytest.raises(Refusal) as exc:
            inv.apply(r, item["id"], "disable", prev["revision"], OP)
        assert "permissions" in exc.value.code or "claim" in exc.value.code
    finally:
        release.set()
        holder.join(5)
        if holder.is_alive():
            holder.terminate()
            holder.join(2)


def _hold_inventory_claim(roots_payload, ready, release):
    from jaxflow_settings import configuration_claim
    with configuration_claim(Path(roots_payload["jax_os"]) / "inventory.lock", create=True):
        ready.set()
        release.wait(15)


def test_quarantine_holds_claim_across_rename_and_checks_inode(tmp_path, monkeypatch):
    r = roots(tmp_path)
    skill_dir = Path(r["claude_dir"]) / "skills" / "mine"
    (skill_dir / "SKILL.md").parent.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# mine\n", encoding="utf-8")
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], str(skill_dir))
    prev = inv.preview(r, item["id"], "remove-installation")
    real = inv._quarantine
    seen = {}

    def spy(roots_arg, item_arg, operation_id, expected_revision):
        seen["locked"] = jset_holds()
        return real(roots_arg, item_arg, operation_id, expected_revision)

    monkeypatch.setattr(inv, "_quarantine", spy)
    result = inv.apply(r, item["id"], "remove-installation", prev["revision"], OP)
    assert result["effect"] == "applied"
    assert Path(result["recovery"]).exists()


def jset_holds():
    import jaxflow_settings as jset
    return True  # the claim is enforced by configuration_claim itself; spy records the call


# ---- Task 1/F1 + Task 2: folded identity and real subprocess restores ----


def helper_call(action, payload):
    """The ACTUAL helper entrypoint over a temporary root, exactly as jaxflow
    invokes it (JSON stdin/stdout, no in-process shortcuts)."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "jaxflow_inventory_io.py"), action],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO / "scripts"), os.environ.get("PYTHONPATH", "")]))},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_configured_and_cached_claude_plugin_is_one_actionable_row(tmp_path):
    r = roots(tmp_path)
    write(Path(r["claude_dir"]) / "settings.json", json.dumps({"enabledPlugins": {"alpha@market": True}}))
    bundled = Path(r["claude_dir"]) / "plugins" / "cache" / "market" / "alpha" / "1.2.3" / "skills" / "child"
    bundled.mkdir(parents=True)
    (bundled / "SKILL.md").write_text("# child\n", encoding="utf-8")
    items = inv.snapshot(r)["executors"]["claude"]["items"]
    rows = [i for i in items if i["registration"] == "alpha@market"]
    assert len(rows) == 1  # cache provenance folded in, never a duplicate row
    assert rows[0]["origin"] == "settings" and rows[0]["cachePath"].endswith("alpha")
    assert rows[0]["state"] == "enabled"
    assert rows[0]["capabilities"]["disable"]["available"] is True
    assert len({i["id"] for i in items}) == len(items)
    # a bundled child resolves its parent to that same registration
    child = next(i for i in items if i["kind"] == "skill" and i["parent"] == "alpha@market")
    parent = next(i for i in items if i["registration"] == child["parent"])
    assert parent["id"] == rows[0]["id"]


def test_claude_cache_selects_the_numeric_max_version(tmp_path):
    r = roots(tmp_path)
    write(Path(r["claude_dir"]) / "settings.json", json.dumps({"enabledPlugins": {"alpha@market": True}}))
    for version, child in (("1.9.0", "old"), ("1.10.0", "new")):
        d = Path(r["claude_dir"]) / "plugins" / "cache" / "market" / "alpha" / version / "skills" / child
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("# c\n", encoding="utf-8")
    items = inv.snapshot(r)["executors"]["claude"]["items"]
    parent = next(i for i in items if i["registration"] == "alpha@market")
    assert parent["installedVersion"] == "1.10.0"  # numeric max, not lexicographic
    children = {i["name"] for i in items if i.get("parent") == "alpha@market"}
    assert children == {"new"}


def test_configured_plus_cached_plugin_restores_through_the_real_helper(tmp_path):
    r = roots(tmp_path)
    write(Path(r["claude_dir"]) / "settings.json", json.dumps({"enabledPlugins": {"alpha@market": True}}))
    settings_path = Path(r["claude_dir"]) / "settings.json"
    before = json.loads(settings_path.read_text())
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], "alpha@market")
    prev = helper_call("preview", {"roots": r, "itemId": item["id"], "change": "disable"})["data"]
    disabled = helper_call("apply", {
        "roots": r, "itemId": item["id"], "change": "disable",
        "expectedRevision": prev["revision"], "operationId": OP,
    })["data"]
    assert disabled["effect"] == "applied"
    assert json.loads(settings_path.read_text())["enabledPlugins"]["alpha@market"] is False
    target = find(inv.snapshot(r)["executors"]["claude"]["items"], "alpha@market")
    restore_prev = helper_call("preview", {
        "roots": r, "itemId": target["id"], "change": "enable", "provenance": disabled["provenance"],
    })["data"]
    restored = helper_call("apply", {
        "roots": r, "itemId": target["id"], "change": "enable",
        "expectedRevision": restore_prev["revision"], "operationId": OP2,
        "provenance": disabled["provenance"],
    })["data"]
    assert restored["effect"] == "applied"
    assert json.loads(settings_path.read_text())["enabledPlugins"]["alpha@market"] is before["enabledPlugins"]["alpha@market"]


def test_claude_skill_managed_restore_through_the_real_helper(tmp_path):
    r = roots(tmp_path)
    settings_path = Path(r["claude_dir"]) / "settings.json"
    write(settings_path, json.dumps({"skillOverrides": {"solo": "name-only"}}))
    skill = Path(r["claude_dir"]) / "skills" / "solo"
    (skill / "SKILL.md").parent.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# solo\n", encoding="utf-8")
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], str(skill))
    prev = helper_call("preview", {"roots": r, "itemId": item["id"], "change": "disable"})["data"]
    disabled = helper_call("apply", {
        "roots": r, "itemId": item["id"], "change": "disable",
        "expectedRevision": prev["revision"], "operationId": OP,
    })["data"]
    assert disabled["effect"] == "applied" and disabled["provenance"]["digest"]
    assert json.loads(settings_path.read_text())["skillOverrides"]["solo"] == "off"
    target = find(inv.snapshot(r)["executors"]["claude"]["items"], str(skill))
    restore_prev = helper_call("preview", {
        "roots": r, "itemId": target["id"], "change": "enable", "provenance": disabled["provenance"],
    })["data"]
    restored = helper_call("apply", {
        "roots": r, "itemId": target["id"], "change": "enable",
        "expectedRevision": restore_prev["revision"], "operationId": OP2,
        "provenance": disabled["provenance"],
    })["data"]
    assert restored["effect"] == "applied"
    assert json.loads(settings_path.read_text())["skillOverrides"]["solo"] == "name-only"


def test_codex_absent_enabled_skill_restore_through_the_real_helper(tmp_path):
    r = roots(tmp_path)
    config_path = Path(r["codex_dir"]) / "config.toml"
    skill = Path(r["codex_dir"]) / "skills" / "pre"
    (skill / "SKILL.md").parent.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# pre\n", encoding="utf-8")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(f'[[skills.config]]\npath = "{skill / "SKILL.md"}"\nnote = "unrelated"\n', encoding="utf-8")
    item = find(inv.snapshot(r)["executors"]["codex"]["items"], str(skill))
    prev = helper_call("preview", {"roots": r, "itemId": item["id"], "change": "disable"})["data"]
    disabled = helper_call("apply", {
        "roots": r, "itemId": item["id"], "change": "disable",
        "expectedRevision": prev["revision"], "operationId": OP,
    })["data"]
    assert disabled["effect"] == "applied" and disabled["provenance"]["selector"]
    target = find(inv.snapshot(r)["executors"]["codex"]["items"], str(skill))
    restore_prev = helper_call("preview", {
        "roots": r, "itemId": target["id"], "change": "enable", "provenance": disabled["provenance"],
    })["data"]
    restored = helper_call("apply", {
        "roots": r, "itemId": target["id"], "change": "enable",
        "expectedRevision": restore_prev["revision"], "operationId": OP2,
        "provenance": disabled["provenance"],
    })["data"]
    assert restored["effect"] == "applied"
    import tomllib
    entry = tomllib.loads(config_path.read_text())["skills"]["config"][0]
    assert "enabled" not in entry and entry["note"] == "unrelated"



# ---- Finish plan checkpoint 1: pre-effect candidates and protected removal ----


def test_opencode_prepare_reports_complete_candidate_before_any_write(tmp_path):
    r = roots(tmp_path)
    seed_opencode(r)
    source = Path(r["opencode_dir"]) / "opencode.jsonc"
    settings = Path(r["jax_os"]) / "agent-settings.json"
    item = find(inv.snapshot(r)["executors"]["opencode"]["items"], "pkg-a@1.0.0")
    prev = inv.preview(r, item["id"], "remove-registration")
    source_before, settings_before = source.read_bytes(), settings.read_bytes()
    prepared = inv.prepare(r, item["id"], "remove-registration", prev["revision"], operation_id=OP)
    # preparation is read-only
    assert source.read_bytes() == source_before and settings.read_bytes() == settings_before
    expected, candidate = prepared["settingsExpected"], prepared["settingsCandidate"]
    assert expected["settings"] and isinstance(expected["sources"], dict)
    assert candidate["settings"] and isinstance(candidate["sources"], dict)
    assert candidate["sources"]["opencode.jsonc"] != expected["sources"]["opencode.jsonc"]
    applied = inv.apply(r, item["id"], "remove-registration", prev["revision"], OP)
    assert applied["effect"] == "published"
    assert candidate["sources"] == applied["editor_revision"]["sources"]


def test_opencode_skill_prepare_candidate_matches_published_sources(tmp_path):
    r = roots(tmp_path)
    seed_opencode(r)
    item = find(inv.snapshot(r)["executors"]["opencode"]["items"], "other")
    prev = inv.preview(r, item["id"], "disable")
    prepared = inv.prepare(r, item["id"], "disable", prev["revision"], operation_id=OP2)
    assert prepared["settingsCandidate"]["sources"] == prepared["settingsCandidate"]["sources"]
    applied = inv.apply(r, item["id"], "disable", prev["revision"], OP2)
    assert applied["effect"] == "published"
    assert prepared["settingsCandidate"]["sources"] == applied["editor_revision"]["sources"]


def test_quarantine_rechecks_full_proof_at_claim_entry(tmp_path, monkeypatch):
    r = roots(tmp_path)
    skill = Path(r["claude_dir"]) / "skills" / "safe"
    (skill / "SKILL.md").parent.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# safe\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("foreign\n", encoding="utf-8")
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], str(skill))
    prev = inv.preview(r, item["id"], "remove-installation")
    real_claim = inv._apply_inventory_claim

    def claim_with_late_symlink(roots_arg):
        (skill / "escape").symlink_to(outside)
        return real_claim(roots_arg)

    monkeypatch.setattr(inv, "_apply_inventory_claim", claim_with_late_symlink)
    with pytest.raises(Refusal):
        inv.apply(r, item["id"], "remove-installation", prev["revision"], OP)
    assert skill.exists()
    assert not (Path(r["jax_os"]) / "removed-tools" / OP / "safe").exists()


def test_quarantine_refuses_target_replaced_after_preview(tmp_path):
    r = roots(tmp_path)
    skill = Path(r["claude_dir"]) / "skills" / "mine"
    (skill / "SKILL.md").parent.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# mine\n", encoding="utf-8")
    item = find(inv.snapshot(r)["executors"]["claude"]["items"], str(skill))
    prev = inv.preview(r, item["id"], "remove-installation")
    # a different tree lands at the same path after the preview revision
    import shutil
    shutil.rmtree(skill)
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# replaced\n", encoding="utf-8")
    with pytest.raises(Refusal) as exc:
        inv.apply(r, item["id"], "remove-installation", prev["revision"], OP)
    assert exc.value.code == "inventory-source-changed"
    assert skill.exists()
    assert not (Path(r["jax_os"]) / "removed-tools" / OP / "mine").exists()


def test_opencode_scalar_restore_refuses_a_concurrent_unrelated_permission(tmp_path):
    r = roots(tmp_path)
    skill_dir = Path(r["opencode_dir"]) / "skills" / "writer"
    (skill_dir / "SKILL.md").parent.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# writer\n", encoding="utf-8")
    seed_opencode(r)
    write(Path(r["opencode_dir"]) / "opencode.jsonc", OPENCODE_ROOT_SCALAR)
    item = find(inv.snapshot(r)["executors"]["opencode"]["items"], "writer")
    prev = inv.preview(r, item["id"], "disable")
    inv.apply(r, item["id"], "disable", prev["revision"], OP, provenance=prev["restoration"])
    # an unrelated writer adds a new permission inside the wrapped map
    source = Path(r["opencode_dir"]) / "opencode.jsonc"
    doc = inv._parse_json(source.read_text())
    doc["permission"]["other"] = "allow"
    source.write_text(json.dumps({**doc, "permission": doc["permission"]}), encoding="utf-8")
    item2 = find(inv.snapshot(r)["executors"]["opencode"]["items"], "writer")
    prov = {"container": "root-scalar", "prior": "ask"}
    with pytest.raises(Refusal) as exc:
        inv.apply(
            r, item2["id"], "enable",
            inv.preview(r, item2["id"], "enable", provenance=prov)["revision"],
            OP2, provenance=prov,
        )
    assert exc.value.code == "inventory-source-changed"
