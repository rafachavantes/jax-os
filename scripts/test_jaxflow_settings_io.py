#!/usr/bin/env python3
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from jax_init import Refusal
from jaxflow_settings import LOCK_PATH, SETTINGS_PATH, builder_environment, resolve_profile
import jaxflow_settings_io as io
from jaxflow_settings_io import apply, edit_source, preview, snapshot

OPENROUTER = "@openrouter/ai-sdk-provider"
CONN = "fixture"
MODEL = "wire-real-model"
PROBE_SOURCE = """{
  // Keep this comment and every unrelated field.
  "provider": {
    "fixture": {
      "models": {
        "jaxflow-builder-default": {
          "id": "old-model",
          "variants": { "high": { "reasoning": { "effort": "high" } } },
          "options": { "provider": { "sort": "price", "only": ["primary"], "allow_fallbacks": true } }
        },
        "jaxflow-builder-fallback": {
          "id": "same-upstream",
          "variants": { "high": { "reasoning": { "effort": "high" } } },
          "options": { "provider": { "sort": "price", "only": ["fallback"], "allow_fallbacks": false } }
        },
        "native-override": { "id": "native-model", "name": "Remove this explicit override" }
      },
      "blacklist": ["native-hidden"],
      "whitelist": ["jaxflow-builder-default", "native-visible"]
    }
  },
  "unrelated": { "url": "https://example.invalid//preserve", "keep": true }
}
"""
CLEAN_SOURCE = """{
  // Keep this comment and every unrelated field.
  "provider": {
    "fixture": {
      "npm": "@openrouter/ai-sdk-provider",
      "models": {
        "wire-real-model": {
          "id": "wire-real-model",
          "name": "Wire",
          "limit": {"context": 128000, "output": 8192},
          "variants": { "high": { "reasoning": { "effort": "high" } } }
        }
      }
    }
  },
  "unrelated": { "url": "https://example.invalid//preserve", "keep": true }
}
"""
JSON_SOURCE = """{
  "provider": {
    "fixture": {
      "npm": "@openrouter/ai-sdk-provider",
      "models": {
        "wire-real-model": {
          "id": "wire-real-model",
          "name": "Wire",
          "limit": {"context": 128000, "output": 8192},
          "variants": { "high": { "reasoning": { "effort": "high" } } }
        }
      }
    }
  },
  "unrelated": { "url": "https://example.invalid//preserve", "keep": true }
}
"""


def _token(raw):
    return hashlib.sha256(raw).hexdigest()


def _paths(tmp_path):
    cfg = tmp_path / "config" / "opencode"
    cfg.mkdir(parents=True)
    jax = tmp_path / "jax-os"
    hermes = tmp_path / "hermes"
    hermes.mkdir()
    return {
        "settings": jax / "agent-settings.json",
        "lock": jax / "agent-settings.lock",
        "opencode.json": cfg / "opencode.json",
        "opencode.jsonc": cfg / "opencode.jsonc",
        "env": hermes / ".env",
    }


def _metadata():
    return {
        "providers": {
            CONN: {
                "npm": OPENROUTER,
                "models": {
                    MODEL: {
                        "id": MODEL,
                        "name": "Wire",
                        "limit": {"context": 128000, "output": 8192},
                        "variants": {"high": {"reasoning": {"effort": "high"}}},
                    }
                },
            }
        }
    }


def _intent(routing_default=None, routing_fallback=None, model=MODEL):
    if routing_default is None:
        routing_default = {"sort": "price", "allow_fallbacks": True, "only": ["primary"]}
    if routing_fallback is None:
        routing_fallback = {"sort": "price", "allow_fallbacks": False, "only": ["fallback"]}

    def profile(routing):
        return {
            "connection": CONN,
            "model": model,
            "effort": "high",
            "credential": {"kind": "native"},
            "routing": routing,
        }

    return {
        "kind": "save-settings",
        "reviewers": {
            "claude": {"model": "sonnet", "effort": "xhigh"},
            "codex": {"model": "gpt-5.6-luna", "effort": "xhigh"},
        },
        "builders": {"default": profile(routing_default), "fallback": profile(routing_fallback)},
    }


def _alias_set(bindings):
    return {(row["profile"], row["alias"], row["model"], row["connection"]) for row in bindings}


def _write_jsonc(paths, text):
    raw = text.encode("utf-8")
    paths["opencode.jsonc"].write_bytes(raw)
    return raw


def test_snapshot_does_not_create_authority_or_lock(tmp_path):
    paths = _paths(tmp_path)
    raw = _write_jsonc(paths, CLEAN_SOURCE)
    result = snapshot(paths, _metadata())
    assert result["editor_revision"]["settings"] == "absent"
    assert result["editor_revision"]["sources"]["opencode.json"] == "absent"
    assert result["editor_revision"]["sources"]["opencode.jsonc"] == _token(raw)
    assert result["settings"] is None
    assert not paths["settings"].exists()
    assert not paths["lock"].exists()
    assert not paths["settings"].parent.exists()


def test_preview_does_not_create_or_mutate(tmp_path):
    paths = _paths(tmp_path)
    raw = _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    preview(_intent(), expected, paths, _metadata())
    assert paths["opencode.jsonc"].read_bytes() == raw
    assert not paths["opencode.json"].exists()
    assert not paths["settings"].exists()
    assert not paths["lock"].exists()
    assert not paths["settings"].parent.exists()


def test_preview_projects_both_aliases_with_distinct_routing(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    result = preview(_intent(), expected, paths, _metadata())
    assert result["affected"] == ["opencode.jsonc"]
    after = _alias_set(result["aliases"]["after"])
    assert ("default", "jaxflow-builder-default", MODEL, CONN) in after
    assert ("fallback", "jaxflow-builder-fallback", MODEL, CONN) in after
    routed = {row["alias"]: row["routing"] for row in result["aliases"]["after"]}
    assert routed["jaxflow-builder-default"]["only"] == ["primary"]
    assert routed["jaxflow-builder-fallback"]["only"] == ["fallback"]
    assert routed["jaxflow-builder-default"]["allow_fallbacks"] is True
    assert routed["jaxflow-builder-fallback"]["allow_fallbacks"] is False
    effort = {row["alias"]: row.get("effort") for row in result["aliases"]["after"]}
    assert effort == {"jaxflow-builder-default": "high", "jaxflow-builder-fallback": "high"}


def test_preview_before_rows_carry_effort(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    apply(_intent(), expected, OP, paths, _metadata())
    updated = snapshot(paths, _metadata())["editor_revision"]
    result = preview(_intent(), updated, paths, _metadata())
    before = {row["alias"]: row.get("effort") for row in result["aliases"]["before"]}
    assert before == {"jaxflow-builder-default": "high", "jaxflow-builder-fallback": "high"}


def test_preview_refuses_stale_revision(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    paths["opencode.jsonc"].write_bytes(CLEAN_SOURCE.replace("Wire", "Changed").encode())
    with pytest.raises(Refusal) as exc:
        preview(_intent(), expected, paths, _metadata())
    assert exc.value.code == "agent-settings-source-changed"


def test_preview_refuses_unrelated_alias_collision(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, PROBE_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    with pytest.raises(Refusal) as exc:
        preview(_intent(), expected, paths, _metadata())
    assert exc.value.code == "agent-alias-collision"


def test_apply_moves_profile_back_to_connection_with_its_old_alias(tmp_path):
    # Live bug 2026-09-16: fallback moved openrouter -> tokenharbor left its alias behind,
    # and moving it back refused agent-alias-collision.
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    meta = _metadata()
    meta["providers"]["other"] = meta["providers"][CONN]
    moved = _intent()
    moved["builders"]["fallback"]["connection"] = "other"
    ops = iter(f"aaaaaaaa-bbbb-4ccc-8ddd-{n:012d}" for n in range(3))
    for intent in (_intent(), moved, _intent()):
        expected = snapshot(paths, meta)["editor_revision"]
        apply(intent, expected, next(ops), paths, meta)
    assert snapshot(paths, meta)["settings"]["builders"]["fallback"]["connection"] == CONN
    value = edit_source(paths["opencode.jsonc"].read_text(encoding="utf-8"), [], jsonc=True)["value"]
    assert value["provider"][CONN]["models"]["jaxflow-builder-fallback"]["id"] == MODEL


def test_preview_uses_existing_jsonc_provenance(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    result = preview(_intent(), expected, paths, _metadata())
    assert result["affected"] == ["opencode.jsonc"]


def test_preview_uses_existing_json_when_jsonc_absent(tmp_path):
    paths = _paths(tmp_path)
    paths["opencode.json"].write_text(JSON_SOURCE, encoding="utf-8")
    expected = snapshot(paths, _metadata())["editor_revision"]
    result = preview(_intent(), expected, paths, _metadata())
    assert result["affected"] == ["opencode.json"]


def test_preview_refuses_ambiguous_provider_in_both_files(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    paths["opencode.json"].write_text(JSON_SOURCE, encoding="utf-8")
    expected = snapshot(paths, _metadata())["editor_revision"]
    with pytest.raises(Refusal) as exc:
        preview(_intent(), expected, paths, _metadata())
    assert exc.value.code == "native-config-ambiguous"


def test_editor_jsonc_localized_edit_preserves_comment():
    result = edit_source(
        PROBE_SOURCE,
        [{"path": ["provider", "fixture", "models", "jaxflow-builder-default", "id"], "value": "same-upstream"}],
        jsonc=True,
    )
    assert "Keep this comment" in result["text"]
    models = result["value"]["provider"]["fixture"]["models"]
    assert models["jaxflow-builder-default"]["id"] == "same-upstream"
    assert models["jaxflow-builder-fallback"]["id"] == "same-upstream"
    assert result["value"]["unrelated"]["url"] == "https://example.invalid//preserve"
    assert models["jaxflow-builder-default"]["options"]["provider"] != models["jaxflow-builder-fallback"]["options"]["provider"]


def test_editor_preserves_crlf():
    crlf = PROBE_SOURCE.replace("\n", "\r\n")
    result = edit_source(
        crlf,
        [{"path": ["provider", "fixture", "models", "jaxflow-builder-default", "id"], "value": "same-upstream"}],
        jsonc=True,
    )
    assert "\r\n" in result["text"]
    assert "\n" not in result["text"].replace("\r\n", "")
    assert result["value"]["provider"]["fixture"]["models"]["jaxflow-builder-default"]["id"] == "same-upstream"


def test_editor_delete_preserves_blacklist_whitelist():
    localized = edit_source(
        PROBE_SOURCE,
        [{"path": ["provider", "fixture", "models", "jaxflow-builder-default", "id"], "value": "same-upstream"}],
        jsonc=True,
    )
    removed = edit_source(
        localized["text"],
        [{"path": ["provider", "fixture", "models", "native-override"], "remove": True}],
        jsonc=True,
    )
    fixture = removed["value"]["provider"]["fixture"]
    assert "native-override" not in fixture["models"]
    assert fixture["blacklist"] == ["native-hidden"]
    assert fixture["whitelist"] == ["jaxflow-builder-default", "native-visible"]
    assert "jaxflow-builder-default" in fixture["models"]
    assert "jaxflow-builder-fallback" in fixture["models"]


def test_editor_rejects_duplicate_keys_and_array_root():
    with pytest.raises(Refusal) as exc:
        edit_source('{"a": 1, "a": 2}', [], jsonc=True)
    assert exc.value.code == "native-config-malformed"
    with pytest.raises(Refusal) as exc:
        edit_source("[1, 2, 3]", [], jsonc=True)
    assert exc.value.code == "native-config-malformed"


OP = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
REV = "aaaaaaaabbbb4ccc8dddeeeeeeeeeeee"


def test_apply_publishes_native_then_settings_and_resolves(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    result = apply(_intent(), expected, OP, paths, _metadata())
    assert result["effect"] == "published"
    assert result["editor_revision"]["settings"] == REV
    settings = result["settings"]
    assert settings["revision"] == REV
    assert (paths["settings"].stat().st_mode & 0o777) == 0o600
    text = paths["opencode.jsonc"].read_text(encoding="utf-8")
    assert "Keep this comment" in text
    value = edit_source(text, [], jsonc=True)["value"]
    resolve_profile(settings, "default", effective_config=value)
    resolve_profile(settings, "fallback", effective_config=value)


def test_apply_projects_quantizations_into_alias_options(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    quantizations = ["fp8", "fp4"]
    intent = _intent(routing_default={
        "sort": "price", "allow_fallbacks": True, "only": ["primary"], "quantizations": quantizations,
    })
    apply(intent, expected, OP, paths, _metadata())
    value = edit_source(paths["opencode.jsonc"].read_text(encoding="utf-8"), [], jsonc=True)["value"]
    saved = value["provider"][CONN]["models"]["jaxflow-builder-default"]["options"]["provider"]
    assert saved["quantizations"] == quantizations


def test_apply_fault_before_native_leaves_targets(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    raw = _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    monkeypatch.setattr(io, "_FAULT", "before-native")
    with pytest.raises(Refusal):
        apply(_intent(), expected, OP, paths, _metadata())
    assert paths["opencode.jsonc"].read_bytes() == raw
    assert not paths["settings"].exists()


def test_apply_fault_before_settings_is_activation_pending(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    monkeypatch.setattr(io, "_FAULT", "before-settings")
    result = apply(_intent(), expected, OP, paths, _metadata())
    assert result["effect"] == "activation-pending"
    assert "jaxflow-builder-default" in paths["opencode.jsonc"].read_text(encoding="utf-8")
    assert not paths["settings"].exists()


def test_apply_refuses_stale_revision(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    paths["opencode.jsonc"].write_bytes(CLEAN_SOURCE.replace("Wire", "Changed").encode())
    with pytest.raises(Refusal) as exc:
        apply(_intent(), expected, OP, paths, _metadata())
    assert exc.value.code == "agent-settings-source-changed"
    assert not paths["settings"].exists()


def _hold_claim(lock_path, ready, release):
    from jaxflow_settings import configuration_claim
    with configuration_claim(Path(lock_path), create=True):
        ready.set()
        release.wait(15)


def test_apply_two_processes_contend_for_claim(tmp_path):
    paths = _paths(tmp_path)
    raw = _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold_claim, args=(str(paths["lock"]), ready, release))
    holder.start()
    try:
        assert ready.wait(15)
        with pytest.raises(Refusal) as exc:
            apply(_intent(), expected, OP, paths, _metadata())
        assert exc.value.code == "agent-settings-permissions"
        assert paths["opencode.jsonc"].read_bytes() == raw
        assert not paths["settings"].exists()
    finally:
        release.set()
        holder.join(5)
        if holder.is_alive():
            holder.terminate()
            holder.join(2)


def test_sync_status_op_is_gone(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=BytesIO(b"{}")))
    assert io.main(["prog", "sync-status"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"ok": False, "error": "agent-settings-malformed"}


def test_no_src_file_imports_the_deleted_sync_scripts():
    out = subprocess.run(
        ["grep", "-rl", "--exclude=test_*.py", "--exclude=*.test.*", "--exclude=conftest.py",
         "--exclude-dir=__pycache__", "provision_agent_settings\\|hermes_env_sync", "src", "scripts"],
        capture_output=True, text=True,
    )
    assert out.stdout.strip() == "", out.stdout


def test_read_credential_native_refuses_and_bws_reads_env(tmp_path):
    paths = _paths(tmp_path)
    with pytest.raises(Refusal):
        io.read_credential({"kind": "native"}, paths)
    paths["env"].write_text('JAX_PROVIDER_FIXTURE_API_KEY="s3cret"\n', encoding="utf-8")
    paths["env"].chmod(0o600)
    assert io.read_credential({
        "kind": "env",
        "env": "JAX_PROVIDER_FIXTURE_API_KEY",
    }, paths) == "s3cret"


def test_read_credential_refuses_the_retired_bws_env_shape(tmp_path):
    paths = _paths(tmp_path)
    paths["env"].write_text('JAX_PROVIDER_FIXTURE_API_KEY="s3cret"\n', encoding="utf-8")
    paths["env"].chmod(0o600)
    with pytest.raises(Refusal):
        io.read_credential({
            "kind": "bws-env",
            "env": "JAX_PROVIDER_FIXTURE_API_KEY",
            "secret_id": "11111111-1111-4111-8111-111111111111",
        }, paths)


def test_connection_health_reports_the_bound_env_name_when_present(monkeypatch):
    monkeypatch.setenv("JAX_PROVIDER_FIXTURE_API_KEY", "sk-live")
    native = {"options": {"apiKey": "{env:JAX_PROVIDER_FIXTURE_API_KEY}"}}
    auth, health, credential = io._connection_health(native, None, os.environ)
    assert credential == {"kind": "env", "env": "JAX_PROVIDER_FIXTURE_API_KEY"}
    assert health == "registered"


def test_connection_health_reports_the_bound_env_name_when_missing(monkeypatch):
    monkeypatch.delenv("JAX_PROVIDER_FIXTURE_API_KEY", raising=False)
    native = {"options": {"apiKey": "{env:JAX_PROVIDER_FIXTURE_API_KEY}"}}
    auth, health, credential = io._connection_health(native, None, os.environ)
    assert credential == {"kind": "env", "env": "JAX_PROVIDER_FIXTURE_API_KEY"}
    assert health == "missing"


LEAKY_SOURCE = """{
  "provider": {
    "fixture": {
      "npm": "@openrouter/ai-sdk-provider",
      "name": "Fixture",
      "options": {"baseURL": "https://openrouter.ai/api/v1", "apiKey": "s3cret"},
      "models": {
        "wire-real-model": {
          "id": "wire-real-model",
          "name": "Wire",
          "limit": {"context": 128000, "output": 8192},
          "variants": { "high": { "reasoning": { "effort": "high" } } }
        },
        "jaxflow-builder-default": {
          "id": "wire-real-model",
          "variants": { "high": { "reasoning": { "effort": "high" } } }
        }
      }
    }
  }
}
"""
UNSUPPORTED_SOURCE = """{
  "provider": {
    "fixture": {
      "npm": "@fixture/unsupported",
      "models": {
        "wire-real-model": {"id": "wire-real-model", "name": "Wire"}
      }
    }
  }
}
"""
OP2 = "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
OP3 = "cccccccc-dddd-4eee-8fff-aaaaaaaaaaaa"
OP4 = "dddddddd-eeee-4fff-8aaa-bbbbbbbbbbbb"
OP5 = "eeeeeeee-ffff-4aaa-8bbb-cccccccccccc"


def _lstat_fp(path):
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size, st.st_mode)


def _provider_meta(cred=None, extra_providers=None):
    meta = _metadata()
    meta["providers"][CONN]["name"] = "Fixture"
    meta["providers"]["catalog-only"] = {
        "npm": "@ai-sdk/xai",
        "name": "Catalog Only",
        "models": {"grok": {"id": "grok", "name": "Grok"}},
    }
    if extra_providers:
        meta["providers"].update(extra_providers)
    if cred is not None:
        meta["credentials"] = {CONN: cred}
    return meta


def _row(connections, connection_id=CONN):
    return next(row for row in connections if row["id"] == connection_id)


def test_snapshot_lists_configured_credentialed_not_catalog_or_aliases(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, LEAKY_SOURCE)
    result = snapshot(paths, _provider_meta({"auth": "api-key", "status": "present"}))
    dumped = json.dumps(result)
    assert "s3cret" not in dumped
    assert "catalog-only" not in dumped
    ids = [row["id"] for row in result["connections"]]
    assert ids == [CONN]
    row = _row(result["connections"])
    assert row["label"] == "Fixture"
    assert row["adapter"] == "openrouter"
    assert row["base_url"] == "https://openrouter.ai/api/v1"
    assert row["auth"] == "api-key"
    assert row["health"] == "registered"
    assert row["credential"] == {"kind": "native"}
    assert row["editable"] is True
    assert row["used_by"] == []
    assert [model["id"] for model in row["models"]] == [MODEL]
    assert "apiKey" not in row
    assert "options" not in row
    model = row["models"][0]
    assert model["origin"] == "override"
    assert model["efforts"] == ["high"]
    assert model["no_effort"] is False
    assert model["compatible"] is True


@pytest.mark.parametrize(
    "cred,auth,health,credential",
    [
        ({"auth": "oauth", "status": "present"}, "oauth", "oauth-managed", {"kind": "native"}),
        ({"auth": "none", "status": "missing"}, "none", "missing", None),
        ({"auth": "api-key", "status": "unreadable"}, "api-key", "unavailable", None),
    ],
)
def test_snapshot_maps_credential_health(tmp_path, cred, auth, health, credential):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    row = _row(snapshot(paths, _provider_meta(cred))["connections"])
    assert row["auth"] == auth
    assert row["health"] == health
    assert row["credential"] == credential


def test_snapshot_marks_ambiguous_provenance_read_only(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    paths["opencode.json"].write_text(JSON_SOURCE, encoding="utf-8")
    row = _row(snapshot(paths, _provider_meta({"auth": "api-key", "status": "present"}))["connections"])
    assert row["editable"] is False
    assert row["reason"] == "native-config-ambiguous"


def test_snapshot_used_by_both_saved_profiles(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    apply(_intent(), expected, OP, paths, _metadata())
    row = _row(snapshot(paths, _provider_meta({"auth": "api-key", "status": "present"}))["connections"])
    assert row["used_by"] == ["default", "fallback"]


def test_save_connection_maps_adapter_and_leaves_settings_absent(tmp_path):
    paths = _paths(tmp_path)
    expected = snapshot(paths, _metadata())["editor_revision"]
    apply({
        "kind": "save-connection",
        "connection": {
            "id": "custom",
            "label": "Custom",
            "adapter": "openai-compatible",
            "base_url": "https://example.com/v1",
        },
    }, expected, OP, paths, _metadata())
    assert not paths["settings"].exists()
    value = edit_source(paths["opencode.json"].read_text(encoding="utf-8"), [], jsonc=False)["value"]
    native = value["provider"]["custom"]
    assert native["npm"] == "@ai-sdk/openai-compatible"
    assert native["name"] == "Custom"
    assert native["options"]["baseURL"] == "https://example.com/v1"
    assert "apiKey" not in native.get("options", {})


def test_save_connection_rejects_localhost_and_client_npm(tmp_path):
    paths = _paths(tmp_path)
    expected = snapshot(paths, _metadata())["editor_revision"]
    apply({
        "kind": "save-connection",
        "connection": {"id": "ok", "label": "Ok", "adapter": "xai"},
    }, expected, OP, paths, _metadata())
    expected = snapshot(paths, _metadata())["editor_revision"]
    with pytest.raises(Refusal) as exc:
        apply({
            "kind": "save-connection",
            "connection": {
                "id": "bad",
                "label": "Bad",
                "adapter": "openai-compatible",
                "base_url": "http://127.0.0.1/v1",
            },
        }, expected, OP, paths, _metadata())
    assert exc.value.code == "agent-settings-malformed"
    with pytest.raises(Refusal) as exc:
        apply({
            "kind": "save-connection",
            "connection": {
                "id": "bad",
                "label": "Bad",
                "adapter": "@ai-sdk/openai-compatible",
            },
        }, expected, OP, paths, _metadata())
    assert exc.value.code == "agent-settings-malformed"


def test_save_and_remove_model_override(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    apply({
        "kind": "save-model",
        "connection": CONN,
        "model": {
            "id": "extra-model",
            "label": "Extra",
            "limit": {"context": 1000, "output": 100},
            "reasoning": False,
            "tool_call": True,
            "effort_template": "none",
        },
    }, expected, OP, paths, _metadata())
    assert not paths["settings"].exists()
    value = edit_source(paths["opencode.jsonc"].read_text(encoding="utf-8"), [], jsonc=True)["value"]
    extra = value["provider"][CONN]["models"]["extra-model"]
    assert extra == {
        "id": "extra-model", "name": "Extra", "limit": {"context": 1000, "output": 100},
        "reasoning": False, "tool_call": True,
    }
    expected = snapshot(paths, _metadata())["editor_revision"]
    apply({"kind": "remove-model", "connection": CONN, "model": "extra-model"}, expected, OP2, paths, _metadata())
    value = edit_source(paths["opencode.jsonc"].read_text(encoding="utf-8"), [], jsonc=True)["value"]
    assert "extra-model" not in value["provider"][CONN]["models"]


def test_remove_catalog_model_adds_blacklist(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, PROBE_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    apply({"kind": "remove-model", "connection": CONN, "model": "native-visible"}, expected, OP, paths, _metadata())
    value = edit_source(paths["opencode.jsonc"].read_text(encoding="utf-8"), [], jsonc=True)["value"]
    fixture = value["provider"][CONN]
    assert "native-visible" not in fixture.get("whitelist", [])
    assert "native-hidden" in fixture["blacklist"]
    assert "native-visible" in fixture["blacklist"]
    assert "jaxflow-builder-default" in fixture["whitelist"]


def test_remove_connection_excludes_provider_without_settings(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    apply({"kind": "remove-connection", "connection": CONN}, expected, OP, paths, _provider_meta())
    assert not paths["settings"].exists()
    value = edit_source(paths["opencode.jsonc"].read_text(encoding="utf-8"), [], jsonc=True)["value"]
    assert CONN not in value.get("provider", {})
    assert CONN in value.get("disabled_providers", [])


def test_remove_referenced_connection_or_model_refuses(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    apply(_intent(), expected, OP, paths, _metadata())
    expected = snapshot(paths, _metadata())["editor_revision"]
    with pytest.raises(Refusal) as exc:
        apply({"kind": "remove-connection", "connection": CONN}, expected, OP2, paths, _metadata())
    assert exc.value.code == "agent-profile-conflict"
    with pytest.raises(Refusal) as exc:
        apply({"kind": "remove-model", "connection": CONN, "model": MODEL}, expected, OP3, paths, _metadata())
    assert exc.value.code == "agent-profile-conflict"


def test_provider_mutation_after_init_republishes_settings_revision(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    published = apply(_intent(), expected, OP, paths, _metadata())
    expected = snapshot(paths, _metadata())["editor_revision"]
    result = apply({
        "kind": "save-model",
        "connection": CONN,
        "model": {
            "id": "extra-model",
            "label": "Extra",
            "limit": {"context": 1000, "output": 100},
            "reasoning": False,
            "tool_call": True,
            "effort_template": "none",
        },
    }, expected, OP2, paths, _metadata())
    assert result["effect"] == "published"
    assert result["settings"]["revision"] == OP2.replace("-", "")
    assert result["settings"]["revision"] != published["settings"]["revision"]
    assert result["settings"]["builders"] == published["settings"]["builders"]


def test_unsupported_transport_is_incompatible_and_refuses_activation(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, UNSUPPORTED_SOURCE)
    row = _row(snapshot(paths, _provider_meta({"auth": "api-key", "status": "present"}))["connections"])
    assert row["models"][0]["compatible"] is False
    assert row["models"][0]["reason"] == "unsupported-transport"
    expected = snapshot(paths, _metadata())["editor_revision"]
    intent = _intent()
    intent["builders"]["default"]["routing"] = None
    intent["builders"]["fallback"]["routing"] = None
    with pytest.raises(Refusal) as exc:
        apply(intent, expected, OP, paths, _metadata())
    assert exc.value.code == "agent-profile-conflict"


VISIBILITY = {
    "provider": {
        "fixture": {
            "npm": "@ai-sdk/openai-compatible",
            "options": {"baseURL": "http://127.0.0.1:1/v1", "apiKey": "fixture"},
            "models": {
                "native-visible": {"id": "visible-wire"},
                "native-hidden": {"id": "hidden-wire"},
                "wire-real-model": {
                    "id": "wire-real-model",
                    "variants": {"high": {"reasoningEffort": "high"}},
                },
            },
            "whitelist": ["native-visible", "wire-real-model"],
        }
    },
    "permission": {"read": "deny"},
}


def _visibility_meta():
    return {
        "providers": {
            CONN: {
                "npm": "@ai-sdk/openai-compatible",
                "models": {
                    "native-visible": {"id": "visible-wire", "name": "Visible"},
                    "extra-model": {"id": "extra-model", "name": "Extra"},
                    "wire-real-model": {
                        "id": "wire-real-model",
                        "name": "Wire",
                        "variants": {"high": {"reasoningEffort": "high"}},
                    },
                },
            }
        },
        "credentials": {CONN: {"auth": "api-key", "status": "present"}},
    }


def _list_models(config, tmp_path, connection=CONN):
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path / "vis-home"),
        "OPENCODE_CONFIG": str(config),
        "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
        "XDG_CONFIG_HOME": str(tmp_path / "vis-xdg-config"),
        "XDG_DATA_HOME": str(tmp_path / "vis-xdg-data"),
        "XDG_CACHE_HOME": str(tmp_path / "vis-xdg-cache"),
        "XDG_STATE_HOME": str(tmp_path / "vis-xdg-state"),
        "NO_COLOR": "1",
        "TERM": "dumb",
        "LANG": "C.UTF-8",
    }
    argv = ["opencode", "models", "--pure"] if connection is None else ["opencode", "models", connection, "--pure"]
    proc = subprocess.run(argv, cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, f"opencode models failed: {proc.stderr[-1000:]}"
    prefix = f"{CONN}/"
    names = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if connection is None:
            if line.startswith(prefix):
                names.append(line[len(prefix):])
        else:
            names.append(line[len(prefix):] if line.startswith(prefix) else line)
    return names


def _whitelist(config):
    """The connection's whitelist as written to disk (no opencode spawn needed)."""
    value = edit_source(config.read_text(encoding="utf-8"), [], jsonc=False)["value"]
    return value["provider"][CONN]["whitelist"]


@pytest.mark.opencode
def test_native_listing_add_hide_and_alias_eligibility(tmp_path):
    if shutil.which("opencode") is None:
        pytest.skip("opencode binary not installed")
    paths = _paths(tmp_path)
    paths["opencode.json"].write_text(json.dumps(VISIBILITY), encoding="utf-8")
    config = paths["opencode.json"]
    meta = _visibility_meta()
    # Intermediate states are asserted on the written whitelist; ONE real `opencode models`
    # listing at the end proves the final state (opencode spawn budget, D12).
    assert _whitelist(config) == ["native-visible", "wire-real-model"]
    expected = snapshot(paths, meta)["editor_revision"]
    apply({
        "kind": "save-model",
        "connection": CONN,
        "model": {
            "id": "extra-model", "label": "Extra", "limit": {"context": 1, "output": 1},
            "reasoning": False, "tool_call": True, "effort_template": "none",
        },
    }, expected, OP, paths, meta)
    assert "extra-model" in _whitelist(config)
    expected = snapshot(paths, meta)["editor_revision"]
    apply({"kind": "remove-model", "connection": CONN, "model": "native-visible"}, expected, OP2, paths, meta)
    assert "native-visible" not in _whitelist(config)
    assert "extra-model" in _whitelist(config)
    expected = snapshot(paths, meta)["editor_revision"]
    intent = _intent()
    intent["builders"]["default"]["routing"] = None
    intent["builders"]["fallback"]["routing"] = None
    apply(intent, expected, OP3, paths, meta)
    whitelist = _whitelist(config)
    assert "jaxflow-builder-default" in whitelist
    assert "jaxflow-builder-fallback" in whitelist
    names = _list_models(config, tmp_path)
    assert "wire-real-model" in names
    assert "native-hidden" not in names
    assert "native-visible" not in names
    assert "extra-model" in names
    assert "jaxflow-builder-default" in names
    assert "jaxflow-builder-fallback" in names


@pytest.mark.opencode
def test_native_listing_hides_removed_provider_despite_catalog(tmp_path):
    if shutil.which("opencode") is None:
        pytest.skip("opencode binary not installed")
    paths = _paths(tmp_path)
    paths["opencode.json"].write_text(json.dumps(VISIBILITY), encoding="utf-8")
    meta = _visibility_meta()
    expected = snapshot(paths, meta)["editor_revision"]
    apply({"kind": "remove-connection", "connection": CONN}, expected, OP, paths, meta)
    value = edit_source(paths["opencode.json"].read_text(encoding="utf-8"), [], jsonc=False)["value"]
    assert CONN not in value["provider"]
    assert value["disabled_providers"] == [CONN]
    restored = json.loads(json.dumps(VISIBILITY))
    restored["disabled_providers"] = [CONN]
    paths["opencode.json"].write_text(json.dumps(restored), encoding="utf-8")
    assert _list_models(paths["opencode.json"], tmp_path, connection=None) == []


def test_provider_mutation_refuses_ambiguous_sources(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    paths["opencode.json"].write_text(JSON_SOURCE, encoding="utf-8")
    expected = snapshot(paths, _metadata())["editor_revision"]
    with pytest.raises(Refusal) as exc:
        apply({
            "kind": "save-model",
            "connection": CONN,
            "model": {
                "id": "extra-model", "limit": {"context": 1, "output": 1},
                "reasoning": False, "tool_call": True, "effort_template": "none",
            },
        }, expected, OP, paths, _metadata())
    assert exc.value.code == "native-config-ambiguous"


def test_writer_part1_consumer_lifecycle_never_touches_production(tmp_path):
    before = (_lstat_fp(SETTINGS_PATH), _lstat_fp(LOCK_PATH))
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    meta = _metadata()

    got = snapshot(paths, meta)
    assert got["settings"] is None
    assert got["editor_revision"]["settings"] == "absent"
    assert not paths["settings"].exists()
    assert not paths["lock"].exists()

    expected = got["editor_revision"]
    apply({
        "kind": "save-connection",
        "connection": {
            "id": "custom",
            "label": "Custom",
            "adapter": "openai-compatible",
            "base_url": "https://example.com/v1",
        },
    }, expected, OP, paths, meta)
    assert not paths["settings"].exists()

    expected = snapshot(paths, meta)["editor_revision"]
    intent = _intent()
    published = apply(intent, expected, OP2, paths, meta)
    assert published["effect"] == "published"
    native = edit_source(paths["opencode.jsonc"].read_text(encoding="utf-8"), [], jsonc=True)["value"]
    resolve_profile(published["settings"], "default", effective_config=native)
    resolve_profile(published["settings"], "fallback", effective_config=native)

    expected = snapshot(paths, meta)["editor_revision"]
    modelled = apply({
        "kind": "save-model",
        "connection": CONN,
        "model": {
            "id": "extra-model",
            "label": "Extra",
            "limit": {"context": 1000, "output": 100},
            "reasoning": False,
            "tool_call": True,
            "effort_template": "none",
        },
    }, expected, OP3, paths, meta)
    assert modelled["effect"] == "published"
    assert modelled["settings"]["builders"] == published["settings"]["builders"]
    native = edit_source(paths["opencode.jsonc"].read_text(encoding="utf-8"), [], jsonc=True)["value"]
    resolve_profile(modelled["settings"], "default", effective_config=native)
    resolve_profile(modelled["settings"], "fallback", effective_config=native)

    paths["env"].write_text('JAX_PROVIDER_FIXTURE_API_KEY="s3cret"\n', encoding="utf-8")
    paths["env"].chmod(0o600)
    intent["builders"]["default"]["credential"] = {
        "kind": "env",
        "env": "JAX_PROVIDER_FIXTURE_API_KEY",
    }
    expected = snapshot(paths, meta)["editor_revision"]
    credded = apply(intent, expected, OP4, paths, meta)
    assert credded["effect"] == "published"
    binding = credded["settings"]["builders"]["default"]["credential"]
    assert io.read_credential(binding, paths) == "s3cret"
    child = builder_environment(
        credded["settings"]["builders"]["default"],
        {"PATH": "/bin", "BWS_ACCESS_TOKEN": "tok"},
        env_path=paths["env"],
    )
    assert child["JAX_PROVIDER_FIXTURE_API_KEY"] == "s3cret"
    assert "BWS_ACCESS_TOKEN" not in child
    assert child["PATH"] == "/bin"

    stale = snapshot(paths, meta)["editor_revision"]
    prior = paths["settings"].read_bytes()
    raw = paths["opencode.jsonc"].read_bytes()
    assert b"example.invalid" in raw
    paths["opencode.jsonc"].write_bytes(raw.replace(b"example.invalid", b"example.changed"))
    with pytest.raises(Refusal) as exc:
        apply(intent, stale, OP5, paths, meta)
    assert exc.value.code == "agent-settings-source-changed"
    assert paths["settings"].read_bytes() == prior

    fresh = snapshot(paths, meta)["editor_revision"]
    current = json.loads(prior.decode("utf-8"))
    reconciled = apply({
        "kind": "save-settings",
        "reviewers": current["reviewers"],
        "builders": current["builders"],
    }, fresh, OP5, paths, meta)
    assert reconciled["effect"] == "published"
    native = edit_source(paths["opencode.jsonc"].read_text(encoding="utf-8"), [], jsonc=True)["value"]
    resolve_profile(reconciled["settings"], "default", effective_config=native)
    resolve_profile(reconciled["settings"], "fallback", effective_config=native)

    assert (_lstat_fp(SETTINGS_PATH), _lstat_fp(LOCK_PATH)) == before
    if before[0] is None:
        assert not SETTINGS_PATH.exists()


ENV_SOURCE = """{
  "provider": {
    "fixture": {
      "npm": "@openrouter/ai-sdk-provider",
      "options": {"apiKey": "{env:MISSING_JAX_PROVIDER_KEY}"},
      "models": {"wire-real-model": {"id": "wire-real-model", "name": "Wire"}}
    }
  }
}
"""
CLEAN_SOURCE_ENV = """{
  // Keep this comment and every unrelated field.
  "provider": {
    "fixture": {
      "npm": "@openrouter/ai-sdk-provider",
      "options": {"apiKey": "{env:JAX_PROVIDER_FIXTURE_API_KEY}"},
      "models": {"wire-real-model": {"id": "wire-real-model", "name": "Wire"}}
    }
  },
  "unrelated": { "url": "https://example.invalid//preserve", "keep": true }
}
"""
META_SOURCE = """{
  "provider": {
    "fixture": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {"apiKey": "k", "baseURL": "https://example.com/v1"},
      "models": {}
    }
  }
}
"""
VERBOSE = """fixture/metadata-model
{
  "id": "metadata-model",
  "api": {"id": "same-upstream", "npm": "@ai-sdk/openai-compatible"},
  "name": "Sanitized metadata model",
  "limit": {"context": 123456, "output": 4096},
  "capabilities": {"toolcall": true},
  "variants": {"high": {"reasoning": {"effort": "high"}, "reasoningEffort": "high"}}
}
"""
SAVE_MODEL = {"kind": "save-model", "connection": CONN, "model": {"id": "x", "limit": {"context": 1, "output": 1}, "reasoning": False, "tool_call": True, "effort_template": "none"}}


def _auth_paths(tmp_path):
    paths = _paths(tmp_path)
    auth = tmp_path / "xdg-data" / "opencode" / "auth.json"
    auth.parent.mkdir(parents=True)
    paths["auth"] = auth
    return paths


def _fake_run(config, verbose_by_id=None):
    verbose_by_id = {} if verbose_by_id is None else verbose_by_id

    def run(argv, cwd=None, env=None, **_kw):
        argv = list(argv)
        if argv[:4] == ["opencode", "debug", "config", "--pure"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps(config))
        if argv[:2] == ["opencode", "models"] and "--verbose" in argv:
            cid = argv[2] if len(argv) > 2 else None
            return SimpleNamespace(returncode=0, stdout=verbose_by_id.get(cid, ""))
        return SimpleNamespace(returncode=1, stdout="")

    return run


@pytest.mark.parametrize("path_present", [True, False])
def test_default_pure_run_resolves_user_opencode_without_mutating_inputs(tmp_path, monkeypatch, path_present):
    monkeypatch.setattr(io.Path, "home", lambda: tmp_path)
    user_bin = tmp_path / ".opencode" / "bin"
    user_bin.mkdir(parents=True)
    executable = user_bin / "opencode"
    executable.write_text("#!/bin/sh\nprintf 'safe-marker\\n'\n", encoding="utf-8")
    executable.chmod(0o755)
    env = {"PATH": str(user_bin if path_present else tmp_path / "empty")}
    argv = ["opencode", "models", "fixture", "--verbose", "--pure"]
    original_argv, original_env = list(argv), dict(env)
    result = io._default_pure_run(argv, env=env)
    assert result.stdout == b"safe-marker\n"
    assert argv == original_argv
    assert env == original_env


def test_discovery_configured_api_key_uses_short_adapter(tmp_path):
    paths = _auth_paths(tmp_path)
    _write_jsonc(paths, LEAKY_SOURCE)
    result = snapshot(paths, run=_fake_run({"provider": {}}), env={})
    dumped = json.dumps(result)
    assert "s3cret" not in dumped
    row = _row(result["connections"])
    assert row["adapter"] == "openrouter"
    assert row["auth"] == "api-key"
    assert row["health"] == "registered"
    assert row["credential"] == {"kind": "native"}
    assert [m["id"] for m in row["models"]] == [MODEL]


def test_discovery_unresolved_env_is_missing_not_registered(tmp_path):
    paths = _auth_paths(tmp_path)
    _write_jsonc(paths, ENV_SOURCE)
    result = snapshot(paths, run=_fake_run({"provider": {}}), env={})
    row = _row(result["connections"])
    assert row["health"] == "missing"
    # MOA-498: the missing env NAME is reported (it is not a secret); no value exists to leak.
    assert row["credential"] == {"kind": "env", "env": "MISSING_JAX_PROVIDER_KEY"}


def test_discovery_native_auth_only_connection(tmp_path):
    paths = _auth_paths(tmp_path)
    paths["auth"].write_text(json.dumps({"openai": {"type": "api", "key": "sk-auth"}}), encoding="utf-8")
    run = _fake_run({
        "provider": {
            "openai": {
                "npm": "@ai-sdk/openai",
                "name": "OpenAI",
                "models": {"gpt-4": {"id": "gpt-4", "name": "GPT-4"}},
            }
        }
    })
    result = snapshot(paths, run=run, env={})
    assert "sk-auth" not in json.dumps(result)
    assert [r["id"] for r in result["connections"]] == ["openai"]
    row = _row(result["connections"], "openai")
    assert row["auth"] == "api-key"
    assert row["health"] == "registered"
    assert row["credential"] == {"kind": "native"}


def test_discovery_oauth_is_selectable_native(tmp_path):
    paths = _auth_paths(tmp_path)
    paths["auth"].write_text(json.dumps({
        "fixture": {"type": "oauth", "refresh": "r-token", "access": "a-token", "expires": 1},
    }), encoding="utf-8")
    _write_jsonc(paths, CLEAN_SOURCE)
    result = snapshot(paths, run=_fake_run({"provider": {}}), env={})
    dumped = json.dumps(result)
    assert "r-token" not in dumped
    assert "a-token" not in dumped
    row = _row(result["connections"])
    assert row["auth"] == "oauth"
    assert row["health"] == "oauth-managed"
    assert row["credential"] == {"kind": "native"}


def test_discovery_missing_auth_is_missing(tmp_path):
    paths = _auth_paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    row = _row(snapshot(paths, run=_fake_run({"provider": {}}), env={})["connections"])
    assert row["health"] == "missing"
    assert row["credential"] is None


def test_discovery_unreadable_auth_is_unavailable(tmp_path):
    paths = _auth_paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    paths["auth"].symlink_to(tmp_path / "elsewhere.json")
    row = _row(snapshot(paths, run=_fake_run({"provider": {}}), env={})["connections"])
    assert row["health"] == "unavailable"
    assert row["credential"] is None


def test_discovery_catalog_only_is_not_listed(tmp_path):
    paths = _auth_paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    run = _fake_run({
        "provider": {
            "catalog-only": {"npm": "@ai-sdk/xai", "name": "Catalog Only", "models": {"grok": {"id": "grok"}}},
            CONN: {"npm": OPENROUTER},
        }
    })
    result = snapshot(paths, run=run, env={})
    assert [r["id"] for r in result["connections"]] == [CONN]
    assert "catalog-only" not in json.dumps(result)


def test_discovery_unsupported_adapter_stays_npm(tmp_path):
    paths = _auth_paths(tmp_path)
    _write_jsonc(paths, UNSUPPORTED_SOURCE)
    row = _row(snapshot(paths, run=_fake_run({"provider": {}}), env={})["connections"])
    assert row["adapter"] == "@fixture/unsupported"
    assert row["models"][0]["compatible"] is False


def test_discovery_disabled_provider_is_hidden(tmp_path):
    paths = _auth_paths(tmp_path)
    restored = json.loads(json.dumps(VISIBILITY))
    restored["disabled_providers"] = [CONN]
    paths["opencode.json"].write_text(json.dumps(restored), encoding="utf-8")
    assert snapshot(paths, run=_fake_run({"provider": {CONN: {"npm": "@ai-sdk/openai-compatible"}}}), env={})["connections"] == []


def test_discovery_hidden_model_is_omitted(tmp_path):
    paths = _auth_paths(tmp_path)
    paths["opencode.json"].write_text(json.dumps(VISIBILITY), encoding="utf-8")
    row = _row(snapshot(paths, run=_fake_run({"provider": {}}), env={})["connections"])
    assert [m["id"] for m in row["models"]] == ["native-visible", MODEL]


@pytest.mark.parametrize("declared_native", [True, False])
@pytest.mark.parametrize("npm", ["@ai-sdk/xai", OPENROUTER, "@ai-sdk/openai-compatible"])
@pytest.mark.parametrize("count", [1, 2])
def test_discovery_catalog_metadata_from_verbose(tmp_path, declared_native, npm, count):
    paths = _auth_paths(tmp_path)
    source = json.loads(META_SOURCE)
    native = source["provider"][CONN]
    native.pop("npm", None)
    if declared_native:
        native["npm"] = npm
    _write_jsonc(paths, json.dumps(source))
    verbose = VERBOSE.replace("@ai-sdk/openai-compatible", npm)
    if count == 2:
        verbose += verbose.replace("metadata-model", "second-model")
    run = _fake_run(source, verbose_by_id={CONN: verbose})
    result = snapshot(paths, run=run, env={})
    row = _row(result["connections"])
    dumped = json.dumps(result)
    assert '"k"' not in dumped
    expected_ids = ["metadata-model"] if count == 1 else ["metadata-model", "second-model"]
    assert [m["id"] for m in row["models"]] == expected_ids
    model = next(m for m in row["models"] if m["id"] == "metadata-model")
    assert model["label"] == "Sanitized metadata model"
    assert all("high" in m["efforts"] for m in row["models"])
    assert all(m["compatible"] for m in row["models"])
    assert row["adapter"] == io.NPM_ADAPTER[npm]


@pytest.mark.parametrize("selected", ["openai-compatible", "openai", "anthropic"])
def test_mixed_catalog_transports_survive_settings_publication(tmp_path, selected):
    paths = _auth_paths(tmp_path)
    source = {"provider": {CONN: {"options": {"apiKey": "fixture-key"}}}}
    _write_jsonc(paths, json.dumps(source))
    models = [{"id": mid, "api": {"npm": "@ai-sdk/" + mid},
               "variants": {"high": {"reasoningEffort": "high"}}}
              for mid in ("openai-compatible", "openai", "anthropic", "unknown")]
    verbose = "\n".join(f"{CONN}/{m['id']}\n{json.dumps(m)}" for m in models)
    run = _fake_run(source, {CONN: verbose})
    current = snapshot(paths, run=run, env={})
    choices = {m["id"]: m for m in _row(current["connections"])["models"]}
    assert all(choices[mid]["compatible"] for mid in ("openai-compatible", "openai", "anthropic"))
    assert choices["unknown"]["compatible"] is False
    intent = _intent(model=selected)
    for profile in intent["builders"].values():
        profile["routing"] = None
    intent["builders"]["fallback"]["model"] = "anthropic" if selected != "anthropic" else "openai"
    preview(intent, current["editor_revision"], paths, run=run, env={})
    published = apply(intent, current["editor_revision"], OP, paths, run=run, env={})
    written = json.loads(paths["opencode.jsonc"].read_text())
    assert "npm" not in written["provider"][CONN]
    for name in ("default", "fallback"):
        resolved = resolve_profile(published["settings"], name, effective_config=written)
        assert resolved["model"] == f"{CONN}/{intent['builders'][name]['model']}"
        assert resolved["effort"] == "high"
    current = snapshot(paths, run=run, env={})
    intent["builders"]["default"]["model"] = "unknown"
    with pytest.raises(Refusal, match="agent-profile-conflict"):
        preview(intent, current["editor_revision"], paths, run=run, env={})


def test_discovery_both_source_files_refuse_writes(tmp_path):
    paths = _auth_paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    paths["opencode.json"].write_text(JSON_SOURCE, encoding="utf-8")
    run = _fake_run({"provider": {}})
    expected = snapshot(paths, run=run, env={})["editor_revision"]
    with pytest.raises(Refusal) as exc:
        apply(SAVE_MODEL, expected, OP, paths, run=run, env={})
    assert exc.value.code == "native-config-ambiguous"


def test_discovery_extra_override_refuses_writes(tmp_path):
    paths = _auth_paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    extra = tmp_path / "project" / "opencode.json"
    extra.parent.mkdir()
    extra.write_text(JSON_SOURCE, encoding="utf-8")
    env = {"OPENCODE_CONFIG": str(extra)}
    run = _fake_run({"provider": {CONN: {"npm": OPENROUTER, "options": {"apiKey": "from-extra"}}}})
    expected = snapshot(paths, run=run, env=env)["editor_revision"]
    assert _row(snapshot(paths, run=run, env=env)["connections"])["id"] == CONN
    assert "from-extra" not in json.dumps(snapshot(paths, run=run, env=env))
    with pytest.raises(Refusal) as exc:
        preview(SAVE_MODEL, expected, paths, run=run, env=env)
    assert exc.value.code == "native-config-ambiguous"
    with pytest.raises(Refusal) as exc:
        apply(SAVE_MODEL, expected, OP, paths, run=run, env=env)
    assert exc.value.code == "native-config-ambiguous"



def test_bind_credential_writes_string_env_reference_and_refuse_invalid(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    apply({
        "kind": "bind-credential",
        "connection": CONN,
        "credential": {
            "kind": "env",
            "env": "JAX_PROVIDER_FIXTURE_API_KEY",
        },
    }, expected, OP, paths, _metadata())
    assert not paths["settings"].exists()
    value = edit_source(paths["opencode.jsonc"].read_text(encoding="utf-8"), [], jsonc=True)["value"]
    native = value["provider"][CONN]
    assert native["options"]["apiKey"] == "{env:JAX_PROVIDER_FIXTURE_API_KEY}"
    with pytest.raises(Refusal) as exc:
        expected = snapshot(paths, _metadata())["editor_revision"]
        preview({
            "kind": "bind-credential",
            "connection": CONN,
            "credential": {"kind": "env", "env": "NOT A VALID NAME"},
        }, expected, paths, _metadata())
    assert exc.value.code == "agent-settings-malformed"


def test_bind_credential_accepts_native_and_removes_the_apikey_override(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE_ENV)
    expected = snapshot(paths, _metadata())["editor_revision"]
    result = apply({
        "kind": "bind-credential",
        "connection": CONN,
        "credential": {"kind": "native"},
    }, expected, OP, paths, _metadata())
    assert result["effect"] == "published"
    value = edit_source(paths["opencode.jsonc"].read_text(encoding="utf-8"), [], jsonc=True)["value"]
    assert "apiKey" not in value["provider"][CONN]["options"]


def test_snapshot_binding_reads_env_reference_only(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE_ENV)
    binding = {
        "provider_id": CONN,
        "env_name": "JAX_PROVIDER_FIXTURE_API_KEY",
        "secret_id": OP,
    }
    result = snapshot(paths, _metadata(), [binding], env={"JAX_PROVIDER_FIXTURE_API_KEY": "s3cret"})
    row = _row(result["connections"])
    assert row["credential"] == {
        "kind": "env",
        "env": "JAX_PROVIDER_FIXTURE_API_KEY",
    }
    assert row["health"] == "registered"
    assert "s3cret" not in json.dumps(result)


def test_snapshot_binding_health_missing_when_env_absent(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE_ENV)
    binding = {
        "provider_id": CONN,
        "env_name": "JAX_PROVIDER_FIXTURE_API_KEY",
        "secret_id": OP,
    }
    result = snapshot(paths, _metadata(), [binding], env={})
    row = _row(result["connections"])
    assert row["credential"] == {"kind": "env", "env": "JAX_PROVIDER_FIXTURE_API_KEY"}
    assert row["health"] == "missing"


def test_save_model_reasoning_writes_variants_and_rejects_bad_limits(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    apply({
        "kind": "save-model",
        "connection": CONN,
        "model": {
            "id": "extra-reasoning",
            "label": "Extra Reasoning",
            "limit": {"context": 1000, "output": 100},
            "reasoning": True,
            "tool_call": True,
            "effort_template": "reasoning",
            "efforts": ["high", "medium"],
        },
    }, expected, OP, paths, _metadata())
    value = edit_source(paths["opencode.jsonc"].read_text(encoding="utf-8"), [], jsonc=True)["value"]
    extra = value["provider"][CONN]["models"]["extra-reasoning"]
    assert extra["limit"] == {"context": 1000, "output": 100}
    assert extra["reasoning"] is True
    assert extra["tool_call"] is True
    assert set(extra["variants"]) == {"high", "medium"}
    assert extra["variants"]["high"]["reasoning"]["effort"] == "high"
    assert extra["variants"]["medium"]["reasoning"]["effort"] == "medium"


def test_save_model_rejects_unbalanced_reasoning_and_bad_limits(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    for model in [
        {"id": "x", "reasoning": False, "effort_template": "reasoning", "efforts": ["high"]},
        {"id": "x", "reasoning": True, "effort_template": "none"},
        {"id": "x", "reasoning": True, "effort_template": "reasoning", "efforts": ["high", "high"]},
        {"id": "x", "limit": {"context": 0, "output": 100}, "effort_template": "none"},
        {"id": "x", "extra": 1, "effort_template": "none"},
    ]:
        with pytest.raises(Refusal) as exc:
            apply({"kind": "save-model", "connection": CONN, "model": model}, expected, OP, paths, _metadata())
        assert exc.value.code == "agent-settings-malformed"
    with pytest.raises(Refusal) as exc:
        apply({
            "kind": "save-model",
            "connection": CONN,
            "model": {"id": "x", "reasoning": True, "tool_call": False, "effort_template": "reasoning", "efforts": ["high"]},
        }, expected, OP, paths, _metadata())
    assert exc.value.code == "agent-settings-malformed"


def test_bind_credential_with_authority_updates_profile_refs_and_revises(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    apply(_intent(), expected, OP, paths, _metadata())
    expected = snapshot(paths, _metadata())["editor_revision"]
    result = apply({
        "kind": "bind-credential",
        "connection": CONN,
        "credential": {
            "kind": "env",
            "env": "JAX_PROVIDER_FIXTURE_API_KEY",
        },
    }, expected, OP2, paths, _metadata())
    assert result["effect"] == "published"
    assert result["settings"]["revision"] == OP2.replace("-", "")
    for profile_name in ("default", "fallback"):
        assert result["settings"]["builders"][profile_name]["credential"] == {
            "kind": "env",
            "env": "JAX_PROVIDER_FIXTURE_API_KEY",
        }
        assert result["settings"]["builders"][profile_name]["model"] == _intent()["builders"][profile_name]["model"]
        assert result["settings"]["builders"][profile_name]["connection"] == CONN
    value = edit_source(paths["opencode.jsonc"].read_text(encoding="utf-8"), [], jsonc=True)["value"]
    assert value["provider"][CONN]["options"]["apiKey"] == "{env:JAX_PROVIDER_FIXTURE_API_KEY}"


def test_referenced_model_edit_refuses_incompatible_effort(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    apply(_intent(), expected, OP, paths, _metadata())
    expected = snapshot(paths, _metadata())["editor_revision"]
    with pytest.raises(Refusal) as exc:
        apply({
            "kind": "save-model",
            "connection": CONN,
            "model": {
                "id": MODEL,
                "limit": {"context": 1, "output": 1},
                "reasoning": True,
                "tool_call": True,
                "effort_template": "reasoning",
                "efforts": ["low"],
            },
        }, expected, OP2, paths, _metadata())
    assert exc.value.code == "agent-profile-conflict"


def test_save_model_appears_in_discovered_choices(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, CLEAN_SOURCE)
    expected = snapshot(paths, _metadata())["editor_revision"]
    apply({
        "kind": "save-model",
        "connection": CONN,
        "model": {
            "id": "brand-new-reasoning",
            "label": "Brand New",
            "limit": {"context": 4096, "output": 512},
            "reasoning": True,
            "tool_call": True,
            "effort_template": "reasoning_effort",
            "efforts": ["high"],
        },
    }, expected, OP, paths, _metadata())
    row = _row(snapshot(paths, _provider_meta({"auth": "api-key", "status": "present"}))["connections"])
    chosen = next(m for m in row["models"] if m["id"] == "brand-new-reasoning")
    assert chosen["origin"] == "override"
    assert chosen["efforts"] == ["high"]
    assert chosen["no_effort"] is False
    assert chosen["compatible"] is True


def test_read_credential_native_resolves_real_env_reference(tmp_path):
    paths = _paths(tmp_path)
    paths["env"].write_text('OPENROUTER_API_KEY="native-key-9f3c"\n', encoding="utf-8")
    paths["env"].chmod(0o600)
    source = """{
      "provider": {
        "fixture": {
          "npm": "@openrouter/ai-sdk-provider",
          "options": {"apiKey": "{env:OPENROUTER_API_KEY}"},
          "models": {}
        }
      }
    }
    """
    _write_jsonc(paths, source)
    assert io.read_credential({"kind": "native", "connection": CONN}, paths) == "native-key-9f3c"
    with pytest.raises(Refusal):
        io.read_credential({"kind": "native", "connection": "missing"}, paths)


@pytest.mark.parametrize(
    ("source_key", "auth", "expected"),
    [
        (None, {"type": "api", "key": "auth-only-key"}, "auth-only-key"),
        ("literal-source-key", {"type": "api", "key": "auth-key"}, "literal-source-key"),
        ("{env:JAX_MISSING_NATIVE_KEY}", {"type": "api", "key": "auth-key"}, None),
        (None, {"type": "oauth", "refresh": "refresh-token"}, None),
    ],
)
def test_read_credential_native_uses_auth_only_as_fallback(tmp_path, monkeypatch, source_key, auth, expected):
    monkeypatch.delenv("JAX_MISSING_NATIVE_KEY", raising=False)
    paths = _auth_paths(tmp_path)
    paths["auth"].write_text(json.dumps({CONN: auth}), encoding="utf-8")
    native = {"models": {}}
    if source_key is not None:
        native["options"] = {"apiKey": source_key}
    _write_jsonc(paths, json.dumps({"provider": {CONN: native}}))
    binding = {"kind": "native", "connection": CONN}
    if expected is None:
        with pytest.raises(Refusal):
            io.read_credential(binding, paths)
    else:
        assert io.read_credential(binding, paths) == expected


RICH_MODEL_SOURCE = """{
  // keep this comment
  "provider": {
    "fixture": {
      "npm": "@openrouter/ai-sdk-provider",
      "models": {
        "rich-model": {
          "id": "rich-model",
          "name": "Rich",
          "limit": { "context": 1000, "output": 163840, "custom": "keep" },
          "api": { "npm": "@custom/pkg" },
          "reasoning": true,
          "tool_call": true,
          "variants": {
            "high": { "reasoning": { "effort": "high" }, "extra": true },
            "low": { "reasoning": { "effort": "low" }, "extra": false }
          },
          "options": { "custom": { "nested": [1, 2, 3] } }
        },
        "flat-model": {
          "id": "flat-model",
          "name": "Flat",
          "limit": { "context": 2000, "output": 100 },
          "reasoning": true,
          "tool_call": true,
          "variants": { "high": { "reasoningEffort": "high" } }
        },
        "naked-model": { "id": "naked-model", "name": "Naked" },
        "zero-model": { "id": "zero-model", "name": "Zero", "limit": { "context": 10, "output": 0 } },
        "weird-model": {
          "id": "weird-model",
          "name": "Weird",
          "limit": { "context": 10, "output": 1 },
          "variants": { "high": { "custom": "x" } }
        }
      }
    }
  },
  "unrelated": { "keep": true }
}
"""


def _rich_meta(cred=None):
    meta = _metadata()
    meta["providers"][CONN]["models"].update({
        "rich-model": {
            "id": "rich-model",
            "name": "Rich",
            "limit": {"context": 1000, "output": 163840},
            "variants": {
                "high": {"reasoning": {"effort": "high"}},
                "low": {"reasoning": {"effort": "low"}},
            },
        },
        "flat-model": {
            "id": "flat-model",
            "name": "Flat",
            "limit": {"context": 2000, "output": 100},
            "variants": {"high": {"reasoningEffort": "high"}},
        },
    })
    if cred is not None:
        meta["credentials"] = {CONN: cred}
    return meta


def _model_at(paths, mid):
    text = paths["opencode.jsonc"].read_text(encoding="utf-8")
    return edit_source(text, [], jsonc=True)["value"]["provider"][CONN]["models"][mid]


def test_narrow_label_edit_preserves_unrelated_native_metadata(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, RICH_MODEL_SOURCE)
    expected = snapshot(paths, _rich_meta())["editor_revision"]
    apply({
        "kind": "save-model",
        "connection": CONN,
        "model": {"id": "rich-model", "label": "Renamed"},
    }, expected, OP, paths, _rich_meta())
    text = paths["opencode.jsonc"].read_text(encoding="utf-8")
    model = _model_at(paths, "rich-model")
    assert model["name"] == "Renamed"
    assert model["limit"] == {"context": 1000, "output": 163840, "custom": "keep"}
    assert model["api"] == {"npm": "@custom/pkg"}
    assert model["variants"]["high"] == {"reasoning": {"effort": "high"}, "extra": True}
    assert model["options"] == {"custom": {"nested": [1, 2, 3]}}
    assert "keep this comment" in text
    assert edit_source(text, [], jsonc=True)["value"]["unrelated"] == {"keep": True}


def test_narrow_context_edit_preserves_existing_output_and_extra_keys(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, RICH_MODEL_SOURCE)
    expected = snapshot(paths, _rich_meta())["editor_revision"]
    apply({
        "kind": "save-model",
        "connection": CONN,
        "model": {"id": "rich-model", "context": 200000},
    }, expected, OP, paths, _rich_meta())
    model = _model_at(paths, "rich-model")
    assert model["limit"] == {"context": 200000, "output": 163840, "custom": "keep"}
    assert model["variants"]["high"] == {"reasoning": {"effort": "high"}, "extra": True}
    assert model["variants"]["low"] == {"reasoning": {"effort": "low"}, "extra": False}


def test_narrow_context_creates_output_zero_only_when_absent(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, RICH_MODEL_SOURCE)
    expected = snapshot(paths, _rich_meta())["editor_revision"]
    apply({
        "kind": "save-model",
        "connection": CONN,
        "model": {"id": "naked-model", "context": 5000},
    }, expected, OP, paths, _rich_meta())
    assert _model_at(paths, "naked-model")["limit"] == {"context": 5000, "output": 0}
    expected = snapshot(paths, _rich_meta())["editor_revision"]
    apply({
        "kind": "save-model",
        "connection": CONN,
        "model": {"id": "zero-model", "label": "Zero Two"},
    }, expected, OP2, paths, _rich_meta())
    assert _model_at(paths, "zero-model")["limit"] == {"context": 10, "output": 0}


def test_effort_edits_use_the_proven_representation(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, RICH_MODEL_SOURCE)
    expected = snapshot(paths, _rich_meta())["editor_revision"]
    apply({
        "kind": "save-model",
        "connection": CONN,
        "model": {"id": "rich-model", "efforts": ["high", "medium"]},
    }, expected, OP, paths, _rich_meta())
    model = _model_at(paths, "rich-model")
    assert set(model["variants"]) == {"high", "medium"}
    assert model["variants"]["high"] == {"reasoning": {"effort": "high"}, "extra": True}
    assert model["variants"]["medium"] == {"reasoning": {"effort": "medium"}}
    expected = snapshot(paths, _rich_meta())["editor_revision"]
    apply({
        "kind": "save-model",
        "connection": CONN,
        "model": {"id": "flat-model", "efforts": ["low"]},
    }, expected, OP2, paths, _rich_meta())
    assert _model_at(paths, "flat-model")["variants"] == {"low": {"reasoningEffort": "low"}}


def test_unknown_or_mixed_representation_refuses_changed_efforts(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, RICH_MODEL_SOURCE)
    expected = snapshot(paths, _rich_meta())["editor_revision"]
    apply({
        "kind": "save-model",
        "connection": CONN,
        "model": {"id": "weird-model", "label": "Weird Two", "context": 20},
    }, expected, OP, paths, _rich_meta())
    model = _model_at(paths, "weird-model")
    assert model["name"] == "Weird Two"
    assert model["variants"] == {"high": {"custom": "x"}}
    expected = snapshot(paths, _rich_meta())["editor_revision"]
    with pytest.raises(Refusal) as exc:
        apply({
            "kind": "save-model",
            "connection": CONN,
            "model": {"id": "weird-model", "efforts": ["low"]},
        }, expected, OP2, paths, _rich_meta())
    assert exc.value.code == "agent-profile-conflict"


COMPAT_CONN = "compat"
COMPAT_SOURCE = """{
  "provider": {
    "compat": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "https://compat.example.com/v1" },
      "models": {
        "native-a": { "id": "native-a", "name": "Native A", "limit": { "context": 4096, "output": 1024 } }
      }
    }
  },
  "unrelated": { "keep": true }
}
"""


def _compat_meta():
    return {
        "providers": {COMPAT_CONN: {"npm": "@ai-sdk/openai-compatible", "models": {}}},
        "credentials": {COMPAT_CONN: {"auth": "api-key", "status": "present"}},
    }


def _compat_model_at(paths, mid):
    text = paths["opencode.jsonc"].read_text(encoding="utf-8")
    return edit_source(text, [], jsonc=True)["value"]["provider"][COMPAT_CONN]["models"][mid]


@pytest.mark.parametrize("model", [
    {"id": "custom-id-only"},
    {"id": "custom-label", "label": "Custom Label"},
    {"id": "custom-context", "context": 5000},
    {"id": "custom-both", "label": "Custom Both", "context": 5000},
])
def test_uncatalogued_model_persists_and_resolves_as_builder_profile(tmp_path, model):
    paths = _paths(tmp_path)
    _write_jsonc(paths, COMPAT_SOURCE)
    meta = _compat_meta()
    expected = snapshot(paths, meta)["editor_revision"]
    apply({"kind": "save-model", "connection": COMPAT_CONN, "model": model}, expected, OP, paths, meta)
    entry = _compat_model_at(paths, model["id"])
    assert entry["id"] == model["id"]
    if "label" in model:
        assert entry["name"] == model["label"]
    if "context" in model:
        assert entry["limit"] == {"context": model["context"], "output": 0}
    else:
        assert "limit" not in entry

    row = next(c for c in snapshot(paths, meta)["connections"] if c["id"] == COMPAT_CONN)
    chosen = next(m for m in row["models"] if m["id"] == model["id"])
    assert chosen["origin"] == "override"
    assert chosen["no_effort"] is True
    assert chosen["effort_template"] is None

    intent = _intent(model=model["id"])
    intent["builders"]["default"]["model"] = model["id"]
    intent["builders"]["fallback"]["model"] = "native-a"
    for profile in intent["builders"].values():
        profile["connection"] = COMPAT_CONN
        profile["effort"] = None
        profile["routing"] = None
    expected = snapshot(paths, meta)["editor_revision"]
    published = apply(intent, expected, OP2, paths, meta)
    native = edit_source(paths["opencode.jsonc"].read_text(encoding="utf-8"), [], jsonc=True)["value"]
    resolved = resolve_profile(published["settings"], "default", effective_config=native)
    assert resolved["model"] == f"{COMPAT_CONN}/{model['id']}"
    fallback = resolve_profile(published["settings"], "fallback", effective_config=native)
    assert fallback["model"] == f"{COMPAT_CONN}/native-a"
    assert native["provider"][COMPAT_CONN]["models"]["native-a"] == {
        "id": "native-a", "name": "Native A", "limit": {"context": 4096, "output": 1024},
    }


def test_uncatalogued_model_refuses_unproven_efforts(tmp_path):
    paths = _paths(tmp_path)
    raw = _write_jsonc(paths, COMPAT_SOURCE)
    meta = _compat_meta()
    expected = snapshot(paths, meta)["editor_revision"]
    with pytest.raises(Refusal) as exc:
        apply({
            "kind": "save-model",
            "connection": COMPAT_CONN,
            "model": {"id": "custom-effort", "efforts": ["high"]},
        }, expected, OP, paths, meta)
    assert exc.value.code == "agent-profile-conflict"
    assert paths["opencode.jsonc"].read_bytes() == raw
    expected = snapshot(paths, meta)["editor_revision"]
    apply({
        "kind": "save-model",
        "connection": COMPAT_CONN,
        "model": {"id": "custom-empty", "efforts": []},
    }, expected, OP2, paths, meta)
    assert _compat_model_at(paths, "custom-empty") == {"id": "custom-empty"}


def test_used_by_effort_removal_refuses(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, RICH_MODEL_SOURCE)
    meta = _rich_meta()
    intent = _intent(model="rich-model")
    expected = snapshot(paths, meta)["editor_revision"]
    apply(intent, expected, OP, paths, meta)
    expected = snapshot(paths, meta)["editor_revision"]
    with pytest.raises(Refusal) as exc:
        apply({
            "kind": "save-model",
            "connection": CONN,
            "model": {"id": "rich-model", "efforts": ["low"]},
        }, expected, OP2, paths, meta)
    assert exc.value.code == "agent-profile-conflict"


def test_oauth_connections_and_models_refuse_writes(tmp_path):
    paths = _paths(tmp_path)
    raw = _write_jsonc(paths, CLEAN_SOURCE)
    meta = _provider_meta({"auth": "oauth", "status": "present"})
    row = _row(snapshot(paths, meta)["connections"])
    assert row["editable"] is False
    assert row["reason"] == "oauth-managed"
    expected = snapshot(paths, meta)["editor_revision"]
    for intent in (
        {"kind": "save-model", "connection": CONN, "model": {"id": MODEL, "label": "x"}},
        {"kind": "remove-model", "connection": CONN, "model": MODEL},
        {"kind": "save-connection", "connection": {"id": CONN, "adapter": "xai"}},
        {"kind": "remove-connection", "connection": CONN},
        {"kind": "bind-credential", "connection": CONN, "credential": {
            "kind": "env", "env": "JAX_PROVIDER_OAUTH_API_KEY",
        }},
    ):
        with pytest.raises(Refusal) as exc:
            preview(intent, expected, paths, meta)
        assert exc.value.code == "agent-profile-conflict"
        with pytest.raises(Refusal) as exc:
            apply(intent, expected, OP, paths, meta)
        assert exc.value.code == "agent-profile-conflict"
    assert paths["opencode.jsonc"].read_bytes() == raw
    assert not paths["opencode.json"].exists()
    assert not paths["settings"].exists()


def test_discovered_only_connection_refuses_adapter_edits(tmp_path):
    paths = _paths(tmp_path)
    meta = _provider_meta({"auth": "api-key", "status": "present"})
    expected = snapshot(paths, meta)["editor_revision"]
    with pytest.raises(Refusal) as exc:
        apply({
            "kind": "save-connection",
            "connection": {"id": CONN, "adapter": "xai"},
        }, expected, OP, paths, meta)
    assert exc.value.code == "agent-profile-conflict"


def test_public_model_dto_exposes_context_and_effort_template(tmp_path):
    paths = _paths(tmp_path)
    _write_jsonc(paths, RICH_MODEL_SOURCE)
    row = _row(snapshot(paths, _rich_meta({"auth": "api-key", "status": "present"}))["connections"])
    rich = next(model for model in row["models"] if model["id"] == "rich-model")
    flat = next(model for model in row["models"] if model["id"] == "flat-model")
    naked = next(model for model in row["models"] if model["id"] == "naked-model")
    assert rich["context"] == 1000
    assert rich["effort_template"] == "reasoning"
    assert flat["effort_template"] == "reasoning_effort"
    assert naked["context"] is None
    assert naked["effort_template"] is None
    dumped = json.dumps(row)
    assert "163840" not in dumped
    assert "@custom/pkg" not in dumped
