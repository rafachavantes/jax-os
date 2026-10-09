#!/usr/bin/env python3
"""Contract tests for jaxflow_settings (MOA-473 Part 1 Task 1)."""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from jaxflow import _pure_config_run
import jaxflow_run as jr
from jax_init import Refusal
import jaxflow_settings as jset
from jaxflow_settings import (
    SETTINGS_CAP, builder_environment, configuration_claim, decode_settings,
    read_effective_opencode_config, read_settings, require_source_revisions,
    resolve_profile, select_builder, select_reviewer,
)
from jaxflow_settings_io import apply, snapshot

FIXTURES = Path(__file__).resolve().parents[1] / "workflow" / "fixtures"
FIXTURE = FIXTURES / "agent-settings-v1.json"
CORPUS = FIXTURES / "agent-settings-validation.json"
ALL_ON = {"claude": True, "codex": True, "opencode": True}


@pytest.fixture
def fixture_settings():
    return decode_settings(FIXTURE.read_bytes())


def test_assignment_is_not_configurable(fixture_settings):
    settings = fixture_settings
    # Distinct valid efforts catch a field-level swap as well as a model swap.
    settings["reviewers"]["claude"]["effort"] = "low"
    settings["reviewers"]["codex"]["effort"] = "high"
    for caller, runtime, configured_model, configured_effort in (
        ("codex", "claude", "sonnet", "low"),
        ("claude", "codex", "gpt-5.6-luna", "high"),
    ):
        for model_override, effort_override in (
            (None, None), ("alternate-model", None),
            (None, "medium"), ("alternate-model", "medium"),
        ):
            selected = select_reviewer(
                settings, caller, model=model_override, effort=effort_override, agents=ALL_ON,
            )
            assert selected["runtime"] == runtime
            assert selected["model"] == (
                configured_model if model_override is None else model_override
            )
            assert selected["effort"] == (
                configured_effort if effort_override is None else effort_override
            )
    settings["reviewers"]["claude"]["runtime"] = "codex"
    with pytest.raises(Refusal):
        decode_settings(json.dumps(settings).encode())


def test_profile_selection_is_explicit(fixture_settings):
    assert select_builder(fixture_settings, fallback=False, builder=None, model=None, effort=None)["profile_name"] == "default"
    assert select_builder(fixture_settings, fallback=True, builder=None, model=None, effort=None)["profile_name"] == "fallback"
    for overrides in ({"builder": "opencode-deepseek"}, {"model": "xai/another"}, {"effort": "low"}):
        args = {"fallback": False, "builder": None, "model": None, "effort": None}
        args.update(overrides)
        with pytest.raises(Refusal):
            select_builder(fixture_settings, **args)


def test_only_absence_bootstraps(tmp_path):
    target = tmp_path / "agent-settings.json"
    assert read_settings(target) is None
    target.write_text("{broken", encoding="utf-8")
    target.chmod(0o600)
    with pytest.raises(Refusal):
        read_settings(target)


def _copy(settings):
    return json.loads(json.dumps(settings))


def _refuse_decode(settings, code="agent-settings-malformed"):
    with pytest.raises(Refusal) as exc:
        decode_settings(json.dumps(settings).encode())
    assert exc.value.code == code


def _refuse_raw(raw, code="agent-settings-malformed"):
    with pytest.raises(Refusal) as exc:
        decode_settings(raw)
    assert exc.value.code == code


def test_unknown_version_and_keys(fixture_settings):
    settings = _copy(fixture_settings)
    settings["schema_version"] = 2
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["extra"] = 1
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["adapter"] = "openrouter"
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["base_url"] = "https://example.invalid"
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["model_config"] = {}
    _refuse_decode(settings)


def test_null_root_bool_as_number_bad_utf8_and_oversized(fixture_settings):
    _refuse_raw(b"null")
    settings = _copy(fixture_settings)
    settings["schema_version"] = True
    _refuse_decode(settings)
    _refuse_raw(b"\xff\xfe{")
    _refuse_raw(b"x" * (SETTINGS_CAP + 1), "agent-settings-too-large")
    duplicate = FIXTURE.read_bytes().replace(
        b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1', 1,
    )
    _refuse_raw(duplicate)
    _refuse_raw(b'{"schema_version": NaN}')
    _refuse_raw(b'{"schema_version": Infinity}')


def test_model_and_connection_charset(fixture_settings):
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["model"] = "-leading-dash"
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["model"] = "has space"
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["model"] = "has\ncontrol"
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["model"] = "モデル"
    decode_settings(json.dumps(settings).encode())
    for path in (
        ("builders", "default", "model"),
        ("builders", "fallback", "model"),
        ("reviewers", "claude", "model"),
        ("reviewers", "codex", "model"),
    ):
        settings = _copy(fixture_settings)
        cursor = settings
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = "ok-\ud800"
        _refuse_decode(settings)
    for connection in ("-dash", "a/b", "has space", "café"):
        settings = _copy(fixture_settings)
        settings["builders"]["default"]["connection"] = connection
        _refuse_decode(settings)


def test_credential_effort_and_routing_rejects(fixture_settings):
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["credential"] = {"kind": "native", "env": "XAI_API_KEY"}
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["credential"] = {"kind": "env", "env": "BWS_ACCESS_TOKEN"}
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["credential"] = {"kind": "env", "env": "PATH"}
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["credential"] = {
        "kind": "env", "env": "JAX_PROVIDER_FIXTURE_API_KEY",
        "secret_id": "not-a-uuid",
    }
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["effort"] = "not valid"
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["reviewers"]["claude"]["effort"] = None
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["routing"] = {"sort": "latency", "allow_fallbacks": False}
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["routing"] = {
        "sort": "price", "allow_fallbacks": True, "order": ["x"],
    }
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["routing"] = {
        "sort": "price", "allow_fallbacks": True,
        "preferred_min_throughput": {"p90": 1, "p50": 2},
    }
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["routing"] = {
        "sort": "price", "allow_fallbacks": True, "max_price": {"prompt": -1},
    }
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["routing"] = {
        "sort": "price", "allow_fallbacks": True,
        "only": ["same"], "ignore": ["same"],
    }
    _refuse_decode(settings)
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["credential"] = {
        "kind": "env",
        "env": "JAX_PROVIDER_FIXTURE_API_KEY",
    }
    decode_settings(json.dumps(settings).encode())
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["routing"] = {
        "sort": "price",
        "allow_fallbacks": True,
        "preferred_min_throughput": {"p90": 40},
        "preferred_max_latency": {"p90": 3},
        "max_price": {"prompt": 0.2, "completion": 0.8},
        "only": ["fixture-primary"],
        "ignore": ["fixture-excluded"],
    }
    decode_settings(json.dumps(settings).encode())
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["routing"] = None
    settings["builders"]["default"]["effort"] = None
    decode_settings(json.dumps(settings).encode())


def test_credential_refuses_the_retired_bws_env_shape():
    with pytest.raises(Refusal) as exc:
        jset._credential({"kind": "bws-env", "env": "JAX_PROVIDER_FIXTURE_API_KEY", "secret_id": "x"})
    assert exc.value.code == "agent-settings-malformed"


def test_credential_accepts_the_new_env_shape():
    jset._credential({"kind": "env", "env": "JAX_PROVIDER_FIXTURE_API_KEY"})  # must not raise


def test_symlink_permission_and_not_file(tmp_path):
    target = tmp_path / "agent-settings.json"
    target.write_bytes(FIXTURE.read_bytes())
    target.chmod(0o600)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(Refusal) as exc:
        read_settings(link)
    assert exc.value.code == "agent-settings-symlink"
    target.chmod(0o644)
    with pytest.raises(Refusal) as exc:
        read_settings(target)
    assert exc.value.code == "agent-settings-permissions"
    target.chmod(0o600)
    with pytest.raises(Refusal) as exc:
        read_settings(tmp_path)
    assert exc.value.code == "agent-settings-not-file"
    real = tmp_path / "real"
    real.mkdir()
    parent_link = tmp_path / "via"
    parent_link.symlink_to(real)
    nested = real / "agent-settings.json"
    nested.write_bytes(FIXTURE.read_bytes())
    nested.chmod(0o600)
    with pytest.raises(Refusal) as exc:
        read_settings(parent_link / "agent-settings.json")
    assert exc.value.code == "agent-settings-symlink"
    huge = tmp_path / "huge.json"
    huge.write_bytes(b"x" * (SETTINGS_CAP + 1))
    huge.chmod(0o600)
    with pytest.raises(Refusal) as exc:
        read_settings(huge)
    assert exc.value.code == "agent-settings-too-large"


def test_legacy_and_reviewer_overrides(fixture_settings):
    assert select_reviewer(None, "claude", model=None, effort=None, agents=ALL_ON) == {
        "runtime": "codex", "model": "gpt-5.6-luna", "effort": "xhigh", "fallback": None,
    }
    assert select_reviewer(None, "codex", model=None, effort=None, agents=ALL_ON) == {
        "runtime": "claude", "model": "sonnet", "effort": "xhigh", "fallback": None,
    }
    assert select_builder(None, fallback=False, builder=None, model=None, effort=None) is None
    with pytest.raises(Refusal) as exc:
        select_builder(None, fallback=True, builder=None, model=None, effort=None)
    assert exc.value.code == "agent-settings-uninitialized"
    with pytest.raises(Refusal) as exc:
        select_builder(None, fallback=False, builder="opencode-builder", model=None, effort=None)
    assert exc.value.code == "agent-settings-uninitialized"
    selected = select_builder(
        fixture_settings, fallback=False, builder="opencode-builder", model=None, effort=None,
    )
    assert selected == {"runtime": "opencode-builder", "profile_name": "default"}
    with pytest.raises(Refusal) as exc:
        select_reviewer(fixture_settings, "claude", model="-bad", effort=None, agents=ALL_ON)
    assert exc.value.code == "reviewer-model-invalid"
    with pytest.raises(Refusal) as exc:
        select_reviewer(fixture_settings, "claude", model=None, effort="max", agents=ALL_ON)
    assert exc.value.code == "reviewer-effort-invalid"
    assert select_reviewer(fixture_settings, "codex", model=None, effort="max", agents=ALL_ON)["effort"] == "max"


def test_decode_prints_no_secrets(capsys, fixture_settings):
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["credential"] = {
        "kind": "env",
        "env": "JAX_PROVIDER_FIXTURE_API_KEY",
    }
    decode_settings(json.dumps(settings).encode())
    select_reviewer(fixture_settings, "claude", model=None, effort=None, agents=ALL_ON)
    select_builder(fixture_settings, fallback=False, builder=None, model=None, effort=None)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert "11111111-1111-4111-8111-111111111111" not in captured.out
    assert "JAX_PROVIDER_FIXTURE_API_KEY" not in captured.out


def test_deep_nested_json_is_malformed():
    raw = b"[" * 60000 + b"]" * 60000
    assert len(raw) == 120000
    assert len(raw) < SETTINGS_CAP
    with pytest.raises(Refusal) as exc:
        decode_settings(raw)
    assert exc.value.code == "agent-settings-malformed"


def _corpus_bytes(case):
    kind = case["kind"]
    if kind == "file":
        return (FIXTURES / case["path"]).read_bytes()
    if kind == "utf8":
        return case["text"].encode("utf-8")
    if kind == "hex":
        return bytes.fromhex(case["hex"])
    if kind == "fill":
        return bytes([case["byte"]]) * case["count"]
    if kind == "repeat":
        prefix = bytes.fromhex(case["hex"]) * case["count"]
        suffix = bytes.fromhex(case.get("suffix_hex", "")) * case.get("suffix_count", 0)
        return prefix + suffix
    raise AssertionError(kind)


def test_shared_validation_corpus():
    for case in json.loads(CORPUS.read_text(encoding="utf-8"))["cases"]:
        raw = _corpus_bytes(case)
        if case["expect"] is None:
            decode_settings(raw)
            continue
        with pytest.raises(Refusal) as exc:
            decode_settings(raw)
        assert exc.value.code == case["expect"], case["name"]


def test_jsonc_parser_exposes_comment_offsets():
    root = Path(__file__).resolve().parents[1]
    pkg = json.loads((root / "package.json").read_text(encoding="utf-8"))
    assert pkg["dependencies"]["jsonc-parser"] == "3.3.1"
    script = (
        "const parser = require('jsonc-parser');\n"
        "const text = '{\\n  // keep\\n  \"a\": 1\\n}\\n';\n"
        "const errors = [];\n"
        "const tree = parser.parseTree(text, errors);\n"
        "if (!tree || errors.length) process.exit(2);\n"
        "if (typeof tree.offset !== 'number' || typeof tree.length !== 'number') process.exit(3);\n"
        "const comments = [];\n"
        "parser.visit(text, { onComment(offset, length) { comments.push({offset, length}); } });\n"
        "if (comments.length !== 1 || comments[0].offset !== 4 || comments[0].length !== 7) process.exit(4);\n"
        "if (typeof parser.modify !== 'function' || typeof parser.applyEdits !== 'function') process.exit(5);\n"
    )
    proc = subprocess.run(["node", "-e", script], cwd=root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def _listing(root):
    found = []
    for dirpath, names, files in os.walk(root, followlinks=False):
        for name in names + files:
            found.append(os.path.relpath(os.path.join(dirpath, name), root))
    return sorted(found)


def test_missing_target_ancestry_does_not_write(tmp_path):
    missing_root = tmp_path / "absent" / "agent-settings.json"
    before = _listing(tmp_path)
    assert read_settings(missing_root) is None
    assert read_settings(tmp_path / "agent-settings.json") is None
    assert _listing(tmp_path) == before
    assert not missing_root.exists()
    real = tmp_path / "real"
    real.mkdir()
    via = tmp_path / "via"
    via.symlink_to(real)
    after_setup = _listing(tmp_path)
    with pytest.raises(Refusal) as exc:
        read_settings(via / "agent-settings.json")
    assert exc.value.code == "agent-settings-symlink"
    assert not (real / "agent-settings.json").exists()
    assert _listing(tmp_path) == after_setup


def test_percentile_and_price_key_matrix(fixture_settings):
    percentiles = ("p50", "p75", "p90", "p99")
    fields = ("preferred_min_throughput", "preferred_max_latency")
    for profile in ("default", "fallback"):
        for field in fields:
            for key in percentiles:
                settings = _copy(fixture_settings)
                settings["builders"][profile]["routing"] = {
                    "sort": "price", "allow_fallbacks": True, field: {key: 1},
                }
                decode_settings(json.dumps(settings).encode())
            for bad in (
                {}, {"p51": 1}, {"unexpected": 1},
                {"p50": 1, "unexpected": 2}, {"p50": 1, "p90": 2},
                {"p50": 0}, {"p50": True}, {"p50": -1}, {"p50": "1"},
            ):
                settings = _copy(fixture_settings)
                settings["builders"][profile]["routing"] = {
                    "sort": "price", "allow_fallbacks": True, field: bad,
                }
                _refuse_decode(settings)
            settings = _copy(fixture_settings)
            settings["builders"][profile]["routing"] = {
                "sort": "price", "allow_fallbacks": True, field: {"p50": 1},
            }
            dumped = json.dumps(settings).replace('"p50": 1', '"p50": NaN', 1)
            _refuse_raw(dumped.encode())
            dumped = json.dumps(settings).replace('"p50": 1', '"p50": Infinity', 1)
            _refuse_raw(dumped.encode())
        for good in (
            {"prompt": 1}, {"completion": 1}, {"prompt": 1, "completion": 2},
            {"prompt": 0}, {"completion": 0}, {"prompt": 0, "completion": 0},
        ):
            settings = _copy(fixture_settings)
            settings["builders"][profile]["routing"] = {
                "sort": "price", "allow_fallbacks": True, "max_price": good,
            }
            decode_settings(json.dumps(settings).encode())
        for bad in (
            {}, {"unexpected": 1}, {"prompt": 1, "unexpected": 2},
            {"prompt": 1, "completion": 2, "unexpected": 3},
            {"prompt": True}, {"prompt": -1}, {"prompt": "1"},
        ):
            settings = _copy(fixture_settings)
            settings["builders"][profile]["routing"] = {
                "sort": "price", "allow_fallbacks": True, "max_price": bad,
            }
            _refuse_decode(settings)
        settings = _copy(fixture_settings)
        settings["builders"][profile]["routing"] = {
            "sort": "price", "allow_fallbacks": True, "max_price": {"prompt": 1},
        }
        dumped = json.dumps(settings).replace('"prompt": 1', '"prompt": NaN', 1)
        _refuse_raw(dumped.encode())
        dumped = json.dumps(settings).replace('"prompt": 1', '"prompt": Infinity', 1)
        _refuse_raw(dumped.encode())


@pytest.fixture
def native_config():
    def model_entry(allow_fallbacks):
        return {
            "id": "wire-real-model",
            "variants": {"high": {"reasoning": {"effort": "high"}}},
            "options": {"provider": {"sort": "price", "allow_fallbacks": allow_fallbacks}},
        }
    return {
        "provider": {
            "fixture": {
                "npm": "@openrouter/ai-sdk-provider",
                "models": {
                    "jaxflow-builder-default": model_entry(True),
                    "jaxflow-builder-fallback": model_entry(False),
                },
            },
        },
        "permission": {"read": "deny"},
        "agent": {"build": {}},
    }


def test_resolution_uses_current_saved_profile(fixture_settings, native_config):
    settings = fixture_settings
    before = resolve_profile(settings, "default", effective_config=native_config)
    settings["revision"] = "22222222222222222222222222222222"
    settings["builders"]["default"]["model"] = "changed-model"
    native_config["provider"]["fixture"]["models"]["jaxflow-builder-default"]["id"] = "changed-model"
    after = resolve_profile(settings, "default", effective_config=native_config)
    assert before["model"] == "fixture/wire-real-model"
    assert after["model"] == "fixture/changed-model"
    assert after["runtime_model"] == "fixture/jaxflow-builder-default"
    assert after["settings_revision"] == settings["revision"]


def test_resolution_compares_full_routing_policies(fixture_settings, native_config):
    default = resolve_profile(fixture_settings, "default", effective_config=native_config)
    fallback = resolve_profile(fixture_settings, "fallback", effective_config=native_config)
    assert default["runtime_model"] == "fixture/jaxflow-builder-default"
    assert fallback["runtime_model"] == "fixture/jaxflow-builder-fallback"
    assert default["credential"] == {"kind": "env", "env": "JAX_PROVIDER_FIXTURE_API_KEY"}
    assert set(default) == {
        "profile_name", "settings_revision", "model", "runtime_model", "effort", "credential",
    }
    native_config["provider"]["fixture"]["models"]["jaxflow-builder-default"]["options"]["provider"]["allow_fallbacks"] = False
    with pytest.raises(Refusal) as exc:
        resolve_profile(fixture_settings, "default", effective_config=native_config)
    assert exc.value.code == "agent-profile-conflict"
    native_config["provider"]["fixture"]["models"]["jaxflow-builder-default"]["options"]["provider"]["allow_fallbacks"] = True
    for name in ("default", "fallback"):
        variant = native_config["provider"]["fixture"]["models"][f"jaxflow-builder-{name}"]["variants"]["high"]
        variant["provider"] = {"sort": "throughput", "allow_fallbacks": True}
        with pytest.raises(Refusal) as exc:
            resolve_profile(fixture_settings, name, effective_config=native_config)
        assert exc.value.code == "agent-profile-conflict"
        del variant["provider"]
        resolve_profile(fixture_settings, name, effective_config=native_config)
    assert native_config["permission"]["read"] == "deny"


def test_resolution_refuses_conflicts(fixture_settings, native_config):
    models = native_config["provider"]["fixture"]["models"]
    del models["jaxflow-builder-default"]
    with pytest.raises(Refusal) as exc:
        resolve_profile(fixture_settings, "default", effective_config=native_config)
    assert exc.value.code == "agent-profile-conflict"
    native_config["provider"]["fixture"]["models"]["jaxflow-builder-default"] = {
        "id": "other-model",
        "variants": {"high": {"reasoning": {"effort": "high"}}},
        "options": {"provider": {"sort": "price", "allow_fallbacks": True}},
    }
    with pytest.raises(Refusal) as exc:
        resolve_profile(fixture_settings, "default", effective_config=native_config)
    assert exc.value.code == "agent-profile-conflict"
    native_config["provider"]["fixture"]["models"]["jaxflow-builder-default"]["id"] = "wire-real-model"
    del native_config["provider"]["fixture"]["models"]["jaxflow-builder-default"]["variants"]["high"]
    with pytest.raises(Refusal) as exc:
        resolve_profile(fixture_settings, "default", effective_config=native_config)
    assert exc.value.code == "agent-profile-conflict"
    native_config["provider"]["fixture"]["models"]["jaxflow-builder-default"]["variants"]["high"] = {
        "reasoning": {"effort": "high"},
    }
    native_config["provider"]["fixture"]["models"]["jaxflow-builder-default"]["options"]["provider"]["order"] = ["x"]
    with pytest.raises(Refusal) as exc:
        resolve_profile(fixture_settings, "default", effective_config=native_config)
    assert exc.value.code == "agent-profile-conflict"
    del native_config["provider"]["fixture"]["models"]["jaxflow-builder-default"]["options"]["provider"]["order"]
    native_config["agent"]["build"] = {"options": {"provider": {"sort": "price", "allow_fallbacks": False}}}
    with pytest.raises(Refusal) as exc:
        resolve_profile(fixture_settings, "default", effective_config=native_config)
    assert exc.value.code == "agent-profile-conflict"
    native_config["agent"]["build"] = {}
    native_config["provider"]["fixture"]["npm"] = "@ai-sdk/openai"
    with pytest.raises(Refusal) as exc:
        resolve_profile(fixture_settings, "default", effective_config=native_config)
    assert exc.value.code == "agent-profile-conflict"
    settings = _copy(fixture_settings)
    settings["builders"]["default"]["routing"] = None
    settings["builders"]["default"]["effort"] = None
    native_config["provider"]["fixture"]["npm"] = "@ai-sdk/openai"
    del native_config["provider"]["fixture"]["models"]["jaxflow-builder-default"]["options"]
    del native_config["provider"]["fixture"]["models"]["jaxflow-builder-default"]["variants"]
    none = resolve_profile(settings, "default", effective_config=native_config)
    assert none["effort"] is None
    native_config["agent"]["build"] = {"variant": "high"}
    with pytest.raises(Refusal) as exc:
        resolve_profile(settings, "default", effective_config=native_config)
    assert exc.value.code == "agent-profile-conflict"


def test_resolve_profile_openrouter_and_native_variant_shapes(fixture_settings, native_config):
    for name in ("default", "fallback"):
        resolve_profile(fixture_settings, name, effective_config=native_config)
        models = native_config["provider"]["fixture"]["models"]
        models[f"jaxflow-builder-{name}"]["variants"]["high"] = {"reasoning": {"effort": "low"}}
        with pytest.raises(Refusal) as exc:
            resolve_profile(fixture_settings, name, effective_config=native_config)
        assert exc.value.code == "agent-profile-conflict"
        models[f"jaxflow-builder-{name}"]["variants"]["high"] = {"reasoning": {"effort": "high"}}
        del models[f"jaxflow-builder-{name}"]["variants"]["high"]["reasoning"]
        with pytest.raises(Refusal) as exc:
            resolve_profile(fixture_settings, name, effective_config=native_config)
        assert exc.value.code == "agent-profile-conflict"
        models[f"jaxflow-builder-{name}"]["variants"]["high"] = {"reasoning": {"effort": "high"}}

    settings = _copy(fixture_settings)
    native = {
        "provider": {
            "fixture": {
                "npm": "@ai-sdk/openai",
                "models": {
                    "jaxflow-builder-default": {
                        "id": "wire-real-model",
                        "variants": {"high": {"reasoningEffort": "high"}},
                    },
                    "jaxflow-builder-fallback": {
                        "id": "wire-real-model",
                        "variants": {"high": {"reasoningEffort": "high"}},
                    },
                },
            },
        },
        "permission": {"read": "deny"},
        "agent": {"build": {}},
    }
    for name in ("default", "fallback"):
        settings["builders"][name]["routing"] = None
        resolved = resolve_profile(settings, name, effective_config=native)
        assert resolved["effort"] == "high"
        assert resolved["runtime_model"] == f"fixture/jaxflow-builder-{name}"
        native["provider"]["fixture"]["models"][f"jaxflow-builder-{name}"]["variants"]["high"] = {}
        resolve_profile(settings, name, effective_config=native)
        native["provider"]["fixture"]["models"][f"jaxflow-builder-{name}"]["variants"]["high"] = None
        with pytest.raises(Refusal) as exc:
            resolve_profile(settings, name, effective_config=native)
        assert exc.value.code == "agent-profile-conflict"
        native["provider"]["fixture"]["models"][f"jaxflow-builder-{name}"]["variants"]["high"] = "high"
        with pytest.raises(Refusal) as exc:
            resolve_profile(settings, name, effective_config=native)
        assert exc.value.code == "agent-profile-conflict"
        del native["provider"]["fixture"]["models"][f"jaxflow-builder-{name}"]["variants"]
        settings["builders"][name]["effort"] = None
        none = resolve_profile(settings, name, effective_config=native)
        assert none["effort"] is None
        settings["builders"][name]["effort"] = "high"
        native["provider"]["fixture"]["models"][f"jaxflow-builder-{name}"]["variants"] = {
            "high": {"reasoningEffort": "high"},
        }


@pytest.mark.parametrize("filename", ["opencode.json", "opencode.jsonc"])
def test_source_revisions_match_current_bytes(fixture_settings, tmp_path, filename):
    settings = _copy(fixture_settings)
    paths = {
        "opencode.json": tmp_path / "opencode.json",
        "opencode.jsonc": tmp_path / "opencode.jsonc",
    }
    other = "opencode.jsonc" if filename == "opencode.json" else "opencode.json"
    require_source_revisions(settings, paths)
    settings["revision"] = "22222222222222222222222222222222"
    require_source_revisions(settings, paths)
    path = paths[filename]
    path.write_bytes(b"{}")
    with pytest.raises(Refusal) as exc:
        require_source_revisions(settings, paths)
    assert exc.value.code == "agent-settings-source-changed"
    settings["source_revisions"][filename] = hashlib.sha256(b"{}").hexdigest()
    require_source_revisions(settings, paths)
    path.write_bytes(b"{ }\n")
    with pytest.raises(Refusal) as exc:
        require_source_revisions(settings, paths)
    assert exc.value.code == "agent-settings-source-changed"
    settings["source_revisions"][filename] = hashlib.sha256(b"{ }\n").hexdigest()
    require_source_revisions(settings, paths)
    path.unlink()
    with pytest.raises(Refusal) as exc:
        require_source_revisions(settings, paths)
    assert exc.value.code == "agent-settings-source-changed"
    settings["source_revisions"][filename] = "absent"
    require_source_revisions(settings, paths)
    assert settings["source_revisions"][other] == "absent"


def test_configuration_claim_is_exclusive(tmp_path):
    lock = tmp_path / "agent-settings.lock"
    with configuration_claim(lock):
        with pytest.raises(Refusal):
            with configuration_claim(lock):
                pass
    with configuration_claim(lock):
        pass
    assert lock.is_file()
    assert (lock.stat().st_mode & 0o777) == 0o600


def test_configuration_claim_refuses_symlink_ancestors(tmp_path):
    real = tmp_path / "real" / "jax"
    real.mkdir(parents=True)
    via = tmp_path / "via"
    via.symlink_to(tmp_path / "real")
    lock = via / "jax" / "agent-settings.lock"
    with pytest.raises(Refusal) as exc:
        with configuration_claim(lock):
            pass
    assert exc.value.code == "agent-settings-symlink"
    assert not (real / "agent-settings.lock").exists()
    nested = tmp_path / "home" / ".jax-os"
    nested.mkdir(parents=True)
    (tmp_path / "link-home").symlink_to(tmp_path / "home")
    lock = tmp_path / "link-home" / ".jax-os" / "agent-settings.lock"
    with pytest.raises(Refusal) as exc:
        with configuration_claim(lock):
            pass
    assert exc.value.code == "agent-settings-symlink"
    assert not (nested / "agent-settings.lock").exists()


def test_builder_environment_native_skips_lookup(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "JAX_PROVIDER_FIXTURE_API_KEY=secret-value\nBWS_ACCESS_TOKEN=machine-token\n",
        encoding="utf-8",
    )
    env_path.chmod(0o600)
    inherited = {"PATH": "/bin", "BWS_ACCESS_TOKEN": "machine-token", "HOME": "/home"}
    out = builder_environment({"credential": {"kind": "native"}}, inherited, env_path=env_path)
    assert out["PATH"] == "/bin"
    assert "BWS_ACCESS_TOKEN" not in out
    assert "JAX_PROVIDER_FIXTURE_API_KEY" not in out
    assert inherited["BWS_ACCESS_TOKEN"] == "machine-token"


def test_builder_environment_bws_env_parsing(tmp_path):
    env_path = tmp_path / ".env"
    profile = {
        "credential": {
            "kind": "env",
            "env": "JAX_PROVIDER_FIXTURE_API_KEY",
        },
    }
    inherited = {"PATH": "/bin", "BWS_ACCESS_TOKEN": "tok", "OTHER": "keep"}

    def write(text):
        env_path.write_text(text, encoding="utf-8")
        env_path.chmod(0o600)

    write("JAX_PROVIDER_FIXTURE_API_KEY=unquoted-key\nUNRELATED=nope\n")
    out = builder_environment(profile, inherited, env_path=env_path)
    assert out["JAX_PROVIDER_FIXTURE_API_KEY"] == "unquoted-key"
    assert "UNRELATED" not in out
    assert "BWS_ACCESS_TOKEN" not in out
    assert out["OTHER"] == "keep"
    assert inherited.get("JAX_PROVIDER_FIXTURE_API_KEY") is None

    write("JAX_PROVIDER_FIXTURE_API_KEY='single-quoted'\n")
    assert builder_environment(profile, inherited, env_path=env_path)["JAX_PROVIDER_FIXTURE_API_KEY"] == "single-quoted"
    write('JAX_PROVIDER_FIXTURE_API_KEY="double-quoted"\n')
    assert builder_environment(profile, inherited, env_path=env_path)["JAX_PROVIDER_FIXTURE_API_KEY"] == "double-quoted"
    write('JAX_PROVIDER_FIXTURE_API_KEY="foo\\nbar"\n')
    assert builder_environment(profile, inherited, env_path=env_path)["JAX_PROVIDER_FIXTURE_API_KEY"] == "foo\nbar"
    write("JAX_PROVIDER_FIXTURE_API_KEY=crlf-value\r\nOTHER=x\r\n")
    assert builder_environment(profile, inherited, env_path=env_path)["JAX_PROVIDER_FIXTURE_API_KEY"] == "crlf-value"
    write("JAX_PROVIDER_FIXTURE_API_KEY=one\nJAX_PROVIDER_FIXTURE_API_KEY=two\n")
    with pytest.raises(Refusal):
        builder_environment(profile, inherited, env_path=env_path)
    write("UNRELATED=only\n")
    with pytest.raises(Refusal):
        builder_environment(profile, inherited, env_path=env_path)
    write("JAX_PROVIDER_FIXTURE_API_KEY=ok\n")
    env_path.chmod(0o644)
    with pytest.raises(Refusal) as exc:
        builder_environment(profile, inherited, env_path=env_path)
    assert exc.value.code == "agent-settings-permissions"


def test_read_effective_opencode_config_uses_pure_introspection():
    seen = {}

    def run(argv, **kw):
        seen["argv"] = argv
        seen["kw"] = kw
        return type("R", (), {"returncode": 0, "stdout": '{"ok": true}', "stderr": "secret-stderr"})()

    parsed = read_effective_opencode_config(run=run, env={"K": "V"}, repo=Path("/repo"))
    assert seen["argv"] == ["opencode", "debug", "config", "--pure"]
    assert seen["kw"]["cwd"] == Path("/repo") or seen["kw"]["cwd"] == "/repo"
    assert seen["kw"]["env"] == {"K": "V"}
    assert parsed == {"ok": True}


class _CaptureHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        size = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(size))
        self.requests.append(request)
        if request.get("stream"):
            frames = [
                {"id": "fixture", "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
                {"id": "fixture", "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            body = ("".join("data: " + json.dumps(frame) + "\n\n" for frame in frames) + "data: [DONE]\n\n").encode()
            content_type = "text/event-stream"
        else:
            body = json.dumps({
                "id": "fixture", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


_FULL_DEFAULT = {
    "sort": "price",
    "preferred_min_throughput": {"p90": 40},
    "preferred_max_latency": {"p90": 3},
    "max_price": {"prompt": 0.2, "completion": 0.8},
    "only": ["fixture-primary"],
    "ignore": ["fixture-excluded"],
    "allow_fallbacks": True,
}
_FULL_FALLBACK = {
    "sort": "price",
    "preferred_min_throughput": {"p90": 10},
    "preferred_max_latency": {"p90": 9},
    "max_price": {"prompt": 0.9, "completion": 1.8},
    "only": ["fixture-fallback"],
    "ignore": ["fixture-excluded"],
    "allow_fallbacks": False,
}


def _model_entry(routing, *, variants=True):
    entry = {"id": "wire-real-model", "options": {"provider": dict(routing)}}
    if variants:
        entry["variants"] = {"high": {"reasoning": {"effort": "high"}}}
    return entry


def _isolated_env(work):
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(work / "home"),
        "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
        "XDG_CONFIG_HOME": str(work / "xdg-config"),
        "XDG_DATA_HOME": str(work / "xdg-data"),
        "XDG_CACHE_HOME": str(work / "xdg-cache"),
        "XDG_STATE_HOME": str(work / "xdg-state"),
        "NO_COLOR": "1",
        "TERM": "dumb",
        "LANG": "C.UTF-8",
    }
    assert not any("SECRET" in value.upper() or "TOKEN" in value.upper() for value in env.values())
    return env


def _spawn_builder(argv, work, env):
    proc = subprocess.Popen(
        argv, cwd=work, env=env, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    try:
        proc.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    return proc


def test_saved_profiles_forward_model_reasoning_and_full_routing(tmp_path, fixture_settings):
    if shutil.which("opencode") is None:
        pytest.fail("installed opencode CLI is unavailable")
    _CaptureHandler.requests = []
    work = tmp_path / "work"
    work.mkdir()
    prompt = work / "prompt.txt"
    prompt.write_text("ok\n", encoding="utf-8")
    last = work / "last.md"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        native = {
            "provider": {
                "fixture": {
                    "npm": "@openrouter/ai-sdk-provider",
                    "options": {
                        "baseURL": f"http://127.0.0.1:{server.server_port}/api/v1",
                        "apiKey": "fixture",
                    },
                    "models": {
                        "jaxflow-builder-default": _model_entry(_FULL_DEFAULT),
                        "jaxflow-builder-fallback": _model_entry(_FULL_FALLBACK),
                    },
                },
            },
            "permission": {"read": "deny"},
            "agent": {"build": {}},
        }
        (work / "opencode.json").write_text(json.dumps(native), encoding="utf-8")
        settings = _copy(fixture_settings)
        settings["builders"]["default"]["routing"] = dict(_FULL_DEFAULT)
        settings["builders"]["fallback"]["routing"] = dict(_FULL_FALLBACK)
        decode_settings(json.dumps(settings).encode())
        env = _isolated_env(work)
        expected = {"default": _FULL_DEFAULT, "fallback": _FULL_FALLBACK}
        for name, routing in expected.items():
            before = len(_CaptureHandler.requests)
            effective = read_effective_opencode_config(run=_pure_config_run, env=env, repo=work)
            resolved = resolve_profile(settings, name, effective_config=effective)
            argv = jr.runtime_argv(
                "opencode-builder", "builder", work, prompt, last,
                model=resolved["runtime_model"], effort=resolved["effort"],
            )
            assert argv[:2] == ["opencode", "run"]
            assert "--pure" not in argv
            assert argv[argv.index("--model") + 1] == f"fixture/jaxflow-builder-{name}"
            proc = _spawn_builder(argv, work, env)
            captured = _CaptureHandler.requests[before:]
            if not captured:
                err = (proc.stderr.read()[-1000:] if proc.stderr else "") or ""
                pytest.fail(
                    f"opencode {name} emitted no request (exit {proc.returncode}): {err}"
                )
            for request in captured:
                assert request["model"] == "wire-real-model"
                assert request["reasoning"] == {"effort": "high"}
                assert request["provider"] == routing
                assert request["provider"]["sort"] == "price"
                assert not {"reasoningEffort", "reasoning_effort", "models", "order"}.intersection(request)

            variant = native["provider"]["fixture"]["models"][f"jaxflow-builder-{name}"]["variants"]["high"]
            variant["provider"] = {**routing, "only": ["fixture-alternate"]}
            (work / "opencode.json").write_text(json.dumps(native), encoding="utf-8")
            effective = read_effective_opencode_config(run=_pure_config_run, env=env, repo=work)
            with pytest.raises(Refusal) as exc:
                resolve_profile(settings, name, effective_config=effective)
            assert exc.value.code == "agent-profile-conflict"
            del variant["provider"]
            (work / "opencode.json").write_text(json.dumps(native), encoding="utf-8")

        native["agent"]["build"]["model"] = "fixture/agent-conflict"
        native["provider"]["fixture"]["models"]["agent-conflict"] = _model_entry(_FULL_DEFAULT)
        native["provider"]["fixture"]["models"]["agent-conflict"]["id"] = "agent-wire-model"
        (work / "opencode.json").write_text(json.dumps(native), encoding="utf-8")
        resolved = resolve_profile(settings, "default", effective_config=native)
        argv = jr.runtime_argv(
            "opencode-builder", "builder", work, prompt, last,
            model=resolved["runtime_model"], effort=resolved["effort"],
        )
        assert argv[argv.index("--model") + 1] == "fixture/jaxflow-builder-default"
        before = len(_CaptureHandler.requests)
        proc = _spawn_builder(argv, work, env)
        captured = _CaptureHandler.requests[before:]
        if not captured:
            err = (proc.stderr.read()[-1000:] if proc.stderr else "") or ""
            pytest.fail(
                f"opencode F4 emitted no request (exit {proc.returncode}): {err}"
            )
        for request in captured:
            assert request["model"] == "wire-real-model"
            assert request["model"] != "agent-wire-model"

        conflict = json.loads(json.dumps(native))
        conflict["provider"]["fixture"]["models"]["jaxflow-builder-default"]["options"]["provider"]["allow_fallbacks"] = False
        with pytest.raises(Refusal) as exc:
            resolve_profile(settings, "default", effective_config=conflict)
        assert exc.value.code == "agent-profile-conflict"
        assert json.loads((work / "opencode.json").read_text(encoding="utf-8"))["permission"]["read"] == "deny"

        settings["builders"]["default"]["effort"] = None
        del native["provider"]["fixture"]["models"]["jaxflow-builder-default"]["variants"]
        (work / "opencode.json").write_text(json.dumps(native), encoding="utf-8")
        none = resolve_profile(settings, "default", effective_config=native)
        argv = jr.runtime_argv(
            "opencode-builder", "builder", work, prompt, last,
            model=none["runtime_model"], effort=none["effort"],
        )
        assert "--variant" not in argv
        before = len(_CaptureHandler.requests)
        proc = _spawn_builder(argv, work, env)
        captured = _CaptureHandler.requests[before:]
        if not captured:
            err = (proc.stderr.read()[-1000:] if proc.stderr else "") or ""
            pytest.fail(f"non-reasoning profile emitted no request (exit {proc.returncode}): {err}")
        for request in captured:
            assert request["model"] == "wire-real-model"
            assert "reasoning" not in request
            assert "reasoningEffort" not in request and "reasoning_effort" not in request
            assert request["provider"] == _FULL_DEFAULT
        assert json.loads((work / "opencode.json").read_text(encoding="utf-8"))["permission"]["read"] == "deny"
    finally:
        server.shutdown()
        server.server_close()


def test_native_xai_captured_request_uses_own_wire_format(tmp_path, fixture_settings):
    if shutil.which("opencode") is None:
        pytest.fail("installed opencode CLI is unavailable")
    _CaptureHandler.requests = []
    work = tmp_path / "work"
    work.mkdir()
    prompt = work / "prompt.txt"
    prompt.write_text("ok\n", encoding="utf-8")
    last = work / "last.md"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        native = {
            "provider": {
                "fixture": {
                    "npm": "@ai-sdk/xai",
                    "options": {
                        "baseURL": f"http://127.0.0.1:{server.server_port}/v1",
                        "apiKey": "fixture",
                    },
                    "models": {
                        "jaxflow-builder-default": {
                            "id": "wire-real-model",
                            "variants": {"high": {"reasoningEffort": "high"}},
                        },
                        "jaxflow-builder-fallback": {
                            "id": "wire-real-model",
                            "variants": {"high": {"reasoningEffort": "high"}},
                        },
                    },
                },
            },
            "permission": {"read": "deny"},
            "agent": {"build": {}},
        }
        (work / "opencode.json").write_text(json.dumps(native), encoding="utf-8")
        settings = _copy(fixture_settings)
        for name in ("default", "fallback"):
            settings["builders"][name]["routing"] = None
        decode_settings(json.dumps(settings).encode())
        env = _isolated_env(work)
        resolved = resolve_profile(settings, "default", effective_config=native)
        argv = jr.runtime_argv(
            "opencode-builder", "builder", work, prompt, last,
            model=resolved["runtime_model"], effort=resolved["effort"],
        )
        assert argv[:2] == ["opencode", "run"]
        assert "--pure" not in argv
        assert argv[argv.index("--model") + 1] == "fixture/jaxflow-builder-default"
        assert argv[argv.index("--variant") + 1] == "high"
        proc = _spawn_builder(argv, work, env)
        captured = _CaptureHandler.requests
        if not captured:
            err = (proc.stderr.read()[-1000:] if proc.stderr else "") or ""
            pytest.fail(
                f"opencode xAI emitted no request (exit {proc.returncode}): {err}"
            )
        for request in captured:
            assert request["model"] == "wire-real-model"
            provider = request.get("provider")
            assert not (isinstance(provider, dict) and "sort" in provider)
            chat = request.get("reasoning_effort") == "high"
            responses = (
                isinstance(request.get("reasoning"), dict)
                and request["reasoning"].get("effort") == "high"
                and request.get("store") is False
            )
            assert chat or responses
            if chat:
                assert request.get("reasoning") != {"effort": "high"}
        assert json.loads((work / "opencode.json").read_text(encoding="utf-8"))["permission"]["read"] == "deny"
    finally:
        server.shutdown()
        server.server_close()


_WRITER_OP = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


def test_writer_produced_aliases_capture_supported_transports(tmp_path):
    if shutil.which("opencode") is None:
        pytest.fail("installed opencode CLI is unavailable")
    cases = (
        {
            "name": "openrouter",
            "npm": "@openrouter/ai-sdk-provider",
            "suffix": "/api/v1",
            "variants": {"high": {"reasoning": {"effort": "high"}}},
            "routing_default": dict(_FULL_DEFAULT),
            "routing_fallback": dict(_FULL_FALLBACK),
            "default_effort": "high",
            "fallback_effort": "high",
        },
        {
            "name": "xai",
            "npm": "@ai-sdk/xai",
            "suffix": "/v1",
            "variants": {"high": {"reasoningEffort": "high"}},
            "routing_default": None,
            "routing_fallback": None,
            "default_effort": "high",
            "fallback_effort": "high",
        },
        {
            "name": "openai-compatible",
            "npm": "@ai-sdk/openai-compatible",
            "suffix": "/v1",
            "variants": {"high": {"reasoningEffort": "high"}},
            "routing_default": None,
            "routing_fallback": None,
            "default_effort": "high",
            "fallback_effort": "high",
        },
        {
            "name": "openai-compatible-none",
            "npm": "@ai-sdk/openai-compatible",
            "suffix": "/v1",
            "variants": None,
            "routing_default": None,
            "routing_fallback": None,
            "default_effort": None,
            "fallback_effort": None,
        },
    )
    reviewers = {
        "claude": {"model": "sonnet", "effort": "xhigh"},
        "codex": {"model": "gpt-5.6-luna", "effort": "xhigh"},
    }
    for case in cases:
        _CaptureHandler.requests = []
        root = tmp_path / case["name"]
        work = root / "work"
        work.mkdir(parents=True)
        prompt = work / "prompt.txt"
        prompt.write_text("ok\n", encoding="utf-8")
        last = work / "last.md"
        hermes = root / "hermes"
        hermes.mkdir()
        paths = {
            "settings": root / "jax-os" / "agent-settings.json",
            "lock": root / "jax-os" / "agent-settings.lock",
            "opencode.json": work / "opencode.json",
            "opencode.jsonc": root / "config" / "opencode" / "opencode.jsonc",
            "env": hermes / ".env",
        }
        server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            catalog = {"id": "wire-real-model", "name": "Wire"}
            if case["variants"] is not None:
                catalog["variants"] = case["variants"]
            native = {
                "provider": {
                    "fixture": {
                        "npm": case["npm"],
                        "options": {
                            "baseURL": f"http://127.0.0.1:{server.server_port}{case['suffix']}",
                            "apiKey": "fixture",
                        },
                        "models": {"wire-real-model": catalog},
                    }
                },
                "permission": {"read": "deny"},
                "agent": {"build": {}},
            }
            (work / "opencode.json").write_text(json.dumps(native), encoding="utf-8")
            meta = {"providers": {"fixture": {"npm": case["npm"], "models": {"wire-real-model": catalog}}}}
            intent = {
                "kind": "save-settings",
                "reviewers": reviewers,
                "builders": {
                    "default": {
                        "connection": "fixture",
                        "model": "wire-real-model",
                        "effort": case["default_effort"],
                        "credential": {"kind": "native"},
                        "routing": case["routing_default"],
                    },
                    "fallback": {
                        "connection": "fixture",
                        "model": "wire-real-model",
                        "effort": case["fallback_effort"],
                        "credential": {"kind": "native"},
                        "routing": case["routing_fallback"],
                    },
                },
            }
            expected = snapshot(paths, meta)["editor_revision"]
            published = apply(intent, expected, _WRITER_OP, paths, meta)
            settings = published["settings"]
            env = _isolated_env(work)
            written = json.loads((work / "opencode.json").read_text(encoding="utf-8"))
            assert written["permission"]["read"] == "deny"
            for name in ("default", "fallback"):
                before = len(_CaptureHandler.requests)
                effective = read_effective_opencode_config(run=_pure_config_run, env=env, repo=work)
                resolved = resolve_profile(settings, name, effective_config=effective)
                argv = jr.runtime_argv(
                    "opencode-builder", "builder", work, prompt, last,
                    model=resolved["runtime_model"], effort=resolved["effort"],
                )
                assert argv[:2] == ["opencode", "run"]
                assert "--pure" not in argv
                assert argv[argv.index("--model") + 1] == f"fixture/jaxflow-builder-{name}"
                effort = settings["builders"][name]["effort"]
                if effort is None:
                    assert "--variant" not in argv
                else:
                    assert argv[argv.index("--variant") + 1] == effort
                proc = _spawn_builder(argv, work, env)
                captured = _CaptureHandler.requests[before:]
                if not captured:
                    err = (proc.stderr.read()[-1000:] if proc.stderr else "") or ""
                    pytest.fail(
                        f"writer {case['name']} {name} emitted no request (exit {proc.returncode}): {err}"
                    )
                routing = settings["builders"][name]["routing"]
                for request in captured:
                    assert request["model"] == "wire-real-model"
                    if routing is not None:
                        assert request["provider"] == routing
                        assert request["provider"]["sort"] == "price"
                        assert request["reasoning"] == {"effort": "high"}
                        assert not {"reasoningEffort", "reasoning_effort", "models", "order"}.intersection(request)
                    elif effort is None:
                        assert "reasoning" not in request
                        assert "reasoningEffort" not in request and "reasoning_effort" not in request
                    else:
                        provider = request.get("provider")
                        assert not (isinstance(provider, dict) and "sort" in provider)
                        chat = request.get("reasoning_effort") == "high"
                        nested = (
                            isinstance(request.get("reasoning"), dict)
                            and request["reasoning"].get("effort") == "high"
                        )
                        assert chat or nested
            assert json.loads((work / "opencode.json").read_text(encoding="utf-8"))["permission"]["read"] == "deny"
        finally:
            server.shutdown()
            server.server_close()



def test_select_reviewer_accepts_jaxos_like_claude(fixture_settings):
    assert select_reviewer(None, "jaxos", model=None, effort=None, agents=ALL_ON) == {
        "runtime": "codex", "model": "gpt-5.6-luna", "effort": "xhigh", "fallback": None,
    }
    assert select_reviewer(fixture_settings, "jaxos", model=None, effort=None, agents=ALL_ON) == \
        select_reviewer(fixture_settings, "claude", model=None, effort=None, agents=ALL_ON)

# ---- model defaults from the shared fixture (spec: One source of model defaults) ----

import jaxflow_settings

_MODEL_DEFAULTS = json.loads((Path(__file__).resolve().parents[1] / "workflow" / "fixtures" / "model-defaults-v1.json").read_text())
_OPPOSITE_RUNTIME = {"claude": "codex", "codex": "claude", "jaxos": "codex"}


def test_reviewer_defaults_derives_from_the_fixture_and_the_opposite_runtime_table():
    assert jaxflow_settings.REVIEWER_DEFAULTS == {
        caller: {"runtime": opposite, **_MODEL_DEFAULTS["reviewers"][opposite]}
        for caller, opposite in _OPPOSITE_RUNTIME.items()
    }


@pytest.mark.parametrize("caller,off,runtime,fallback", [
    ("claude", None, "codex", None),
    ("codex", None, "claude", None),
    ("jaxos", None, "codex", None),
    ("claude", "codex", "claude", "codex off"),
    ("codex", "claude", "codex", "claude off"),
    ("jaxos", "codex", "claude", "codex off"),
])
def test_select_reviewer_opposite_else_the_other_reviewer_runtime(fixture_settings, caller, off, runtime, fallback):
    agents = {**ALL_ON, **({off: False} if off else {})}
    got = select_reviewer(fixture_settings, caller, model=None, effort=None, agents=agents)
    assert got["runtime"] == runtime and got["fallback"] == fallback
    assert got["model"] == fixture_settings["reviewers"][runtime]["model"]
    assert got["effort"] == fixture_settings["reviewers"][runtime]["effort"]


def test_select_reviewer_without_agent_settings_uses_the_shared_model_defaults_of_the_chosen_runtime():
    got = select_reviewer(None, "claude", model=None, effort=None, agents={**ALL_ON, "codex": False})
    assert got["runtime"] == "claude"
    assert got["model"] == jset._MODEL_DEFAULTS["reviewers"]["claude"]["model"]


@pytest.mark.parametrize("caller", ["claude", "codex", "jaxos"])
@pytest.mark.parametrize("agents", [
    {"claude": False, "codex": False, "opencode": False},
    {"claude": False, "codex": False, "opencode": True},  # OpenCode alone cannot review
])
def test_select_reviewer_refuses_when_neither_reviewer_agent_is_on(caller, agents):
    with pytest.raises(Refusal) as exc:
        select_reviewer(None, caller, model=None, effort=None, agents=agents)
    assert exc.value.code == "no-reviewer-agent"
    assert exc.value.hint == "hint: a review needs claude or codex: turn one on in /settings"


def test_select_reviewer_validates_overrides_against_the_chosen_runtime():
    # effort "max" is valid for the claude reviewer only
    got = select_reviewer(None, "claude", model=None, effort="max", agents={**ALL_ON, "codex": False})
    assert got["runtime"] == "claude" and got["effort"] == "max"
    with pytest.raises(Refusal) as exc:
        select_reviewer(None, "claude", model=None, effort="max", agents=ALL_ON)
    assert exc.value.code == "reviewer-effort-invalid"


def test_reviewer_fallback_is_derivable_from_caller_and_runtime():
    assert jset.reviewer_fallback("claude", "codex") is None
    assert jset.reviewer_fallback("claude", "claude") == "codex off"
    assert jset.reviewer_fallback("codex", "codex") == "claude off"
    assert jset.reviewer_fallback("jaxos", "claude") == "codex off"
    assert jset.reviewer_fallback("nobody", "claude") is None
