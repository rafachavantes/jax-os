import json
from pathlib import Path

import pytest

import general_settings as gs

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "workflow" / "fixtures"
FIXTURE = FIXTURES_DIR / "general-settings-v1.json"
FIXTURE_V2 = FIXTURES_DIR / "general-settings-v2.json"
NULL_FIXTURE = FIXTURES_DIR / "general-settings-null.json"
DEFAULTS = gs.parse_settings(gs.ABSENT)

EXPECTED = {
    "ownerName": "Owner", "locale": "pt-BR", "reposRoot": "/home/rafa/repos",
    "vaultPath": "/home/rafa/obsidian-vault",
    "monitoredUnits": {"user": ["hermes-gateway.service"], "system": ["tailscaled.service"]},
    "integrations": {
        "linear": True, "ttyd": False, "webhook": True, "hermesTokens": False,
        "agents": {"claude": True, "codex": False, "opencode": False},
        "classifier": True, "github": False, "vault": False,
    },
}


EXPECTED_V2 = {**EXPECTED, "integrations": {**EXPECTED["integrations"],
                                           "agents": {"claude": True, "codex": True, "opencode": False}}}


def test_the_shared_fixture_parses_to_the_exact_expected_object():
    assert gs.parse_settings(json.loads(FIXTURE.read_text())) == EXPECTED


def test_missing_input_returns_every_default():
    assert DEFAULTS == {
        "ownerName": "", "locale": "en-US", "reposRoot": str(Path.home() / "repos"),
        "vaultPath": None, "monitoredUnits": {"user": [], "system": []},
        "integrations": {
            "linear": False, "ttyd": False, "webhook": False, "hermesTokens": False,
            "agents": {"claude": False, "codex": False, "opencode": False},
            "classifier": False, "github": False, "vault": False,
        },
    }


def test_a_partial_nested_object_defaults_its_missing_child_independently():
    result = gs.parse_settings({"monitoredUnits": {"user": ["a.service"]}})
    assert result["monitoredUnits"] == {"user": ["a.service"], "system": []}


def test_null_is_valid_only_for_vault_path():
    assert gs.parse_settings({"vaultPath": None})["vaultPath"] is None


def test_v1_fixture_with_legacy_subscriptions_migrates_into_agents():
    parsed = gs.parse_settings(json.loads(FIXTURE.read_text()))
    assert parsed == EXPECTED
    assert "subscriptions" not in parsed["integrations"]


def test_v2_fixture_parses_to_the_exact_expected_object():
    assert gs.parse_settings(json.loads(FIXTURE_V2.read_text())) == EXPECTED_V2


def test_legacy_opencode_go_maps_to_agents_opencode():
    got = gs.parse_settings({"integrations": {"subscriptions": {"opencodeGo": True}}})
    assert got["integrations"]["agents"] == {"claude": False, "codex": False, "opencode": True}


def test_both_keys_present_agents_wins_and_subscriptions_is_ignored():
    got = gs.parse_settings({"integrations": {
        "agents": {"codex": True},
        "subscriptions": {"claude": True, "codex": False, "opencodeGo": True},
    }})
    assert got["integrations"]["agents"] == {"claude": False, "codex": True, "opencode": False}
    assert "subscriptions" not in got["integrations"]


def test_each_agent_key_defaults_to_false_independently():
    got = gs.parse_settings({"integrations": {"agents": {"claude": True}}})
    assert got["integrations"]["agents"] == {"claude": True, "codex": False, "opencode": False}


@pytest.mark.parametrize("raw", [
    "not an object", [1, 2], {"locale": "fr-FR"}, {"foo": 1},
    {"monitoredUnits": {"user": [], "system": [], "extra": []}},
    {"integrations": {"subscriptions": {"claude": True, "codex": False, "opencodeGo": False, "extra": True}}},
    {"ownerName": None}, {"locale": None}, None,
    {"monitoredUnits": None}, {"integrations": None}, {"integrations": {"subscriptions": None}},
    {"integrations": {"agents": {"claude": True, "extra": True}}}, {"integrations": {"agents": None}}, {"integrations": {"agents": {"claude": "yes"}}},
], ids=["not-object", "array", "wrong-locale", "unknown-top-key", "unknown-monitored-key",
        "unknown-subscription-key", "null-ownerName", "null-locale", "null-raw-value",
        "null-monitoredUnits-round2-f1", "null-integrations-round2-f1",
        "null-integrations-subscriptions-round2-f1",
        "unknown-agent-key", "null-integrations-agents", "non-bool-agent"])
def test_malformed_shapes_raise_settings_malformed(raw):
    with pytest.raises(gs.SettingsError) as exc:
        gs.parse_settings(raw)
    assert exc.value.code == "settings-malformed"


def test_absent_sentinel_is_the_only_input_that_yields_defaults_round_1_f3():
    # gs.ABSENT means "no file on disk"; a JSON-decoded None (the file literally contains
    # `null`) is a value, not an absence, and is therefore malformed, not defaulted.
    assert gs.parse_settings(gs.ABSENT) == DEFAULTS
    with pytest.raises(gs.SettingsError):
        gs.parse_settings(None)


def test_read_settings_missing_file_returns_ok_true_defaults(tmp_path):
    assert gs.read_settings(path=tmp_path / "nope" / "settings.json") == {"ok": True, "data": DEFAULTS}


def test_read_settings_malformed_json_on_disk(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not valid json")
    assert gs.read_settings(path=path) == {"ok": False, "error": "settings-malformed"}


def test_read_settings_a_directory_not_a_file_is_unreadable(tmp_path):
    path = tmp_path / "settings.json"
    path.mkdir()
    assert gs.read_settings(path=path) == {"ok": False, "error": "settings-unreadable"}


def test_read_settings_f1_non_enoent_stat_failure_is_unreadable_not_defaults(tmp_path):
    # ENOTDIR: a path component is a regular file, not a directory -- deterministic, no
    # chmod/root needed. Only FileNotFoundError means absent; any other OSError must be
    # settings-unreadable, not defaults. Parity with the TS test in settings.test.ts.
    not_a_dir = tmp_path / "notadir"
    not_a_dir.write_text("")
    path = not_a_dir / "settings.json"
    assert gs.read_settings(path=path) == {"ok": False, "error": "settings-unreadable"}


def test_read_settings_the_fixture_round_trips_to_the_same_expected_object():
    assert gs.read_settings(path=FIXTURE) == {"ok": True, "data": EXPECTED}


def test_read_settings_a_top_level_null_on_disk_is_malformed_round_1_f3():
    # Shared fixture with Python's twin in src/server/settings.test.ts -- a file present but
    # containing top-level JSON null must NOT default; only an absent file defaults.
    assert gs.read_settings(path=NULL_FIXTURE) == {"ok": False, "error": "settings-malformed"}
