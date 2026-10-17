#!/usr/bin/env python3
"""Direct tests for jev_client's env-file reader (MOA-498 D1) — this module previously had
no dedicated test file; test_jaxflow_run.py stubs api_key() wholesale and never exercised
_check_secure/ENV_PATH directly."""
import os
import stat
from pathlib import Path

import pytest

import jev_client


def _write(path, text, mode=0o600):
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def test_api_key_prefers_process_env_over_env_local_and_the_file(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    _write(env_path, "TYPESAFE_API=from-file\n")
    local_path = tmp_path / ".env.local"
    _write(local_path, "TYPESAFE_API=from-env-local\n")
    monkeypatch.setattr(jev_client, "ENV_PATH", env_path)
    monkeypatch.setattr(jev_client, "LOCAL_ENV_PATH", local_path)
    monkeypatch.setenv("TYPESAFE_API", "from-process-env")
    assert jev_client.api_key("TYPESAFE_API") == "from-process-env"


def test_api_key_prefers_env_local_over_the_jaxos_home_file(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    _write(env_path, "TYPESAFE_API=from-file\n")
    local_path = tmp_path / ".env.local"
    _write(local_path, "TYPESAFE_API=from-env-local\n")
    monkeypatch.setattr(jev_client, "ENV_PATH", env_path)
    monkeypatch.setattr(jev_client, "LOCAL_ENV_PATH", local_path)
    monkeypatch.delenv("TYPESAFE_API", raising=False)
    assert jev_client.api_key("TYPESAFE_API") == "from-env-local"


def test_api_key_falls_back_to_the_secure_file_when_env_var_and_env_local_are_absent(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    _write(env_path, "TYPESAFE_API=from-file\n")
    monkeypatch.setattr(jev_client, "ENV_PATH", env_path)
    monkeypatch.setattr(jev_client, "LOCAL_ENV_PATH", tmp_path / "nope.env.local")
    monkeypatch.delenv("TYPESAFE_API", raising=False)
    assert jev_client.api_key("TYPESAFE_API") == "from-file"


def test_api_key_returns_empty_when_no_source_has_it(tmp_path, monkeypatch):
    monkeypatch.setattr(jev_client, "ENV_PATH", tmp_path / "nope.env")
    monkeypatch.setattr(jev_client, "LOCAL_ENV_PATH", tmp_path / "nope.env.local")
    monkeypatch.delenv("TYPESAFE_API", raising=False)
    assert jev_client.api_key("TYPESAFE_API") == ""


@pytest.mark.parametrize("build,expected", [
    (lambda p: None, "missing"),
    (lambda p: p.mkdir(), "not-regular"),
    (lambda p: (_write(p, "X=1\n", 0o644)), "bad-mode"),
])
def test_env_file_status_states(tmp_path, monkeypatch, build, expected):
    env_path = tmp_path / ".env"
    build(env_path)
    monkeypatch.setattr(jev_client, "ENV_PATH", env_path)
    assert jev_client.env_file_status() == {"state": expected}


def test_env_file_status_ok(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    _write(env_path, "X=1\n")
    monkeypatch.setattr(jev_client, "ENV_PATH", env_path)
    assert jev_client.env_file_status() == {"state": "ok"}


def test_env_file_status_wrong_owner(tmp_path, monkeypatch):
    # No test here runs as root, so a real different-owner file can't be created deterministically
    # (cold review 0c968393813f F4): mock os.getuid() instead of the file's real owner, which the
    # running test process cannot change anyway.
    env_path = tmp_path / ".env"
    _write(env_path, "X=1\n")
    monkeypatch.setattr(jev_client, "ENV_PATH", env_path)
    # Plan deviation (builder): the plan's `lambda: os.getuid() + 1` recurses, because
    # jev_client.os IS the test's os module and the lambda's lookup hits the patched
    # attribute. Capture the real callable first so the fake delegates to it.
    real_getuid = os.getuid
    monkeypatch.setattr(jev_client.os, "getuid", lambda: real_getuid() + 1)
    assert jev_client.env_file_status() == {"state": "wrong-owner"}


def test_env_file_status_refuses_a_symlink_even_to_a_valid_0600_file(tmp_path, monkeypatch):
    real = tmp_path / "real.env"
    _write(real, "X=1\n")
    link = tmp_path / ".env"
    link.symlink_to(real)
    monkeypatch.setattr(jev_client, "ENV_PATH", link)
    assert jev_client.env_file_status() == {"state": "not-regular"}


def test_env_file_status_never_returns_a_value(tmp_path, monkeypatch):
    # Cold review 0c968393813f F3: this test previously called env_file_status() with no
    # redirection at all, lstat-ing whatever ENV_PATH happened to be (the real configured home
    # file outside a test). Every env_file_status() test redirects ENV_PATH first.
    monkeypatch.setattr(jev_client, "ENV_PATH", tmp_path / ".env")
    for state_dict in [
        jev_client.env_file_status(),
    ]:
        assert set(state_dict) == {"state"}


def test_env_path_is_derived_from_jaxos_home(tmp_path):
    import importlib

    # Cold review 0c968393813f F5: the old version's `finally: importlib.reload(jev_client)`
    # ran BEFORE monkeypatch restored JAXOS_HOME (the fixture's teardown happens after the test
    # function returns), so the "restore" reload just recomputed ENV_PATH from the still-patched
    # JAXOS_HOME, leaving jev_client.ENV_PATH pointed at tmp_path for every test that ran after
    # this one. `pytest.MonkeyPatch.context()` scopes JAXOS_HOME to the `with` block, so it is
    # unset again by the time the block exits — before the reload below ever runs.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("JAXOS_HOME", str(tmp_path))
        importlib.reload(jev_client)
        assert jev_client.ENV_PATH == tmp_path / ".env"
    # JAXOS_HOME is back to its pre-test value here (or unset) — reloading now recomputes
    # ENV_PATH from the real environment, not from tmp_path.
    importlib.reload(jev_client)
    # Neither override is JAXOS_HOME-derived (ENV_PATH is a hardcoded conftest.py path;
    # LOCAL_ENV_PATH is derived from jev_client.py's own file location), so the reload above
    # recomputes LOCAL_ENV_PATH back to this repo's REAL .env.local and does not restore
    # either override by itself — reassert both explicitly for every test that runs after this
    # one (matching conftest.py's own lines verbatim; cold review dab11d7b84a9 F1).
    jev_client.ENV_PATH = Path("/nonexistent/jaxflow-tests-must-never-read-this/.env")
    jev_client.LOCAL_ENV_PATH = Path("/nonexistent/jaxflow-tests-must-never-read-this/.env.local")
