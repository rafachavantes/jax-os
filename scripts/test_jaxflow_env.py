from pathlib import Path

import jaxflow_env as jenv


def test_jaxos_home_default_and_override(monkeypatch):
    monkeypatch.delenv("JAXOS_HOME", raising=False)
    assert jenv.jaxos_home() == Path.home() / ".jax-os"
    monkeypatch.setenv("JAXOS_HOME", "/tmp/y")
    assert jenv.jaxos_home() == Path("/tmp/y")  # read fresh, no caching


def test_api_base_url_default_and_override(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    assert jenv.api_base_url() == "http://127.0.0.1:3100"
    monkeypatch.setenv("PORT", "9999")
    assert jenv.api_base_url() == "http://127.0.0.1:9999"
