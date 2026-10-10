#!/usr/bin/env python3
"""Self-tests for the opencode spawn counter, the budget hook and the skip-without-binary contract (D12/D13/A7)."""
import importlib
import subprocess
from pathlib import Path

import pytest

import conftest


def test_counter_counts_opencode_argv_only(monkeypatch):
    monkeypatch.setattr(conftest, "OPENCODE_CALLS", [])
    count = conftest._count_opencode
    count(["/home/u/.opencode/bin/opencode", "run"])   # list, absolute path
    count(["opencode", "models", "--pure"])            # list, bare name
    count(Path("/tmp/a b/opencode"))                   # scalar PathLike with a space: used whole, never split
    count("\t/usr/bin/opencode\tmodels", shell=True)   # shell string, tab-separated: shlex token 0
    assert len(conftest.OPENCODE_CALLS) == 4
    count(["python3", "x"])
    count(["git", "status"])
    count(["/usr/bin/opencode-not"])
    count("opencode run x")                            # shell=False scalar is ONE executable name
    count([])
    assert len(conftest.OPENCODE_CALLS) == 4


def test_wrapped_popen_and_run_each_count_once(tmp_path, monkeypatch):
    """A stand-in executable named `opencode` (never the real binary) proves the wrapper is
    installed and that `subprocess.run` -> `Popen` is not counted twice."""
    fake = tmp_path / "opencode"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(conftest, "OPENCODE_CALLS", [])
    subprocess.run([str(fake)], check=True)
    assert len(conftest.OPENCODE_CALLS) == 1
    subprocess.Popen([str(fake)]).wait()
    assert len(conftest.OPENCODE_CALLS) == 2


_SKIP_SITES = [
    ("test_jaxflow_settings", "test_saved_profiles_forward_model_reasoning_and_full_routing", lambda m: (None, None)),
    ("test_jaxflow_settings", "test_native_xai_captured_request_uses_own_wire_format", lambda m: (None, None)),
    ("test_jaxflow_settings", "test_writer_produced_aliases_capture_supported_transports",
     lambda m: (None, m._WRITER_CASES[2])),
    ("test_jaxflow_settings_io", "test_native_listing_add_hide_and_alias_eligibility", lambda m: (None,)),
    ("test_jaxflow_settings_io", "test_native_listing_hides_removed_provider_despite_catalog", lambda m: (None,)),
]


@pytest.mark.parametrize("module,name,args", _SKIP_SITES, ids=[site[1] for site in _SKIP_SITES])
def test_opencode_marker_skips_without_binary(monkeypatch, module, name, args):
    """With PATH emptied the real-binary tests must SKIP (never fail) before touching anything."""
    monkeypatch.setenv("PATH", "")
    mod = importlib.import_module(module)
    fn = getattr(mod, name)
    with pytest.raises(pytest.skip.Exception):
        fn(*args(mod))
