#!/usr/bin/env python3
"""Self-tests for the opencode spawn counter, the budget hook and the skip-without-binary contract (D12/D13/A7)."""
import importlib
import subprocess
import types
from pathlib import Path

import pytest

import conftest


def test_counter_counts_opencode_argv_only(monkeypatch):
    monkeypatch.setattr(conftest, "OPENCODE_CALLS", [])
    real = "/usr/bin/opencode"
    monkeypatch.setattr(conftest, "_real_opencode", lambda: real)
    monkeypatch.setattr(
        conftest.shutil, "which",
        lambda name: real if Path(str(name)).name == "opencode" else None,
    )
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
    """A stand-in treated as the real binary proves the wrapper is installed and that
    `subprocess.run` -> `Popen` is not counted twice."""
    fake = tmp_path / "opencode"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(conftest, "OPENCODE_CALLS", [])
    monkeypatch.setattr(conftest, "_real_opencode", lambda: str(fake.resolve()))
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


def test_bare_opencode_on_a_fake_child_path_is_not_counted(tmp_path, monkeypatch):
    fake_dir = tmp_path / "bin"
    fake_dir.mkdir()
    fake = fake_dir / "opencode"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(conftest, "OPENCODE_CALLS", [])
    subprocess.run(["opencode"], env={"PATH": str(fake_dir)}, check=True)
    assert conftest.OPENCODE_CALLS == []


def test_fake_opencode_stand_in_is_not_counted(tmp_path, monkeypatch):
    fake_dir = tmp_path / ".opencode" / "bin"
    fake_dir.mkdir(parents=True)
    fake = fake_dir / "opencode"
    fake.write_text("#!/bin/sh\necho safe-marker\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(conftest, "OPENCODE_CALLS", [])
    monkeypatch.setattr(conftest.shutil, "which", lambda name: "/usr/bin/opencode" if name == "opencode" else None)
    monkeypatch.setattr(conftest.os.path, "realpath", lambda p: str(p))
    conftest._count_opencode([str(fake), "models", "fixture"], shell=False)
    assert conftest.OPENCODE_CALLS == []
    conftest._count_opencode(["/usr/bin/opencode", "models"], shell=False)
    assert len(conftest.OPENCODE_CALLS) == 1


class _Reporter:
    def __init__(self):
        self.lines = []

    def write_line(self, message, **kwargs):
        self.lines.append(message)


@pytest.mark.parametrize("calls,binary,failed", [
    pytest.param(5, "/fake/opencode", False, id="at-budget-leaves-the-session-alone"),
    pytest.param(6, "/fake/opencode", True, id="over-budget-fails-the-session"),
    pytest.param(6, None, False, id="binary-absent-never-enforces"),
])
def test_budget_hook_fails_the_session_only_over_budget(monkeypatch, calls, binary, failed):
    """Direct test of `pytest_sessionfinish`: binary discovery stubbed, fake session/reporter."""
    monkeypatch.setattr(conftest, "OPENCODE_CALLS", [["opencode", "run"]] * calls)
    monkeypatch.setattr(conftest.shutil, "which", lambda name: binary)
    reporter = _Reporter()
    session = types.SimpleNamespace(
        exitstatus=pytest.ExitCode.OK,
        config=types.SimpleNamespace(pluginmanager=types.SimpleNamespace(get_plugin=lambda name: reporter)),
    )
    conftest.pytest_sessionfinish(session, session.exitstatus)
    if failed:
        assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
        assert len(reporter.lines) == 1
        assert "opencode invocation budget exceeded: 6 > 5" in reporter.lines[0]
    else:
        assert session.exitstatus == pytest.ExitCode.OK
        assert reporter.lines == []
