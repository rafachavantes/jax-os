import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent


def _resolve(module, expr, env_overrides):
    env = dict(os.environ)
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    result = subprocess.run(
        [sys.executable, "-c", f"import {module} as m; print({expr})"],
        cwd=SCRIPTS_DIR, env=env, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# (module, expr, path suffix under JAXOS_HOME) -- every path-based row of the spec's table.
PATH_CASES = [
    ("jaxflow_run", "m.DB_PATH", "jaxos.db"),
    ("jaxflow", "m.CALLBACKS_ROOT", "callbacks"),
    ("jaxflow_hook", "m.CALLBACKS_ROOT", "callbacks"),
    ("jaxflow_hook", "m.THROTTLE_ROOT", "hook-throttle"),
    ("jaxflow_settings", "m.SETTINGS_PATH", "agent-settings.json"),
    ("jaxflow_settings", "m.LOCK_PATH", "agent-settings.lock"),
    ("collect_metrics", "m.DB_PATH", "jaxos.db"),
    ("collect_metrics", "m.BACKUP_PATH", "jaxos.db.bak"),
    ("workflow_poll", "m.JEV_SHADOW_PATH", "jev-shadow.jsonl"),
    ("jaxflow_inventory_io", "m._home_roots()['jax_os']", ""),
]

# (module, expr, URL suffix after the host:port) -- every URL-based row.
URL_CASES = [
    ("jaxflow_run", "m.EVENTS_URL", "/api/workflow/events"),
    ("jaxflow", "m.MISSION_BASE_URL", "/api/mission"),
    ("jaxflow_hook", "m.EVENTS_URL", "/api/workflow/events"),
    ("jaxflow_hook", "m.SESSIONS_URL", "/api/workflow/sessions"),
    ("workflow_poll", "m.EVENTS_URL", "/api/workflow/events?pending=1&limit=20"),
    ("workflow_poll", "m.ACK_URL", "/api/workflow/events/ack"),
    ("workflow_poll", "m.STALENESS_URL", "/api/workflow/staleness-check"),
    ("workflow_poll", "m.DEFERRED_URL", "/api/workflow/events/deferred"),
    ("workflow_poll", "m.CLASSIFY_URL", "/api/workflow/events/classify"),
]


@pytest.mark.parametrize("module,expr,suffix", PATH_CASES, ids=[f"{m}:{e}" for m, e, _ in PATH_CASES])
def test_path_consumer_resolves_under_jaxos_home(module, expr, suffix):
    out = _resolve(module, expr, {"JAXOS_HOME": "/tmp/moa497x"})
    assert out == (f"/tmp/moa497x/{suffix}" if suffix else "/tmp/moa497x")


@pytest.mark.parametrize("module,expr,suffix", PATH_CASES, ids=[f"{m}:{e}" for m, e, _ in PATH_CASES])
def test_path_consumer_default_equals_today(module, expr, suffix):
    out = _resolve(module, expr, {"JAXOS_HOME": None})
    home = str(Path.home() / ".jax-os")
    assert out == (f"{home}/{suffix}" if suffix else home)


@pytest.mark.parametrize("module,expr,suffix", URL_CASES, ids=[f"{m}:{e}" for m, e, _ in URL_CASES])
def test_url_consumer_resolves_against_port(module, expr, suffix):
    out = _resolve(module, expr, {"PORT": "9999", "JAXFLOW_EVENTS_URL": None})
    assert out == f"http://127.0.0.1:9999{suffix}"


@pytest.mark.parametrize("module,expr,suffix", URL_CASES, ids=[f"{m}:{e}" for m, e, _ in URL_CASES])
def test_url_consumer_default_equals_today(module, expr, suffix):
    out = _resolve(module, expr, {"PORT": None, "JAXFLOW_EVENTS_URL": None})
    assert out == f"http://127.0.0.1:3100{suffix}"


def test_jaxflow_run_events_url_still_prefers_the_escape_hatch_when_set():
    # Regression for the existing test-isolation escape hatch (scripts/conftest.py,
    # jaxflow_run.py:20's own comment): PORT must NOT override an explicitly set
    # JAXFLOW_EVENTS_URL.
    out = _resolve("jaxflow_run", "m.EVENTS_URL",
                    {"PORT": "9999", "JAXFLOW_EVENTS_URL": "http://127.0.0.1:1/api/workflow/events"})
    assert out == "http://127.0.0.1:1/api/workflow/events"
