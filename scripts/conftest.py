# Session-wide safety net (incident 2026-09-19): a test that gave `dispatch_diff_review`
# a REAL `run` let a real `tmux new-session` fire, and the tmux SERVER (not this pytest
# process) forked the detached `--run-worker` child -- invisible to any in-process
# monkeypatch. Setting this before any test module (hence jaxflow_run.py) is ever
# imported means every child re-importing it fresh, and any child process tmux spawns
# later (env is inherited), picks up the unroutable override too.
import os

os.environ.setdefault("JAXFLOW_EVENTS_URL", "http://127.0.0.1:1/api/workflow/events")
# The stop hook would otherwise start the real jaxos-workflow-poll.service on every text stop.
os.environ["JAXFLOW_NO_CLASSIFY_KICK"] = "1"

# MOA-495/MOA-498: jev_client.api_key() reads real secret files for real -- ENV_PATH
# ($JAXOS_HOME/.env) and LOCAL_ENV_PATH (this repo's own .env.local), and both may exist
# with valid 0600 perms on this machine -- the SAME shape of risk workflow_poll.py's own
# tests guard per-test (`patch.object(jev_client, "ENV_PATH", ...)`), applied once here
# instead of at every call site, so any test that reaches the default (non-injected) Jev
# path fails closed (empty key -> no network) rather than making a live API call. A test
# that wants a real-shaped Jev *answer* injects a fake at the call site (jev=/score_chunks=/
# noul= parameters) -- that path is untouched by this guard.
from pathlib import Path

import jev_client

jev_client.ENV_PATH = Path("/nonexistent/jaxflow-tests-must-never-read-this/.env")
# MOA-498 F1 added a second read source (LOCAL_ENV_PATH, derived from jev_client.py's own file
# location) that resolves to this repo's REAL .env.local when not overridden — neuter it here
# too, for the same reason as ENV_PATH above: any test that reaches the default api_key() path
# must fail closed, never read a real file.
jev_client.LOCAL_ENV_PATH = Path("/nonexistent/jaxflow-tests-must-never-read-this/.env.local")

# Tests that clear os.environ (patch.dict(..., clear=True)) drop JAXFLOW_NO_CLASSIFY_KICK, so
# the in-process hook's kick command is also neutered here.
import jaxflow_hook

jaxflow_hook.CLASSIFY_KICK = ["true"]

# The Codex write path of jaxflow_inventory_io imports `tomlkit`, which lives ONLY in the
# isolated helper venv (never the system interpreter). Expose that venv's site-packages to
# the test process AND to the subprocesses that run the real helper (they rebuild PYTHONPATH
# from os.environ), so `pytest scripts` passes without a PYTHONPATH incantation; when the
# venv is absent those five tests fail on the import, exactly as before.
import glob
import sys

for _sp in glob.glob(os.path.expanduser("~/.jax-os/inventory-venv/lib/python3.*/site-packages")):
    if _sp not in sys.path:
        sys.path.append(_sp)  # append, not insert: the system interpreter's own packages still win
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [os.environ.get("PYTHONPATH", ""), _sp]))


# ---- opencode spawn counter (jaxflow lean spec D12/D13/A7) ---------------------------------
# The real `opencode` binary is slow (75 s of the old 139 s suite). Every process whose
# executable is `opencode` is recorded in OPENCODE_CALLS (budget assertion: Task 4). Tests that
# need the binary carry `@pytest.mark.opencode` and skip when it is absent.
import shlex
import subprocess

import pytest

OPENCODE_CALLS = []


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "opencode: needs the real `opencode` binary (skipped when it is not installed)")


def _argv0(args, shell):
    """The executable `Popen` would start. shell=False: argv[0] of a list, or a scalar
    str/bytes/PathLike taken WHOLE (never split: `Path('/tmp/a b/opencode')` is one path).
    shell=True: `sh -c <command>`, so the executable is the first shell token."""
    if isinstance(args, (str, bytes, os.PathLike)):
        first = os.fsdecode(args)
    else:
        first = os.fsdecode(args[0]) if args else ""
    if not shell:
        return first
    try:
        tokens = shlex.split(first)
    except ValueError:  # unbalanced quote: no executable we can name
        return ""
    return tokens[0] if tokens else ""


def _count_opencode(args, shell=False):
    """Record `args` when its executable's basename is `opencode`."""
    if os.path.basename(_argv0(args, shell)) == "opencode":
        OPENCODE_CALLS.append(args)


@pytest.fixture(scope="session", autouse=True)
def opencode_spawn_counter():
    """Count every `opencode` process the session starts. `subprocess.run` builds a
    `Popen`, so wrapping `Popen` alone sees both without double counting."""
    real_popen = subprocess.Popen

    class _CountingPopen(real_popen):
        def __init__(self, *args, **kwargs):
            # ponytail: `shell` passed positionally (9th argument) is not seen; nobody does that.
            _count_opencode(args[0] if args else kwargs.get("args"), kwargs.get("shell", False))
            super().__init__(*args, **kwargs)

    subprocess.Popen = _CountingPopen
    yield OPENCODE_CALLS
    subprocess.Popen = real_popen
