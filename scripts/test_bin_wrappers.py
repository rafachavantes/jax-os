"""Regression guard (spec MOA-502 'Wrapper resolution'): a `bin/*` wrapper symlinked into a
temp PATH directory from a checkout at an ARBITRARY path (never this repo's own path) resolves
to and execs THAT checkout's own scripts/*.py — never a hardcoded /home/rafa/... path."""
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fake_checkout(tmp_path):
    checkout = tmp_path / "some-other-clone-path"
    (checkout / "bin").mkdir(parents=True)
    (checkout / "scripts").mkdir()
    for name in ("jaxflow", "jaxflow-hook", "jax-init"):
        shutil.copy(REPO_ROOT / "bin" / name, checkout / "bin" / name)
        os.chmod(checkout / "bin" / name, 0o755)
    (checkout / "scripts" / "jaxflow.py").write_text(
        "import sys\nprint('fake-jaxflow-ran')\nsys.exit(0)\n")
    (checkout / "scripts" / "jaxflow_hook.py").write_text(
        "import sys\nprint('fake-jaxflow-hook-ran')\nsys.exit(0)\n")
    (checkout / "scripts" / "jax_init.py").write_text(
        "import sys\nprint('fake-jax-init-ran')\nsys.exit(0)\n")
    return checkout


def test_bin_jaxflow_execs_its_own_checkouts_script_not_a_hardcoded_path(tmp_path):
    checkout = _fake_checkout(tmp_path)
    result = subprocess.run([str(checkout / "bin" / "jaxflow")], capture_output=True, text=True)
    assert result.stdout.strip() == "fake-jaxflow-ran"


def test_bin_jaxflow_hook_execs_its_own_checkouts_script(tmp_path):
    checkout = _fake_checkout(tmp_path)
    result = subprocess.run([str(checkout / "bin" / "jaxflow-hook"), "claude-stop"], capture_output=True, text=True)
    assert result.stdout.strip() == "fake-jaxflow-hook-ran"


def test_bin_jax_init_execs_its_own_checkouts_script(tmp_path):
    checkout = _fake_checkout(tmp_path)
    result = subprocess.run([str(checkout / "bin" / "jax-init")], capture_output=True, text=True)
    assert result.stdout.strip() == "fake-jax-init-ran"


def test_bin_wrapper_symlinked_elsewhere_still_resolves_its_own_checkout(tmp_path):
    # The realistic install shape (spec Decision 6): the owner symlinks bin/jaxflow into their
    # own PATH directory. readlink -f "$0" must still land back on THIS checkout's scripts/.
    checkout = _fake_checkout(tmp_path)
    fake_path_dir = tmp_path / "fake-local-bin"
    fake_path_dir.mkdir()
    (fake_path_dir / "jaxflow").symlink_to(checkout / "bin" / "jaxflow")
    result = subprocess.run([str(fake_path_dir / "jaxflow")], capture_output=True, text=True)
    assert result.stdout.strip() == "fake-jaxflow-ran"


import json


def test_example_hook_files_exist_and_call_the_jaxflow_hook_wrapper():
    for name in ("claude-hooks.example.json", "codex-hooks.example.json"):
        path = REPO_ROOT / "workflow" / "hooks" / name
        data = json.loads(path.read_text(encoding="utf-8"))
        commands = []
        for hook_list in data["hooks"].values():
            for entry in hook_list:
                for h in entry["hooks"]:
                    commands.append(h["command"])
        assert commands, f"{name} has no hook commands"
        assert all(cmd.startswith("jaxflow-hook ") for cmd in commands), \
            f"{name} has a command not calling the wrapper: {commands}"
