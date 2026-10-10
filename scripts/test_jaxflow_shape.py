#!/usr/bin/env python3
"""Structural guards for scripts/jaxflow.py (jaxflow lean spec, A1/A2).

Both guards are xfail(strict=True) while their phase is pending, so the suite stays green
now and the XPASS flips to a failure the moment the phase lands: the phase's plan then
removes the marker (P1 for the function cap, P2 for the module cap).
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent
MAX_FUNCTION_LINES = 250
MAX_MODULE_LINES = 1800


def _oversized_functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sorted(
        f"{node.name} ({node.end_lineno - node.lineno + 1} lines, :{node.lineno})"
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.end_lineno - node.lineno + 1 > MAX_FUNCTION_LINES
    )


def test_no_function_over_250_lines():
    offenders = [
        f"{name}.py: {item}"
        for name in ALL_MODULES
        for item in _oversized_functions(SCRIPTS / f"{name}.py")
    ]
    assert not offenders, f"functions over {MAX_FUNCTION_LINES} lines: {offenders}"


_MODULES = [
    pytest.param("jaxflow.py", id="jaxflow"),
    *(pytest.param(f"{name}.py", id=name) for name in (
        "jaxflow_common", "jaxflow_workerkit", "jaxflow_merge", "jaxflow_review",
        "jaxflow_build", "jaxflow_worker", "jaxflow_cli")),
]


@pytest.mark.parametrize("filename", _MODULES)
def test_jaxflow_modules_under_1800_lines(filename):
    path = SCRIPTS / filename
    assert path.exists(), f"{filename} does not exist"
    lines = len(path.read_text(encoding="utf-8").splitlines())
    assert lines <= MAX_MODULE_LINES, f"{filename} has {lines} lines (cap {MAX_MODULE_LINES})"


ALL_MODULES = (
    "jaxflow", "jaxflow_common", "jaxflow_workerkit", "jaxflow_merge", "jaxflow_review",
    "jaxflow_build", "jaxflow_worker", "jaxflow_cli",
)
# who a module may import (spec "Module map (P2)", plus the workerkit leaf)
ALLOWED_IMPORTS = {
    "jaxflow_common": set(),
    "jaxflow_workerkit": {"jaxflow_common"},
    "jaxflow_merge": {"jaxflow_common"},
    "jaxflow_build": {"jaxflow_common"},
    "jaxflow_review": {"jaxflow_common", "jaxflow_workerkit"},
    "jaxflow_worker": {"jaxflow_common", "jaxflow_workerkit", "jaxflow_review"},
    "jaxflow_cli": {"jaxflow_common", "jaxflow_workerkit", "jaxflow_merge", "jaxflow_review",
                    "jaxflow_build", "jaxflow_worker"},
}


def _sibling_imports(name):
    """Sibling jaxflow modules imported anywhere in the file (top level or inside a function)."""
    tree = ast.parse((SCRIPTS / f"{name}.py").read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [(a.name, "import") for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "", "from")]
        else:
            continue
        for mod, kind in mods:
            if mod in ALL_MODULES:
                found.setdefault(mod, set()).add(kind)
    return found


@pytest.mark.parametrize("name", sorted(ALLOWED_IMPORTS))
def test_module_imports_follow_the_tier_order(name):
    found = _sibling_imports(name)
    assert set(found) <= ALLOWED_IMPORTS[name], f"{name} imports {sorted(set(found) - ALLOWED_IMPORTS[name])}"
    assert "from" not in {k for kinds in found.values() for k in kinds}, (
        f"{name} uses `from jaxflow_x import ...`: reference siblings as `jc.<name>` so patches reach them"
    )


def test_import_graph_is_acyclic():
    graph = {m: set(ALLOWED_IMPORTS[m]) & set(_sibling_imports(m)) for m in ALLOWED_IMPORTS}
    state = {}

    def visit(node, path):
        assert state.get(node) != "open", f"cycle: {' -> '.join(path + [node])}"
        if state.get(node) == "done":
            return
        state[node] = "open"
        for dep in graph[node]:
            visit(dep, path + [node])
        state[node] = "done"

    for module in graph:
        visit(module, [])


@pytest.mark.parametrize("name", ALL_MODULES)
def test_each_module_imports_alone_in_a_fresh_interpreter(name):
    result = subprocess.run([sys.executable, "-c", f"import {name}"], cwd=SCRIPTS,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_jaxflow_py_is_only_the_entry_shim():
    tree = ast.parse((SCRIPTS / "jaxflow.py").read_text(encoding="utf-8"))
    assert not [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert len((SCRIPTS / "jaxflow.py").read_text(encoding="utf-8").splitlines()) <= 80


def test_shim_reexports_everything_other_files_read_from_it():
    import jaxflow
    wanted = set()
    for path in SCRIPTS.glob("*.py"):
        if path.stem in ALL_MODULES or path.name == "test_jaxflow_shape.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id in ("jaxflow", "jf")):
                wanted.add(node.attr)
            if isinstance(node, ast.ImportFrom) and node.module == "jaxflow":
                wanted.update(a.name for a in node.names)
    missing = sorted(n for n in wanted if not hasattr(jaxflow, n))
    assert not missing, f"names read from `jaxflow` but not re-exported by the shim: {missing}"


def test_only_workflow_poll_tests_patch_the_shim():
    offenders = []
    for path in SCRIPTS.glob("test*.py"):
        if path.name in ("test_workflow_poll.py", "test_jaxflow_shape.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            patching = (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("setattr", "delattr") and node.args
                        and isinstance(node.args[0], ast.Name) and node.args[0].id == "jaxflow")
            assigning = (isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "jaxflow"
                for t in node.targets))
            if patching or assigning:
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"tests must patch the DEFINING module, not the shim: {offenders}"


def test_script_path_names_the_entry_script():
    import jaxflow_common
    assert jaxflow_common.SCRIPT_PATH.name == "jaxflow.py"
    assert jaxflow_common.SCRIPT_PATH.parent == SCRIPTS


