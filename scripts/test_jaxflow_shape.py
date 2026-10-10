#!/usr/bin/env python3
"""Structural guards for scripts/jaxflow.py (jaxflow lean spec, A1/A2).

Both guards are xfail(strict=True) while their phase is pending, so the suite stays green
now and the XPASS flips to a failure the moment the phase lands: the phase's plan then
removes the marker (P1 for the function cap, P2 for the module cap).
"""
import ast
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


@pytest.mark.xfail(strict=True, reason="P1 brings the nine functions under 250")
def test_no_function_over_250_lines():
    offenders = _oversized_functions(SCRIPTS / "jaxflow.py")
    assert not offenders, f"functions over {MAX_FUNCTION_LINES} lines: {offenders}"


_MODULES = [
    pytest.param("jaxflow.py", id="jaxflow", marks=pytest.mark.xfail(
        strict=True, reason="P2 splits jaxflow.py under 1800")),
    *(pytest.param(f"{name}.py", id=name) for name in (
        "jaxflow_common", "jaxflow_workerkit", "jaxflow_merge", "jaxflow_review",
        "jaxflow_build", "jaxflow_worker", "jaxflow_cli")),
]


@pytest.mark.parametrize("filename", _MODULES)
def test_jaxflow_modules_under_1800_lines(filename):
    path = SCRIPTS / filename
    if not path.exists():
        pytest.skip(f"{filename} does not exist yet (P2)")
    lines = len(path.read_text(encoding="utf-8").splitlines())
    assert lines <= MAX_MODULE_LINES, f"{filename} has {lines} lines (cap {MAX_MODULE_LINES})"


P1A_FUNCTIONS = (
    "cmd_merge", "_cmd_merge_pr", "run_worker", "_run_builder_worker",
    "_run_diff_reviewer_worker",
)


def test_p1a_functions_within_250_lines():
    # P1a (workers + merge) is done: these five were 286-448 lines on base 523b04f.
    # The full check (`test_no_function_over_250_lines`) stays xfail until P1b.
    tree = ast.parse(Path(__file__).with_name("jaxflow.py").read_text(encoding="utf-8"))
    sizes = {
        node.name: node.end_lineno - node.lineno + 1
        for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert set(P1A_FUNCTIONS) <= set(sizes)
    over = {name: sizes[name] for name in P1A_FUNCTIONS if sizes[name] > 250}
    assert not over, over
