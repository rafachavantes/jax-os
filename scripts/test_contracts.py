#!/usr/bin/env python3
"""Contract-text guards (MOA-471, trimmed by jaxflow-lean P0).

Stdlib-only: reads the contract markdown files as text. Only NEGATIVE guards live here:
each one catches a future edit silently reintroducing an obsolete claim (`--prompt-file`,
a blanket full-read instruction, a manual `git worktree`/`checkout -b` step, a fenced role
example, ...). Positive exact-wording pins were dropped: they froze prose, not behaviour.
"""
import pathlib
import re

CONTRACTS = pathlib.Path(__file__).resolve().parent.parent / "workflow" / "contracts"


def _read(name):
    return (CONTRACTS / name).read_text(encoding="utf-8")


def _section(text, heading):
    """Slices from an exact `##`-heading line through the line before the next `##`
    heading, or EOF -- so an assertion scoped to ONE contract section never sees
    another section's own example block or prose (MOA-472 F2)."""
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line == heading)
    end = start + 1
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    return "\n".join(lines[start:end])


def test_builder_contract_no_blanket_full_read():
    assert "Read the entire handoff package and every referenced document" not in _read("builder-contract.md")


def test_builder_contract_no_final_tests_txt_obligation():
    assert "Success without this evidence is a contract violation" not in _read("builder-contract.md")


def test_builder_contract_report_example_never_has_verdict():
    assert "verdict:" not in _read("builder-contract.md")


def test_reviewer_contract_report_example_never_has_result():
    assert "result:" not in _read("reviewer-contract.md")


def test_orchestrator_handoff_no_prompt_file_claim():
    assert "--prompt-file" not in _read("orchestrator-handoff.md")


def test_orchestrator_handoff_no_manual_git_instructions():
    handoff = _read("orchestrator-handoff.md")
    assert "git checkout -b" not in handoff
    assert "git worktree add" not in handoff
    assert "git rev-parse --verify" not in handoff


def test_orchestrator_handoff_no_goal_task_acceptance_fields():
    handoff = _read("orchestrator-handoff.md")
    assert "goal:" not in handoff
    assert "tasks:" not in handoff
    assert "acceptance:" not in handoff


def test_orchestrator_handoff_quiet_reporter_never_pipes_to_tail():
    assert "| tail" not in _read("orchestrator-handoff.md")


def test_status_report_contract_report_section_has_no_fenced_frontmatter_example():
    # The REPORT FILE section may name `result:`/`verdict:` in prose, but the fenced example
    # showing both roles' keys side by side is gone (split into the builder/reviewer
    # contracts). Scoped to the section so the STATUS FILE section's own fence never makes
    # this pass for the wrong reason.
    report = _section(_read("status-report-contract.md"), "## REPORT FILE (report.md)")
    assert "```markdown\n---\n" not in report


def test_merge_contract_approval_never_demands_the_full_sha():
    assert "full 40-character head SHA" not in _read("merge-contract.md")


TEMPLATE = CONTRACTS.parent / "templates" / "AGENTS-template.md"
_PRESET_HEADING = re.compile(r"^\*\*Preset: `([a-z-]+)`\*\* — (.+)$", re.M)


def test_agents_template_has_the_five_presets_with_market_names_and_protection():
    text = TEMPLATE.read_text(encoding="utf-8")
    headings = _PRESET_HEADING.findall(text)
    assert [name for name, _ in headings] == [
        "single-branch", "single-branch-pr", "dual-branch", "dual-branch-pr", "bubble-buildprint"]
    market = {
        "single-branch": "trunk-based development",
        "single-branch-pr": "GitHub Flow",
        "dual-branch": "environment branches",
        "dual-branch-pr": "GitLab Flow with PRs",
        "bubble-buildprint": "Bubble via Buildprint",
    }
    for name, rest in headings:
        assert market[name] in rest, name
    for block in re.split(r"(?m)^\*\*Preset: ", text)[1:]:
        name = block.split("`")[1]
        has_protection = "branches/" in block and "/protection" in block and "enforce_admins" in block
        assert has_protection == (name in ("single-branch-pr", "dual-branch-pr")), name
