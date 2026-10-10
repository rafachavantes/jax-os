#!/usr/bin/env python3
"""Stdlib-only tests for jaxflow preflight/report helpers."""
import json
import sqlite3
import stat
import subprocess
from subprocess import CompletedProcess
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import general_settings
import jaxflow_run as jr
import jev_client


@pytest.fixture(autouse=True)
def _integration_on_by_default(monkeypatch):
    # MOA-502 Decision 1: `refine_reason` now reads integrations.classifier live through
    # general_settings.read_settings(). Pre-existing tests exercise the real (non-injected)
    # Jev path, so default it ON here; the gate-specific test below re-patches read_settings
    # with its own off fixture, which wins (its setattr applies after this autouse one).
    monkeypatch.setattr(general_settings, "read_settings",
                        lambda: {"ok": True, "data": {"integrations": {"classifier": True}}})

VALID_HANDOFF = """diff: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa..bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
paths:
  spec: /canonical/repo/spec.md
  plan: /canonical/repo/plan.md
  test-output: /canonical/repo/.local/reports/builder123.tests.txt
"""


def test_report_contract_builder_and_reviewer():
    builder = """---
run_id: abc123
project: jax-os
role: builder
phase: workflow-phase-c
result: success
summary: built the wrapper
---
body
"""
    assert jr.validate_report(builder, "abc123", "jax-os", "builder", "workflow-phase-c") == (
        "ok", "built the wrapper", "success"
    )
    reviewer = builder.replace("role: builder", "role: reviewer").replace(
        "result: success", "verdict: approve"
    )
    assert jr.validate_report(reviewer, "abc123", "jax-os", "reviewer", "workflow-phase-c") == (
        "ok", "built the wrapper", "approve"
    )


def test_report_contract_stage_two_recovers_a_duplicate_unknown_mismatched_or_malformed_block_as_ok():
    # f3182ac80171 F1: stage 1 still rejects each of these shapes (a duplicate key, an
    # unknown key, a mismatched injected field, a malformed closing boundary) -- but
    # stage 2 finds the untouched `result:`/`summary:` lines anyway (AC 12).
    base = """---
run_id: abc123
project: jax-os
role: builder
phase: C
result: success
summary: done
---
"""
    for bad in (
        base.replace("summary: done", "summary: done\nsummary: twice"),
        base.replace("summary: done", "extra: nope\nsummary: done"),
        base.replace("run_id: abc123", "run_id: other"),
        base.replace("summary: done\n---\n", "summary: done\n---oops\n"),
    ):
        assert jr.validate_report(bad, "abc123", "jax-os", "builder", "C") == (
            "ok", "done", "success"
        )


def test_report_contract_stray_control_character_in_an_explicit_summary_is_invalid():
    # An explicit summary's own validity check is untouched by decisions 1/2/6 -- this
    # sub-case of the old combined test STAYS, unlike its four siblings above.
    base = """---
run_id: abc123
project: jax-os
role: builder
phase: C
result: success
summary: done
---
"""
    bad = base.replace("summary: done", "summary: bad\rvalue")
    assert jr.validate_report(bad, "abc123", "jax-os", "builder", "C")[0] == "invalid"


def test_report_summary_length_tolerance():
    accepted = (200, 201, 235, 300)
    for role, result_line, outcome in (
        ("builder", "result: success", "success"),
        ("reviewer", "verdict: approve", "approve"),
    ):
        for length in accepted:
            report = f"""---
run_id: abc123
project: jax-os
role: {role}
phase: C
{result_line}
summary: {"x" * length}
---
"""
            status, kept, matched = jr.validate_report(report, "abc123", "jax-os", role, "C")
            assert status == "ok", f"{role} {length} chars must be accepted"
            assert kept == "x" * length, "summary must be retained, not trimmed"
            assert matched == outcome
        report = f"""---
run_id: abc123
project: jax-os
role: {role}
phase: C
{result_line}
summary: {"x" * 301}
---
"""
        assert jr.validate_report(report, "abc123", "jax-os", role, "C")[0] == "invalid"


def test_report_contract_accepts_leading_prose_before_frontmatter():
    # Real case: .local/reports/9d21587206fd.md was rejected only for this reason.
    text = (
        "Some prose the model added before the frontmatter.\n\n"
        "---\nrun_id: abc123\nproject: jax-os\nrole: reviewer\nphase: C\n"
        "verdict: approve\nsummary: fine\n---\nbody\n"
    )
    assert jr.validate_report(text, "abc123", "jax-os", "reviewer", "C") == ("ok", "fine", "approve")


def test_report_contract_accepts_a_markdown_fenced_frontmatter_block():
    text = (
        "```markdown\n---\nrun_id: abc123\nproject: jax-os\nrole: reviewer\nphase: C\n"
        "verdict: approve\nsummary: fine\n---\nbody\n```\n"
    )
    assert jr.validate_report(text, "abc123", "jax-os", "reviewer", "C") == ("ok", "fine", "approve")


def test_report_contract_accepts_prose_then_fenced_frontmatter_the_b81e_shape():
    # Fixture SHAPE only (spec §2.4), not the real file's text: leading prose, a lone
    # "---" body rule, then the frontmatter wrapped in a ```markdown fence.
    text = (
        "Summary sentence before the block.\n\n"
        "---\n\n"
        "```markdown\n"
        "---\nrun_id: abc123\nproject: jax-os\nrole: reviewer\nphase: C\n"
        "verdict: approve\nsummary: fine\n---\n\n"
        "Body prose after the block.\n"
        "```\n"
    )
    assert jr.validate_report(text, "abc123", "jax-os", "reviewer", "C") == ("ok", "fine", "approve")


def test_report_contract_two_frontmatter_blocks_are_stage_one_ambiguous_and_stage_two_recovered():
    # 77f30f829248 F2: stage 1 still refuses two candidates as ambiguous (never silently
    # picks one) -- but stage 2's loose scan then recovers the first occurrence anyway.
    block = (
        "---\nrun_id: abc123\nproject: jax-os\nrole: reviewer\nphase: C\n"
        "verdict: approve\nsummary: fine\n---\n"
    )
    text = block + "\n" + block
    candidates = jr._frontmatter_candidates(text, set(jr.REVIEWER_KEYS))
    assert len(candidates) == 2
    assert jr.validate_report(text, "abc123", "jax-os", "reviewer", "C") == ("ok", "fine", "approve")


def test_report_contract_trailing_horizontal_rule_in_body_still_valid():
    text = (
        "---\nrun_id: abc123\nproject: jax-os\nrole: reviewer\nphase: C\n"
        "verdict: approve\nsummary: fine\n---\n"
        "body text\n\n---\n\nmore body after a body-level rule\n"
    )
    assert jr.validate_report(text, "abc123", "jax-os", "reviewer", "C") == ("ok", "fine", "approve")


def test_report_contract_unrelated_key_value_block_in_body_still_valid():
    text = (
        "---\nrun_id: abc123\nproject: jax-os\nrole: reviewer\nphase: C\n"
        "verdict: approve\nsummary: fine\n---\n"
        "body\n\n---\ncommit: deadbeef\nauthor: someone\n---\nmore body\n"
    )
    assert jr.validate_report(text, "abc123", "jax-os", "reviewer", "C") == ("ok", "fine", "approve")


def test_report_contract_accepts_a_bare_bold_label_with_no_frontmatter():
    # spec line 220: "**label**:" (closed BEFORE the colon) and "**label:**" (closed
    # AFTER the colon) are the same shape -- both must recover cleanly.
    for label in ("result:", "**result**:", "**result:**"):
        text = f"# Report\n\n{label} success\nsummary: built it\n"
        assert jr.validate_report(text, "r1", "demo", "builder", "P") == (
            "ok", "built it", "success"
        )


def test_report_contract_the_a070500df0bc_shape_recovers_the_bold_prefixed_summary():
    # The real report that motivated this whole issue (spec lines 84-95, 214-215):
    # role is "builder", but the report's own label is "**verdict:**" -- the WRONG
    # role's label (row 5) -- with the closing "**" AFTER the colon. Its own
    # "**summary:**" line needs no row-8 fallback: recovered verbatim, bold markup
    # stripped, never a body line.
    text = (
        "# Report — a070500df0bc (builder, jax-os, moa-470)\n"
        "**verdict:** success\n"
        "**commit:** 6127c1153b4070b01e32e6c821d0c6586efa8de9\n"
        "**summary:** Persist every jaxflow worker's child stdout+stderr into redacted "
        "child.log and surface classified reason/tail on missing/invalid run-finished.\n"
    )
    status, summary, outcome = jr.validate_report(
        text, "a070500df0bc", "jax-os", "builder", "moa-470")
    assert status == "invalid"  # row 5: "verdict:" is the wrong label for a builder
    assert outcome is None
    assert summary == (
        "Persist every jaxflow worker's child stdout+stderr into redacted child.log "
        "and surface classified reason/tail on missing/invalid run-finished."
    )


def test_report_contract_accepts_no_label_at_all_as_invalid_but_fallback_eligible():
    # b756335468ba's shape (spec §2.1): invalid like garbage, but the 3rd element is
    # None -- unlike garbage, which is what makes it fallback-eligible internally.
    # Table row 6: the summary source is still rows 7-8, independent of the `invalid`
    # status -- here row 8's qualifying body line, never the fixed diagnostic.
    text = "# Builder report\n\nDid five tasks. All green.\n"
    status, summary, outcome = jr.validate_report(text, "r1", "demo", "builder", "P")
    assert status == "invalid"
    assert outcome is None
    assert summary == "Did five tasks. All green."


def test_report_contract_a_present_out_of_enum_value_is_invalid_and_not_fallback_eligible():
    # Table row 4: `invalid`, but the summary source is still rows 7-8 -- a good
    # explicit summary next to a garbage outcome must survive, not collapse to the
    # fixed diagnostic (that diagnostic is row 7's alone, for a DEFECTIVE summary).
    text = "result: banana\nsummary: built it\n"
    status, summary, outcome = jr.validate_report(text, "r1", "demo", "builder", "P")
    assert status == "invalid"
    assert outcome == "banana"  # NOT None: a claimed value, never fallback-eligible
    assert summary == "built it"


def test_report_contract_asterisk_markup_inside_the_value_does_not_launder_it_into_the_enum():
    # F4/F5: the label is split off the LEFT side of the first ":" only -- normalization
    # never runs on the value, so an asterisk inside it must never be stripped into an
    # accidental enum match.
    text = "result: suc*cess\nsummary: built it\n"
    status, summary, outcome = jr.validate_report(text, "r1", "demo", "builder", "P")
    assert status == "invalid"
    assert outcome == "suc*cess"  # NOT "success" -- the "*" must survive verbatim


def test_report_contract_bold_asterisks_touching_the_value_do_not_launder_it():
    # F4 (the finding this replacement fixes): the previous regex's trailing `\**`
    # consumed asterisks immediately after the colon, so `result: **success` normalized
    # to `success` and wrongly validated. Splitting on the first ":" and never touching
    # the right side closes this exact case, not just the `suc*cess` shape above.
    text = "result: **success\nsummary: built it\n"
    status, summary, outcome = jr.validate_report(text, "r1", "demo", "builder", "P")
    assert status == "invalid"
    assert outcome == "**success"  # NOT "success" -- asterisks must survive verbatim


def test_report_contract_closed_label_markup_never_launders_the_values_own_asterisks():
    # A label's OWN closing "**" (left unclosed, so its match is consumed from the
    # value's start) must never reach past that one run into asterisks that are the
    # VALUE's own markup -- only the exact closing run comes off, nothing else.
    text = "**result:** **success\nsummary: built it\n"
    status, summary, outcome = jr.validate_report(text, "r1", "demo", "builder", "P")
    assert status == "invalid"
    assert outcome == "**success"  # the label's "**" is consumed; the value's own stay


def test_report_contract_asterisk_inside_the_summary_text_survives_verbatim():
    # F5: the same over-eager strip must never touch summary prose either.
    text = "result: success\nsummary: built the *starred* thing\n"
    assert jr.validate_report(text, "r1", "demo", "builder", "P") == (
        "ok", "built the *starred* thing", "success"
    )


def test_split_loose_label_strips_only_a_matching_closing_asterisk_run():
    # F2 (diff review 5e2ef5708068): stripping ANY closing asterisk run regardless of
    # length let a mismatched run launder a malformed value into a clean enum match
    # (`**result: *success` wrongly became `success`). Only a CLOSING run exactly as
    # long as the label's own unclosed OPENING run is markup; anything shorter or
    # longer belongs to the value's own text and must survive verbatim.
    cases = [
        ("**result:** success", "ok", "success"),
        ("**result: *success", "invalid", "*success"),
        ("*result: *success", "ok", "success"),
        ("*result: **success", "invalid", "**success"),
        ("**result: **success", "ok", "success"),
        ("**result:** *success", "invalid", "*success"),
        ("result: **success", "invalid", "**success"),
        ("result: suc*cess", "invalid", "suc*cess"),
        ("***result: ***success", "ok", "success"),
    ]
    for line, expected_status, expected_value in cases:
        text = f"{line}\nsummary: built it\n"
        status, summary, outcome = jr.validate_report(text, "r1", "demo", "builder", "P")
        assert status == expected_status, line
        assert outcome == expected_value, line


def test_report_contract_a_loosely_scanned_blocked_value_resolves_ok_not_fallback_eligible():
    text = "result: blocked\nsummary: stuck on task 2\n"
    assert jr.validate_report(text, "r1", "demo", "builder", "P") == (
        "ok", "stuck on task 2", "blocked"
    )


def test_report_contract_ignores_the_other_roles_label():
    # Table row 5: same `invalid`/summary-source-is-rows-7-8 treatment as row 6 -- a
    # good explicit summary next to the wrong role's label must survive verbatim.
    text = "verdict: approve\nsummary: looks fine\n"
    status, summary, outcome = jr.validate_report(text, "r1", "demo", "builder", "P")
    assert status == "invalid"
    assert outcome is None
    assert summary == "looks fine"


def test_report_contract_first_occurrence_of_a_repeated_label_wins():
    text = "result: success\nresult: failure\nsummary: first summary\nsummary: second\n"
    assert jr.validate_report(text, "r1", "demo", "builder", "P") == (
        "ok", "first summary", "success"
    )


def test_report_contract_summary_fallback_uses_first_non_heading_line():
    text = "# Builder report\n\nDid the work.\n\nresult: success\n"
    assert jr.validate_report(text, "r1", "demo", "builder", "P") == (
        "ok", "Did the work.", "success"
    )


def test_report_contract_summary_fallback_is_the_fixed_literal_with_no_qualifying_line():
    text = "# Report\n\n## Nothing else\n\nresult: success\n"
    assert jr.validate_report(text, "r1", "demo", "builder", "P") == (
        "ok", "no summary in report", "success"
    )


def test_report_contract_summary_fallback_excludes_a_metadata_shaped_body_line():
    # AC 17: the only non-heading line IS a label -- never used as the summary text.
    text = "result: success\n"
    assert jr.validate_report(text, "r1", "demo", "builder", "P") == (
        "ok", "no summary in report", "success"
    )


def test_report_contract_stage_two_explicit_summary_empty_or_over_length_is_invalid():
    # 5520577eb124 F1 (AC 18): widened to match stage 1 -- empty/over-length is invalid
    # too, not just a control character.
    for role, ok_line in (("builder", "result: success"), ("reviewer", "verdict: approve")):
        empty = f"{ok_line}\nsummary: \n"
        overlong = f"{ok_line}\nsummary: {'x' * 301}\n"
        assert jr.validate_report(empty, "r1", "demo", role, "P")[0] == "invalid"
        assert jr.validate_report(overlong, "r1", "demo", role, "P")[0] == "invalid"


def test_report_contract_stage_two_resolves_outcome_and_summary_regardless_of_line_order():
    # 5520577eb124 F2 (AC 19): one order-agnostic pass, neither search gates the other.
    forward = "result: success\nsummary: order one way\n"
    backward = "summary: order the other way\nresult: success\n"
    assert jr.validate_report(forward, "r1", "demo", "builder", "P") == (
        "ok", "order one way", "success"
    )
    assert jr.validate_report(backward, "r1", "demo", "builder", "P") == (
        "ok", "order the other way", "success"
    )


def test_report_contract_ignores_a_result_label_inside_a_fenced_code_block():
    text = "```\nresult: success\n```\nsummary: real summary\n"
    status, summary, outcome = jr.validate_report(text, "r1", "demo", "builder", "P")
    assert status == "invalid"
    assert outcome is None


def test_report_contract_ignores_a_result_label_inside_a_blockquote():
    text = "> result: success\nsummary: real summary\n"
    status, summary, outcome = jr.validate_report(text, "r1", "demo", "builder", "P")
    assert status == "invalid"
    assert outcome is None


def test_report_contract_finds_a_real_label_after_a_fenced_code_block():
    text = "```\nresult: success\n```\nresult: failure\nsummary: real one\n"
    assert jr.validate_report(text, "r1", "demo", "builder", "P") == (
        "ok", "real one", "failure"
    )


def test_report_contract_ignores_a_result_label_inside_a_tilde_fenced_code_block():
    # F6: `_iter_scannable_lines` tracked backtick fences only -- a `~~~` fence let a
    # label inside it count as real.
    text = "~~~\nresult: success\n~~~\nsummary: real summary\n"
    status, summary, outcome = jr.validate_report(text, "r1", "demo", "builder", "P")
    assert status == "invalid"
    assert outcome is None


def test_report_contract_finds_a_real_label_after_a_tilde_fenced_code_block():
    text = "~~~\nresult: success\n~~~\nresult: failure\nsummary: real one\n"
    assert jr.validate_report(text, "r1", "demo", "builder", "P") == (
        "ok", "real one", "failure"
    )


def test_report_contract_mismatched_fence_markers_never_close_a_fence():
    # F6: a fence is closed only by the SAME marker that opened it -- a `~~~` line
    # encountered inside a ``` fence must not end it early (and vice versa). If it did,
    # "summary: still fenced" below would wrongly surface as the real summary.
    text = "```\nresult: banana\n~~~\nsummary: still fenced\n```\nresult: failure\nsummary: real one\n"
    assert jr.validate_report(text, "r1", "demo", "builder", "P") == (
        "ok", "real one", "failure"
    )


def test_runtime_argv_is_pinned_and_prompt_content_stays_out_of_argv():
    with TemporaryDirectory() as raw:
        prompt = Path(raw) / "prompt.txt"
        out = Path(raw) / "last.md"
        grok = [
            "opencode", "run", "--auto", "--format", "json", "--dir", "/repo",
            "--model", "xai/grok-4.6", "Execute the attached handoff exactly.",
            "--file", str(prompt), "--variant", "xhigh",
        ]
        assert jr.runtime_argv("opencode-grok", "builder", Path("/repo"), prompt, out) == grok
        # `--variant high` is part of opencode-deepseek's RUNTIME definition (Rafa's own
        # tested config: openrouter/deepseek/deepseek-v4-flash-0731 at effort high), not a
        # caller override -- `cmd_build` still refuses `--effort` for an OpenCode builder.
        assert jr.runtime_argv("opencode-deepseek", "builder", Path("/repo"), prompt, out) == [
            "opencode", "run", "--auto", "--format", "json", "--dir", "/repo",
            "--model", "openrouter/deepseek/deepseek-v4-flash-0731",
            "Execute the attached handoff exactly.", "--file", str(prompt), "--variant", "high",
        ]
        # `codex` is a REVIEWER runtime only (MOA-451): its workspace-write sandbox cannot
        # write a linked worktree's git dir, so a Codex builder could implement but never
        # commit, and could not spawn `next build`'s TypeScript child either.
        try:
            jr.runtime_argv("codex", "builder", Path("/repo"), prompt, out)
            raise AssertionError("codex must not be accepted as a builder runtime")
        except ValueError:
            pass
        assert jr.runtime_argv("codex", "reviewer", Path("/repo"), prompt, out) == [
            "codex", "exec", "--ephemeral", "--json", "--sandbox", "read-only",
            "-c", 'approval_policy="never"', "--model", "gpt-5.6-luna",
            "-c", 'model_reasoning_effort="xhigh"', "-C", "/tmp",
            "--skip-git-repo-check", "--output-last-message", str(out), "-",
        ]
        joined = " ".join(jr.runtime_argv("opencode-grok", "builder", Path("/repo"), prompt, out))
        assert "secret-handoff" not in joined


def test_runtime_argv_managed_builder():
    with TemporaryDirectory() as raw:
        prompt = Path(raw) / "prompt.txt"
        out = Path(raw) / "last.md"
        argv = jr.runtime_argv(
            "opencode-builder", "builder", Path("/repo"), prompt, out,
            model="fixture/wire-real-model", effort="high",
        )
        assert argv == [
            "opencode", "run", "--auto", "--format", "json", "--dir", "/repo",
            "--agent", "build", "--model", "fixture/wire-real-model",
            "Execute the attached handoff exactly.", "--file", str(prompt),
            "--variant", "high",
        ]
        argv_none = jr.runtime_argv(
            "opencode-builder", "builder", Path("/repo"), prompt, out,
            model="fixture/wire-real-model", effort=None,
        )
        assert "--variant" not in argv_none
        try:
            jr.runtime_argv("opencode-builder", "builder", Path("/repo"), prompt, out, model=None)
            raise AssertionError("missing model accepted")
        except ValueError as exc:
            assert "managed builder model missing" in str(exc)


def test_run_paths_must_remain_beneath_repo():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        paths = jr.safe_run_paths(root, "abc")
        assert paths["scratch"] == root / ".local" / "scratch" / "abc"
        assert paths["report"] == root / ".local" / "reports" / "abc.md"
        assert paths["tests"] == root / ".local" / "reports" / "abc.tests.txt"


def test_run_paths_reject_a_symlinked_local_parent():
    with TemporaryDirectory() as raw:
        base = Path(raw)
        root, outside = base / "repo", base / "outside"
        root.mkdir(); outside.mkdir()
        (root / ".local").symlink_to(outside, target_is_directory=True)
        try:
            jr.safe_run_paths(root.resolve(), "abc")
        except ValueError:
            pass
        else:
            raise AssertionError("symlink escape accepted")


def test_run_paths_reject_a_symlinked_reports_parent():
    with TemporaryDirectory() as raw:
        base = Path(raw)
        root, outside = base / "repo", base / "outside"
        (root / ".local").mkdir(parents=True); outside.mkdir()
        (root / ".local" / "reports").symlink_to(outside, target_is_directory=True)
        try:
            jr.safe_run_paths(root.resolve(), "abc")
        except ValueError:
            pass
        else:
            raise AssertionError("reports escape accepted")


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(root: Path, branch="feat/workflow-phase-c"):
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "t@t.test")
    _git(root, "config", "user.name", "t")
    (root / "README").write_text("x\n", encoding="utf-8")
    (root / ".gitignore").write_text(".local/\n", encoding="utf-8")
    _git(root, "add", "README", ".gitignore")
    _git(root, "commit", "-m", "init")
    if branch != "main":
        _git(root, "checkout", "-b", branch)


def _args(repo, **kw):
    return SimpleNamespace(
        role=kw.get("role", "builder"),
        project=kw.get("project", "jax-os"),
        phase=kw.get("phase", "workflow-phase-c"),
        repo=str(repo),
        prompt_file=kw.get("prompt_file"),
        runtime=kw.get("runtime", "opencode-grok"),
        callback=kw.get("callback"),
    )


def _run_real(argv, cwd=None):
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)


def test_preflight_rejects_detached_dirty_main_wrong_branch_and_bad_tokens():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        assert jr.preflight(_args(root), run=_run_real) == root

        _git(root, "checkout", "--detach")
        try:
            jr.preflight(_args(root), run=_run_real)
        except ValueError:
            pass
        else:
            raise AssertionError("detached HEAD accepted")
        _git(root, "checkout", "feat/workflow-phase-c")

        (root / "dirty.txt").write_text("x\n", encoding="utf-8")
        try:
            jr.preflight(_args(root), run=_run_real)
        except ValueError:
            pass
        else:
            raise AssertionError("untracked dirty tree accepted")
        (root / "dirty.txt").unlink()

        _git(root, "checkout", "main")
        try:
            jr.preflight(_args(root), run=_run_real)
        except ValueError:
            pass
        else:
            raise AssertionError("builder on main accepted")

        _git(root, "checkout", "-b", "feat/other")
        try:
            jr.preflight(_args(root), run=_run_real)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong builder branch accepted")

        _git(root, "checkout", "feat/workflow-phase-c")
        try:
            jr.preflight(_args(root, project="BAD"), run=_run_real)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed project accepted")
        try:
            jr.preflight(_args(root, phase="has space"), run=_run_real)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed phase accepted")


def test_preflight_rejects_stale_callback_via_injected_runner():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)

        def run(argv, cwd=None):
            if argv[:2] == ["tmux", "list-panes"]:
                return SimpleNamespace(returncode=0, stdout="jax-other-lead:1.1\n", stderr="")
            return _run_real(argv, cwd)

        try:
            jr.preflight(_args(root, callback="jax-os-lead:1.1"), run=run)
        except ValueError:
            pass
        else:
            raise AssertionError("stale callback accepted")

        def live(argv, cwd=None):
            if argv[:2] == ["tmux", "list-panes"]:
                return SimpleNamespace(returncode=0, stdout="jax-os-lead:1.1\n", stderr="")
            return _run_real(argv, cwd)

        jr.preflight(_args(root, callback="jax-os-lead:1.1"), run=live)


def test_parse_reviewer_handoff_requires_top_level_diff_and_indented_test_output():
    parsed = jr.parse_reviewer_handoff(VALID_HANDOFF)
    assert parsed["base_sha"] == "a" * 40
    assert parsed["head_sha"] == "b" * 40
    assert parsed["test_output"] == "/canonical/repo/.local/reports/builder123.tests.txt"
    assert parsed["builder_run_id"] == "builder123"
    for bad in (
        VALID_HANDOFF.replace("  test-output:", "paths.test-output:"),
        VALID_HANDOFF.replace("  test-output:", " test-output:"),
        VALID_HANDOFF.replace("  test-output:", "    test-output:"),
        VALID_HANDOFF + "\ndiff: " + "c" * 40 + ".." + "d" * 40 + "\n",
        VALID_HANDOFF.replace("paths:\n", "paths:\n") + "paths:\n",
        VALID_HANDOFF + "\n  test-output: /canonical/repo/.local/reports/other.tests.txt\n",
        "test-output: /canonical/repo/.local/reports/builder123.tests.txt\n" + VALID_HANDOFF,
        VALID_HANDOFF.replace("a" * 40, "main").replace("b" * 40, "HEAD"),
        VALID_HANDOFF.replace("a" * 40, "a" * 12),
        VALID_HANDOFF.replace("builder123.tests.txt", "builder123.md"),
    ):
        try:
            jr.parse_reviewer_handoff(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad handoff accepted: {bad!r}")


def test_reviewer_evidence_rejects_empty_and_symlink_escape():
    with TemporaryDirectory() as raw:
        base = Path(raw).resolve()
        repo, outside = base / "repo", base / "outside"
        reports = repo / ".local" / "reports"
        reports.mkdir(parents=True)
        outside.mkdir()
        empty = reports / "builder123.tests.txt"
        empty.write_text("", encoding="utf-8")
        try:
            jr.validate_reviewer_evidence(str(empty), reports)
        except ValueError:
            pass
        else:
            raise AssertionError("empty evidence accepted")
        empty.write_text("ok\n", encoding="utf-8")
        escaped = outside / "builder123.tests.txt"
        escaped.write_text("ok\n", encoding="utf-8")
        link = reports / "builder123.tests.txt"
        link.unlink()
        link.symlink_to(escaped)
        try:
            jr.validate_reviewer_evidence(str(link), reports)
        except ValueError:
            pass
        else:
            raise AssertionError("symlinked evidence accepted")


def test_preflight_rejects_reviewer_when_head_does_not_match():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        reports = root / ".local" / "reports"
        reports.mkdir(parents=True)
        evidence = reports / "builder123.tests.txt"
        evidence.write_text("COMMAND: true\nEXIT: 0\n", encoding="utf-8")
        prompt = root / "handoff.md"
        prompt.write_text(
            VALID_HANDOFF.replace("/canonical/repo/.local/reports/builder123.tests.txt", str(evidence)),
            encoding="utf-8",
        )
        with patch.object(jr, "reviewer_head_matches", return_value=False):
            try:
                jr.preflight(_args(root, role="reviewer", runtime="codex", prompt_file=str(prompt)), run=_run_real)
            except ValueError:
                pass
            else:
                raise AssertionError("head mismatch accepted")


def test_assemble_prompt_orders_contracts_injection_then_handoff():
    injected = {
        "run_id": "abc123",
        "project": "jax-os",
        "role": "builder",
        "phase": "workflow-phase-c",
        "report": "/repo/.local/reports/abc123.md",
        "tests": "/repo/.local/reports/abc123.tests.txt",
        "scratch": "/repo/.local/scratch/abc123",
        "repo": "/repo",
    }
    text = jr.assemble_prompt("builder", injected, "HANDOFF-BODY")
    assert text.index("Builder Contract") < text.index("Status & Report Contract")
    assert text.index("Status & Report Contract") < text.index("run_id: abc123")
    assert text.index("run_id: abc123") < text.index("HANDOFF-BODY")
    assert text.index("scratch: /repo/.local/scratch/abc123") < text.index("HANDOFF-BODY")


def test_assemble_prompt_builder_prompt_shows_its_own_report_example_block():
    injected = {
        "run_id": "abc123", "project": "jax-os", "role": "builder", "phase": "P",
        "report": "/repo/.local/reports/abc123.md",
        "tests": "/repo/.local/reports/abc123.tests.txt",
        "scratch": "/repo/.local/scratch/abc123", "repo": "/repo",
    }
    text = jr.assemble_prompt("builder", injected, "HANDOFF-BODY")
    builder_block = (
        "## REPORT FRONTMATTER EXAMPLE\n\n```markdown\n---\n"
        "run_id: <injected — copy verbatim>\nproject: <injected — copy verbatim>\n"
        "role: builder\nphase: <injected — copy verbatim>\n"
        "result: success | failure | blocked\nsummary: <ONE line, ≤200 chars>\n---\n```"
    )
    reviewer_block = builder_block.replace("role: builder", "role: reviewer").replace(
        "result: success | failure | blocked",
        "verdict: approve | approve-with-changes | reject",
    )
    assert builder_block in text
    assert reviewer_block not in text


def test_assemble_prompt_reviewer_prompt_shows_its_own_report_example_block():
    injected = {
        "run_id": "abc123", "project": "jax-os", "role": "reviewer", "phase": "P",
        "report": "/repo/.local/reports/abc123.md",
        "tests": "/repo/.local/reports/abc123.tests.txt",
        "scratch": "/repo/.local/scratch/abc123", "repo": "/repo",
    }
    text = jr.assemble_prompt("reviewer", injected, "HANDOFF-BODY")
    builder_block = (
        "## REPORT FRONTMATTER EXAMPLE\n\n```markdown\n---\n"
        "run_id: <injected — copy verbatim>\nproject: <injected — copy verbatim>\n"
        "role: builder\nphase: <injected — copy verbatim>\n"
        "result: success | failure | blocked\nsummary: <ONE line, ≤200 chars>\n---\n```"
    )
    reviewer_block = builder_block.replace("role: builder", "role: reviewer").replace(
        "result: success | failure | blocked",
        "verdict: approve | approve-with-changes | reject",
    )
    assert reviewer_block in text
    assert builder_block not in text


def test_atomic_finalize_replaces_and_locks_valid_bytes():
    with TemporaryDirectory() as raw:
        base = Path(raw).resolve()
        reports = base / "reports"
        reports.mkdir()
        dest = reports / "abc.md"
        jr.atomic_finalize(dest, "hello\n", reports)
        assert dest.read_text(encoding="utf-8") == "hello\n"
        assert stat.S_IMODE(dest.stat().st_mode) == 0o444
        outside = base / "outside.md"
        try:
            jr.atomic_finalize(outside, "nope\n", reports)
        except ValueError:
            pass
        else:
            raise AssertionError("outside sibling accepted")
        assert not outside.exists()
        link = reports / "link.md"
        link.symlink_to(dest)
        try:
            jr.atomic_finalize(link, "x\n", reports)
        except ValueError:
            pass
        else:
            raise AssertionError("symlink destination overwritten")
        escaped = base / "escaped"
        escaped.mkdir()
        nested = reports / "nested"
        nested.symlink_to(escaped, target_is_directory=True)
        try:
            jr.atomic_finalize(nested / "abc.md", "x\n", reports)
        except ValueError:
            pass
        else:
            raise AssertionError("ancestor symlink escape accepted")


def test_atomic_finalize_rejects_local_replaced_by_symlink_after_preflight():
    with TemporaryDirectory() as raw:
        base = Path(raw).resolve()
        repo, outside = base / "repo", base / "outside"
        repo.mkdir()
        paths = jr.safe_run_paths(repo, "abc")
        report, reports_root = paths["report"], paths["reports_root"]
        local = repo / ".local"
        aside = base / "local-aside"
        local.rename(aside)
        (outside / "reports").mkdir(parents=True)
        local.symlink_to(outside, target_is_directory=True)
        try:
            jr.atomic_finalize(report, "escaped\n", reports_root)
        except ValueError:
            pass
        else:
            raise AssertionError("post-preflight .local symlink accepted")
        assert not (outside / "reports" / "abc.md").exists()
        assert not report.exists()


def test_reviewer_head_matches_against_real_sqlite():
    sha = "b" * 40
    other = "a" * 40
    with TemporaryDirectory() as raw:
        db = Path(raw) / "jaxos.db"
        con = sqlite3.connect(db)
        con.execute(
            "CREATE TABLE workflow_events ("
            "id INTEGER PRIMARY KEY, ts TEXT NOT NULL, run_id TEXT, "
            "project TEXT NOT NULL, role TEXT, type TEXT NOT NULL, "
            "source TEXT NOT NULL, emitter TEXT NOT NULL, payload TEXT, "
            "delivery TEXT NOT NULL, forwarded_at TEXT)"
        )

        def insert(run_id, role, typ, head_sha):
            con.execute(
                "INSERT INTO workflow_events "
                "(ts, run_id, project, role, type, source, emitter, payload, delivery) "
                "VALUES ('t', ?, 'jax-os', ?, ?, 'deterministic', 'wrapper', ?, 'local')",
                (run_id, role, typ, json.dumps({"head_sha": head_sha})),
            )

        insert("builder123", "builder", "run-finished", sha)
        insert("wrong-role", "reviewer", "run-finished", sha)
        insert("wrong-type", "builder", "run-started", sha)
        insert("bad-sha", "builder", "run-finished", "not-a-sha")
        con.commit()
        con.close()

        assert jr.reviewer_head_matches(db, "builder123", sha) is True
        assert jr.reviewer_head_matches(db, "builder123", other) is False
        assert jr.reviewer_head_matches(db, "wrong-role", sha) is False
        assert jr.reviewer_head_matches(db, "wrong-type", sha) is False
        assert jr.reviewer_head_matches(db, "bad-sha", sha) is False
        assert jr.reviewer_head_matches(Path(raw) / "missing.db", "builder123", sha) is False

        corrupt = Path(raw) / "corrupt.db"
        corrupt.write_text("not a sqlite database", encoding="utf-8")
        assert jr.reviewer_head_matches(corrupt, "builder123", sha) is False

        broken = Path(raw) / "broken-schema.db"
        bad = sqlite3.connect(broken)
        bad.execute("CREATE TABLE workflow_events (id INTEGER PRIMARY KEY)")
        bad.commit()
        bad.close()
        assert jr.reviewer_head_matches(broken, "builder123", sha) is False


def test_preflight_softened_kwargs_relax_dirty_and_branch_checks():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        (root / "untracked.txt").write_text("x\n", encoding="utf-8")
        args = SimpleNamespace(
            role="builder", project="demo", phase="P", repo=str(root),
            prompt_file=None, runtime="opencode-grok", callback=None,
        )
        _git(root, "checkout", "-b", "feat/P")
        jr.preflight(args, run=_run_real, allow_untracked=True)

        (root / "tracked.txt").write_text("y\n", encoding="utf-8")
        _git(root, "add", "tracked.txt")
        try:
            jr.preflight(args, run=_run_real, allow_untracked=True)
        except ValueError as exc:
            assert str(exc) == "dirty-tracked-tree"
        else:
            raise AssertionError("dirty-tracked-tree not raised")

        _git(root, "commit", "-m", "add tracked")
        (root / "untracked.txt").unlink()  # clean tree again: the branch rule is what's under test below
        _git(root, "checkout", "-b", "not-feat-branch")
        jr.preflight(args, run=_run_real, require_feat_branch=False)
        try:
            jr.preflight(args, run=_run_real)
        except ValueError as exc:
            assert str(exc) == "builder branch must be feat/<phase>"
        else:
            raise AssertionError("builder branch must be feat/<phase> not raised (legacy default)")


def test_preflight_skip_handoff_and_claude_runtime_accepted():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        _git(root, "checkout", "-b", "any-branch")
        for runtime in ("codex", "claude"):
            args = SimpleNamespace(
                role="reviewer", project="demo", phase="P", repo=str(root),
                prompt_file=None, runtime=runtime, callback=None,
            )
            jr.preflight(args, run=_run_real, allow_untracked=True, skip_handoff=True)


def test_runtime_argv_model_effort_override_and_claude_branch_and_legacy_bytes_identical():
    legacy = jr.runtime_argv("codex", "reviewer", "/repo", Path("/p"), Path("/o"))
    assert legacy == [
        "codex", "exec", "--ephemeral", "--json", "--sandbox", "read-only",
        "-c", 'approval_policy="never"', "--model", "gpt-5.6-luna",
        "-c", 'model_reasoning_effort="xhigh"', "-C", "/tmp",
        "--skip-git-repo-check", "--output-last-message", "/o", "-",
    ]
    overridden = jr.runtime_argv("codex", "reviewer", "/repo", Path("/p"), Path("/o"), model="foo", effort="high")
    assert "--model" in overridden and overridden[overridden.index("--model") + 1] == "foo"
    assert 'model_reasoning_effort="high"' in overridden
    claude_argv = jr.runtime_argv("claude", "reviewer", "/repo", Path("/p"), Path("/o"), model="sonnet", effort="high")
    assert claude_argv == [
        "claude", "-p", "--model", "sonnet", "--effort", "high", "--output-format", "text",
        "--no-session-persistence", "--setting-sources", "", "--tools", "Read,Glob,Grep",
        "--add-dir", "/repo",
    ]
    diff_argv = jr.runtime_argv("claude", "reviewer", "/repo", Path("/p"), Path("/o"), model="sonnet", effort="high",
                                diff_range=("a" * 40, "b" * 40))
    # a diff review gets Bash pinned to three exact shapes on this run's SHAs -- never an
    # open `Bash(git diff:*)` prefix (`--output=` would write; review 85a6b2a16cfa)
    assert diff_argv[diff_argv.index("--tools") + 1] == "Read,Glob,Grep,Bash"
    allowed = diff_argv[diff_argv.index("--allowedTools") + 1:diff_argv.index("--add-dir")]
    assert allowed == [
        f"Bash(git diff {'a' * 40}..{'b' * 40})",
        f"Bash(git diff --stat {'a' * 40}..{'b' * 40})",
        f"Bash(git diff {'a' * 40}..{'b' * 40} -- :*)",
    ]
    assert "Bash(git diff:*)" not in diff_argv


def test_reviewer_head_matches_ancestor_via_repo_and_legacy_exact_equality():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)
        (root / "a.txt").write_text("a\n", encoding="utf-8")
        _git(root, "add", "a.txt")
        _git(root, "commit", "-m", "base")
        base_sha = _run_real(["git", "rev-parse", "HEAD"], root).stdout.strip()
        (root / "b.txt").write_text("b\n", encoding="utf-8")
        _git(root, "add", "b.txt")
        _git(root, "commit", "-m", "head")
        head_sha = _run_real(["git", "rev-parse", "HEAD"], root).stdout.strip()

        db_path = root / "jaxos.db"
        con = sqlite3.connect(db_path)
        con.execute("CREATE TABLE workflow_events (id INTEGER PRIMARY KEY, run_id TEXT, role TEXT, type TEXT, payload TEXT)")
        con.execute(
            "INSERT INTO workflow_events (run_id, role, type, payload) VALUES ('b1', 'builder', 'run-finished', ?)",
            (json.dumps({"head_sha": base_sha}),),
        )
        con.commit()
        con.close()

        assert jr.reviewer_head_matches(db_path, "b1", head_sha, repo=root, run=_run_real) is True
        assert jr.reviewer_head_matches(db_path, "b1", base_sha, repo=root, run=_run_real) is True
        assert jr.reviewer_head_matches(db_path, "b1", "f" * 40, repo=root, run=_run_real) is False
        assert jr.reviewer_head_matches(db_path, "b1", head_sha) is False


def test_alloc_run_explicit_run_id_collision_and_generated_path_unaffected():
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        _init_repo(root)

        class FakeTmuxRun:
            def __init__(self):
                self.existing = set()

            def __call__(self, argv, cwd=None):
                if argv[:2] == ["tmux", "has-session"]:
                    name = argv[argv.index("-t") + 1]
                    return SimpleNamespace(returncode=0 if name in self.existing else 1)
                raise AssertionError("unexpected call")

        fake = FakeTmuxRun()
        fake.existing.add("jax-demo-spec-taken12345678")
        try:
            jr._alloc_run("demo", "spec", root, fake, run_id="taken12345678")
        except ValueError as exc:
            assert str(exc) == "run path collision"
        else:
            raise AssertionError("run path collision not raised")

        run_id, session, paths = jr._alloc_run("demo", "spec", root, fake, run_id="free1234abcd")
        assert run_id == "free1234abcd"
        assert session == "jax-demo-spec-free1234abcd"

        run_id2, session2, paths2 = jr._alloc_run("demo", "spec", root, fake)
        assert len(run_id2) == 12
        assert session2 == f"jax-demo-spec-{run_id2}"


def test_runtime_argv_opencode_grok_carries_its_variant_and_model_override_still_works():
    """Every builder runtime now names its own variant, and each value is one the model
    actually offers (grok-4.6: low/medium/high/xhigh). A model override must not disturb
    the variant -- the variant belongs to the runtime, the model id to the call."""
    with TemporaryDirectory() as raw:
        prompt = Path(raw) / "prompt.txt"
        out = Path(raw) / "last.md"
        bare = jr.runtime_argv("opencode-grok", "builder", Path("/repo"), prompt, out)
        assert bare == [
            "opencode", "run", "--auto", "--format", "json", "--dir", "/repo",
            "--model", "xai/grok-4.6", "Execute the attached handoff exactly.",
            "--file", str(prompt), "--variant", "xhigh",
        ]
        overridden = jr.runtime_argv(
            "opencode-grok", "builder", Path("/repo"), prompt, out, model="custom-id",
        )
        assert overridden[overridden.index("--model") + 1] == "custom-id"
        assert overridden[overridden.index("--variant") + 1] == "xhigh"

def test_preflight_threads_db_path_into_reviewer_head_gate():
    # fixes cold review round 2 F3(a): preflight()'s skip_handoff=False branch hardcoded
    # jr.DB_PATH with no way to inject a test database -- this new db_path=None keyword
    # (defaults to jr.DB_PATH, so every existing call stays byte-for-byte unaffected) is
    # the fix. A bare call still checks the real DB_PATH (patched here to a missing file,
    # so it cannot accidentally see Rafa's real workflow_events); passing db_path= must
    # reach the seeded database instead.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "repo"
        root.mkdir()
        _init_repo(root)
        reports = root / ".local" / "reports"
        reports.mkdir(parents=True)
        head_sha = "b" * 40
        evidence = reports / "builder123.tests.txt"
        evidence.write_text("COMMAND: true\nEXIT: 0\n", encoding="utf-8")
        prompt = root / "handoff.md"
        prompt.write_text(
            VALID_HANDOFF.replace("/canonical/repo/.local/reports/builder123.tests.txt", str(evidence)),
            encoding="utf-8",
        )

        db = Path(raw) / "jaxos.db"
        con = sqlite3.connect(db)
        con.execute(
            "CREATE TABLE workflow_events (id INTEGER PRIMARY KEY, run_id TEXT, role TEXT, "
            "type TEXT, payload TEXT)"
        )
        con.execute(
            "INSERT INTO workflow_events (run_id, role, type, payload) VALUES "
            "('builder123', 'builder', 'run-finished', ?)",
            (json.dumps({"head_sha": head_sha}),),
        )
        con.commit()
        con.close()

        # fixes plan fixture bug: `handoff.md` is an untracked file at the repo root, so
        # without allow_untracked=True preflight()'s own dirty-tree check raises first --
        # never reaching the reviewer_head_matches line this test targets (matches how
        # both real callers, dispatch_review/dispatch_diff_review, already invoke preflight).
        missing_db = Path(raw) / "missing.db"
        with patch.object(jr, "DB_PATH", missing_db):
            try:
                jr.preflight(
                    _args(root, role="reviewer", runtime="codex", prompt_file=str(prompt)),
                    run=_run_real, allow_untracked=True,
                )
            except ValueError as exc:
                assert str(exc) == "reviewer head mismatch"
            else:
                raise AssertionError("bare preflight() matched against the wrong DB_PATH")

        assert jr.preflight(
            _args(root, role="reviewer", runtime="codex", prompt_file=str(prompt)),
            run=_run_real, db_path=db, allow_untracked=True,
        ) == root


def test_runtime_argv_codex_reviewer_codex_cwd_override_bare_call_stays_tmp():
    # fixes cold review F4: a diff review's cwd override reaches Codex's OWN -C flag,
    # while every existing bare call (no codex_cwd) is unaffected.
    legacy = jr.runtime_argv("codex", "reviewer", "/repo", Path("/p"), Path("/o"))
    assert legacy[legacy.index("-C") + 1] == "/tmp"
    diff_argv = jr.runtime_argv("codex", "reviewer", "/repo", Path("/p"), Path("/o"), codex_cwd="/worktree")
    assert diff_argv[diff_argv.index("-C") + 1] == "/worktree"


def test_runtime_argv_claude_dedupes_read_dir_grants_and_keeps_spaces():
    # MOA-467: the read-only directory grants (control repo, worktree, external document
    # parents) must appear exactly once each, in order, with spaces intact -- argv
    # entries, never a shell-joined string.
    with TemporaryDirectory() as raw:
        prompt = Path(raw) / "prompt.txt"
        out = Path(raw) / "last.md"
        argv = jr.runtime_argv(
            "claude", "reviewer", Path("/control repo"), prompt, out, model="sonnet", effort="high",
            extra_read_dirs=(Path("/control repo"), Path("/work tree"), Path("/docs parent")),
        )
        assert argv[argv.index("--add-dir"):] == [
            "--add-dir", "/control repo", "/work tree", "/docs parent",
        ]


def test_runtime_argv_codex_ignores_extra_read_dirs_unchanged():
    # Codex has no directory-grant mechanism -- passing the same grants must leave its
    # argv byte-for-byte identical.
    bare = jr.runtime_argv("codex", "reviewer", "/repo", Path("/p"), Path("/o"))
    with_grants = jr.runtime_argv(
        "codex", "reviewer", "/repo", Path("/p"), Path("/o"),
        extra_read_dirs=(Path("/control repo"), Path("/work tree")),
    )
    assert with_grants == bare
    assert "--add-dir" not in with_grants


def test_preflight_bare_call_rejects_ancestor_only_match_db_path_call_accepts_it():
    # fixes Part 2 diff-review F4: `preflight()`'s reviewer-handoff branch used to pass
    # `repo=repo, run=run` to `reviewer_head_matches` UNCONDITIONALLY, so even the BARE
    # (db_path=None) call silently gained the ancestor-accepting check -- it must stay
    # exact-equality-only (spec §2.10 #3: "Without repo: today's exact-equality check,
    # unchanged"). Only `dispatch_diff_review`'s own explicit `db_path=` call may opt in.
    with TemporaryDirectory() as raw:
        root = Path(raw).resolve() / "repo"
        root.mkdir()
        _init_repo(root)
        (root / "a.txt").write_text("a\n", encoding="utf-8")
        _git(root, "add", "a.txt")
        _git(root, "commit", "-m", "base")
        base_sha = _run_real(["git", "rev-parse", "HEAD"], root).stdout.strip()
        (root / "b.txt").write_text("b\n", encoding="utf-8")
        _git(root, "add", "b.txt")
        _git(root, "commit", "-m", "head")
        head_sha = _run_real(["git", "rev-parse", "HEAD"], root).stdout.strip()

        reports = root / ".local" / "reports"
        reports.mkdir(parents=True)
        evidence = reports / "builder123.tests.txt"
        evidence.write_text("COMMAND: true\nEXIT: 0\n", encoding="utf-8")
        prompt = root / "handoff.md"
        prompt.write_text(
            VALID_HANDOFF
            .replace("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", head_sha)
            .replace("/canonical/repo/.local/reports/builder123.tests.txt", str(evidence)),
            encoding="utf-8",
        )

        db = Path(raw) / "jaxos.db"
        con = sqlite3.connect(db)
        con.execute(
            "CREATE TABLE workflow_events (id INTEGER PRIMARY KEY, run_id TEXT, role TEXT, "
            "type TEXT, payload TEXT)"
        )
        con.execute(
            "INSERT INTO workflow_events (run_id, role, type, payload) VALUES "
            # the recorded head is only an ANCESTOR of head_sha, never equal to it.
            "('builder123', 'builder', 'run-finished', ?)",
            (json.dumps({"head_sha": base_sha}),),
        )
        con.commit()
        con.close()

        with patch.object(jr, "DB_PATH", db):
            try:
                jr.preflight(
                    _args(root, role="reviewer", runtime="codex", prompt_file=str(prompt)),
                    run=_run_real, allow_untracked=True,
                )
            except ValueError as exc:
                assert str(exc) == "reviewer head mismatch"
            else:
                raise AssertionError("bare preflight() accepted an ancestor-only match")

        assert jr.preflight(
            _args(root, role="reviewer", runtime="codex", prompt_file=str(prompt)),
            run=_run_real, db_path=db, allow_untracked=True,
        ) == root


def test_parse_tests_frames_single_frame_ok():
    text = "COMMAND: true\nsome output\nmore output\nEXIT: 0\n"
    frames, error = jr.parse_tests_frames(text, ["true"])
    assert error is None
    assert frames == [{"exit_code": 0}]


def test_parse_tests_frames_two_frames_ok():
    text = "COMMAND: cmd-a\nout-a\nEXIT: 0\nCOMMAND: cmd-b\nout-b\nEXIT: 1\n"
    frames, error = jr.parse_tests_frames(text, ["cmd-a", "cmd-b"])
    assert error is None
    assert frames == [{"exit_code": 0}, {"exit_code": 1}]


def test_parse_tests_frames_marker_like_content_inside_open_frame_is_content():
    # A captured command's own stdout containing literal `COMMAND: foo` / `EXIT: 0`
    # lines must NOT be mistaken for a new frame boundary -- only an EXACT match of the
    # NEXT expected command's opening line closes the current frame (spec item 8).
    text = (
        "COMMAND: cmd-a\n"
        "some tool printed:\nCOMMAND: foo\nEXIT: 0\n"  # content, not a real boundary
        "EXIT: 7\n"  # the LAST EXIT: line in frame 1 wins
        "COMMAND: cmd-b\nout-b\nEXIT: 0\n"
    )
    frames, error = jr.parse_tests_frames(text, ["cmd-a", "cmd-b"])
    assert error is None
    assert frames == [{"exit_code": 7}, {"exit_code": 0}]


def test_parse_tests_frames_missing_opening_line():
    text = "COMMAND: cmd-a\nout\nEXIT: 0\n"
    frames, error = jr.parse_tests_frames(text, ["cmd-a", "cmd-b"])
    assert frames is None
    assert error == "frame-2-missing"


def test_parse_tests_frames_no_exit_line():
    text = "COMMAND: cmd-a\nno exit line here\n"
    frames, error = jr.parse_tests_frames(text, ["cmd-a"])
    assert frames is None
    assert error == "frame-1-no-exit"


def test_parse_tests_frames_leading_content_before_frame_one():
    text = "some prose before\nCOMMAND: cmd-a\nout\nEXIT: 0\n"
    frames, error = jr.parse_tests_frames(text, ["cmd-a"])
    assert frames is None
    assert error == "leading-content"


def test_parse_tests_frames_empty_text_is_frame_one_missing():
    frames, error = jr.parse_tests_frames("", ["cmd-a"])
    assert frames is None
    assert error == "frame-1-missing"


# ---- MOA-470: child.log path, tail excerpt, reason classifier ----

def test_child_log_path_matches_the_manifest_dir_shape():
    # spec §4.1: child.log is a sibling of manifest.json, which `jaxflow.py`'s
    # `_manifest_dir` roots at `<repo>/.local/runs/<run_id>/` on the CONTROL repo.
    # `jaxflow_run.py` cannot import `_manifest_dir` (jaxflow.py imports THIS module,
    # not the reverse), so this is a one-line independent restatement of that shape.
    assert jr.child_log_path(Path("/r"), "abc123") == Path("/r/.local/runs/abc123/child.log")


def test_child_log_tail_keeps_last_40_lines():
    text = "\n".join(f"line {i}" for i in range(100)) + "\n"
    tail = jr.child_log_tail(text)
    assert "line 60" in tail  # the 100 lines are 0..99; the last 40 are 60..99
    assert "line 59" not in tail


def test_child_log_tail_bounds_to_4096_chars_via_jr_bound():
    # spec §4.3: "additionally hard-capped at 4096 characters via jr._bound" -- even a
    # SINGLE line (so the 40-line cap above never engages) must still be trimmed.
    text = "x" * 10000
    tail = jr.child_log_tail(text)
    assert len(tail) == 4096
    assert tail == "x" * 4096


# Fixture text below reproduces (spec §6 acceptance criterion 5, §2.1) the verbatim
# `message.error`/`message.updated` strings recorded from the three 2026-09-10 failure
# shapes -- NOT replayed from a real run (none survived), built inline as strings.
XAI_SPENDING_LIMIT_WINDOW = (
    '{"type":"message.updated","properties":{"info":{"error":{"name":"APIError",'
    '"data":{"message":"personal-team-blocked:spending-limit: You have run out of '
    'credits or need a Grok subscription.","statusCode":403}}}}}\n'
)
OPENROUTER_PAYMENT_REQUIRED_WINDOW = (
    '{"type":"message.updated","properties":{"info":{"error":{"name":"APIError",'
    '"data":{"message":"This request requires more credits, or fewer max_tokens. '
    'You requested up to 393216 tokens, but can only afford 302941.","statusCode":402}'
    '}}}}\n'
)
DEEPSEEK_CLEAN_EXIT_NO_REPORT_WINDOW = (
    '{"type":"message.updated","properties":{"info":{"finish":"length"}}}\n'
)


def test_classify_reason_xai_spending_limit_is_provider_limit_even_at_nonzero_exit():
    # order matters (spec §4.3): a pattern hit wins even though exit_code != 0 would
    # otherwise classify as "crash" -- the more specific label wins.
    assert jr.classify_reason(
        XAI_SPENDING_LIMIT_WINDOW, exit_code=1, contract_status="missing",
    ) == "provider-limit"


def test_classify_reason_openrouter_payment_required_is_provider_limit():
    assert jr.classify_reason(
        OPENROUTER_PAYMENT_REQUIRED_WINDOW, exit_code=1, contract_status="missing",
    ) == "provider-limit"


def test_classify_reason_deepseek_clean_exit_no_report_is_no_report():
    assert jr.classify_reason(
        DEEPSEEK_CLEAN_EXIT_NO_REPORT_WINDOW, exit_code=0, contract_status="missing",
    ) == "no-report"


def test_classify_reason_nonzero_exit_no_pattern_match_is_crash():
    assert jr.classify_reason(
        "plain traceback text, nothing provider-shaped here\n",
        exit_code=1, contract_status="missing",
    ) == "crash"


def test_classify_reason_zero_exit_invalid_report_is_unknown():
    assert jr.classify_reason(
        "plain output, no error, no patterns\n", exit_code=0, contract_status="invalid",
    ) == "unknown"


def test_classify_reason_billing_and_quota_patterns():
    # spec §4.4's remaining table rows, not exercised by the three named fixtures above.
    assert jr.classify_reason('"quota exceeded"\n', exit_code=1, contract_status="missing") == "provider-limit"
    assert jr.classify_reason("your billing details are invalid\n", exit_code=1, contract_status="missing") == "provider-limit"
    assert jr.classify_reason('"statusCode":402\n', exit_code=1, contract_status="missing") == "provider-limit"
    # deliberately absent from the table (spec §4.4): a bare 403 must NOT match on its
    # own -- only xAI's specific message text does (the fixture above already covers
    # that). A generic "403 Forbidden" line must classify as crash, not provider-limit.
    assert jr.classify_reason("403 Forbidden\n", exit_code=1, contract_status="missing") == "crash"


# ---- MOA-495 2.1: refine_reason (Jev Choice refinement) ------------------------

def test_refine_reason_regex_wins_without_calling_jev():
    def unreachable(_window):
        raise AssertionError("jev must not be called when the regex already matched")
    result = jr.refine_reason(
        XAI_SPENDING_LIMIT_WINDOW, exit_code=1, contract_status="missing", jev=unreachable,
    )
    assert result == "provider-limit"


def test_refine_reason_uses_jev_choice_at_or_above_confidence_threshold():
    result = jr.refine_reason(
        "plain traceback, nothing provider-shaped\n", exit_code=1, contract_status="missing",
        jev=lambda window: ("test-failure", 0.5),
    )
    assert result == "test-failure"


def test_refine_reason_falls_back_below_confidence_threshold():
    result = jr.refine_reason(
        "plain traceback, nothing provider-shaped\n", exit_code=1, contract_status="missing",
        jev=lambda window: ("hung", 0.49),
    )
    assert result == "crash"  # today's classify_reason result, unchanged


def test_refine_reason_falls_back_on_jev_error():
    def raising(_window):
        raise RuntimeError("jev down")
    result = jr.refine_reason(
        "plain output, no error, no patterns\n", exit_code=0, contract_status="invalid",
        jev=raising,
    )
    assert result == "unknown"  # today's classify_reason result, unchanged


def test_refine_reason_returns_the_base_result_with_zero_jev_calls_when_classifier_is_off(monkeypatch):
    # `refine_reason` swallows ANY exception from its `jev` callable (an injected fake that
    # only raises would let this test pass without the gate), so the fake both records its
    # calls and would return a decision-changing answer if it ever ran.
    import general_settings
    monkeypatch.setattr(general_settings, "read_settings",
                         lambda: {"ok": True, "data": {"integrations": {"classifier": False}}})
    calls = []
    def counting_jev(_window):
        calls.append(_window)
        return "hung", 0.99
    result = jr.refine_reason("plain traceback, nothing provider-shaped\n", exit_code=1,
                              contract_status="missing", jev=counting_jev)
    assert calls == []
    assert result == jr.classify_reason("plain traceback, nothing provider-shaped\n", exit_code=1, contract_status="missing")


def test_default_reason_jev_rejects_a_choice_outside_the_criteria(monkeypatch):
    # The validation `refine_reason`'s default path relies on -- an injected `jev=`
    # fake is trusted to already return a valid (choice, confidence) pair, same as
    # workflow_poll.py's own _jev_ok() fixtures.
    monkeypatch.setattr(jev_client, "api_key", lambda name=None: "k")
    monkeypatch.setattr(jev_client, "call", lambda *a, **k: {
        "reason": {"choice": "banana", "confidence": 0.99},
    })
    with pytest.raises(ValueError):
        jr._default_reason_jev("plain traceback\n")


# ---- MOA-495 2.1: log_context (Jev Score over ~2KiB chunks) ---------------------

def test_log_context_keeps_top_scored_chunks_and_always_the_last_one():
    # 7 chunks (A..G); the last (G) scores lowest of all and would not make the top 5 --
    # it must still be surfaced. The middle-lowest scorer (F, index 5) is correctly cut.
    letters = "ABCDEFG"
    window = "".join(letter * 2048 for letter in letters)
    scores = {0: 7, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
    result = jr.log_context(window, score_chunks=lambda chunks: scores)
    assert result == [letter * 500 for letter in "ABCDEG"]  # log order, bounded to 500 chars


def test_log_context_none_on_jev_error():
    def raising(_chunks):
        raise RuntimeError("jev down")
    assert jr.log_context("some log text\n", score_chunks=raising) is None


def test_log_context_none_on_empty_window():
    assert jr.log_context("", score_chunks=lambda chunks: {0: 1.0}) is None


def test_log_context_none_when_no_chunk_gets_a_score():
    assert jr.log_context("some text\n", score_chunks=lambda chunks: {}) is None


def test_log_context_bounds_each_chunk_by_utf8_bytes_not_code_points():
    # Cold review F2: a chunk made of multibyte characters (each '€' is 3 UTF-8 bytes)
    # must be bounded by BYTES, not code points -- 500 code points of '€' would be 1500
    # bytes, blowing past the run-finished payload's 16 KiB ceiling once combined with
    # the other chunks and `tail`.
    window = "€" * (2048 * 2)
    result = jr.log_context(window, score_chunks=lambda chunks: {0: 2, 1: 1})
    assert result == ["€" * 166, "€" * 166]  # 166 * 3 = 498 bytes <= 500
    for chunk in result:
        assert len(chunk.encode("utf-8")) <= 500


def test_log_context_returns_none_without_calling_scorer_when_classifier_is_off(monkeypatch):
    # `log_context` swallows ANY exception from its `score_chunks` callable (a fake that
    # only raises would let this test pass without the gate), so the fake both records
    # its calls and would return a usable score if it ever ran.
    monkeypatch.setattr(general_settings, "read_settings",
                         lambda: {"ok": True, "data": {"integrations": {"classifier": False}}})
    calls = []
    def counting_scorer(chunks):
        calls.append(chunks)
        return {0: 1.0}
    assert jr.log_context("some log text\n" * 200, score_chunks=counting_scorer) is None
    assert calls == []


def test_log_context_returns_none_without_calling_scorer_when_settings_unreadable(monkeypatch):
    monkeypatch.setattr(general_settings, "read_settings", lambda: {"ok": False})
    calls = []
    def counting_scorer(chunks):
        calls.append(chunks)
        return {0: 1.0}
    assert jr.log_context("some log text\n" * 200, score_chunks=counting_scorer) is None
    assert calls == []


def test_log_context_calls_scorer_when_classifier_is_on(monkeypatch):
    monkeypatch.setattr(general_settings, "read_settings",
                         lambda: {"ok": True, "data": {"integrations": {"classifier": True}}})
    result = jr.log_context("some log text\n" * 200, score_chunks=lambda chunks: {0: 1.0})
    assert result is not None


# ---- MOA-495 cold review F2: _bound_bytes / cap_payload_bytes --------------------

def test_bound_bytes_cuts_at_a_valid_utf8_character_boundary():
    assert jr._bound_bytes("€" * 10, 5) == "€"  # 1 whole char (3B); the 2 leftover bytes of the 2nd are dropped
    assert jr._bound_bytes("ab€cd", 5) == "ab€"  # exactly completes the 3-byte char
    assert jr._bound_bytes("ab€cd", 4) == "ab"  # the char's first 2 bytes are incomplete, dropped whole
    assert jr._bound_bytes("hello", 10) == "hello"


def test_cap_payload_bytes_keeps_payload_untouched_when_within_limit():
    payload = {"tail": "hello", "log_context": ["a", "b"], "tally": "tally: 1 findings — 0 repeated, 1 new"}
    result = jr.cap_payload_bytes(dict(payload), limit=1000)
    assert result == payload


def test_cap_payload_bytes_drops_log_context_before_tally():
    payload = {"log_context": ["x" * 200] * 3, "tally": "small"}
    limit = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) - 10
    result = jr.cap_payload_bytes(payload, limit=limit)
    assert "log_context" not in result
    assert result["tally"] == "small"


def test_cap_payload_bytes_drops_tally_too_when_still_over_budget():
    payload = {"log_context": ["x" * 200] * 3, "tally": "y" * 200}
    result = jr.cap_payload_bytes(payload, limit=50)
    assert "log_context" not in result
    assert "tally" not in result


def test_cap_payload_bytes_multibyte_regression():
    # Cold review F2's own worked example (a tail plus six '€'-heavy chunks, 12,288 +
    # 9,000 bytes under the OLD code-point bound) now fits under budget once each chunk
    # is bounded by BYTES instead -- no drop needed.
    payload = {"tail": "€" * jr.CHILD_TAIL_CHARS, "log_context": ["€" * 166] * 6}
    result = jr.cap_payload_bytes(dict(payload), limit=jr.RUN_FINISHED_PAYLOAD_BYTES_MAX)
    assert result == payload
    size = len(json.dumps(result, ensure_ascii=False).encode("utf-8"))
    assert size <= jr.RUN_FINISHED_PAYLOAD_BYTES_MAX

    # An even more extreme multibyte tail (4-byte characters) still exercises the
    # guard: log_context is dropped as the payload's own supplementary detail, even
    # though the tail alone already accounts for the overflow here.
    extreme = {"tail": "😀" * jr.CHILD_TAIL_CHARS, "log_context": ["😀" * 125] * 6}
    result_extreme = jr.cap_payload_bytes(dict(extreme), limit=jr.RUN_FINISHED_PAYLOAD_BYTES_MAX)
    assert "log_context" not in result_extreme


# ---- MOA-495 2.2: extract_findings ----------------------------------------------

def test_extract_findings_splits_on_each_numbered_line():
    text = (
        "# Review\n"
        "F1. HIGH — cache race in the writer\n"
        "  fix: lock the write path\n"
        "F2 — MEDIUM — misleading empty-file UI state\n"
        "F3.  LOW — typo in comment\n"
    )
    findings = jr.extract_findings(text)
    assert set(findings) == {"F1", "F2", "F3"}
    assert findings["F1"] == "F1. HIGH — cache race in the writer\n  fix: lock the write path"
    assert findings["F2"] == "F2 — MEDIUM — misleading empty-file UI state"


def test_extract_findings_returns_empty_dict_when_no_numbered_line_exists():
    assert jr.extract_findings("approve, nothing to report\n") == {}


def test_child_log_classify_window_is_byte_bounded_not_char_bounded():
    # Each "€" is 3 UTF-8 bytes. A character-based slice (`text[-limit:]`) would keep
    # `limit` CHARACTERS here -- 3x more bytes than the window promises. The
    # byte-bounded window must land mid-character at this cut (120000 bytes total,
    # window of 65536 does not divide evenly by 3), exercising `errors="replace"`
    # instead of raising.
    text = "€" * 40000  # 40000 chars == 120000 UTF-8 bytes
    window = jr.child_log_classify_window(text, limit=65536)
    assert len(window) < len(text)
    assert len(window) <= 65536 // 3 + 1  # at most ~limit/3 chars fit in a 3-byte-per-char stream


# ---- MOA-474: derive_outcome fixtures -------------------------------------------
# Trimmed, hand-built envelopes matching the exact key shapes spec §2.1 describes for
# the xAI/OpenRouter/unknown incidents -- never copied from a real .local/runs log.

# Real OpenCode shape: the cause is NESTED under `error` (spec §2.1) -- {type, timestamp,
# sessionID, error: {name, data: {message, statusCode, ...}}}. A flat {type, name, data}
# is NOT what the runtime emits (post-build verification of run c86664823a0a caught the
# builder reading the flat shape and losing every real 403 message).
def _error_line(name, data):
    return json.dumps({"type": "error", "timestamp": 1, "sessionID": "ses_x",
                       "error": {"name": name, "data": data}})


_XAI_SPENDING_LIMIT_LINE = _error_line("APIError", {
    "statusCode": 403, "message": "xAI spending limit reached for this account",
    "isRetryable": False, "responseHeaders": {"x-secret": "never-read"},
    "responseBody": "never-read", "metadata": {},
})
_OPENROUTER_BUDGET_LINE = _error_line("APIError", {
    "statusCode": 403, "message": "API key budget limit exceeded (monthly limit). Contact your org admin.",
})
_UNSEEN_ERROR_LINE = _error_line("APIError", {"message": "Connection reset by server"})
# Verbatim last line of incident 1d9c379e5810's child.log (after redact()), trimmed of
# responseHeaders/responseBody -- the wiring check the synthetic lines cannot give.
_REAL_OPENROUTER_403_LINE = json.dumps({
    "type": "error", "timestamp": 1758066032946, "sessionID": "ses_real",
    "error": {"name": "APIError", "data": {
        "message": "API key budget limit exceeded (monthly limit). Contact your org admin.",
        "statusCode": 403, "isRetryable": False, "responseHeaders": {},
        "responseBody": "{\"error\":{\"message\":\"API key budget limit exceeded\"}}", "metadata": {},
    }},
}, separators=(",", ":"))
_CLEAN_STREAM_LAST_LINE = json.dumps({"type": "step_finish", "reason": "stop"})
_MALFORMED_LAST_LINE = "{not json at all"
_EMPTY_ERROR_LINE = _error_line("", {})
_FLAT_ERROR_LINE = json.dumps({"type": "error", "name": "APIError", "data": {"statusCode": 500, "message": "flat"}})


def _completed(argv, returncode, stdout):
    return CompletedProcess(argv, returncode, stdout, "")


def _builder_evidence(**overrides):
    base = dict(
        role="builder", report_result=None, report_summary=None, contract_status="missing",
        stream_last_line="", exit_code=0, signal=None, verify_frames=[],
        tests_written=True, head_sha="a" * 40, base_sha="b" * 40, is_descendant=True,
    )
    base.update(overrides)
    return base


def _reviewer_evidence(**overrides):
    base = dict(
        role="reviewer", report_result=None, contract_status="missing",
        stream_last_line="", exit_code=0, signal=None, verify_frames=[],
        tests_written=True, head_sha=None, base_sha=None, is_descendant=False,
    )
    base.update(overrides)
    return base


# ---- rows 9/4: terminal stream error (builder, unlabeled then labeled-success) ----

def test_derive_outcome_row9_xai_spending_limit_is_a_terminal_failure():
    out = jr.derive_outcome(_builder_evidence(stream_last_line=_XAI_SPENDING_LIMIT_LINE))
    assert out["result"] == "failure"
    assert out["stage"] == "runtime"
    assert out["diagnostic"] == "APIError 403 xAI spending limit reached for this account"


def test_derive_outcome_row9_openrouter_budget_is_a_terminal_failure():
    out = jr.derive_outcome(_builder_evidence(stream_last_line=_OPENROUTER_BUDGET_LINE))
    assert out["result"] == "failure"
    assert out["stage"] == "runtime"
    assert out["diagnostic"] == "APIError 403 API key budget limit exceeded (monthly limit). Contact your org admin."


def test_derive_outcome_row9_unseen_structured_error_with_no_statuscode():
    out = jr.derive_outcome(_builder_evidence(stream_last_line=_UNSEEN_ERROR_LINE))
    assert out["result"] == "failure"
    assert out["stage"] == "runtime"
    # no statusCode segment: name + message only, no double space.
    assert out["diagnostic"] == "APIError Connection reset by server"


def test_derive_outcome_row4_labeled_success_vetoed_by_terminal_stream_error():
    out = jr.derive_outcome(_builder_evidence(
        report_result="success", contract_status="ok",
        stream_last_line=_XAI_SPENDING_LIMIT_LINE,
    ))
    assert out["result"] == "failure"
    assert out["stage"] == "runtime"


def test_derive_outcome_row9_malformed_last_line_falls_to_exit_code_rows():
    # An error appears mid-stream is never modeled here (derive_outcome only ever sees
    # the LAST line, per the caller's contract) -- this fixture proves a malformed last
    # line is NOT terminal-error evidence and, with exit 0 and a clean tree, resolves
    # via rows 13/14 instead.
    out = jr.derive_outcome(_builder_evidence(stream_last_line=_MALFORMED_LAST_LINE))
    assert out["result"] == "success"
    assert out["stage"] == "report"


def test_derive_outcome_row9_truncated_stream_with_nonzero_exit_says_no_cause():
    out = jr.derive_outcome(_builder_evidence(
        stream_last_line=_MALFORMED_LAST_LINE, exit_code=1,
    ))
    assert out["result"] == "failure"
    assert out["stage"] == "runtime"
    assert out["diagnostic"] == "exit 1 — no cause emitted"


def test_derive_outcome_row9_empty_error_fields_say_runtime_error_no_cause():
    out = jr.derive_outcome(_builder_evidence(stream_last_line=_EMPTY_ERROR_LINE))
    assert out["result"] == "failure"
    assert out["diagnostic"] == "runtime error — no cause emitted"


# ---- row 3: recovered/retried mid-stream errors are never terminal evidence ----

def test_derive_outcome_row8_recovered_tool_error_mid_stream_never_fires_row9():
    # derive_outcome only ever receives the LAST line -- a caller that fed it the whole
    # stream would break this contract; this test documents the caller's obligation by
    # asserting the clean last-line fixture alone (representing "ends step_finish/stop
    # after a mid-stream tool_use error") resolves success, never failure.
    out = jr.derive_outcome(_builder_evidence(stream_last_line=_CLEAN_STREAM_LAST_LINE))
    assert out["result"] == "success"
    assert out["stage"] == "report"
    # Rows 13/14 produce this secondary note (spec §7 Part 3); the None diagnostic the
    # plan's draft asserted here belongs to Part 2 row 8, a different branch.
    assert out["diagnostic"] in ("report missing", "report invalid")


# ---- rows 5/10: exit/signal, no stream error ----

def test_derive_outcome_row10_nonzero_exit_with_clean_last_line_is_still_failure():
    out = jr.derive_outcome(_builder_evidence(
        stream_last_line=_CLEAN_STREAM_LAST_LINE, exit_code=1,
    ))
    # step_finish/stop is not an error event, but a nonzero exit is still evidence.
    assert out["result"] == "failure"
    assert out["stage"] == "runtime"
    assert out["diagnostic"] == "exit 1 — no cause emitted"


def test_derive_outcome_row10_nonzero_exit_uses_last_stream_text_when_present():
    # MOA-474 cold review F4: `last_stream_text`, once populated by the caller via
    # _last_stream_text_line, replaces the "no cause emitted" tail.
    out = jr.derive_outcome(_builder_evidence(
        stream_last_line=_CLEAN_STREAM_LAST_LINE, exit_code=1,
        last_stream_text="ran out of disk space writing the worktree",
    ))
    assert out["diagnostic"] == "exit 1 — ran out of disk space writing the worktree"


def test_last_stream_text_line_extracts_the_last_text_events_first_line():
    log = "\n".join([
        json.dumps({"type": "step_start"}),
        json.dumps({"type": "text", "part": {"text": "first thought\nmore"}}),
        json.dumps({"type": "text", "part": {"text": "final cause line\nignored rest"}}),
        "{not json}",
    ])
    assert jr._last_stream_text_line(log) == "final cause line"


def test_last_stream_text_line_returns_none_with_no_text_event():
    log = "\n".join([json.dumps({"type": "step_start"}), json.dumps({"type": "step_finish"})])
    assert jr._last_stream_text_line(log) is None


# ---- rows 6/11 vs 7/12: build checked before verify (table order, spec §7) ----

def test_derive_outcome_row11_verify_red_reports_verify_stage():
    out = jr.derive_outcome(_builder_evidence(
        stream_last_line=_CLEAN_STREAM_LAST_LINE,
        verify_frames=[("pytest -q", _completed("pytest -q", 1, "1 failed\n"))],
    ))
    assert out["result"] == "failure"
    assert out["stage"] == "verify"
    assert out["diagnostic"] == "pytest -q exit 1: 1 failed"


def test_derive_outcome_verify_diagnostic_redacts_secrets():
    # MOA-474 cold review F2: a verify/build frame's command and output can both carry
    # a secret (an inline env assignment, a leaked token in a stack trace) -- both must
    # come out through the same redact() _write_verify_tests_file already applies.
    command = "TOKEN=sk-abc1234567890abcd1234 pytest -q"
    frame = _completed(
        command, 1,
        "1 failed\n"
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123\n",
    )
    out = jr.derive_outcome(_builder_evidence(
        stream_last_line=_CLEAN_STREAM_LAST_LINE, verify_frames=[(command, frame)],
    ))
    assert out["result"] == "failure"
    assert "sk-abc1234567890abcd1234" not in out["diagnostic"]
    assert "Bearer eyJ" not in out["diagnostic"]
    assert "[REDACTED]" in out["diagnostic"]


def test_derive_outcome_row11_build_red_reports_build_stage_over_a_red_verify():
    # Table order (spec §7 Part 2/3 notes): build is checked BEFORE verify when both
    # are red -- frames[1] is the build frame (jaxflow.py:2105-2108's own emission
    # order is verify-then-build; the DECISION order is the reverse).
    out = jr.derive_outcome(_builder_evidence(
        stream_last_line=_CLEAN_STREAM_LAST_LINE,
        verify_frames=[
            ("pytest -q", _completed("pytest -q", 1, "1 failed\n")),
            ("pnpm build", _completed("pnpm build", 2, "type error\n")),
        ],
    ))
    assert out["result"] == "failure"
    assert out["stage"] == "build"
    assert out["diagnostic"] == "pnpm build exit 2: type error"


def test_derive_outcome_row7_tests_evidence_file_write_failure_reports_verify_stage():
    out = jr.derive_outcome(_builder_evidence(
        report_result="success", contract_status="ok",
        stream_last_line=_CLEAN_STREAM_LAST_LINE, tests_written=False,
    ))
    assert out["result"] == "failure"
    assert out["stage"] == "verify"


# ---- rows 13/14: MOA-472 fallback, gated behind 9-12 ----

def test_derive_outcome_row13_all_green_and_descendant_is_success():
    out = jr.derive_outcome(_builder_evidence(stream_last_line=_CLEAN_STREAM_LAST_LINE))
    assert out["result"] == "success"
    assert out["stage"] == "report"
    assert out["diagnostic"] in ("report missing", "report invalid")


def test_derive_outcome_row14_all_green_but_not_a_descendant_is_failure():
    out = jr.derive_outcome(_builder_evidence(
        stream_last_line=_CLEAN_STREAM_LAST_LINE, is_descendant=False,
    ))
    assert out["result"] == "failure"
    assert out["stage"] == "report"
    assert out["diagnostic"] == "no commit beyond base"


def test_derive_outcome_row5_crashed_runtime_beats_green_checks_and_a_commit():
    # Traceability item 5: a confirmed unrecovered failure never becomes success from
    # green checks/a commit existing.
    out = jr.derive_outcome(_builder_evidence(
        stream_last_line=_XAI_SPENDING_LIMIT_LINE, is_descendant=True,
    ))
    assert out["result"] == "failure"
    assert out["stage"] == "runtime"


# ---- Part 2 row 3: a claimed blocked/failure is never overridden ----

def test_derive_outcome_row3_claimed_blocked_stays_blocked_even_with_red_verify():
    # §5.1 conflict 1 -- the pinned regression.
    out = jr.derive_outcome(_builder_evidence(
        report_result="blocked", report_summary="blocked: waiting on API access",
        contract_status="ok",
        verify_frames=[("pytest -q", _completed("pytest -q", 1, "boom\n"))],
    ))
    assert out["result"] == "blocked"
    assert out["stage"] == "report"
    # MOA-474 cold review F8: row 3's diagnostic is the report's own summary, verbatim
    # -- never None (a None diagnostic renders as the literal string "None" on the
    # callback line, spec §12.1).
    assert out["diagnostic"] == "blocked: waiting on API access"


def test_derive_outcome_row3_claimed_failure_stays_failure_regardless_of_evidence():
    out = jr.derive_outcome(_builder_evidence(
        report_result="failure", report_summary="failure: three tests still red",
        contract_status="ok",
        stream_last_line=_CLEAN_STREAM_LAST_LINE,
    ))
    assert out["result"] == "failure"
    assert out["stage"] == "report"
    assert out["diagnostic"] == "failure: three tests still red"


def test_derive_outcome_row3_claimed_failure_with_no_report_summary_is_never_none():
    # F8's own edge case: a caller that (wrongly) omits report_summary must still never
    # surface a bare None -- the documented evidence shape always supplies it in
    # practice (Task 3 always passes the report's own `summary`), but this pins the
    # function's own contract independent of any one caller.
    out = jr.derive_outcome(_builder_evidence(
        report_result="failure", report_summary=None, contract_status="ok",
    ))
    assert out["diagnostic"] is not None


def test_derive_outcome_out_of_enum_report_label_normalizes_to_no_label():
    # Judgment call (spec review, MOA-474 plan split): a garbage result:/verdict:
    # label out-of-enum for RESULTS never reaches Part 2 -- it folds into Part 3
    # (no-label) exactly like report_result is None, same green-checks-plus-
    # descendant success this fixture would get if the label were absent entirely.
    out = jr.derive_outcome(_builder_evidence(
        report_result="bogus-label", contract_status="invalid",
        stream_last_line=_CLEAN_STREAM_LAST_LINE,
    ))
    assert out["result"] == "success"
    assert out["stage"] == "report"


def test_derive_outcome_row6_claimed_success_vetoed_by_red_build():
    out = jr.derive_outcome(_builder_evidence(
        report_result="success", contract_status="ok",
        stream_last_line=_CLEAN_STREAM_LAST_LINE,
        verify_frames=[
            ("pytest -q", _completed("pytest -q", 0, "ok\n")),
            ("pnpm build", _completed("pnpm build", 1, "boom\n")),
        ],
    ))
    assert out["result"] == "failure"
    assert out["stage"] == "build"


def test_derive_outcome_row8_claimed_success_with_all_green_stays_success_no_stage():
    out = jr.derive_outcome(_builder_evidence(
        report_result="success", contract_status="ok",
        stream_last_line=_CLEAN_STREAM_LAST_LINE,
        verify_frames=[("pytest -q", _completed("pytest -q", 0, "ok\n"))],
    ))
    assert out["result"] == "success"
    assert out["stage"] is None
    assert out["diagnostic"] is None


# ---- Reviewer table rows 3-6 ----

def test_derive_outcome_reviewer_row4_terminal_stream_error_no_verdict():
    out = jr.derive_outcome(_reviewer_evidence(stream_last_line=_XAI_SPENDING_LIMIT_LINE))
    assert "verdict" not in out or out["verdict"] is None
    assert out["stage"] == "runtime"


def test_derive_outcome_reviewer_row5_nonzero_exit_no_verdict():
    out = jr.derive_outcome(_reviewer_evidence(
        stream_last_line=_CLEAN_STREAM_LAST_LINE, exit_code=1,
    ))
    assert out.get("verdict") is None
    assert out["stage"] == "runtime"


def test_derive_outcome_reviewer_row6_clean_missing_report_says_report_missing():
    out = jr.derive_outcome(_reviewer_evidence(
        stream_last_line=_CLEAN_STREAM_LAST_LINE, contract_status="missing",
    ))
    assert out.get("verdict") is None
    assert out["stage"] == "report"
    assert out["diagnostic"] == "report missing"


def test_derive_outcome_reviewer_row6_clean_invalid_report_says_report_invalid():
    out = jr.derive_outcome(_reviewer_evidence(
        stream_last_line=_CLEAN_STREAM_LAST_LINE, contract_status="invalid",
    ))
    assert out["diagnostic"] == "report invalid"


def test_derive_outcome_reviewer_never_reads_a_parsed_but_invalid_verdict_label():
    # Test matrix: a reviewer report that DID parse a verdict: label but is still
    # contract_status invalid must never emit it -- rows 4-6 never look at
    # report_result at all for a reviewer.
    out = jr.derive_outcome(_reviewer_evidence(
        report_result="approve", contract_status="invalid",
        stream_last_line=_CLEAN_STREAM_LAST_LINE,
    ))
    assert out.get("verdict") is None


# ---- redaction: only three fields ever read; bounded; whitespace-collapsed ----

def test_derive_outcome_redaction_never_reads_headers_body_or_metadata():
    line = json.dumps({
        "type": "error", "name": "E", "data": {
            "statusCode": 403, "message": "plain cause",
            "responseHeaders": {"authorization": "Bearer sk-leaked-forever"},
            "responseBody": "sk-another-leaked-secret-value-1234567890",
            "metadata": {"apiKey": "sk-should-never-appear-1234567890"},
        },
    })
    out = jr.derive_outcome(_builder_evidence(stream_last_line=line))
    assert "leaked" not in out["diagnostic"]
    assert "sk-" not in out["diagnostic"] or "[REDACTED]" in out["diagnostic"]
    assert out["diagnostic"] == "E 403 plain cause"


def test_derive_outcome_diagnostic_is_hard_cut_to_300_with_no_ellipsis():
    long_message = "x" * 400
    line = json.dumps({"type": "error", "name": "E", "data": {"message": long_message}})
    out = jr.derive_outcome(_builder_evidence(stream_last_line=line))
    assert len(out["diagnostic"]) == 300
    assert "..." not in out["diagnostic"]


def test_derive_outcome_diagnostic_collapses_doubled_up_whitespace():
    line = json.dumps({
        "type": "error", "name": "E  ", "data": {"statusCode": 403, "message": "  msg  "},
    })
    out = jr.derive_outcome(_builder_evidence(stream_last_line=line))
    assert "  " not in out["diagnostic"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all tests passed")


def test_derive_outcome_reads_the_real_nested_opencode_error_envelope():
    out = jr.derive_outcome(_builder_evidence(stream_last_line=_REAL_OPENROUTER_403_LINE))
    assert out["result"] == "failure"
    assert out["stage"] == "runtime"
    assert out["diagnostic"] == "APIError 403 API key budget limit exceeded (monthly limit). Contact your org admin."
    assert "responseBody" not in out["diagnostic"] and "never-read" not in out["diagnostic"]


def test_runtime_stream_diagnostic_tolerates_the_flat_shape_too():
    assert jr._runtime_stream_diagnostic(json.loads(_FLAT_ERROR_LINE)) == "APIError 500 flat"


# ---- MOA-474 §16: cross-language boundary fixtures ------------------------------
# One {runStarted, runFinished, expected} JSON file per representative decision-table
# row, regenerated by this test and committed to scripts/fixtures/moa-474/ -- read
# back by src/server/db/moa-474-boundary.test.ts (this task, parseIngress + persisted
# payload only) and later by Part B's src/server/db/moa-474-boundary.test.ts additions
# (lastRunFor read-back, not built here). Shape is fixed by Part B's existing
# contract -- do not rename these three top-level keys.

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "scripts" / "fixtures" / "moa-474"


def _run_started_event(run_id, role, repo="/home/rafa/repos/demo", kind="build"):
    # F3 (cold review 938043c1d949): every field `parseIngress`'s `run-started` case
    # requires (workflow-events.ts:85-119) -- phase/runtime/kind/target/caller/
    # caller_session/model/effort/session/repo, plus `verify` when kind is `build`.
    # `runtime: "opencode-grok"` (not "opencode-builder") keeps the managed-builder
    # lineage fields (requested_profile/root_build_run_id) out of scope -- these
    # fixtures aren't proving lineage, only the run-finished boundary.
    payload = {
        "phase": "P1", "runtime": "opencode-grok", "kind": kind, "target": "feat/x",
        "caller": "claude", "caller_session": "01234567-89ab-4cde-8f01-23456789abcd",
        "model": "xai/grok-4.6", "effort": "n/a",
        "session": f"jax-demo-{kind}-{run_id}", "repo": repo,
    }
    if kind == "build":
        payload["verify"] = "true"
    elif kind == "diff":
        # F7 (round 4): `builder_run_id` is now required at ingress for every diff
        # run-started row -- `dispatch_diff_review` always sets it for real
        # (`scripts/jaxflow.py:2074`), so a fixture claiming to be current ingress
        # output must carry one too.
        payload["builder_run_id"] = "aaaabbbbcccc"
    return {
        "run_id": run_id, "project": "demo", "role": role, "type": "run-started",
        "source": "deterministic", "emitter": "wrapper", "payload": payload,
    }


def _run_finished_event(run_id, role, payload):
    return {
        "run_id": run_id, "project": "demo", "role": role, "type": "run-finished",
        "source": "deterministic", "emitter": "wrapper", "payload": payload,
    }


def _write_fixture(name, run_id, role, finished_payload, expected, kind="build"):
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fixture = {
        "runStarted": _run_started_event(run_id, role, kind=kind),
        "runFinished": _run_finished_event(run_id, role, finished_payload),
        "expected": expected,
    }
    (_FIXTURES_DIR / f"{name}.json").write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


def test_export_moa474_boundary_fixtures():
    import jaxflow as jf

    happy = jr.derive_outcome(_builder_evidence(stream_last_line=_CLEAN_STREAM_LAST_LINE))
    _write_fixture(
        "builder-ok-success", "aaaabbbbcccc", "builder",
        {
            "phase": "P1", "exit_code": 0, "contract_status": "ok", "report_path": "/r.md",
            "summary": "all tasks done", "head_sha": "a" * 40, "result": happy["result"],
        },
        {"outcome": happy["result"], "stage": None, "diagnostic": None,
         "contractStatus": "ok", "headSha": "a" * 40},
    )

    runtime = jr.derive_outcome(_builder_evidence(stream_last_line=_XAI_SPENDING_LIMIT_LINE))
    _write_fixture(
        "builder-runtime-failure", "bbbbccccdddd", "builder",
        {
            "phase": "P1", "exit_code": 0, "contract_status": "missing", "report_path": "/r.md",
            "summary": "Report missing", "head_sha": None, "result": runtime["result"],
            "stage": runtime["stage"], "diagnostic": runtime["diagnostic"],
        },
        {"outcome": runtime["result"], "stage": runtime["stage"], "diagnostic": runtime["diagnostic"],
         "contractStatus": "missing", "headSha": None},
    )

    verify_red = jr.derive_outcome(_builder_evidence(
        stream_last_line=_CLEAN_STREAM_LAST_LINE,
        verify_frames=[("pytest -q", _completed("pytest -q", 1, "1 failed\n"))],
    ))
    _write_fixture(
        "builder-verify-failure", "ccccddddeeee", "builder",
        {
            "phase": "P1", "exit_code": 0, "contract_status": "missing", "report_path": "/r.md",
            "summary": "Report missing", "head_sha": "a" * 40, "result": verify_red["result"],
            "stage": verify_red["stage"], "diagnostic": verify_red["diagnostic"],
        },
        {"outcome": verify_red["result"], "stage": verify_red["stage"], "diagnostic": verify_red["diagnostic"],
         "contractStatus": "missing", "headSha": "a" * 40},
    )

    interrupted = jf._interrupted_payload(role="builder", phase="P1", signal_name="SIGTERM")
    _write_fixture(
        "builder-interrupted-worker", "ddddeeeeffff", "builder", interrupted,
        {"outcome": interrupted.get("result"), "stage": interrupted.get("stage"),
         "diagnostic": interrupted.get("diagnostic"), "contractStatus": "interrupted",
         "headSha": interrupted.get("head_sha")},
    )

    reviewer_missing = jr.derive_outcome(_reviewer_evidence(
        stream_last_line=_CLEAN_STREAM_LAST_LINE, contract_status="missing",
    ))
    _write_fixture(
        "reviewer-no-verdict", "eeeeffff0000", "reviewer",
        {
            "phase": "P1", "exit_code": 0, "contract_status": "missing", "report_path": "/r.md",
            "summary": "Report missing", "stage": reviewer_missing["stage"],
            "diagnostic": reviewer_missing["diagnostic"],
        },
        {"outcome": None, "stage": reviewer_missing["stage"], "diagnostic": reviewer_missing["diagnostic"],
         "contractStatus": "missing", "headSha": None},
        kind="diff",  # RUN_KINDS: "diff", never "diff-review" (F3)
    )

    _write_fixture(
        "reviewer-ok-approve", "ffff00001111", "reviewer",
        {
            "phase": "P1", "exit_code": 0, "contract_status": "ok", "report_path": "/r.md",
            "summary": "looks fine", "verdict": "approve",
        },
        {"outcome": "approve", "stage": None, "diagnostic": None, "contractStatus": "ok", "headSha": None},
        kind="diff",  # RUN_KINDS: "diff", never "diff-review" (F3)
    )

    # Pins the [...v].length fix (A1 Task 1 Step 3.3): 200 non-BMP emoji = 200 code points
    # (producer-valid, _bound-sliced) but 400 UTF-16 units.
    emoji_summary = "\U0001F600" * 200
    _write_fixture(
        "builder-summary-200-non-bmp-emoji", "000011112222", "builder",
        {
            "phase": "P1", "exit_code": 0, "contract_status": "ok", "report_path": "/r.md",
            "summary": emoji_summary, "head_sha": "a" * 40, "result": "success",
        },
        {"outcome": "success", "stage": None, "diagnostic": None, "contractStatus": "ok", "headSha": "a" * 40},
    )

    cancelled_payload = {
        "phase": "P1", "exit_code": None, "contract_status": "cancelled", "report_path": None,
        "summary": "cancelled by claude at 2026-01-01T00:00:00Z", "head_sha": None,
    }
    _write_fixture(
        "cancelled-row-unchanged-shape", "111122223333", "builder", cancelled_payload,
        {"outcome": None, "stage": None, "diagnostic": None, "contractStatus": "cancelled", "headSha": None},
    )

    assert len(list(_FIXTURES_DIR.glob("*.json"))) == 8


def test_count_findings_numbered_lines_each_accepted_separator_form():
    text = (
        "---\nrun_id: abc\n---\n"
        "F1. HIGH — something bad\n"
        "F2 — MEDIUM — something wrong\n"
        "F3-LOW — a nit\n"
    )
    assert jr.count_findings(text, "some summary") == {"high": 1, "medium": 1, "low": 1}


def test_count_findings_falls_back_to_the_summary_line_when_no_numbered_lines_exist():
    text = "# Review\n\nEverything reads fine, no numbered findings here.\n"
    assert jr.count_findings(text, "reject — 2 HIGH, 1 MEDIUM") == {"high": 2, "medium": 1, "low": 0}


def test_count_findings_returns_none_when_neither_grammar_matches():
    text = "# Review\n\nNo numbered findings anywhere in this body.\n"
    assert jr.count_findings(text, "approve, nothing to report") is None


def test_count_findings_accepts_the_999_boundary_and_omits_over_1000():
    assert jr.count_findings("no numbered lines", "999 HIGH") == {"high": 999, "medium": 0, "low": 0}
    # a report claiming 1000+ findings of one severity is malformed for this purpose --
    # omitted entirely, never clamped to 999 (round-3 F1).
    assert jr.count_findings("no numbered lines", "1000 HIGH") is None


def test_count_findings_counts_one_match_per_line_case_sensitive():
    # lowercase "high" must not match (case-sensitive); a line matched once never
    # double-counts even if it contains the word twice.
    text = "F1. high — this must not count\nF2. HIGH HIGH — counts once\n"
    assert jr.count_findings(text, "irrelevant") == {"high": 1, "medium": 0, "low": 0}


def test_events_url_is_guarded_against_the_real_hub_under_pytest():
    """Incident 2026-09-19: a test giving `dispatch_diff_review` a real `run` let a
    real `tmux new-session` spawn a detached `--run-worker` child that posted a live
    event to Rafa's hub -- invisible to any in-process monkeypatch. conftest.py sets
    JAXFLOW_EVENTS_URL before any test module (hence this one) is imported; this
    proves it took effect and that a real POST to it fails fast, never the live port."""
    import urllib.request
    assert jr.EVENTS_URL != "http://127.0.0.1:3100/api/workflow/events"
    req = urllib.request.Request(jr.EVENTS_URL, data=b"{}", method="POST")
    try:
        urllib.request.urlopen(req, timeout=1)
    except OSError:
        return
    raise AssertionError("guarded EVENTS_URL unexpectedly reachable")

# ---- model defaults from the shared fixture (spec: One source of model defaults) ----

_MODEL_DEFAULTS = json.loads((Path(__file__).resolve().parents[1] / "workflow" / "fixtures" / "model-defaults-v1.json").read_text())


def test_model_by_runtime_derives_from_the_fixture_with_provider_prefixes():
    assert jr.MODEL_BY_RUNTIME == {
        "opencode-grok": "xai/" + _MODEL_DEFAULTS["builders"]["default"]["model"],
        "opencode-deepseek": "openrouter/" + _MODEL_DEFAULTS["builders"]["fallback"]["model"],
    }
    assert jr.VARIANT_BY_RUNTIME == {
        "opencode-grok": _MODEL_DEFAULTS["builders"]["default"]["effort"],
        "opencode-deepseek": _MODEL_DEFAULTS["builders"]["fallback"]["effort"],
    }
