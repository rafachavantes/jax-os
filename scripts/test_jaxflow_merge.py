"""jaxflow merge tests (P2 split of test_jaxflow.py)."""
import contextlib
import json
import os
import signal
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
import general_settings
import jaxflow_cli
import jaxflow_merge
import jaxflow_common
import jaxflow_run as jr
import jax_init as ji

from testkit import (  # noqa: F401  (autouse fixtures included on purpose)
    _DUAL_PR_AGENTS,
    _MergeArgs,
    _agents,
    _completed,
    _fixed_now,
    _fresh_db,
    _insert,
    _integration_on_by_default,
    _isolate_agent_settings,
    _isolate_callbacks,
    _ledger_post,
    _merge_env,
    _merge_runner,
    _never_post,
    _pr_open_args,
    _restore_signal_handlers,
    _run_real,
    _switch_aware_runner,
    _worktree_path,
)


_TEMPLATE_DUAL_BRANCH_BLOCK = """## Deploy policy

**Preset: `dual-branch`** — small/internal project, low ceremony.

- Base branch: `staging` (default). Delivery target: `staging`, merged and pushed per
  the merge contract.

## Something else
"""


def test_delivery_target_falls_back_to_default_branch_with_no_preset_block(tmp_path):
    repo = _agents(tmp_path, "# AGENTS.md\n\nNo deploy policy here.\n")
    calls = []

    def fake_run(argv, cwd=None):
        calls.append(argv)
        if argv[:2] == ["git", "symbolic-ref"]:
            return _completed(0, "refs/remotes/origin/trunk\n")
        return _completed(1, "")

    target, no_preset, preset = jaxflow_merge._resolve_delivery_target(repo, run=fake_run)
    assert (target, no_preset, preset) == ("trunk", True, None)
    assert calls, "the fallback must actually consult _default_branch"


def test_delivery_target_absent_agents_file_is_the_same_fallback(tmp_path):
    target, no_preset, preset = jaxflow_merge._resolve_delivery_target(
        tmp_path, run=lambda argv, cwd=None: _completed(0, "refs/remotes/origin/main\n")
    )
    assert (target, no_preset, preset) == ("main", True, None)


def test_delivery_target_reads_the_dual_branch_preset_block(tmp_path):
    repo = _agents(tmp_path, _TEMPLATE_DUAL_BRANCH_BLOCK)

    def fake_run(argv, cwd=None):
        raise AssertionError("a resolved preset must never call _default_branch")

    assert jaxflow_merge._resolve_delivery_target(repo, run=fake_run) == ("staging", False, "dual-branch")


@pytest.mark.parametrize("preset", ["single-branch", "single-branch-pr", "bubble-buildprint"])
def test_delivery_target_reads_the_other_presets(tmp_path, preset):
    repo = _agents(
        tmp_path,
        f"## Deploy policy\n\n**Preset: `{preset}`** — notes.\n\n"
        "- Base branch: `main`. Delivery target: `main`.\n",
    )
    target, no_preset, resolved = jaxflow_merge._resolve_delivery_target(
        repo, run=lambda argv, cwd=None: _completed(1, "")
    )
    assert (target, no_preset, resolved) == ("main", False, preset)


def test_delivery_target_reads_the_dual_branch_pr_preset_block(tmp_path):
    repo = _agents(
        tmp_path,
        "**Preset: `dual-branch-pr`** — production project.\n\n"
        "- Base branch: `staging` (default). Delivery target: feature-branch PR against "
        "`staging`; executor and gate per `merge-contract.md`.\n"
        "- Production target: `main`.\n",
    )
    assert jaxflow_merge._resolve_delivery_target(
        repo, run=lambda argv, cwd=None: _completed(1, ""),
    ) == ("staging", False, "dual-branch-pr")


@pytest.mark.parametrize("agents_text", [
    pytest.param(
        "**Preset: `dual-branch`** — a.\n\nDelivery target: `staging`.\n\n"
        "**Preset: `single-branch`** — b.\n\nDelivery target: `main`.\n",
        id="two-preset-lines"),
    pytest.param(
        "**Preset: `dual-branch`** — a.\n\n- Base branch: `staging`.\n",
        id="block-without-a-delivery-target"),
    # A `Delivery target:` that belongs to a LATER section must not be borrowed.
    pytest.param(
        "**Preset: `dual-branch`** — a.\n\n- Base branch: `staging`.\n\n"
        "## Another section\n\nDelivery target: `production`.\n",
        id="block-ends-at-the-next-heading"),
    pytest.param(
        "**Preset: `experimental`** — a.\n\nDelivery target: `main`.\n",
        id="unknown-preset-name"),
])
def test_delivery_target_refuses_with_preset_unknown(tmp_path, agents_text):
    repo = _agents(tmp_path, agents_text)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge._resolve_delivery_target(repo, run=lambda argv, cwd=None: _completed(1, ""))
    assert exc.value.code == "preset-unknown"


_FIVE_PRESETS = ["single-branch", "single-branch-pr", "dual-branch", "dual-branch-pr", "bubble-buildprint"]


@pytest.mark.parametrize("preset,target", [
    ("single-branch", "main"), ("single-branch-pr", "main"), ("dual-branch", "staging"),
    ("dual-branch-pr", "staging"), ("bubble-buildprint", "main"),
])
def test_delivery_target_resolves_each_of_the_five_presets(tmp_path, preset, target):
    repo = _agents(tmp_path, f"**Preset: `{preset}`** — a.\n\n- Delivery target: `{target}`.\n")
    assert jaxflow_merge._resolve_delivery_target(
        repo, run=lambda argv, cwd=None: _completed(1, ""),
    ) == (target, False, preset)


@pytest.mark.parametrize("old", ["strict", "simple", "greenfield"])  # old-name-ok
def test_delivery_target_refuses_an_old_preset_name_and_lists_the_five(tmp_path, old):
    repo = _agents(tmp_path, f"**Preset: `{old}`** — a.\n\nDelivery target: `main`.\n")
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge._resolve_delivery_target(repo, run=lambda argv, cwd=None: _completed(1, ""))
    assert exc.value.code == "preset-unknown"
    assert all(name in exc.value.hint for name in _FIVE_PRESETS)


def test_preset_sets_partition_the_five_names():
    local, pr = set(jaxflow_merge._LOCAL_PRESETS), set(jaxflow_merge._PR_PRESETS)
    assert local | pr == set(_FIVE_PRESETS) and not local & pr
    assert set(jaxflow_merge._RELEASE_PRESETS) == {"dual-branch-pr"}


# cold review e24e7fb33bc8 F1: the production target resolves for RELEASE presets only, so a stray
# `Production target:` line in a non-release block must still refuse.
_PRODUCTION_REFUSALS = [
    *(pytest.param(
        f"**Preset: `{preset}`** — a.\n\nDelivery target: `main`.\n- Production target: `main`.\n",
        id=f"{preset}-with-a-stray-production-line")
      for preset in ("single-branch", "single-branch-pr", "dual-branch", "bubble-buildprint")),
    *(pytest.param(
        f"**Preset: `{preset}`** — a.\n\nDelivery target: `main`.\n",
        id=f"{preset}-without-the-line")
      for preset in ("dual-branch", "single-branch", "single-branch-pr", "bubble-buildprint")),
    pytest.param(
        "**Preset: `dual-branch-pr`** — a.\n\nDelivery target: `staging`.\n",
        id="dual-branch-pr-missing-the-line"),
    pytest.param("# AGENTS.md\n\nNo deploy policy here.\n", id="no-preset-block-at-all"),
]


@pytest.mark.parametrize("agents_text", _PRODUCTION_REFUSALS)
def test_production_target_refuses(tmp_path, agents_text):
    repo = _agents(tmp_path, agents_text)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge._resolve_production_target(repo)
    assert exc.value.code == "production-target-unconfigured"


def test_production_target_reads_the_dual_branch_pr_block(tmp_path):
    repo = _agents(tmp_path, "**Preset: `dual-branch-pr`** — a.\n\nDelivery target: `staging`.\nProduction target: `main`.\n")
    assert jaxflow_merge._resolve_production_target(repo) == "main"


def test_required_target_for_branch_is_delivery_for_a_feature_head(tmp_path):
    repo = _agents(tmp_path, "**Preset: `dual-branch-pr`** — a.\n\nDelivery target: `staging`.\nProduction target: `main`.\n")
    assert jaxflow_merge._required_target_for_branch(
        repo, "feat/x", run=lambda argv, cwd=None: _completed(1, ""),
    ) == ("staging", "dual-branch-pr")


def test_required_target_for_branch_is_production_for_a_release_head(tmp_path):
    repo = _agents(tmp_path, "**Preset: `dual-branch-pr`** — a.\n\nDelivery target: `staging`.\nProduction target: `main`.\n")
    assert jaxflow_merge._required_target_for_branch(
        repo, "release/2026-09-27-staging-promotion", run=lambda argv, cwd=None: _completed(1, ""),
    ) == ("main", "dual-branch-pr")


def test_required_target_for_branch_on_a_local_preset_refuses_a_release_head(tmp_path):
    repo = _agents(tmp_path, "**Preset: `dual-branch`** — a.\n\nDelivery target: `staging`.\n")
    # A release/* branch on a local preset has no Production target configured at all —
    # this only matters once `merge`/`pr open` actually route a release/* head through here
    # (Task 6/7); this test pins the helper's own behavior in isolation.
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge._required_target_for_branch(
            repo, "release/x", run=lambda argv, cwd=None: _completed(1, ""),
        )
    assert exc.value.code == "production-target-unconfigured"


# ------------------------------------------------------------------ slice d: PR delivery (github helpers)

# MOA-502 Decision 2: every gh call site refuses `github-integration-disabled` before any
# git/GitHub mutation when integrations.github is off. The pre-gate probes (each command's
# own toplevel/branch reads) already ran — the assertion is that no gh/push call followed.

def test_pr_open_refuses_github_integration_disabled_before_any_gh_call(tmp_path, monkeypatch):
    import general_settings
    monkeypatch.setattr(general_settings, "read_settings",
                         lambda: {"ok": True, "data": {"integrations": {"github": False}}})
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_pr_open(_pr_open_args(), run=fake_run, env=_merge_env())
    assert exc.value.code == "github-integration-disabled"
    assert not any(c[:2] == ["gh", "pr"] or c[:2] == ["git", "push"] for c in calls), calls


def test_merge_pr_refuses_github_integration_disabled_before_any_gh_call(tmp_path, monkeypatch):
    import general_settings
    monkeypatch.setattr(general_settings, "read_settings",
                         lambda: {"ok": True, "data": {"integrations": {"github": False}}})
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(target="staging"), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "github-integration-disabled"
    assert not any(c[:2] == ["gh", "pr"] or c[:2] == ["git", "push"] for c in calls), calls


def test_release_refuses_github_integration_disabled_before_any_gh_call(tmp_path, monkeypatch):
    import general_settings
    monkeypatch.setattr(general_settings, "read_settings",
                         lambda: {"ok": True, "data": {"integrations": {"github": False}}})
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_release(SimpleNamespace(from_caller="claude"), run=fake_run, env=_merge_env())
    assert exc.value.code == "github-integration-disabled"
    assert not any(c[:2] == ["gh", "pr"] or c[:2] == ["git", "push"] for c in calls), calls


@pytest.mark.parametrize("preset,target,pr_refused,release_refused", [
    ("single-branch", "main", True, True),
    ("single-branch-pr", "main", False, True),
    ("dual-branch", "staging", True, True),
    ("dual-branch-pr", "staging", False, False),
    ("bubble-buildprint", "main", True, True),
])
def test_preset_matrix_pr_open_and_release_refusals(
        tmp_path, monkeypatch, preset, target, pr_refused, release_refused):
    # D2/D4: single-branch refuses both; single-branch-pr refuses only `release`;
    # dual-branch-pr refuses neither (its release path is exercised by the release tests below).
    monkeypatch.chdir(tmp_path)
    agents = f"**Preset: `{preset}`** — a.\n\nDelivery target: `{target}`.\n"
    if preset == "dual-branch-pr":
        agents += "Production target: `main`.\n"
    fake_run, calls = _merge_runner(tmp_path, agents=agents, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "check-ref-format",): _completed(0, ""),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_pr_open(
            _pr_open_args(target="not-the-target"), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == ("preset-not-pr" if pr_refused else "target-mismatch")
    if release_refused:
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_release(
                SimpleNamespace(from_caller="claude"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
        assert exc.value.code == "preset-no-release"
    assert not any(c[0] == "gh" or c[:2] == ["git", "push"] for c in calls)
    assert not any(c[:2] in (["git", "fetch"], ["git", "update-ref"]) for c in calls)


def test_github_repo_slug_parses_ssh_and_https_remotes(tmp_path):
    for url, expected in [
        ("git@github.com:acme/route-converter-se.git", "acme/route-converter-se"),
        ("https://github.com/acme/route-converter-se.git", "acme/route-converter-se"),
        ("https://github.com/acme/route-converter-se", "acme/route-converter-se"),
    ]:
        def run(argv, cwd=None, url=url):
            if argv[:3] == ["git", "remote", "get-url"]:
                return _completed(0, f"{url}\n")
            return _completed(1, "")
        assert jaxflow_merge._github_repo_slug(run, tmp_path) == expected


def test_github_repo_slug_refuses_when_origin_is_unreadable_or_not_github(tmp_path):
    for result in (_completed(1, ""), _completed(0, "https://gitlab.com/acme/x.git\n"),
                   # cold review 75e934eacdca F5: a lookalike host must not match on the
                   # "github.com" substring alone -- _GH_REMOTE_RE is anchored to the real host.
                   _completed(0, "https://evilgithub.com/acme/x.git\n")):
        run = lambda argv, cwd=None, result=result: result
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge._github_repo_slug(run, tmp_path)
        assert exc.value.code == "github-unreachable"


_GH_PR_VIEW_BODY = {
    "number": 7, "url": "https://github.com/acme/x/pull/7", "state": "OPEN",
    "headRefOid": "a" * 40, "headRefName": "feat/x", "baseRefName": "staging",
    "mergeable": "MERGEABLE", "mergeCommit": None,
}


def test_gh_pr_view_parses_json_and_passes_the_exact_argv():
    calls = []
    def run(argv, cwd=None):
        calls.append(argv)
        return _completed(0, json.dumps(_GH_PR_VIEW_BODY))
    view = jaxflow_merge._gh_pr_view(run, Path("/repo"), "acme/x", 7)
    assert view["state"] == "OPEN"
    assert calls == [["gh", "pr", "view", "7", "--repo", "acme/x", "--json",
                       "number,url,state,headRefOid,headRefName,baseRefName,mergeable,mergeCommit"]]


def test_gh_pr_view_refuses_github_unreachable_on_failure_or_bad_json(tmp_path):
    for result in (_completed(1, ""), _completed(0, "not json")):
        run = lambda argv, cwd=None, result=result: result
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge._gh_pr_view(run, tmp_path, "acme/x", 7)
        assert exc.value.code == "github-unreachable"


def test_resolve_pr_number_zero_one_and_ambiguous_matches(tmp_path):
    def run_for(matches):
        return lambda argv, cwd=None: _completed(0, json.dumps(matches))
    assert jaxflow_merge._resolve_pr_number(run_for([]), tmp_path, "acme/x", "feat/x", "staging") is None
    assert jaxflow_merge._resolve_pr_number(run_for([{"number": 9}]), tmp_path, "acme/x", "feat/x", "staging") == 9
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge._resolve_pr_number(run_for([{"number": 9}, {"number": 10}]), tmp_path, "acme/x", "feat/x", "staging")
    assert exc.value.code == "pr-ambiguous"


def test_resolve_pr_number_refuses_github_unreachable_on_failure_or_bad_json(tmp_path):
    for result in (_completed(1, ""), _completed(0, "not json")):
        run = lambda argv, cwd=None, result=result: result
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge._resolve_pr_number(run, tmp_path, "acme/x", "feat/x", "staging")
        assert exc.value.code == "github-unreachable"


def test_find_recorded_pr_returns_the_latest_matching_branch_row(tmp_path):
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, None, "demo", "lead", "pr-opened",
            {"branch": "feat/x", "sha": "a" * 40, "pr_number": 7}, ts="2026-09-27T10:00:00")
    _insert(con, None, "demo", "lead", "pr-opened",
            {"branch": "feat/x", "sha": "b" * 40, "pr_number": 7}, ts="2026-09-27T11:00:00")
    _insert(con, None, "demo", "lead", "pr-opened",
            {"branch": "feat/y", "sha": "c" * 40, "pr_number": 9}, ts="2026-09-27T12:00:00")
    con.close()
    found = jaxflow_merge._find_recorded_pr("demo", "feat/x", db_path=db)
    assert found == {"branch": "feat/x", "sha": "b" * 40, "pr_number": 7}


def test_find_recorded_pr_returns_none_with_no_matching_row(tmp_path):
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    assert jaxflow_merge._find_recorded_pr("demo", "feat/x", db_path=db) is None


def test_find_recorded_pr_returns_none_with_no_db_at_all(tmp_path):
    assert jaxflow_merge._find_recorded_pr("demo", "feat/x", db_path=tmp_path / "nope.db") is None


_SINGLE_PR_AGENTS = (
    "**Preset: `single-branch-pr`** — GitHub Flow.\n\n"
    "- Base branch: `main`. Delivery target: `main`.\n"
)


def test_pr_open_happy_path_pushes_creates_and_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    events = []
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("git", "push", "origin", f"{'a' * 40}:refs/heads/feat/x"): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, "[]"),
        ("gh", "pr", "create"): _completed(0, "https://github.com/acme/x/pull/7\n"),
    })
    with monkeypatch.context() as m:
        # `jr.DB_PATH` alone is enough: `_find_recorded_pr`'s own `db_path = db_path or
        # jr.DB_PATH` fallback already resolves to this test's db (no need to also patch
        # `_open_ro` -- a stray earlier attempt to do that as `jaxflow._open_ro = lambda path:
        # jaxflow._open_ro(db)` was self-referential and would recurse forever the moment it
        # actually ran).
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow_merge.cmd_pr_open(
            _pr_open_args(), run=fake_run, post=_ledger_post(db, events),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result == {"number": 7, "url": "https://github.com/acme/x/pull/7", "repo_slug": "acme/x"}
    assert events[0]["type"] == "pr-opened"
    assert events[0]["payload"] == {
        "repo": "acme/x", "branch": "feat/x", "sha": "a" * 40, "base": "staging",
        "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature",
    }
    assert ["git", "push", "origin", f"{'a' * 40}:refs/heads/feat/x"] in calls
    assert ["gh", "pr", "create", "--repo", "acme/x", "--head", "feat/x", "--base", "staging",
            "--title", "Ship it"] in calls


def test_pr_open_resumes_at_the_same_sha_with_no_mutation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    # project MUST be the real slug cmd_pr_open computes (slugify_project(repo.name), where
    # repo.name is tmp_path's own pytest-generated basename) — a hardcoded "demo" would never
    # match and _find_recorded_pr would silently see nothing (same convention as
    # scripts/test_jaxflow.py:13158's existing resume-ineligible fixture).
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened",
            {"repo": "acme/x", "branch": "feat/x", "sha": "a" * 40, "base": "staging",
             "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature"})
    con.close()
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({
            **_GH_PR_VIEW_BODY, "headRefOid": "a" * 40, "baseRefName": "staging",
            "url": "https://github.com/acme/x/pull/7",
        })),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow_merge.cmd_pr_open(
            _pr_open_args(), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["number"] == 7
    for banned in ("git push", "gh pr create"):
        assert not any(" ".join(c).startswith(banned) for c in calls), f"{banned} ran on a no-op reuse"


def test_pr_open_fast_forward_fix_pushes_and_refreshes_the_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened",
            {"repo": "acme/x", "branch": "feat/x", "sha": "a" * 40, "base": "staging",
             "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature"})
    con.close()
    new_sha = "b" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", f"feat/x^{{commit}}"): _completed(0, f"{new_sha}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({
            **_GH_PR_VIEW_BODY, "headRefOid": "a" * 40, "baseRefName": "staging",
            "url": "https://github.com/acme/x/pull/7",
        })),
        ("git", "merge-base", "--is-ancestor", "a" * 40, new_sha): _completed(0, ""),
        ("git", "push", "origin", f"{new_sha}:refs/heads/feat/x"): _completed(0, ""),
    })
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        jaxflow_merge.cmd_pr_open(
            _pr_open_args(sha=new_sha), run=fake_run, post=_ledger_post(db, events),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert ["git", "push", "origin", f"{new_sha}:refs/heads/feat/x"] in calls
    assert events[0]["payload"]["sha"] == new_sha


def test_pr_open_refuses_a_diverged_remote_head(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened",
            {"repo": "acme/x", "branch": "feat/x", "sha": "a" * 40, "base": "staging",
             "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature"})
    con.close()
    other_sha = "c" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", f"feat/x^{{commit}}"): _completed(0, f"{other_sha}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({
            **_GH_PR_VIEW_BODY, "headRefOid": "a" * 40, "baseRefName": "staging",
        })),
        ("git", "merge-base", "--is-ancestor", "a" * 40, other_sha): _completed(1, ""),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_pr_open(
                _pr_open_args(sha=other_sha), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-remote-diverged"
    assert not any(c[:2] == ["git", "push"] for c in calls)


def test_pr_open_reports_a_closed_unmerged_pr_never_reopening_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened",
            {"repo": "acme/x", "branch": "feat/x", "sha": "a" * 40, "base": "staging",
             "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature"})
    con.close()
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({**_GH_PR_VIEW_BODY, "state": "CLOSED"})),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_pr_open(
                _pr_open_args(), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-closed"
    assert not any(c[:3] == ["gh", "pr", "create"] for c in calls)


def test_pr_open_refuses_a_base_mismatch_on_a_recorded_pr_before_any_push_or_event(tmp_path, monkeypatch):
    # cold review 75e934eacdca F4: reusing a recorded PR whose ACTUAL GitHub base no longer
    # matches the approved --target must refuse before any push/event, even though --target
    # itself already matched the configured Delivery target (a separate, earlier check).
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened",
            {"repo": "acme/x", "branch": "feat/x", "sha": "a" * 40, "base": "staging",
             "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature"})
    con.close()
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({**_GH_PR_VIEW_BODY, "baseRefName": "wrong-base"})),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_pr_open(
                _pr_open_args(), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-identity-mismatch"
    assert not any(c[:2] == ["git", "push"] or c[:3] == ["gh", "pr", "create"] for c in calls)


def test_pr_open_refuses_a_head_branch_mismatch_on_a_recorded_pr_before_any_push_or_event(tmp_path, monkeypatch):
    # cold review 24597072c8ac F2: same guard as the base-mismatch test above, for the head
    # branch -- a recorded/resolved PR number whose actual GitHub head branch isn't `branch`
    # must refuse before any push/event too.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened",
            {"repo": "acme/x", "branch": "feat/x", "sha": "a" * 40, "base": "staging",
             "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature"})
    con.close()
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({**_GH_PR_VIEW_BODY, "headRefName": "feat/other"})),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_pr_open(
                _pr_open_args(), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-identity-mismatch"
    assert not any(c[:2] == ["git", "push"] or c[:3] == ["gh", "pr", "create"] for c in calls)


@pytest.mark.parametrize("agents,target,pushed_or_opened", [
    pytest.param(
        _DUAL_PR_AGENTS, "main",
        lambda c: c[:2] == ["git", "push"] or c[:3] == ["gh", "pr", "create"],
        id="feature-head-against-the-wrong-base"),
    pytest.param(
        _SINGLE_PR_AGENTS, "staging",
        lambda c: c[0] == "gh" or c[:2] == ["git", "push"],
        id="single-branch-pr-refuses-a-staging-target"),
])
def test_pr_open_refuses_target_mismatch(tmp_path, monkeypatch, agents, target, pushed_or_opened):
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(tmp_path, agents=agents, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "check-ref-format",): _completed(0, ""),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_pr_open(
            _pr_open_args(target=target), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "target-mismatch"
    assert not any(pushed_or_opened(c) for c in calls)


def test_pr_open_argparse_wiring_end_to_end(monkeypatch, tmp_path):
    seen = {}
    def fake_cmd_pr_open(args, **kwargs):
        seen["branch"] = args.branch
        seen["title"] = args.title
        return {"number": 1, "url": "https://github.com/acme/x/pull/1", "repo_slug": "acme/x"}
    monkeypatch.setattr(jaxflow_merge, "cmd_pr_open", fake_cmd_pr_open)
    argv = ["pr", "open", "feat/x", "--sha", "a" * 40, "--target", "staging", "--title", "Ship it"]
    assert jaxflow_cli.main(argv) == jaxflow_common.OK
    assert seen == {"branch": "feat/x", "title": "Ship it"}


def test_pr_open_rejects_a_blank_title(tmp_path):
    with pytest.raises(SystemExit):
        jaxflow_cli.parse_args(["pr", "open", "feat/x", "--sha", "a" * 40, "--target", "staging", "--title", "   "])


def test_pr_open_refuses_a_local_preset_before_any_push_or_gh_call(tmp_path, monkeypatch):
    # cold review e441aa1770e3 F1: pr-open is PR-preset-only -- a single-branch/dual-branch/bubble-buildprint/no-preset repo must refuse before any push or gh call, not silently deliver.
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(
        tmp_path, agents="**Preset: `dual-branch`** — a.\n\nDelivery target: `staging`.\n", script={
            ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
            ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
            ("git", "check-ref-format",): _completed(0, ""),
        })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_pr_open(
            _pr_open_args(), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "preset-not-pr"
    assert not any(c[0] == "gh" for c in calls)
    assert not any(c[:2] == ["git", "push"] for c in calls)


def test_pr_open_refuses_a_no_preset_repo_before_any_push_or_gh_call(tmp_path, monkeypatch):
    # Same refusal for a repo with no Deploy policy block at all -- the common case (most repos
    # on this machine predate the preset system) must never be treated as implicitly a PR preset.
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(
        tmp_path, agents="# AGENTS.md\n\nNo deploy policy here.\n", script={
            ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
            ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
            ("git", "check-ref-format",): _completed(0, ""),
        })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_pr_open(
            _pr_open_args(), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "preset-not-pr"
    assert not any(c[0] == "gh" for c in calls)
    assert not any(c[:2] == ["git", "push"] for c in calls)


@pytest.mark.parametrize("agents", [
    "**Preset: `single-branch`** — a.\n\nDelivery target: `main`.\n",
    "**Preset: `dual-branch`** — a.\n\nDelivery target: `staging`.\n",
    "**Preset: `bubble-buildprint`** — a.\n\nDelivery target: `main`.\n",
    "# AGENTS.md\n\nNo deploy policy here.\n",
])
def test_pr_open_refuses_preset_not_pr_for_a_release_head_before_target_resolution(
        tmp_path, monkeypatch, agents):
    # cold review 9f7f7290c510 F2: the preset check precedes `_required_target_for_branch`, which
    # would otherwise raise production-target-unconfigured for a release/* head first.
    monkeypatch.chdir(tmp_path)
    head = "release/2026-10-07-staging-promotion"
    fake_run, calls = _merge_runner(tmp_path, agents=agents, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", f"{head}^{{commit}}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "check-ref-format",): _completed(0, ""),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_pr_open(
            _pr_open_args(branch=head, target="main"), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "preset-not-pr"
    assert not any(c[0] == "gh" or c[:2] == ["git", "push"] for c in calls)


def test_pr_open_single_branch_pr_targets_main_and_records_the_pr(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    events = []
    fake_run, calls = _merge_runner(tmp_path, agents=_SINGLE_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("git", "push", "origin", f"{'a' * 40}:refs/heads/feat/x"): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, "[]"),
        ("gh", "pr", "create"): _completed(0, "https://github.com/acme/x/pull/7\n"),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow_merge.cmd_pr_open(
            _pr_open_args(target="main"), run=fake_run, post=_ledger_post(db, events),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["number"] == 7
    assert events[0]["type"] == "pr-opened" and events[0]["payload"]["base"] == "main"
    assert ["gh", "pr", "create", "--repo", "acme/x", "--head", "feat/x", "--base", "main",
            "--title", "Ship it"] in calls


def test_pr_open_single_branch_pr_refuses_a_release_head_even_with_a_production_target(
        tmp_path, monkeypatch):
    # cold review e24e7fb33bc8 F1. Picked REFUSE over "treat it as a feature targeting main":
    # spec D3 says a release/* head on single-branch-pr refuses production-target-unconfigured, and
    # the preset has no release step, so the branch name must never be read as a feature head.
    monkeypatch.chdir(tmp_path)
    head = "release/2026-10-07-staging-promotion"
    fake_run, calls = _merge_runner(
        tmp_path, agents=_SINGLE_PR_AGENTS + "- Production target: `main`.\n", script={
            ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
            ("git", "rev-parse", "--verify", f"{head}^{{commit}}"): _completed(0, f"{'a' * 40}\n"),
            ("git", "check-ref-format",): _completed(0, ""),
        })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_pr_open(
            _pr_open_args(branch=head, target="main"), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "production-target-unconfigured"
    assert not any(c[0] == "gh" or c[:2] == ["git", "push"] for c in calls)


_GH_PR_VIEW_MERGED = {
    "number": 7, "url": "https://github.com/acme/x/pull/7", "state": "MERGED",
    "headRefOid": "a" * 40, "headRefName": "feat/x", "baseRefName": "staging",
    "mergeable": "MERGEABLE", "mergeCommit": {"oid": "d" * 40},
}


def _pr_merge_setup(tmp_path, db, *, pr_state="OPEN", mergeable="MERGEABLE", target="staging",
                         branch="feat/x", sha="a" * 40, worktree_registered=True,
                         merge_result=None):
    # project MUST be the real slug cmd_merge computes (slugify_project(repo.name)) — reuses
    # the existing `_worktree_path` helper (scripts/test_jaxflow.py:12971-12973) for the exact
    # same reason every OTHER merge test does: a literal like "demo" would never match.
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened",
            {"repo": "acme/x", "branch": branch, "sha": sha, "base": target,
             "pr_number": 7, "pr_url": "https://github.com/acme/x/pull/7", "kind": "feature"})
    con.close()
    worktree = _worktree_path(tmp_path, branch)
    if worktree_registered:
        # A feature branch's PR-path merge runs --checks IN this worktree (decision 7), and
        # `_cmd_merge_pr` gates on `checks_dir.is_dir()` being a REAL directory before it
        # ever calls `_is_registered_worktree` -- unlike the fully-mocked release path (which
        # never reaches this branch's `worktree`), that check is not fakeable through `run`.
        # Every existing local-preset merge test that needs this same gate creates the directory
        # too (e.g. `_worktree_path(tmp_path).mkdir(parents=True)` at :13066/:13154/:13178/...).
        worktree.mkdir(parents=True, exist_ok=True)
    view = {"number": 7, "url": "https://github.com/acme/x/pull/7", "state": pr_state,
            "headRefOid": sha, "headRefName": branch, "baseRefName": target,
            "mergeable": mergeable, "mergeCommit": None}
    script = {
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", f"{branch}^{{commit}}"): _completed(0, f"{sha}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps(view)),
        ("git", "rev-parse", "HEAD"): _completed(0, f"{sha}\n"),
        ("git", "status", "--porcelain", "--untracked-files=no"): _completed(0, ""),
        ("/bin/bash", "-lc"): _completed(0, ""),
        ("gh", "pr", "merge",): merge_result or _completed(0, ""),
        ("git", "worktree", "list", "--porcelain"): _completed(
            0, f"worktree {worktree}\nbranch refs/heads/{branch}\n" if worktree_registered else ""),
        ("git", "fetch", "origin", target): _completed(0, ""),
        ("git", "switch", target): _completed(0, ""),
        ("git", "merge", "--ff-only"): _completed(0, ""),
        ("git", "worktree", "remove", str(worktree)): _completed(0, ""),
        ("git", "branch", "-d", branch): _completed(0, ""),
    }
    return script, worktree


def test_merge_pr_happy_path_verifies_checks_merges_and_syncs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, worktree = _pr_merge_setup(tmp_path, db)
    view_calls = {"n": 0}
    def gh_view_then_merged(argv, cwd=None):
        view_calls["n"] += 1
        if view_calls["n"] == 1:
            return _completed(0, json.dumps({
                "number": 7, "url": "https://github.com/acme/x/pull/7", "state": "OPEN",
                "headRefOid": "a" * 40, "headRefName": "feat/x", "baseRefName": "staging",
                "mergeable": "MERGEABLE", "mergeCommit": None,
            }))
        return _completed(0, json.dumps(_GH_PR_VIEW_MERGED))
    # `run()` intercepts every "gh pr view" call itself (stateful: OPEN first, MERGED on the
    # post-merge re-read) — `_pr_merge_setup`'s own static "gh pr view" script entry is
    # never reached for THIS test, only the entries after it (checks, merge, cleanup).
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    def run(argv, cwd=None):
        if tuple(argv[:3]) == ("gh", "pr", "view"):
            calls.append(list(argv))
            return gh_view_then_merged(argv, cwd)
        return fake_run(argv, cwd)
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow_merge.cmd_merge(
            _MergeArgs(target="staging"), run=run, post=_ledger_post(db, events),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result == jaxflow_common.OK
    merge_call = next(c for c in calls if c[:3] == ["gh", "pr", "merge"])
    assert merge_call == ["gh", "pr", "merge", "7", "--repo", "acme/x", "--merge",
                           "--match-head-commit", "a" * 40, "--subject", "feat: Phase X (merge feat/x)"]
    assert events[-1]["type"] == "merge-approved"
    assert events[-1]["payload"]["merge_sha"] == "d" * 40
    assert events[-1]["payload"]["pr_number"] == 7


def test_merge_pr_github_merge_refused_leaves_nothing_recorded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(
        tmp_path, db, merge_result=_completed(1, "", "required status check \"ci\" is expected"))
    script[("gh", "pr", "checks",)] = _completed(0, "ci\tpending\thttps://x\n")
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_ledger_post(db, events),
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "github-merge-refused"
    assert "required status check" in exc.value.hint
    assert events == []


def test_merge_pr_merge_queue_required(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(
        tmp_path, db, merge_result=_completed(1, "", "Pull request is in a merge queue"))
    script[("gh", "pr", "checks",)] = _completed(0, "")
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "merge-queue-required"


def test_merge_pr_reports_a_success_that_did_not_actually_merge(tmp_path, monkeypatch):
    # cold review 24597072c8ac F1 (downgraded to LOW -- no project uses a merge queue --
    # relabelled): `gh pr merge` can exit 0 without the PR actually merging (a merge queue or
    # auto-merge accepted the request asynchronously). The post-merge re-read still showing
    # OPEN must refuse the specific `merge-not-completed`, not the generic `github-unreachable`,
    # and nothing may be recorded as merged.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    still_open = {"number": 7, "url": "https://github.com/acme/x/pull/7", "state": "OPEN",
                  "headRefOid": "a" * 40, "headRefName": "feat/x", "baseRefName": "staging",
                  "mergeable": "MERGEABLE", "mergeCommit": None}
    view_calls = {"n": 0}
    def gh_view(argv, cwd=None):
        view_calls["n"] += 1
        return _completed(0, json.dumps(still_open))
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    def run(argv, cwd=None):
        if tuple(argv[:3]) == ("gh", "pr", "view"):
            calls.append(list(argv))
            return gh_view(argv, cwd)
        return fake_run(argv, cwd)
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_merge(
                _MergeArgs(target="staging"), run=run, post=_ledger_post(db, events),
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "merge-not-completed"
    assert events == []


def test_merge_pr_identity_mismatch_on_base(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    view = {"number": 7, "url": "https://github.com/acme/x/pull/7", "state": "OPEN",
            "headRefOid": "a" * 40, "headRefName": "feat/x", "baseRefName": "wrong-base",
            "mergeable": "MERGEABLE", "mergeCommit": None}
    script[("gh", "pr", "view")] = _completed(0, json.dumps(view))
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-identity-mismatch"
    _assert_no_mutation(calls)


def test_merge_pr_refuses_a_head_branch_mismatch(tmp_path, monkeypatch):
    # cold review 24597072c8ac F2: the resolved PR's actual head branch must be verified
    # against args.branch, not just its head sha -- a same-sha PR whose recorded/resolved
    # number actually points at a different branch must refuse before any GitHub mutation.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    view = {"number": 7, "url": "https://github.com/acme/x/pull/7", "state": "OPEN",
            "headRefOid": "a" * 40, "headRefName": "feat/other", "baseRefName": "staging",
            "mergeable": "MERGEABLE", "mergeCommit": None}
    script[("gh", "pr", "view")] = _completed(0, json.dumps(view))
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-identity-mismatch"
    _assert_no_mutation(calls)
    assert not any(c[:3] == ["gh", "pr", "merge"] for c in calls)


def test_merge_pr_head_moved_since_approval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    view = {"number": 7, "url": "https://github.com/acme/x/pull/7", "state": "OPEN",
            "headRefOid": "e" * 40, "headRefName": "feat/x", "baseRefName": "staging",
            "mergeable": "MERGEABLE", "mergeCommit": None}
    script[("gh", "pr", "view")] = _completed(0, json.dumps(view))
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-head-moved"


def test_merge_pr_mergeability_unknown_after_bounded_retries_mutates_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db, mergeable="UNKNOWN")
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "mergeability-unknown"
    assert not any(c[:3] == ["gh", "pr", "merge"] for c in calls)


def test_merge_pr_rerun_after_a_real_github_merge_is_recording_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, worktree = _pr_merge_setup(tmp_path, db)
    script[("gh", "pr", "view")] = _completed(0, json.dumps(_GH_PR_VIEW_MERGED))
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow_merge.cmd_merge(
            _MergeArgs(target="staging"), run=fake_run, post=_ledger_post(db, events),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result == jaxflow_common.OK
    assert not any(c[:3] == ["gh", "pr", "merge"] for c in calls)
    assert not any(c[:2] == ["/bin/bash", "-lc"] for c in calls)
    assert events[-1]["payload"]["merge_sha"] == "d" * 40


def test_merge_pr_already_merged_head_mismatch_refuses_pr_head_moved(tmp_path, monkeypatch):
    # cold review e441aa1770e3 F2: the already-merged recovery path must verify GitHub's
    # headRefOid against the approved sha before recording -- a PR merged with extra commits
    # (pushed and merged outside the approval flow) must never be recorded as an approval of a
    # DIFFERENT sha. Base is covered separately by
    # test_merge_pr_identity_mismatch_on_base (that check runs unconditionally, before
    # state is even inspected).
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    script[("gh", "pr", "view")] = _completed(0, json.dumps({**_GH_PR_VIEW_MERGED, "headRefOid": "e" * 40}))
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-head-moved"
    _assert_no_mutation(calls)


@pytest.mark.parametrize("branch,target,expected", [
    ("release/2026-09-27-staging-promotion", "main", True),
    ("release/2026-09-27-staging-promotion", "staging", False),
    ("feat/x", "main", False),
])
def test_merge_pr_target_mutual_exclusion_by_head_shape(tmp_path, monkeypatch, branch, target, expected):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    sha = "a" * 40
    if expected:
        script, checks_dir = _pr_merge_setup(
            tmp_path, db, branch=branch, sha=sha, target=target, worktree_registered=False)
        script[("git", "worktree", "add", "--detach")] = _completed(0, "")
        # cold review 75e934eacdca F2: cleanup no longer passes --force (plain removal).
        script[("git", "worktree", "remove", str(checks_dir))] = _completed(0, "")
    else:
        script, _ = _pr_merge_setup(tmp_path, db, branch=branch, sha=sha, target="staging")
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    # Plan deviation: a successful release merge needs the SAME stateful post-merge re-read
    # `test_merge_pr_happy_path...` already overrides for the feature case -- a static
    # OPEN view would make the implementation's own post-merge verification refuse
    # github-unreachable (the plan's fixture does not wire this; see report).
    if expected:
        view_calls = {"n": 0}
        def run(argv, cwd=None):
            if tuple(argv[:3]) == ("gh", "pr", "view"):
                calls.append(list(argv))
                view_calls["n"] += 1
                if view_calls["n"] == 1:
                    return _completed(0, json.dumps({
                        "number": 7, "url": "https://github.com/acme/x/pull/7", "state": "OPEN",
                        "headRefOid": sha, "headRefName": branch, "baseRefName": target,
                        "mergeable": "MERGEABLE", "mergeCommit": None,
                    }))
                return _completed(0, json.dumps({
                    **_GH_PR_VIEW_MERGED, "baseRefName": target, "headRefOid": sha,
                    "headRefName": branch,
                }))
            return fake_run(argv, cwd)
    else:
        run = fake_run
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        if expected:
            result = jaxflow_merge.cmd_merge(
                _MergeArgs(branch=branch, sha=sha, target=target), run=run,
                post=_ledger_post(db, events), env=_merge_env(), now=_fixed_now,
                allowlist_root=tmp_path.parent,
            )
            assert result == jaxflow_common.OK
        else:
            with pytest.raises(ji.Refusal) as exc:
                jaxflow_merge.cmd_merge(
                    _MergeArgs(branch=branch, sha=sha, target=target), run=run,
                    post=_never_post, env=_merge_env(), now=_fixed_now,
                    allowlist_root=tmp_path.parent,
                )
            assert exc.value.code == "target-mismatch"


def test_merge_single_branch_pr_feature_head_takes_the_pr_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    sha = "a" * 40
    script, _ = _pr_merge_setup(tmp_path, db, branch="feat/x", sha=sha, target="main")
    fake_run, calls = _merge_runner(tmp_path, agents=_SINGLE_PR_AGENTS, script=script)
    # Same post-merge re-read the dual-branch-pr happy path overrides: a static OPEN view
    # makes the implementation refuse merge-not-completed. The plan fixture does not wire this.
    view_calls = {"n": 0}
    def run(argv, cwd=None):
        if tuple(argv[:3]) == ("gh", "pr", "view"):
            calls.append(list(argv))
            view_calls["n"] += 1
            body = {
                "number": 7, "url": "https://github.com/acme/x/pull/7",
                "headRefOid": sha, "headRefName": "feat/x", "baseRefName": "main",
                "mergeable": "MERGEABLE",
            }
            if view_calls["n"] == 1:
                return _completed(0, json.dumps({**body, "state": "OPEN", "mergeCommit": None}))
            return _completed(0, json.dumps({**body, "state": "MERGED", "mergeCommit": {"oid": "d" * 40}}))
        return fake_run(argv, cwd)
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow_merge.cmd_merge(
            _MergeArgs(branch="feat/x", sha=sha, target="main"), run=run,
            post=_ledger_post(db, events), env=_merge_env(), now=_fixed_now,
            allowlist_root=tmp_path.parent,
        )
    assert result == jaxflow_common.OK
    assert any(c[:3] == ["gh", "pr", "merge"] for c in calls)
    # Local delivery is `git merge --no-ff`. The PR path's best-effort sync (decision 10)
    # does call `git merge --ff-only origin/<target>`, so that prefix is not the local path.
    assert not any(c[:4] == ["git", "merge", "--no-ff", "--no-commit"] for c in calls)


def test_merge_single_branch_pr_refuses_a_release_head(tmp_path, monkeypatch):
    # spec D3: single-branch-pr has no release path, even with a Production target line.
    monkeypatch.chdir(tmp_path)
    agents = _SINGLE_PR_AGENTS + "Production target: `main`.\n"
    fake_run, calls = _merge_runner(tmp_path, agents=agents, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(
            _MergeArgs(branch="release/2026-10-07", sha="a" * 40, target="main"),
            run=fake_run, post=_never_post, env=_merge_env(), now=_fixed_now,
            allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "production-target-unconfigured"
    _assert_no_mutation(calls)


def test_merge_pr_no_recorded_pr_and_zero_gh_list_matches_refuses_pr_not_found(tmp_path, monkeypatch):
    # Ledger & card reconciliation: merge ALSO falls back to `gh pr list` when the hub has no
    # record (same lookup `pr open` uses) — pr-not-found only fires once THAT also finds
    # nothing, never before.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()  # no pr-opened row recorded at all
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        # cold review 75e934eacdca F1: without a real GitHub origin, _github_repo_slug refuses
        # github-unreachable before ever reaching gh pr list -- this fixture must reach the
        # intended pr-not-found refusal instead.
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "list"): _completed(0, "[]"),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-not-found"
    _assert_no_mutation(calls)


def test_merge_pr_no_recorded_pr_and_ambiguous_gh_list_refuses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()  # no pr-opened row recorded at all
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{'a' * 40}\n"),
        # cold review 75e934eacdca F1: same reasoning as the pr-not-found fixture above -- reach
        # gh pr list (and its ambiguity) instead of refusing github-unreachable first.
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "list"): _completed(0, json.dumps([{"number": 7}, {"number": 8}])),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_merge(
                _MergeArgs(target="staging"), run=fake_run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "pr-ambiguous"
    _assert_no_mutation(calls)


def test_merge_pr_checks_dirtied_tree_refuses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    dirty_calls = {"n": 0}
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    def run(argv, cwd=None):
        if tuple(argv[:3]) == ("git", "status", "--porcelain"):
            dirty_calls["n"] += 1
            return _completed(0, "" if dirty_calls["n"] == 1 else " M dirty.txt\n")
        return fake_run(argv, cwd)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_merge(
                _MergeArgs(target="staging"), run=run, post=_never_post,
                env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
            )
    assert exc.value.code == "checks-dirtied-tree"


def test_release_happy_path_fetches_snapshots_and_opens_the_pr(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    snapshot = "e" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("git", "fetch", "origin", "staging"): _completed(0, ""),
        ("git", "rev-parse", "--verify", "origin/staging^{commit}"): _completed(0, f"{snapshot}\n"),
        ("git", "rev-parse", "--verify", "refs/remotes/origin/release/2026-01-01-staging-promotion"): _completed(1, ""),
        ("git", "update-ref",): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, "[]"),
        ("gh", "pr", "create"): _completed(0, "https://github.com/acme/x/pull/11\n"),
    })
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow_merge.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_ledger_post(db, events),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["branch"] == "release/2026-01-01-staging-promotion"
    assert result["snapshot_sha"] == snapshot
    assert result["number"] == 11
    assert events[0]["payload"] == {
        "repo": "acme/x", "branch": "release/2026-01-01-staging-promotion", "sha": snapshot,
        "base": "main", "pr_number": 11, "pr_url": "https://github.com/acme/x/pull/11",
        "kind": "release", "snapshot_sha": snapshot,
    }
    assert ["git", "update-ref", "refs/heads/release/2026-01-01-staging-promotion", snapshot] in calls


def test_release_name_collision_same_day_gets_a_numeric_suffix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    snapshot = "e" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("git", "fetch", "origin", "staging"): _completed(0, ""),
        ("git", "rev-parse", "--verify", "origin/staging^{commit}"): _completed(0, f"{snapshot}\n"),
        ("git", "rev-parse", "--verify", "refs/remotes/origin/release/2026-01-01-staging-promotion"): _completed(0, f"{'f' * 40}\n"),
        ("git", "rev-parse", "--verify", "refs/remotes/origin/release/2026-01-01-staging-promotion-2"): _completed(1, ""),
        ("git", "update-ref",): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, "[]"),
        ("gh", "pr", "create"): _completed(0, "https://github.com/acme/x/pull/12\n"),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow_merge.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_ledger_post(db, []),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["branch"] == "release/2026-01-01-staging-promotion-2"


def test_release_name_collision_local_ref_gets_a_numeric_suffix(tmp_path, monkeypatch):
    # F2: the collision an ORIGIN-only check misses -- an interrupted run's local
    # `refs/heads/<branch>` with no remote counterpart yet must still trigger the suffix.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    snapshot = "e" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("git", "fetch", "origin", "staging"): _completed(0, ""),
        ("git", "rev-parse", "--verify", "origin/staging^{commit}"): _completed(0, f"{snapshot}\n"),
        ("git", "rev-parse", "--verify", "refs/heads/release/2026-01-01-staging-promotion"): _completed(0, f"{'f' * 40}\n"),
        ("git", "rev-parse", "--verify", "refs/remotes/origin/release/2026-01-01-staging-promotion-2"): _completed(1, ""),
        ("git", "update-ref",): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, "[]"),
        ("gh", "pr", "create"): _completed(0, "https://github.com/acme/x/pull/14\n"),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow_merge.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_ledger_post(db, []),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["branch"] == "release/2026-01-01-staging-promotion-2"


def test_release_reuses_an_open_release_pr_without_advancing_the_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    old_snapshot = "b" * 40
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened", {
        "repo": "acme/x", "branch": "release/2026-09-20-staging-promotion", "sha": old_snapshot,
        "base": "main", "pr_number": 9, "pr_url": "https://github.com/acme/x/pull/9",
        "kind": "release", "snapshot_sha": old_snapshot,
    })
    con.close()
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({
            "number": 9, "url": "https://github.com/acme/x/pull/9", "state": "OPEN",
            "headRefOid": old_snapshot, "baseRefName": "main", "mergeable": "MERGEABLE",
            "mergeCommit": None,
        })),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow_merge.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["snapshot_sha"] == old_snapshot
    assert result["number"] == 9
    assert not any(c[:2] == ["git", "fetch"] for c in calls)
    assert not any(c[:3] == ["gh", "pr", "create"] for c in calls)


def test_release_reconciles_an_interrupted_attempt_instead_of_duplicating(tmp_path, monkeypatch):
    # decision 9: a PARTIALLY-completed release (PR created on GitHub, ledger post never
    # landed) is found via _open_pr's own gh-list reconciliation, never re-created.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    snapshot = "e" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("git", "fetch", "origin", "staging"): _completed(0, ""),
        ("git", "rev-parse", "--verify", "origin/staging^{commit}"): _completed(0, f"{snapshot}\n"),
        ("git", "rev-parse", "--verify", "refs/remotes/origin/release/2026-01-01-staging-promotion"): _completed(0, f"{snapshot}\n"),
        ("git", "update-ref",): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, json.dumps([{"number": 13}])),
        ("gh", "pr", "view"): _completed(0, json.dumps({
            "number": 13, "url": "https://github.com/acme/x/pull/13", "state": "OPEN",
            "headRefOid": snapshot, "headRefName": "release/2026-01-01-staging-promotion",
            "baseRefName": "main", "mergeable": "MERGEABLE", "mergeCommit": None,
        })),
    })
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow_merge.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_ledger_post(db, events),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["number"] == 13
    assert not any(c[:3] == ["gh", "pr", "create"] for c in calls)


def test_release_refuses_a_repo_outside_the_allowlist_before_any_mutation(tmp_path, monkeypatch):
    # cold review e83ed7ea5eeb F1: refused right after the toplevel resolves -- before the
    # ledger read, the fetch, or the local ref write. No DB or ledger involved: the check
    # runs before either is ever reached.
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_never_post,
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path / "elsewhere",
        )
    assert exc.value.code == "path-outside-allowlist"
    assert not any(c[:2] == ["git", "fetch"] for c in calls)
    assert not any(c[:2] == ["git", "update-ref"] for c in calls)


def test_release_does_not_reuse_a_same_numbered_pr_recorded_for_another_repo(tmp_path, monkeypatch):
    # cold review e83ed7ea5eeb F2: `payload["repo"]` must match the CURRENT repo_slug -- a
    # same-named repo under a different owner must never donate its PR number.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened", {
        "repo": "acme/other", "branch": "release/2026-09-20-staging-promotion", "sha": "b" * 40,
        "base": "main", "pr_number": 9, "pr_url": "https://github.com/acme/other/pull/9",
        "kind": "release", "snapshot_sha": "b" * 40,
    })
    con.close()
    snapshot = "e" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("git", "fetch", "origin", "staging"): _completed(0, ""),
        ("git", "rev-parse", "--verify", "origin/staging^{commit}"): _completed(0, f"{snapshot}\n"),
        ("git", "update-ref",): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, "[]"),
        ("gh", "pr", "create"): _completed(0, "https://github.com/acme/x/pull/20\n"),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow_merge.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_ledger_post(db, []),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["number"] == 20
    assert ["git", "fetch", "origin", "staging"] in calls
    assert not any(c[:3] == ["gh", "pr", "view"] for c in calls)


def test_release_does_not_reuse_when_the_live_pr_head_no_longer_matches_the_recorded_snapshot(
    tmp_path, monkeypatch,
):
    # cold review e83ed7ea5eeb F2: a recorded snapshot_sha the live PR's head has moved past
    # (something pushed to the release branch outside jaxflow) must not be handed back as
    # current -- falls through to a fresh fetch/reconcile instead.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    old_snapshot = "b" * 40
    con = _fresh_db(db)
    _insert(con, None, jaxflow_common.slugify_project(tmp_path.name), "lead", "pr-opened", {
        "repo": "acme/x", "branch": "release/2026-09-20-staging-promotion", "sha": old_snapshot,
        "base": "main", "pr_number": 9, "pr_url": "https://github.com/acme/x/pull/9",
        "kind": "release", "snapshot_sha": old_snapshot,
    })
    con.close()
    snapshot = "e" * 40
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "remote", "get-url", "origin"): _completed(0, "git@github.com:acme/x.git\n"),
        ("gh", "pr", "view"): _completed(0, json.dumps({
            "number": 9, "url": "https://github.com/acme/x/pull/9", "state": "OPEN",
            "headRefOid": "c" * 40, "baseRefName": "main", "mergeable": "MERGEABLE",
            "mergeCommit": None,
        })),
        ("git", "fetch", "origin", "staging"): _completed(0, ""),
        ("git", "rev-parse", "--verify", "origin/staging^{commit}"): _completed(0, f"{snapshot}\n"),
        ("git", "update-ref",): _completed(0, ""),
        ("gh", "pr", "list"): _completed(0, "[]"),
        ("gh", "pr", "create"): _completed(0, "https://github.com/acme/x/pull/21\n"),
    })
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        result = jaxflow_merge.cmd_release(
            SimpleNamespace(from_caller="claude"), run=fake_run, post=_ledger_post(db, []),
            env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent,
        )
    assert result["number"] == 21
    assert ["git", "fetch", "origin", "staging"] in calls


def test_release_argparse_wiring_end_to_end(monkeypatch):
    seen = {}
    def fake_cmd_release(args, **kwargs):
        seen["from_caller"] = args.from_caller
        return {"number": 1, "url": "https://github.com/acme/x/pull/1",
                "snapshot_sha": "a" * 40, "branch": "release/x"}
    monkeypatch.setattr(jaxflow_merge, "cmd_release", fake_cmd_release)
    assert jaxflow_cli.main(["release", "--from", "claude"]) == jaxflow_common.OK
    assert seen == {"from_caller": "claude"}


def _assert_no_mutation(calls):
    """Spec §7.1 #19/#22: a refusal stops after the reads that detected it. Nothing that
    changes git state, the filesystem or the ledger may follow — the full set, not a
    sample of it (cold review F4)."""
    flat = [" ".join(c) for c in calls]
    for banned in ("git merge --no-ff", "git commit", "git push", "git worktree",
                    "git branch -d", "/bin/bash -lc"):
        assert not any(f.startswith(banned) for f in flat), f"{banned} ran after a refusal"


def _merge_args_without_target(**kw):
    args = _MergeArgs(**kw)
    delattr(args, "target")
    return args


def _merge_run_real_ref_format(tmp_path, *, script=None, agents=None):
    """Fake merge runner except `git check-ref-format`, which is the real binary."""
    fake_run, calls = _merge_runner(tmp_path, script=script, agents=agents)

    def run(argv, cwd=None):
        if tuple(argv[:2]) == ("git", "check-ref-format"):
            calls.append(list(argv))
            return _run_real(argv, cwd=cwd)
        return fake_run(argv, cwd=cwd)

    return run, calls


def _assert_no_delivery(calls, *, policy_target="release"):
    """MOA-458: mismatch/invalid target must not reach resume, switch, or delivery."""
    _assert_no_mutation(calls)
    flat = [" ".join(c) for c in calls]
    for banned in (f"git rev-parse --verify {policy_target}^2", "git switch",
                   "git write-tree"):
        assert not any(f.startswith(banned) for f in flat), f"{banned} ran after a refusal"


def test_merge_refuses_a_sha_that_is_not_full_40_hex(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(
        tmp_path, script={("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")}
    )
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha="abc1234"), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "sha-mismatch"
    _assert_no_mutation(calls)


def test_merge_refuses_when_head_is_not_the_approved_sha(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha, other = "a" * 40, "b" * 40
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        # the branch RESOLVES -- to a commit that is not the approved one. A missing ref is
        # a different case and would pass this test without proving anything (part-1 cold
        # review), so the specific entry comes first and the broad one only answers the
        # resume probe.
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{other}\n"),
        ("git", "rev-parse", "--verify"): _completed(128, ""),      # not a resume
        ("git", "branch", "--show-current"): _completed(0, "feat/x\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "sha-mismatch"
    _assert_no_mutation(calls)  # §7.1 #19: zero mutating calls past the detecting reads


def test_merge_refuses_a_dirty_tracked_tree_untracked_files_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    base = {
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "--verify"): _completed(128, ""),      # not a resume
        ("git", "branch", "--show-current"): _completed(0, "feat/x\n"),
    }
    dirty = {**base, ("git", "status", "--porcelain"): _completed(0, " M src/app.py\n")}
    fake_run, calls = _merge_runner(tmp_path, script=dirty)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "dirty-tracked-tree"
    _assert_no_mutation(calls)

    # a FAILED probe is not a clean tree either: empty stdout from a git that errored
    # would otherwise read as "nothing dirty" and let the merge proceed on a tree whose
    # state was never established (round-4 F4)
    broken = {**base, ("git", "status", "--porcelain"):
              _completed(128, "", "fatal: not a git repository\n")}
    fake_run3, calls3 = _merge_runner(tmp_path, script=broken)
    with pytest.raises(ji.Refusal) as exc3:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run3, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc3.value.code == "dirty-tracked-tree"
    assert "git status failed" in exc3.value.hint
    _assert_no_mutation(calls3)

    # untracked-only output must NOT refuse: the status probe asks git to omit untracked
    # files, so an untracked artifact produces empty output and the flow continues past
    # this gate (it will fail later for an unrelated reason, which this test ignores).
    clean = {**base, ("git", "status", "--porcelain"): _completed(0, "")}
    fake_run2, calls2 = _merge_runner(tmp_path, script=clean)
    with contextlib.suppress(Exception):
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run2, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    # The whole argv, not a 3-element prefix (diff review 2026-09-07): what actually
    # makes an untracked artifact pass this gate is `--untracked-files=no` on the probe,
    # and a prefix assertion stays green if that flag is ever dropped -- which would turn
    # every untracked build artifact into a refused delivery.
    assert ["git", "status", "--porcelain", "--untracked-files=no"] in calls2
    assert any(c[:2] == ["git", "switch"] for c in calls2), \
        "a clean tracked tree must reach the target switch — the only switch there is"


def test_merge_refuses_when_the_target_switch_lands_elsewhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "--verify"): _completed(128, ""),      # not a resume
        ("git", "status", "--porcelain"): _completed(0, ""),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        # the switch to `main` returned 0 but silently did not take effect — exactly what
        # target-mismatch guards against
        ("git", "branch", "--show-current"): _completed(0, "feat/x\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "target-mismatch"
    _assert_no_mutation(calls)


def test_merge_happy_path_call_order_and_printed_outcome_with_remote(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    posted = []
    _worktree_path(tmp_path).mkdir(parents=True)   # the checkout the cleanup block removes
    fake_run, calls = _switch_aware_runner(tmp_path, sha)

    def record_post(event):
        # The POST goes into the SAME list as the git calls, so its position is assertable
        # against the push instead of merely "something was posted" (cold review F6).
        calls.append(["POST", event["type"]])
        posted.append(event)

    rc = jaxflow_merge.cmd_merge(_MergeArgs(sha=sha, checks="pytest -q"), run=fake_run,
                           post=record_post, env=_merge_env(), now=_fixed_now,
                           allowlist_root=tmp_path.parent)
    assert rc == jaxflow_common.OK
    flat = [" ".join(c) for c in calls]

    def idx(pred):
        return next(i for i, f in enumerate(flat) if pred(f))

    assert not any(f.startswith("git switch feat/x") for f in flat), \
        "the source branch is held by the builder worktree and is never checked out (F12)"
    # Every required READ, not only the mutations (diff review 2026-09-07): asserting
    # the mutation chain alone stays green if the source-ref, status, branch or remote
    # probe drifts across a mutation -- and a gate that runs after what it gates is not
    # a gate. §7.1 #21 asks for the FULL order, so pin the reads to it.
    i_srcref = idx(lambda f: f == "git rev-parse --verify feat/x^{commit}")
    i_status = idx(lambda f: f.startswith("git status --porcelain --untracked-files=no"))
    i_switch_target = idx(lambda f: f == "git switch main")
    i_oncheck = idx(lambda f: f == "git branch --show-current")
    i_merge = idx(lambda f: f.startswith("git merge --no-ff --no-commit"))
    i_checks = idx(lambda f: "pytest -q" in f)
    i_diff = idx(lambda f: f.startswith("git diff --quiet"))
    i_commit = idx(lambda f: f.startswith("git commit"))
    i_post = idx(lambda f: f.startswith("POST"))
    i_push = idx(lambda f: f.startswith("git push"))
    assert (i_srcref < i_status < i_switch_target < i_oncheck
            < i_merge < i_checks < i_diff < i_commit < i_post < i_push)
    i_remote = idx(lambda f: f.startswith("git remote get-url origin"))
    assert i_commit < i_remote < i_push, "the remote is probed only after a durable commit"
    # the index is fingerprinted on both sides of the checks (cold review F5)
    i_tree_before = idx(lambda f: f == "git write-tree")
    i_tree_after = max(i for i, f in enumerate(flat) if f == "git write-tree")
    assert i_merge < i_tree_before < i_checks < i_tree_after < i_commit
    assert flat[i_checks].startswith("/bin/bash -lc")
    assert flat[i_push] == "git push origin main"
    # the commit subject is the contract's own, never a caller-supplied message
    assert "feat: Phase X (merge feat/x)" in flat[i_commit]

    assert len(posted) == 1
    ev = posted[0]
    assert ev["type"] == "merge-approved" and ev["role"] == "lead" and ev["emitter"] == "wrapper"
    assert ev["payload"] == {
        "phase": "Phase X", "branch": "feat/x", "sha": sha, "target": "main",
        "approved_by": "rafa", "merge_sha": "c" * 40,
        "checks": {"mode": "run"},
    }
    assert ev["pane"] == "%1"
    assert "run_id" not in ev, "merge-approved is not run-scoped"
    # worktree removal strictly before branch deletion, both after the push
    # (§7.1 #21 full order, cold review C2/F6).
    i_worktree = idx(lambda f: f.startswith("git worktree remove"))
    i_branch_del = idx(lambda f: f == "git branch -d feat/x")
    assert i_push < i_worktree < i_branch_del

    out = capsys.readouterr().out.strip().splitlines()
    assert out[-2:] == [f"merged {'c' * 40} pushed origin/main", "checks: pytest -q exit 0"]


def test_merge_refuses_when_worktree_claim_held(tmp_path, monkeypatch):
    import jaxflow_resume as jresume

    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    worktree = _worktree_path(tmp_path)
    worktree.mkdir(parents=True)
    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    with jresume.worktree_claim(tmp_path, worktree):
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_merge(
                _MergeArgs(sha=sha, checks="true"), run=fake_run,
                post=lambda e: {"ok": True}, env=_merge_env(), now=_fixed_now,
                allowlist_root=tmp_path.parent,
            )
    assert exc.value.code in ("resume-ineligible", "agent-settings-permissions")
    _assert_no_mutation(calls)


def test_merge_refuses_nonterminal_newer_build(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    _worktree_path(tmp_path).mkdir(parents=True)
    db = tmp_path / "jaxos.db"
    monkeypatch.setattr(jr, "DB_PATH", db)
    con = _fresh_db(db)
    _insert(con, "bbbbbbbbbbbb", jaxflow_common.slugify_project(tmp_path.name), "builder", "run-started", {
        "phase": "P", "runtime": "opencode-builder", "kind": "build",
        "target": "feat/x", "session": "jax-demo-build-bbbbbbbbbbbb",
        "repo": str(tmp_path.resolve()),
    })
    con.close()
    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(
            _MergeArgs(sha=sha, checks="true"), run=fake_run,
            post=lambda e: {"ok": True}, env=_merge_env(), now=_fixed_now,
            allowlist_root=tmp_path.parent,
        )
    assert exc.value.code == "resume-ineligible"
    _assert_no_mutation(calls)


def test_merge_runs_checks_on_a_non_fast_forward_merge(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    _worktree_path(tmp_path).mkdir(parents=True)
    fake_run, calls = _switch_aware_runner(tmp_path, sha)  # default: not an ancestor
    rc = jaxflow_merge.cmd_merge(_MergeArgs(sha=sha, checks="pytest -q"), run=fake_run,
                           post=lambda e: {"ok": True}, env=_merge_env(), now=_fixed_now,
                           allowlist_root=tmp_path.parent)
    assert rc == jaxflow_common.OK
    flat = [" ".join(c) for c in calls]
    assert any("pytest -q" in f for f in flat), "a non-fast-forward merge must still run checks_cmd"
    assert any(f.startswith("git diff --quiet") for f in flat)
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "checks: pytest -q exit 0"


def test_merge_still_refuses_checks_failed_on_a_non_fast_forward(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha, overrides={("/bin/bash", "-lc"): _completed(1, "boom")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "checks-failed"
    flat = [" ".join(c) for c in calls]
    assert any(f.startswith("git merge --no-ff") for f in flat), "must still attempt the merge"


def test_merge_refuses_an_equal_tip_merge_as_merge_failed(tmp_path, monkeypatch):
    # F3, AC 10: an equal tip is excluded from the skip, runs the full sequence, and
    # fails at `git commit` with nothing staged -- identical to today.
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha,
        overrides={
            ("git", "rev-parse", "HEAD"): _completed(0, sha + "\n"),  # target tip == approved sha
            # Scripted to say "yes" on purpose: this proves the equal-tip short-circuit
            # inside `_is_strict_descendant` -- not an unlucky default -- is what excludes
            # this merge from the skip.
            ("git", "merge-base", "--is-ancestor"): _completed(0, ""),
            ("git", "commit"): _completed(1, "nothing to commit, working tree clean"),
        },
    )
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha, checks="true"), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "merge-failed"
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("git merge-base --is-ancestor") for f in flat), \
        "equal tip must be excluded before ever consulting ancestry"
    assert any(f.startswith("/bin/bash -lc") for f in flat), "checks_cmd must still run"


def test_merge_fast_forward_uses_the_captured_pre_merge_tip_not_a_later_head_read(
        tmp_path, monkeypatch, capsys):
    # AC 8: the fast-forward decision is pinned to the tip captured BEFORE `git merge`
    # runs -- never a later HEAD re-read.
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    _worktree_path(tmp_path).mkdir(parents=True)
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha, overrides={("git", "merge-base", "--is-ancestor"): _completed(0, "")})
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: {"ok": True},
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    flat = [" ".join(c) for c in calls]
    i_tip = flat.index("git rev-parse HEAD")
    i_merge = next(i for i, f in enumerate(flat) if f.startswith("git merge --no-ff --no-commit"))
    i_ancestor = next(i for i, f in enumerate(flat) if f.startswith("git merge-base --is-ancestor"))
    assert i_tip < i_merge < i_ancestor
    assert "git rev-parse HEAD" not in flat[i_merge:i_ancestor], \
        "the fast-forward check must reuse the tip captured before the merge, never a fresh HEAD read"


def test_merge_a_non_fast_forward_target_does_not_skip_checks(tmp_path, monkeypatch, capsys):
    # F1, AC 9: pins the DIRECTION -- a reversed base/head would report a fast-forward
    # here (args.sha IS an ancestor of target_tip -- the WRONG relationship), and this
    # test fails if that ever recurs.
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    target_tip = "c" * 40
    _worktree_path(tmp_path).mkdir(parents=True)

    def is_ancestor(argv):
        if argv[3:] == [sha, target_tip]:
            return _completed(0, "")   # the WRONG direction: would wrongly say "yes"
        if argv[3:] == [target_tip, sha]:
            return _completed(1, "")   # the correct direction: no, not an ancestor
        raise AssertionError(f"unexpected merge-base call: {argv}")

    base_fake, calls = _switch_aware_runner(
        tmp_path, sha, overrides={("git", "rev-parse", "HEAD"): _completed(0, target_tip + "\n")})

    def wrapped(argv, cwd=None):
        if argv[:3] == ["git", "merge-base", "--is-ancestor"]:
            calls.append(list(argv))
            return is_ancestor(argv)
        return base_fake(argv, cwd=cwd)

    rc = jaxflow_merge.cmd_merge(_MergeArgs(sha=sha, checks="pytest -q"), run=wrapped,
                           post=lambda e: {"ok": True}, env=_merge_env(), now=_fixed_now,
                           allowlist_root=tmp_path.parent)
    assert rc == jaxflow_common.OK
    flat = [" ".join(c) for c in calls]
    assert any("pytest -q" in f for f in flat), \
        "the correct direction finds no fast-forward, so checks_cmd must still run"


def test_merge_without_a_remote_skips_push_and_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha, remote=False)
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert not any(c[:2] == ["git", "push"] for c in calls)
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-2] == f"merged {'c' * 40} — no remote delivery configured"
    assert out[-1] == "checks: true exit 0"


def test_merge_prints_the_no_preset_block_line(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert "target: main (no preset block)" in capsys.readouterr().out


def test_merge_omits_pane_when_tmux_pane_is_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha, posted = "a" * 40, []
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    env = _merge_env()
    del env["TMUX_PANE"]
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=posted.append,
                      env=env, now=_fixed_now, allowlist_root=tmp_path.parent)
    assert "pane" not in posted[0], "absent pane is omitted, never sent as null"


@pytest.mark.parametrize("failing,code", [
    (("git", "merge", "--no-ff"), "merge-failed"),
    (("/bin/bash", "-lc"), "checks-failed"),
    (("git", "diff", "--quiet"), "checks-dirtied-tree"),
])
def test_merge_aborts_cleanly_with_zero_further_calls(tmp_path, monkeypatch, failing, code):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha,
                                           overrides={failing: _completed(1, "boom")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == code
    flat = [" ".join(c) for c in calls]
    assert "git merge --abort" in flat
    # everything after the abort is READ-ONLY state reporting (_abort_merge's two probes);
    # nothing that changes git state, the filesystem or the ledger may follow.
    tail = flat[flat.index("git merge --abort") + 1:]
    assert all(f.startswith("git rev-parse -q --verify MERGE_HEAD") or
               f.startswith("git status --porcelain") for f in tail), tail
    assert not any(f.startswith("git commit") or f.startswith("git push") or
                   f.startswith("git worktree") or f.startswith("git branch -d") or
                   f.startswith("git reset") for f in flat)


def test_merge_post_failure_keeps_the_commit_and_prints_the_resume_line(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha)

    def failing_post(event):
        raise RuntimeError("hub down")

    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=failing_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "hub-unreachable"
    flat = [" ".join(c) for c in calls]
    assert any(f.startswith("git commit") for f in flat), "the commit is kept"
    assert not any(f.startswith("git push") or f.startswith("git worktree") for f in flat)
    assert f"merge commit {'c' * 40} kept; re-run the same jaxflow merge to resume" in capsys.readouterr().out


def test_merge_push_failure_keeps_everything_and_cleans_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    posted = []
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha, overrides={("git", "push"): _completed(1, "rejected")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=posted.append,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "push-failed"
    assert len(posted) == 1, "the audit row was already posted and is not retried"
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("git worktree") or f.startswith("git branch -d") for f in flat)


def test_merge_copies_worktree_reports_before_removing_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    # the worktree `build` would have created for this branch, with a builder report in it
    worktree = tmp_path.parent / f"{jaxflow_common.slugify_project(tmp_path.name)}-feat-x"
    (worktree / ".local" / "reports").mkdir(parents=True)
    (worktree / ".local" / "reports" / "abc123abc123.md").write_text("report", encoding="utf-8")
    # the spec requires the verification evidence to survive the removal too, not just the
    # report markdown (part-1 cold review round 2)
    (worktree / ".local" / "reports" / "abc123abc123.tests.txt").write_text(
        "42 passed", encoding="utf-8")
    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    dest = tmp_path / ".local" / "reports"
    assert dest.joinpath("abc123abc123.md").read_text(encoding="utf-8") == "report"
    assert dest.joinpath("abc123abc123.tests.txt").read_text(encoding="utf-8") == "42 passed", \
        "the verification evidence survives too (spec §5.3)"
    flat = [" ".join(c) for c in calls]
    assert any(f.startswith("git worktree remove") for f in flat)
    assert any(f == "git branch -d feat/x" for f in flat)


def test_merge_writes_status_md_stage_ship_and_drops_the_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    (tmp_path / ".jax-os").mkdir()
    (tmp_path / ".jax-os" / "status.md").write_text(
        # fixture fix (execution rule): `_fixed_now()` (already defined at the top of this
        # file) returns 2026-01-01, not the 2026-09-07 the plan's own reference definition
        # assumed -- an `updated` stamp of 2026-09-01 would read as NEWER than
        # `dispatch_start` and trip the staleness guard, skipping the write this test
        # means to exercise. Any timestamp before 2026-01-01 keeps the same intent.
        "---\nproject: Demo\nstage: review\nbuilder: codex\nbranch: feat/x\n"
        "gate: awaiting-approval\nupdated: 2020-01-01T00:00:00-03:00\n---\n\n"
        "## Now\nOld text.\n\n## Residuals\nKeep me.\n",
        encoding="utf-8",
    )
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    text = (tmp_path / ".jax-os" / "status.md").read_text(encoding="utf-8")
    assert "stage: ship" in text
    assert "gate:" not in text, "a completed delivery has no pending decision"
    assert "project: Demo" in text and "## Residuals\nKeep me." in text
    assert "merged" in text.split("## Now")[1]


# ---- resume path (moved here from Task 2 so every commit gate stays green) -------------

def test_merge_detects_a_resume_and_skips_merge_checks_and_commit(tmp_path, monkeypatch):
    """§7.1 #26: target HEAD already carries the merge of --sha."""
    monkeypatch.chdir(tmp_path)
    sha, posted = "a" * 40, []
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        # the merge commit is read from TARGET, never from HEAD (round-2 F4)
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        # ...and it records the branch the approval named (the round-2 F4 binding)
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        # ...and the source branch still resolves to the approved sha (the F2 gate)
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{sha}\n"),
        ("git", "remote", "get-url"): _completed(2, ""),   # 2 == no origin
    })
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=posted.append,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    flat = [" ".join(c) for c in calls]
    for banned in ("git merge --no-ff", "git commit", "git diff --quiet", "/bin/bash -lc",
                   "git write-tree"):
        assert not any(f.startswith(banned) for f in flat), f"{banned} ran on a resume"
    assert posted and posted[0]["type"] == "merge-approved"
    assert posted[0]["payload"]["merge_sha"] == "c" * 40, "the target's merge, not HEAD"
    assert posted[0]["payload"]["checks"] == {"mode": "resumed"}


def test_merge_does_not_mistake_an_unrelated_merge_commit_for_a_resume(tmp_path, monkeypatch):
    """§7.1 #26, negative half: the normal Step 1 sequence runs in full."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    # main's HEAD IS a merge commit, but of a DIFFERENT sha. The override has to name the
    # SPECIFIC key `_merge_happy_script` already defines -- a broad prefix is inserted after
    # it and never reached, which made this test pass without testing anything (part-1 cold
    # review).
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha,
        overrides={("git", "rev-parse", "--verify", "main^2"): _completed(0, "d" * 40 + "\n")})
    _worktree_path(tmp_path).mkdir(parents=True)
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run,
                      post=lambda e: calls.append(["POST", e["type"]]),
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    flat = [" ".join(c) for c in calls]
    # the FULL normal sequence in order, not merely "some switch ran" (cold review C2/F6)
    at = -1
    for expected in ("git switch main", "git merge --no-ff",
                     "/bin/bash -lc", "git write-tree", "git diff --quiet", "git commit",
                     "POST", "git push", "git worktree remove", "git branch -d feat/x"):
        nxt = next((i for i, f in enumerate(flat) if i > at and f.startswith(expected)), None)
        assert nxt is not None, f"{expected} missing after position {at}: {flat}"
        at = nxt


# ---- cold-review findings F1/F2/F3 ------------------------------------------------------

def test_merge_reports_when_the_abort_could_not_clean_the_checkout(tmp_path, monkeypatch):
    """F1: `git merge --abort` can exit non-zero and leave an AM merge state. The refusal
    must SAY so instead of implying a clean restore."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha, dirty_after_abort=" M src/app.py\n",
        overrides={
            ("/bin/bash", "-lc"): _completed(1, "boom"),
            ("git", "merge", "--abort"): _completed(128, "fatal: could not abort"),
            ("git", "rev-parse", "-q", "--verify", "MERGE_HEAD"): _completed(0, "e" * 40 + "\n"),
        })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "checks-failed"
    assert "exited 128" in exc.value.hint, "the abort's own exit code is reported (F11)"
    assert "MERGE_HEAD" in exc.value.hint
    assert "src/app.py" in exc.value.hint, "the leftover state is named, not just hinted at"
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("git reset") for f in flat), "jaxflow never resets --hard on its own"


def test_merge_abort_that_succeeds_cleanly_adds_no_state_note(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha,
        overrides={
            ("/bin/bash", "-lc"): _completed(1, "boom"),
            ("git", "rev-parse", "-q", "--verify", "MERGE_HEAD"): _completed(1, ""),
        })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "checks-failed"
    assert "MERGE_HEAD" not in exc.value.hint


def test_merge_abort_that_exits_zero_still_reports_a_modified_tracked_file(tmp_path, monkeypatch):
    """F1, the second real case (verified on git 2026-09-07): when the checks modified a
    tracked file that was NOT part of the merge, `git merge --abort` exits 0 and leaves
    that modification in place. Exit 0 is not proof of a clean checkout, so the refusal
    still has to name what survived."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha, dirty_after_abort=" M untouched.py\n",
        overrides={
            ("/bin/bash", "-lc"): _completed(1, "boom"),
            ("git", "merge", "--abort"): _completed(0, ""),          # the abort SUCCEEDED
            ("git", "rev-parse", "-q", "--verify", "MERGE_HEAD"): _completed(1, ""),
        })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "checks-failed"
    assert "MERGE_HEAD" not in exc.value.hint, "there is no merge state left"
    assert "tracked changes remain" in exc.value.hint and "untouched.py" in exc.value.hint


def test_merge_refuses_when_the_checks_stage_a_tracked_change(tmp_path, monkeypatch):
    """F5: `git diff --quiet` is blind to a check that ran `git add` (verified on real git
    2026-09-07: it exits 0 while `git diff --cached --quiet` exits 1, and the following
    commit carried the staged change). The index fingerprint is what catches it."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        # fixture fix (execution rule): both values must be real 40-hex shapes -- the
        # implementation's shape check (round-5 F3) would otherwise reject the FIRST one
        # and refuse `merge-failed` before the staged-change comparison this test targets
        # is ever reached. Two DIFFERENT valid tree ids is what "the index moved under the
        # checks" actually means.
        tmp_path, sha, trees=["3" * 40, "4" * 40],
        overrides={("git", "rev-parse", "-q", "--verify", "MERGE_HEAD"): _completed(1, "")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "checks-dirtied-tree"
    assert "(staged)" in exc.value.hint
    flat = [" ".join(c) for c in calls]
    assert "git merge --abort" in flat
    assert not any(f.startswith("git commit") or f.startswith("git push") for f in flat)


@pytest.mark.parametrize("branch,code", [
    ("x" * 513, "branch-invalid"),   # a valid git ref, over the audit payload's bound
    ("", "branch-invalid"),
    ("feat/ x", "branch-invalid"),
])
def test_merge_refuses_a_branch_the_audit_event_cannot_carry(tmp_path, monkeypatch,
                                                             branch, code):
    """`git check-ref-format` accepts a 2048-character ref; `LIMITS.target` is 512. An
    oversized-but-valid branch would reach the commit subject AND the payload, commit,
    and only then fail its POST -- the stranded-merge shape the phase bound already
    prevents, one field over (round-5 F11). Bounded before any mutation."""
    monkeypatch.chdir(tmp_path)
    # the toplevel read must SUCCEED, or the refusal is `not-a-git-toplevel` and the gate
    # under test is never reached (round-6 F15)
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(branch=branch), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == code
    _assert_no_mutation(calls)


def test_merge_refuses_a_delivery_target_the_audit_event_cannot_carry(tmp_path, monkeypatch):
    """The target is payload too, and it comes from a hand-written `AGENTS.md` line or
    from a symbolic-ref read -- neither validates a shape (round-5 F8/F11)."""
    monkeypatch.chdir(tmp_path)
    # the parser wants the real heading shape, bold + backticks, or the block is never
    # recognised and the long target is never extracted (round-6 F15)
    _agents(tmp_path, "## Deploy policy\n\n**Preset: `dual-branch`** — notes.\n\n"
                      "Delivery target: `" + "y" * 513 + "`.\n")
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "preset-unknown"
    _assert_no_mutation(calls)


def test_merge_resume_refuses_when_the_subject_read_fails_but_prints_the_right_text(
        tmp_path, monkeypatch):
    """#27c covers EVERY value reader, not just one (diff review 2026-09-07). A failed
    `git log` whose stdout happens to be the expected subject must not authorise a resume
    -- that read is the identity binding that stops an alias branch being deleted."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        # non-zero, but printing exactly what the caller wanted to see
        ("git", "log", "-1", "--format=%s"): _completed(128, "feat: Phase X (merge feat/x)\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "sha-mismatch"
    _assert_no_mutation(calls)


def test_merge_refuses_when_the_post_switch_branch_read_fails_but_prints_the_target(
        tmp_path, monkeypatch):
    """The other #27c reader: `git branch --show-current` failing while echoing the
    target would let the merge proceed on a checkout it never confirmed."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha,
        overrides={("git", "branch", "--show-current"): _completed(128, "main\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "target-mismatch"
    assert not any(" ".join(c).startswith("git merge --no-ff") for c in calls), \
        "nothing is merged onto a checkout the tool could not confirm"


def test_merge_refuses_when_a_git_read_fails_but_prints_a_plausible_value(tmp_path,
                                                                          monkeypatch):
    """The whole point of `_git_read` (round-5 F2): git writes to stdout before it fails,
    so a NON-ZERO call whose output happens to look like a sha must not be trusted. Two
    of the eight call sites had exactly this bug -- checking the shape but not the exit."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha,
        overrides={("git", "rev-parse", "HEAD"): _completed(128, "c" * 40 + "\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "merge-failed"
    assert not any(" ".join(c).startswith("git push") for c in calls)


@pytest.mark.parametrize("trees,code", [
    (["not-a-tree-object", "not-a-tree-object"], "merge-failed"),
    (["b" * 40, "still-not-a-tree"], "checks-dirtied-tree"),
])
def test_merge_refuses_a_write_tree_value_that_is_not_a_tree_object(tmp_path, monkeypatch,
                                                                    trees, code):
    """The two fingerprints are compared to EACH OTHER, so two malformed-but-equal values
    would report `staged_changed = False` -- and `git diff --quiet` is blind to a staged
    change, which is the whole reason the fingerprints exist (round-5 F3)."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha, trees=trees)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == code
    assert not any(" ".join(c).startswith("git commit") for c in calls)


def test_merge_refuses_when_the_remote_probe_itself_fails(tmp_path, monkeypatch):
    """Exit 2 is `No such remote` and nothing else is (verified on real git). Treating
    every non-zero result as `no remote` would report a complete local delivery because
    the repository was broken or unreadable (round-5 F4). The commit is durable here, so
    the refusal is `push-failed` and the documented resume finishes the job."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    posted = []
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha,
        overrides={("git", "remote", "get-url"): _completed(128, "", "fatal: not a git repo")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=posted.append,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "push-failed"
    assert "could not probe origin" in exc.value.hint
    assert len(posted) == 1, "the audit event is posted BEFORE the push, and stays posted"
    assert not any(" ".join(c).startswith(("git push", "git worktree")) for c in calls)


def test_merge_resume_refuses_when_the_branch_read_fails_rather_than_being_gone(
        tmp_path, monkeypatch):
    """`rev-parse --verify` returns 128 both for `no such ref` and for a broken
    repository; `show-ref --verify` separates them (exit 1 vs 128, verified on real git).
    Only `gone` may skip cleanup -- a failed read must not post an audit for a delivery
    whose source branch could not be checked (round-5 F6)."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        ("git", "show-ref", "--verify"): _completed(128, "", "fatal: not a git repository"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "sha-mismatch"
    assert "could not read feat/x" in exc.value.hint
    _assert_no_mutation(calls)


def test_merge_reports_a_branch_that_could_not_be_deleted(tmp_path, monkeypatch, capsys):
    """Best-effort, but never silent (round-5 F10): the merge still succeeds, and a
    branch that outlived it is state the tech lead has to know about."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    worktree = _worktree_path(tmp_path)
    (worktree / ".local" / "reports").mkdir(parents=True)
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha,
        overrides={("git", "branch", "-d"): _completed(1, "", "error: not fully merged")})
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    out = capsys.readouterr().out
    assert "branch feat/x kept" in out and "not fully merged" in out
    assert "merged " in out, "the delivery still reports success"


@pytest.mark.parametrize("phase", [
    "Release\nsecond line",          # `git commit -m` takes it; `%s` flattens it (F15)
    "x" * 201,                       # over the event validator's LIMITS.summary bound (F16)
    "",
    "   ",
    " Phase X ",                     # untrimmed: the commit subject would not round-trip
    "\ufeffPhase X",                  # the ONE character JS `\s` strips and Python's
                                     # `strip()` does not, so only refusing it here keeps
                                     # the producer from being the looser side (part-2
                                     # cold review round 4)
    pytest.param("\U0001f680" * 200,      # 200 emoji = 400 UTF-16 units; `len()` would accept it
                 id="utf16-code-units-over-the-bound"),
    pytest.param("Phase \udcff X",        # POSIX argv surrogateescape: `encode("utf-16-le")` raises
                 id="undecodable-argv-byte"),
])
def test_merge_refuses_an_unusable_phase_title(tmp_path, monkeypatch, phase):
    """F15/F16: one canonical title serves the commit subject, the audit payload and the
    resume comparison, so anything that cannot survive all three is refused BEFORE any git
    state is touched."""
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(
        tmp_path, script={("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(phase=phase), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "phase-invalid"
    _assert_no_mutation(calls)


def test_merge_phase_title_is_identical_in_the_commit_and_the_audit_event(tmp_path, monkeypatch):
    """F16: the commit subject and the `merge-approved` payload carry the SAME title —
    the earlier draft committed the raw one and posted a truncated one."""
    monkeypatch.chdir(tmp_path)
    sha, posted = "a" * 40, []
    title = "Phase C.2 — Jax Rules (canonical rule-file management)"
    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha, phase=title), run=fake_run, post=posted.append,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    commit = next(c for c in calls if c[:2] == ["git", "commit"])
    assert commit[-1] == f"feat: {title} (merge feat/x)"
    assert posted[0]["payload"]["phase"] == title


def test_merge_refuses_when_the_merged_index_cannot_be_fingerprinted(tmp_path, monkeypatch):
    """F9: a PRE-check `git write-tree` failure is a broken merge, not a dirty check. It
    refuses `merge-failed` and the checks command never runs at all."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha, trees=[_completed(128, "", "fatal: unable to write new index file")])
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "merge-failed"
    assert "unable to write new index file" in exc.value.hint
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("/bin/bash -lc") for f in flat), "the checks never ran"
    assert "git merge --abort" in flat


def test_merge_report_copy_skips_a_symlinked_entry(tmp_path, monkeypatch, capsys):
    """F17: everything under the builder's `.local/reports/` is builder-controlled.
    `shutil.copytree` follows symlinks, which would copy whatever one points at — outside
    the allowlist included — into the control repo (spec §6/I4)."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    secret = tmp_path.parent / "outside-the-allowlist.txt"
    secret.write_text("SECRET", encoding="utf-8")
    reports = _worktree_path(tmp_path) / ".local" / "reports"
    reports.mkdir(parents=True)
    (reports / "abc123abc123.md").write_text("real report", encoding="utf-8")
    # An ALLOWED name (diff review 2026-09-07): `evil.md` was refused by the name filter
    # before the open, so this test passed without ever exercising `O_NOFOLLOW` -- it
    # would have stayed green if symlinked entries were followed.
    (reports / "def456def456.md").symlink_to(secret)
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    copied = tmp_path / ".local" / "reports"
    assert (copied / "abc123abc123.md").read_text(encoding="utf-8") == "real report"
    assert not (copied / "def456def456.md").exists(), "a symlinked report is never copied"
    assert "def456def456.md skipped" in capsys.readouterr().out
    assert secret.read_text(encoding="utf-8") == "SECRET"


def test_merge_accepts_a_phase_at_exactly_the_utf16_bound(tmp_path, monkeypatch):
    """The bound is 200 code units, not 200 bytes and not 100 characters: 100 emoji are
    exactly 200 units and must pass."""
    monkeypatch.chdir(tmp_path)
    sha, posted = "a" * 40, []
    title = "\U0001f680" * 100
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha, phase=title), run=fake_run, post=posted.append,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert posted[0]["payload"]["phase"] == title


@pytest.mark.parametrize("plant,expected_skip", [
    ("symlinked-dir", "reports copy skipped"),          # the reports dir itself
    ("symlinked-ancestor", "reports copy skipped"),     # `.local`, one level up
    ("dest-symlinked-ancestor", "reports copy skipped"),  # the CONTROL repo's `.local`
    ("hardlink", "not a single-linked regular file"),
    ("wrong-name", "not a run report"),
    ("dest-symlink", "existing destination unreadable"),
    ("oversize", "over the"),
    ("fifo", "not a single-linked regular file"),
])
def test_merge_report_copy_refuses_every_escape(tmp_path, monkeypatch, capsys, plant, expected_skip):
    """The four escapes a name-only check let through, each reproduced. `merge` reads a
    builder-written file exactly here and nowhere else, so this is the whole attack
    surface (spec §6/I4)."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    secret = tmp_path.parent / "outside-the-allowlist.txt"
    secret.write_text("SECRET", encoding="utf-8")
    worktree = _worktree_path(tmp_path)
    reports = worktree / ".local" / "reports"

    outside = tmp_path.parent / "elsewhere"
    # fixture fix (execution rule): `tmp_path.parent` is the SAME session tmp root for
    # every one of this test's 8 parametrized invocations (pytest nests each `tmp_path`
    # directly under one shared session dir), so a fixed name here collides with the
    # previous case's leftover directory. `exist_ok=True` is enough: the decoy content is
    # identical across cases, so re-using the same directory changes nothing under test.
    (outside / "reports").mkdir(parents=True, exist_ok=True)
    (outside / "reports" / "abc123abc123.md").write_text("SECRET", encoding="utf-8")

    if plant == "symlinked-dir":
        # the reports directory itself is a symlink pointing outside the worktree
        (worktree / ".local").mkdir(parents=True)
        reports.symlink_to(outside / "reports")
    elif plant == "symlinked-ancestor":
        # `.local` is the symlink — one level ABOVE the component O_NOFOLLOW would see
        # if only the leaf were checked (part-1 cold review round 2)
        worktree.mkdir(parents=True)
        (worktree / ".local").symlink_to(outside)
    elif plant == "dest-symlinked-ancestor":
        # the escape pointing the other way: the CONTROL repo's `.local` redirects writes
        reports.mkdir(parents=True)
        (reports / "abc123abc123.md").write_text("real report", encoding="utf-8")
        (tmp_path / ".local").symlink_to(outside)
    else:
        reports.mkdir(parents=True)
        if plant == "hardlink":
            os.link(secret, reports / "abc123abc123.md")
        elif plant == "wrong-name":
            (reports / ".env").write_text("TOKEN=1", encoding="utf-8")
        elif plant == "oversize":
            (reports / "abc123abc123.md").write_bytes(b"x" * (jaxflow_common.REPORT_COPY_MAX + 1))
        elif plant == "dest-symlink":
            (reports / "abc123abc123.md").write_text("real report", encoding="utf-8")
            (tmp_path / ".local" / "reports").mkdir(parents=True)
            (tmp_path / ".local" / "reports" / "abc123abc123.md").symlink_to(secret)
        elif plant == "fifo":
            # opening a FIFO for reading blocks until a writer shows up, and this copy
            # runs AFTER the merge commit -- a regression would hang the merge, not fail
            # it, so the alarm below turns the hang into an ordinary test failure.
            os.mkfifo(reports / "abc123abc123.md")

    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    if plant == "dest-symlinked-ancestor":
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                              env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
        assert exc.value.code == "agent-settings-symlink"
        assert secret.read_text(encoding="utf-8") == "SECRET"
        assert not list((outside / "reports").glob("*.tmp"))
        return
    previous = None
    if plant == "fifo":
        def _blocked(signum, frame):
            raise AssertionError("the report copy blocked on the FIFO")
        # Save BOTH the handler and any timer already armed: the autouse fixture at the
        # top of this file restores only SIGTERM and SIGHUP, so leaving `_blocked`
        # installed would make a later alarm anywhere in the suite fail in this test's
        # name (round-4 F2).
        previous = (signal.signal(signal.SIGALRM, _blocked),
                    signal.setitimer(signal.ITIMER_REAL, 5))
    try:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    finally:
        if previous is not None:
            handler, (delay, interval) = previous
            signal.setitimer(signal.ITIMER_REAL, delay, interval)
            signal.signal(signal.SIGALRM, handler)
    assert expected_skip in capsys.readouterr().out
    assert secret.read_text(encoding="utf-8") == "SECRET", "the outside file is never written"
    assert not list((outside / "reports").glob("*.tmp")), "nothing is written outside either"
    if plant == "dest-symlink":
        copied = tmp_path / ".local" / "reports" / "abc123abc123.md"
        assert copied.is_symlink() and copied.readlink() == secret, "the plant is left alone"
    elif plant != "dest-symlinked-ancestor":
        copied = tmp_path / ".local" / "reports" / "abc123abc123.md"
        assert not copied.exists() or copied.read_text(encoding="utf-8") != "SECRET"
    assert not (tmp_path / ".local" / "reports" / ".env").exists()


def test_merge_report_copy_continues_past_one_unreadable_entry(tmp_path, monkeypatch, capsys):
    """One bad report must not cost the others: the delivery still reports success, so a
    loop that aborts would lose evidence silently."""
    if os.geteuid() == 0:
        pytest.skip("mode 000 is readable for root, so the skip path never fires")
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    reports = _worktree_path(tmp_path) / ".local" / "reports"
    reports.mkdir(parents=True)
    (reports / "aaa111aaa111.md").write_text("first", encoding="utf-8")
    (reports / "bbb222bbb222.md").write_text("unreadable", encoding="utf-8")
    (reports / "bbb222bbb222.md").chmod(0o000)
    (reports / "ccc333ccc333.md").write_text("third", encoding="utf-8")
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    try:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
        dest = tmp_path / ".local" / "reports"
        assert (dest / "aaa111aaa111.md").read_text(encoding="utf-8") == "first"
        assert (dest / "ccc333ccc333.md").read_text(encoding="utf-8") == "third"
        assert "bbb222bbb222.md skipped" in capsys.readouterr().out
    finally:
        (reports / "bbb222bbb222.md").chmod(0o600)


def test_merge_refuses_when_the_commit_sha_cannot_be_read(tmp_path, monkeypatch):
    """`merge-approved` requires a full 40-hex sha and the validator rejects anything
    else, so an unchecked `rev-parse HEAD` would turn a durable commit into an
    un-auditable one (round-4 F5). Nothing is aborted -- the commit landed, and the same
    command resumes from it."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha, overrides={("git", "rev-parse", "HEAD"): _completed(128, "")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "merge-failed"
    assert "re-run the same jaxflow merge to resume" in exc.value.hint
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("git merge --abort") for f in flat), \
        "there is nothing to abort after a successful commit"
    assert not any(f.startswith(("git push", "git worktree", "git branch -d")) for f in flat)


@pytest.mark.parametrize("mutated,expected_len", [("ori", 3), ("original-grown", 14)])
def test_merge_report_copy_skips_a_report_that_changed_under_the_read(
        tmp_path, monkeypatch, capsys, mutated, expected_len):
    """`st_size` is a snapshot taken before the read. A report that shrinks would be
    copied truncated and one that grows would be copied as a stale prefix -- both are a
    silent evidence loss reported as a successful copy. Deterministic rather than
    timing-based: the first `os.read` rewrites the file, exactly as a concurrent writer
    would. The claim is about THIS path, not about `os.read` globally (round-4 F3):
    `subprocess` does call it when capturing pipe output, but `run` is injected here so no
    subprocess exists, and `real_read` is captured before the patch so the fake cannot
    recurse. `monkeypatch` restores the attribute at teardown."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    reports = _worktree_path(tmp_path) / ".local" / "reports"
    reports.mkdir(parents=True)
    victim = reports / "abc123abc123.md"
    victim.write_text("original", encoding="utf-8")          # 8 bytes at fstat time
    real_read, fired = os.read, []

    def racing_read(fd, n):
        if not fired:                                        # only the first read races
            fired.append(True)
            victim.write_text(mutated, encoding="utf-8")
        return real_read(fd, n)

    monkeypatch.setattr(os, "read", racing_read)
    fake_run, _ = _switch_aware_runner(tmp_path, sha)
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert fired, "the race never fired -- the copy loop no longer calls os.read"
    assert len(victim.read_bytes()) == expected_len
    assert "abc123abc123.md skipped (changed while being read)" in capsys.readouterr().out
    assert not (tmp_path / ".local" / "reports" / "abc123abc123.md").exists(), \
        "a report whose bytes could not be trusted is not copied at all"


def test_merge_keeps_the_worktree_when_a_report_copy_fails(tmp_path, monkeypatch, capsys):
    """F3 (spec §4 guard 7 / §11): `cmd_merge` inherits the shared helper's stricter
    copy-then-remove rule -- a report that exists on disk but cannot be preserved now
    keeps the worktree instead of today's catch-and-continue removal."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    reports = _worktree_path(tmp_path) / ".local" / "reports"
    reports.mkdir(parents=True)
    (reports / "abc123abc123.md").write_bytes(b"x" * (jaxflow_common.REPORT_COPY_MAX + 1))
    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert "kept (report copy failed)" in capsys.readouterr().out
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("git worktree remove") for f in flat), \
        "a copy failure aborts before removal"
    assert not any(f == "git branch -d feat/x" for f in flat)


def test_merge_refuses_an_agents_md_that_exists_but_cannot_be_read(tmp_path, monkeypatch):
    """A policy that cannot be read is not the same as no policy: treating it as empty
    would fall through to the default branch and could direct-merge a `dual-branch-pr` repo."""
    monkeypatch.chdir(tmp_path)
    policy = tmp_path / "AGENTS.md"
    policy.write_bytes(b"**Preset: `dual-branch-pr`**\n\xff\xfe not utf-8\n")
    fake_run, calls = _merge_runner(
        tmp_path, script={("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "preset-unknown"
    assert "could not be read" in exc.value.hint
    _assert_no_mutation(calls)


def test_merge_resume_writes_the_target_as_the_status_branch(tmp_path, monkeypatch):
    """A resume never switches to the target, so `_current_branch()` would report whatever
    the tech lead was standing on — or nothing, from a detached HEAD."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    (tmp_path / ".jax-os").mkdir()
    (tmp_path / ".jax-os" / "status.md").write_text(
        # fixture fix (execution rule): see the same note in
        # test_merge_writes_status_md_stage_ship_and_drops_the_gate -- `_fixed_now()`
        # returns 2026-01-01, so `updated` must predate it or the staleness guard skips
        # the write.
        "---\nproject: Demo\nstage: review\nbuilder: codex\nbranch: feat/x\n"
        "updated: 2020-01-01T00:00:00-03:00\n---\n\n## Now\nOld.\n",
        encoding="utf-8")
    fake_run, _ = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        # fixture fix (execution rule): without this entry the fake's default (exit 0)
        # reads as "the branch is present", so `branch_present` never flips to False and
        # the flow falls through to the `feat/x^{commit}` probe below -- which THIS
        # fixture also scripts as gone (128), correctly raising sha-mismatch instead of
        # exercising this test's actual resume-completes path. Exit 1 is "no such ref".
        ("git", "show-ref", "--verify"): _completed(1, ""),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(128, ""),
        ("git", "remote", "get-url"): _completed(2, ""),   # 2 == no origin
        # a detached HEAD: `git branch --show-current` prints nothing
        ("git", "branch", "--show-current"): _completed(0, "\n"),
    })
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    text = (tmp_path / ".jax-os" / "status.md").read_text(encoding="utf-8")
    assert "branch: main" in text, "the delivery target, not the caller's checkout"
    assert "stage: ship" in text


def test_merge_refuses_when_the_switch_to_the_target_fails(tmp_path, monkeypatch):
    """F8: a failed `git switch` returns non-zero while leaving the previous branch checked
    out, so the return code has to be checked and not just the branch afterwards. This is
    the verb's ONLY switch (round-4 F12)."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha, switch_fails="main")
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "target-mismatch"
    assert "git switch main failed" in exc.value.hint
    _assert_no_mutation(calls)


def test_merge_resume_refuses_a_subject_that_merely_contains_the_branch(tmp_path, monkeypatch):
    """F7: `--phase` is free text. A real merge of `other` titled `Release (merge feat/x)`
    writes the subject `feat: Release (merge feat/x) (merge other)`; a substring check
    would accept a resume for feat/x and delete it. The exact subject shape does not."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        ("git", "log", "-1", "--format=%s"): _completed(
            0, "feat: Release (merge feat/x) (merge other)\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{sha}\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha, phase="Release (merge feat/x)"), run=fake_run,
                          post=_never_post, env=_merge_env(), now=_fixed_now,
                          allowlist_root=tmp_path.parent)
    assert exc.value.code == "sha-mismatch"
    _assert_no_mutation(calls)


def test_merge_resume_refuses_a_branch_the_merge_commit_does_not_name(tmp_path, monkeypatch):
    """F4: an alias branch pointing at the approved sha passes every sha check there is --
    reproduced on real git, where the alias and its registered worktree were both removed
    while the delivered branch survived. Only the merge commit's own subject says which
    branch the approval delivered."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        # jaxflow's own merge names feat/x; the caller asked to resume `alias`
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        ("git", "rev-parse", "--verify", "alias^{commit}"): _completed(0, f"{sha}\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(branch="alias", sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "sha-mismatch"
    assert "does not record alias" in exc.value.hint
    _assert_no_mutation(calls)


def test_merge_resume_refuses_when_the_branch_no_longer_names_the_approved_sha(tmp_path, monkeypatch):
    """F2: a resume must not delete a branch it never verified."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        # the caller-named branch points somewhere else entirely
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, "9" * 40 + "\n"),
    })
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "sha-mismatch"
    _assert_no_mutation(calls)


def test_merge_resume_skips_cleanup_when_the_branch_is_already_gone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha, posted = "a" * 40, []
    fake_run, calls = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        # fixture fix (execution rule): without this, the fake's default (exit 0) reads
        # as "branch present" and the flow never reaches the "already gone" path this
        # test is named for. Exit 1 is "no such ref".
        ("git", "show-ref", "--verify"): _completed(1, ""),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(128, ""),
        ("git", "remote", "get-url"): _completed(2, ""),   # 2 == no origin
    })
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=posted.append,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("git worktree remove") or f.startswith("git branch -d") for f in flat)
    assert posted, "an already-cleaned resume still completes its audit"


def test_merge_never_removes_a_worktree_git_does_not_register_for_that_branch(tmp_path, monkeypatch):
    """F2, second half: the derived path must be the REGISTERED worktree of that branch."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    worktree = tmp_path.parent / f"{jaxflow_common.slugify_project(tmp_path.name)}-feat-x"
    worktree.mkdir()
    # git knows this path, but as the worktree of a DIFFERENT branch
    fake_run, calls = _switch_aware_runner(tmp_path, sha, overrides={
        ("git", "worktree", "list"): _completed(
            0, f"worktree {worktree}\nHEAD {'f' * 40}\nbranch refs/heads/other\n\n")})
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=lambda e: None,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("git worktree remove") for f in flat)
    assert not any(f == "git branch -d feat/x" for f in flat)
    assert worktree.is_dir(), "an unregistered path is never removed"


def test_merge_commit_failure_aborts_and_never_audits_or_pushes(tmp_path, monkeypatch):
    """F3: this repo's own commit-msg hook is exactly the failure class this guards."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    fake_run, calls = _switch_aware_runner(tmp_path, sha, overrides={
        ("git", "commit",): _completed(1, "hook refused: Co-Authored-By is banned")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "merge-failed"
    assert "hook refused" in exc.value.hint
    flat = [" ".join(c) for c in calls]
    assert "git merge --abort" in flat
    # ZERO cleanup, branch deletion included (cold review F3): nothing was delivered
    assert not any(f.startswith("git push") or f.startswith("git worktree")
                   or f.startswith("git branch -d") for f in flat)


_RELEASE_POLICY = (
    "## Deploy policy\n\n**Preset: `dual-branch`** — notes.\n\n"
    "Delivery target: `release`.\n"
)


_TARGET_512 = "ab/" * 170 + "cd"


@pytest.mark.parametrize("release_carries_merge", [True, False])
def test_merge_refuses_when_approved_target_differs_from_policy(
        tmp_path, monkeypatch, release_carries_merge):
    """MOA-458: approve main, policy now release — refuse before resume or fresh delivery."""
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    status = tmp_path / ".jax-os" / "status.md"
    status.parent.mkdir()
    original = "---\nproject: Demo\nstage: review\n---\n\n## Now\nOld.\n"
    status.write_text(original, encoding="utf-8")
    script = {
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "rev-parse", "--verify", "release^2"): _completed(
            0 if release_carries_merge else 128,
            f"{sha}\n" if release_carries_merge else ""),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{sha}\n"),
        ("git", "status", "--porcelain"): _completed(0, ""),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/release\n"),
    }
    fake_run, calls = _merge_runner(tmp_path, script=script, agents=_RELEASE_POLICY)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha, target="main"), run=fake_run,
                          post=_never_post, env=_merge_env(), now=_fixed_now,
                          allowlist_root=tmp_path.parent)
    assert exc.value.code == "target-mismatch"
    assert "main" in exc.value.hint and "release" in exc.value.hint
    _assert_no_delivery(calls, policy_target="release")
    assert status.read_text(encoding="utf-8") == original


def test_merge_happy_path_with_matching_non_main_policy_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha, posted = "a" * 40, []
    _agents(tmp_path, _TEMPLATE_DUAL_BRANCH_BLOCK)
    fake_run, calls = _switch_aware_runner(tmp_path, sha, target="staging")
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha, target="staging"), run=fake_run,
                      post=posted.append, env=_merge_env(), now=_fixed_now,
                      allowlist_root=tmp_path.parent)
    assert ["git", "switch", "staging"] in calls
    assert posted[0]["payload"]["target"] == "staging"


@pytest.mark.parametrize("target", [
    None, "", " main", "main ", "ma in", "-main", "x" * 513, "a\udcff",
])
def test_merge_refuses_an_unusable_approved_target(tmp_path, monkeypatch, target):
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(
        tmp_path, script={("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(target=target), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "target-invalid"
    _assert_no_delivery(calls, policy_target="main")
    assert not any(c[:2] == ["git", "check-ref-format"] for c in calls)


def test_merge_direct_call_missing_target_is_target_invalid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(
        tmp_path, script={("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_merge_args_without_target(), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "target-invalid"
    _assert_no_delivery(calls, policy_target="main")


@pytest.mark.parametrize("target,code", [
    ("main^", "target-invalid"),
    ("main..x", "target-invalid"),
    ("main@{1}", "target-invalid"),
    ("feat/foo/bar", "target-mismatch"),
    ("feat/café", "target-mismatch"),
    (_TARGET_512, "target-mismatch"),
])
def test_merge_approved_target_uses_real_check_ref_format(tmp_path, monkeypatch, target, code):
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_run_real_ref_format(
        tmp_path, script={("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(target=target), run=fake_run, post=_never_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == code
    assert ["git", "check-ref-format", f"refs/heads/{target}"] in calls
    _assert_no_delivery(calls, policy_target="main")


@pytest.mark.parametrize("override,code", [
    pytest.param({"sha": "abc1234"}, "sha-mismatch", id="sha"),
    pytest.param({"branch": ""}, "branch-invalid", id="branch"),
    pytest.param({"phase": ""}, "phase-invalid", id="phase"),
])
def test_merge_gates_win_over_an_invalid_target(tmp_path, monkeypatch, override, code):
    monkeypatch.chdir(tmp_path)
    fake_run, calls = _merge_runner(
        tmp_path, script={("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(target="-nope", **override), run=fake_run,
                          post=_never_post, env=_merge_env(), now=_fixed_now,
                          allowlist_root=tmp_path.parent)
    assert exc.value.code == code
    _assert_no_mutation(calls)
    assert not any(c[:2] == ["git", "check-ref-format"] for c in calls)


def test_merge_identical_retry_resumes_after_push_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    posted = []
    fake_run, calls = _switch_aware_runner(
        tmp_path, sha, overrides={("git", "push"): _completed(1, "rejected")})
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=posted.append,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "push-failed"
    assert any(" ".join(c).startswith("git commit") for c in calls)

    posted2 = []
    fake_run2, calls2 = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{sha}\n"),
        ("git", "remote", "get-url"): _completed(0, "git@github:x/y.git\n"),
    })
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run2, post=posted2.append,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    flat2 = [" ".join(c) for c in calls2]
    for banned in ("git merge --no-ff", "git commit", "/bin/bash -lc", "git write-tree",
                   "git switch"):
        assert not any(f.startswith(banned) for f in flat2), f"{banned} ran on a resume"
    assert posted2 and posted2[0]["type"] == "merge-approved"
    assert any(f.startswith("git push") for f in flat2)


def test_merge_identical_retry_resumes_after_audit_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40

    def failing_post(event):
        raise RuntimeError("hub down")

    fake_run, calls = _switch_aware_runner(tmp_path, sha)
    with pytest.raises(ji.Refusal) as exc:
        jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run, post=failing_post,
                          env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    assert exc.value.code == "hub-unreachable"
    assert any(" ".join(c).startswith("git commit") for c in calls)

    posted2 = []
    fake_run2, calls2 = _merge_runner(tmp_path, script={
        ("git", "rev-parse", "--show-toplevel"): _completed(0, f"{tmp_path}\n"),
        ("git", "symbolic-ref"): _completed(0, "refs/remotes/origin/main\n"),
        ("git", "rev-parse", "--verify", "main^2"): _completed(0, f"{sha}\n"),
        ("git", "rev-parse", "main^{commit}"): _completed(0, "c" * 40 + "\n"),
        ("git", "log", "-1", "--format=%s"): _completed(0, "feat: Phase X (merge feat/x)\n"),
        ("git", "rev-parse", "--verify", "feat/x^{commit}"): _completed(0, f"{sha}\n"),
        ("git", "remote", "get-url"): _completed(2, ""),
    })
    jaxflow_merge.cmd_merge(_MergeArgs(sha=sha), run=fake_run2, post=posted2.append,
                      env=_merge_env(), now=_fixed_now, allowlist_root=tmp_path.parent)
    flat2 = [" ".join(c) for c in calls2]
    for banned in ("git merge --no-ff", "git commit", "/bin/bash -lc", "git write-tree"):
        assert not any(f.startswith(banned) for f in flat2), f"{banned} ran on a resume"
    assert posted2 and posted2[0]["type"] == "merge-approved"


def test_merge_subparser_requires_every_flag_including_target_and_rejects_message():
    args = jaxflow_cli.parse_args(["merge", "feat/x", "--sha", "a" * 40,
                              "--phase", "Phase X", "--checks", "pytest -q",
                              "--target", "main"])
    assert (args.command, args.branch, args.sha) == ("merge", "feat/x", "a" * 40)
    assert (args.phase, args.checks, args.target) == ("Phase X", "pytest -q", "main")
    for missing in (["merge", "feat/x", "--sha", "a" * 40, "--phase", "P", "--target", "main"],
                    ["merge", "feat/x", "--sha", "a" * 40, "--checks", "true", "--target", "main"],
                    ["merge", "--sha", "a" * 40, "--phase", "P", "--checks", "true",
                     "--target", "main"],
                    ["merge", "feat/x", "--sha", "a" * 40, "--phase", "P", "--checks", "true"]):
        with pytest.raises(SystemExit):
            jaxflow_cli.parse_args(missing)
    with pytest.raises(SystemExit):
        jaxflow_cli.parse_args(["merge", "feat/x", "--sha", "a" * 40, "--phase", "P",
                            "--checks", "true", "--target", "main", "--message", "x"])


def test_merge_rejects_a_blank_checks_command():
    # `required=True` only demands the flag; `bash -lc ""` exits 0, so a blank value would
    # let the merge commit and audit with nothing verified (whole-branch review, F1).
    for blank in ("", "   ", "\t", "\n", " \t\n "):
        with pytest.raises(SystemExit):
            jaxflow_cli.parse_args(["merge", "feat/x", "--sha", "a" * 40,
                                "--phase", "P", "--checks", blank, "--target", "main"])
    # A command that merely LOOKS trivial is the caller's business, not the parser's.
    assert jaxflow_cli.parse_args(["merge", "feat/x", "--sha", "a" * 40,
                               "--phase", "P", "--checks", "true",
                               "--target", "main"]).checks == "true"


# ---- MOA-510: merge reuses the test evidence a diff review recorded ----

_REUSE_SHA = "a" * 40


def _tests_txt(*frames):
    """One COMMAND/output/EXIT frame per (command, exit_code), the shape
    `_write_verify_tests_file` writes and `jr.parse_tests_frames` reads."""
    return "".join(f"COMMAND: {cmd}\nsome output\nEXIT: {code}\n" for cmd, code in frames)


def _seed_reusable_review(tmp_path, db, *, branch="feat/x", verify="pnpm test", build="pnpm build",
                          tests="ok", review_head=_REUSE_SHA, review_id="e" * 12, review_ts="t2",
                          review_branch=None, manifest=True, with_builder=True):
    """Seeds everything `_merge_checks_reuse` reads: a finished builder run (started payload
    carries `verify`/`build`), a diff review run-started row for `branch`, that review's
    manifest (`kind: diff`, `builder_run_id`, `verify.head_sha`) under
    `<tmp_path>/.local/runs/<review_id>/`, and the build worktree's tests.txt under
    `<worktree>/.local/reports/<builder>.tests.txt`.
    `tests`: "ok" = both frames EXIT 0; None = no evidence file; any other str = written raw.
    `manifest`: True = regular file; False = none; "symlink" = `manifest.json` is a symlink to a
    valid manifest OUTSIDE the run directory (the escape the helper must refuse)."""
    project = jaxflow_common.slugify_project(tmp_path.name)
    builder_id = "b" * 12
    con = sqlite3.connect(db)
    if not con.execute("SELECT 1 FROM sqlite_master WHERE name = 'workflow_events'").fetchone():
        con.close()
        con = _fresh_db(db)
    if with_builder:
        started = {"kind": "build", "target": branch, "repo": str(tmp_path), "verify": verify}
        if build:
            started["build"] = build
        _insert(con, builder_id, project, "builder", "run-started", started, ts="t0")
        _insert(con, builder_id, project, "builder", "run-finished",
                {"exit_code": 0, "contract_status": "ok", "head_sha": _REUSE_SHA}, ts="t0")
    _insert(con, review_id, project, "reviewer", "run-started",
            {"kind": "diff", "target": review_branch or branch, "repo": str(tmp_path),
             "builder_run_id": builder_id}, ts=review_ts)
    con.close()
    if manifest:
        mdir = tmp_path / ".local" / "runs" / review_id
        mdir.mkdir(parents=True, exist_ok=True)
        body = json.dumps({
            "kind": "diff", "builder_run_id": builder_id,
            "verify": {"mode": "rerun", "reason": "head-sha-changed",
                       "source_build_run_id": builder_id, "head_sha": review_head},
        })
        if manifest == "symlink":
            outside = tmp_path / "outside-manifest.json"
            outside.write_text(body, encoding="utf-8")
            (mdir / "manifest.json").symlink_to(outside)
        else:
            (mdir / "manifest.json").write_text(body, encoding="utf-8")
    worktree = _worktree_path(tmp_path, branch)
    if tests is not None:
        reports = worktree / ".local" / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        text = _tests_txt((verify, 0), (build, 0)) if tests == "ok" else tests
        (reports / f"{builder_id}.tests.txt").write_text(text, encoding="utf-8")
    else:
        worktree.mkdir(parents=True, exist_ok=True)


def _reuse_run(worktree, *, porcelain="", registered=True):
    listing = f"worktree {worktree}\nbranch refs/heads/feat/x\n" if registered else ""

    def run(argv, cwd=None):
        if argv[:3] == ["git", "worktree", "list"]:
            return _completed(0, listing)
        if argv == ["git", "status", "--porcelain", "--untracked-files=all"]:
            return _completed(0, porcelain)
        raise AssertionError(f"unexpected git call: {argv}")
    return run


def _call_reuse(tmp_path, db, *, checks="pnpm test", sha=_REUSE_SHA, recheck=False, **run_kw):
    worktree = _worktree_path(tmp_path)
    return jaxflow_merge._merge_checks_reuse(
        _reuse_run(worktree, **run_kw), tmp_path, jaxflow_common.slugify_project(tmp_path.name),
        worktree, "feat/x", sha, checks, allowlist_root=tmp_path.parent, recheck=recheck,
        db_path=db)


def test_merge_checks_reuse_accepts_matching_evidence(tmp_path):
    db = tmp_path / "jaxos.db"
    _seed_reusable_review(tmp_path, db)
    assert _call_reuse(tmp_path, db) == (True, "all-conditions-met", "e" * 12)


def test_merge_checks_reuse_accepts_a_legacy_build_without_a_build_command(tmp_path):
    db = tmp_path / "jaxos.db"
    _seed_reusable_review(tmp_path, db, build=None, tests=_tests_txt(("pnpm test", 0)))
    assert _call_reuse(tmp_path, db) == (True, "all-conditions-met", "e" * 12)


@pytest.mark.parametrize("seed,call,reason", [
    ({}, {"recheck": True}, "recheck-forced"),
    ({"review_head": "9" * 40}, {}, "head-sha-changed"),
    ({}, {"sha": "9" * 40}, "head-sha-changed"),
    ({}, {"checks": "pnpm other"}, "verify-command-changed"),
    ({}, {"porcelain": "?? scratch.txt\n"}, "worktree-dirty"),
    ({}, {"porcelain": " M src/a.ts\n"}, "worktree-dirty"),
    ({"tests": None}, {}, "tests-file-missing"),
    ({"tests": "garbage\n"}, {}, "frame-1-missing"),
    ({"tests": _tests_txt(("pnpm test", 0), ("pnpm build", 1))}, {}, "prior-verify-failed"),
    ({"manifest": False}, {}, "manifest-unreadable"),
    ({"manifest": "symlink"}, {}, "lookup-failed"),
    ({"with_builder": False}, {}, "no-build-run"),
    ({"review_branch": "feat/other"}, {}, "no-diff-review"),
    ({}, {"registered": False}, "worktree-missing"),
])
def test_merge_checks_reuse_reasons_when_not_reusable(tmp_path, seed, call, reason):
    db = tmp_path / "jaxos.db"
    _seed_reusable_review(tmp_path, db, **seed)
    assert _call_reuse(tmp_path, db, **call) == (False, reason, None)


def test_merge_checks_reuse_refuses_a_failed_verify_masked_by_a_passing_build(tmp_path):
    # Review Focus 1 (spec F1): the review's own failed rerun leaves `verify EXIT 1` then
    # `build EXIT 0` before `verify-failed`. The frames must be parsed with BOTH commands, so
    # the build frame's EXIT 0 can never stand in for the verify frame's EXIT 1.
    db = tmp_path / "jaxos.db"
    _seed_reusable_review(
        tmp_path, db, tests=_tests_txt(("pnpm test", 1), ("pnpm build", 0)))
    assert _call_reuse(tmp_path, db) == (False, "prior-verify-failed", None)


def test_merge_checks_reuse_uses_only_the_newest_diff_review_of_the_branch(tmp_path):
    # Review Focus 5: the OLDER review matches the SHA, the NEWER one does not -> not reusable.
    db = tmp_path / "jaxos.db"
    _seed_reusable_review(tmp_path, db, review_id="d" * 12, review_ts="t1")
    _seed_reusable_review(tmp_path, db, review_id="f" * 12, review_ts="t2", review_head="9" * 40)
    assert _call_reuse(tmp_path, db) == (False, "head-sha-changed", None)
    # And the other way round: an older stale review does not hide a newer matching one.
    db2 = tmp_path / "jaxos2.db"
    _seed_reusable_review(tmp_path, db2, review_id="d" * 12, review_ts="t1", review_head="9" * 40)
    _seed_reusable_review(tmp_path, db2, review_id="f" * 12, review_ts="t2")
    assert _call_reuse(tmp_path, db2) == (True, "all-conditions-met", "f" * 12)


def test_merge_checks_reuse_never_raises_on_an_unreadable_ledger(tmp_path):
    # Review Focus 3: a lookup failure is "not reusable", never an exception or a refusal.
    worktree = _worktree_path(tmp_path)
    worktree.mkdir(parents=True)
    assert _call_reuse(tmp_path, tmp_path / "missing.db") == (False, "lookup-failed", None)
    bad = tmp_path / "bad.db"
    bad.write_text("not a sqlite file", encoding="utf-8")
    assert _call_reuse(tmp_path, bad) == (False, "lookup-failed", None)


def test_merge_checks_reuse_asks_git_for_untracked_files_explicitly(tmp_path):
    # Review F2: plain `git status --porcelain` obeys `status.showUntrackedFiles`; with it off a
    # worktree holding untracked files would read clean. The argv must override the config.
    db = tmp_path / "jaxos.db"
    _seed_reusable_review(tmp_path, db)
    worktree = _worktree_path(tmp_path)
    seen = []
    inner = _reuse_run(worktree)

    def run(argv, cwd=None):
        seen.append(list(argv))
        return inner(argv, cwd)

    jaxflow_merge._merge_checks_reuse(
        run, tmp_path, jaxflow_common.slugify_project(tmp_path.name), worktree, "feat/x", _REUSE_SHA,
        "pnpm test", allowlist_root=tmp_path.parent, db_path=db)
    assert ["git", "status", "--porcelain", "--untracked-files=all"] in seen


def test_merge_checks_reuse_treats_a_non_dict_verify_field_as_an_unreadable_manifest(tmp_path):
    db = tmp_path / "jaxos.db"
    _seed_reusable_review(tmp_path, db)
    (tmp_path / ".local" / "runs" / ("e" * 12) / "manifest.json").write_text(
        json.dumps({"kind": "diff", "builder_run_id": "b" * 12, "verify": "oops"}),
        encoding="utf-8")
    assert _call_reuse(tmp_path, db) == (False, "manifest-unreadable", None)


def _pr_stateful_run(fake_run, calls, open_view):
    """`gh pr view` answers OPEN once, then MERGED (what the real merge does); every other
    call goes to the scripted fake."""
    seen = {"n": 0}

    def run(argv, cwd=None):
        if tuple(argv[:3]) == ("gh", "pr", "view"):
            calls.append(list(argv))
            seen["n"] += 1
            if seen["n"] == 1:
                return _completed(0, json.dumps(open_view))
            return _completed(0, json.dumps(
                {**open_view, "state": "MERGED", "mergeCommit": {"oid": "d" * 40}}))
        return fake_run(argv, cwd)
    return run


def _run_pr_merge(tmp_path, monkeypatch, *, seed=None, setup_kw=None, dirty_untracked=False,
                  dirty_tracked=False, **args_kw):
    """Runs the PR-path merge of feat/x. Returns (rc_or_Refusal, calls, events)."""
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db, **(setup_kw or {}))
    if seed is not None:
        _seed_reusable_review(tmp_path, db, **seed)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    inner = _pr_stateful_run(fake_run, calls, json.loads(script[("gh", "pr", "view")].stdout))

    def run(argv, cwd=None):
        if argv[:3] == ["git", "status", "--porcelain"]:
            if argv[3:] == ["--untracked-files=no"] and dirty_tracked:
                return _completed(0, " M tracked.txt\n")
            if argv[3:] == ["--untracked-files=all"] and dirty_untracked:
                return _completed(0, "?? scratch.txt\n")
        return inner(argv, cwd)

    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        try:
            rc = jaxflow_merge.cmd_merge(
                _MergeArgs(target="staging", checks="pnpm test", **args_kw), run=run,
                post=_ledger_post(db, events), env=_merge_env(), now=_fixed_now,
                allowlist_root=tmp_path.parent)
        except ji.Refusal as exc:
            rc = exc
    return rc, calls, events


def _bash_ran(calls):
    return any(c[:2] == ["/bin/bash", "-lc"] for c in calls)


def test_merge_pr_reuses_the_review_evidence_and_audits_it(tmp_path, monkeypatch, capsys):
    rc, calls, events = _run_pr_merge(tmp_path, monkeypatch, seed={})
    assert rc == jaxflow_common.OK
    assert not _bash_ran(calls), "the checks command must not run when the evidence is reusable"
    assert events[-1]["payload"]["checks"] == {
        "mode": "reused", "source_review_run_id": "e" * 12, "head_sha": _REUSE_SHA}
    assert any(c[:3] == ["gh", "pr", "merge"] for c in calls), "the merge itself still happens"
    assert capsys.readouterr().out.strip().splitlines()[-1] == f"checks: reused from review {'e' * 12} at {'a' * 12}"


@pytest.mark.parametrize("seed,args_kw", [
    pytest.param(None, {}, id="no-diff-review"),
    pytest.param({"review_head": "9" * 40}, {}, id="head-moved-since-review"),
    pytest.param({"verify": "pnpm other"}, {}, id="checks-differ-from-recorded-verify"),
    pytest.param({"tests": None}, {}, id="evidence-missing"),
    pytest.param({"manifest": "symlink"}, {}, id="manifest-symlink-escape"),
    pytest.param({"tests": _tests_txt(("pnpm test", 0), ("pnpm build", 1))}, {}, id="build-frame-failed"),
    pytest.param({"tests": _tests_txt(("pnpm test", 1), ("pnpm build", 0))}, {}, id="masked-verify"),
    pytest.param({}, {"recheck": True}, id="recheck"),
])
def test_merge_pr_runs_the_checks_when_reuse_does_not_hold(tmp_path, monkeypatch, capsys, seed, args_kw):
    rc, calls, events = _run_pr_merge(tmp_path, monkeypatch, seed=seed, **args_kw)
    assert rc == jaxflow_common.OK
    assert _bash_ran(calls)
    assert events[-1]["payload"]["checks"] == {"mode": "run"}
    assert capsys.readouterr().out.strip().splitlines()[-1] == "checks: pnpm test exit 0"


def test_merge_pr_untracked_dirt_runs_the_checks_without_a_refusal(tmp_path, monkeypatch):
    # Review Focus 3: the PR guard ignores untracked files, so untracked dirt only disables reuse.
    rc, calls, events = _run_pr_merge(tmp_path, monkeypatch, seed={}, dirty_untracked=True)
    assert rc == jaxflow_common.OK
    assert _bash_ran(calls)
    assert events[-1]["payload"]["checks"] == {"mode": "run"}


def test_merge_pr_tracked_dirt_still_refuses_even_with_reusable_evidence(tmp_path, monkeypatch):
    rc, calls, _ = _run_pr_merge(tmp_path, monkeypatch, seed={}, dirty_tracked=True)
    assert isinstance(rc, ji.Refusal) and rc.code == "dirty-tracked-tree"
    assert not _bash_ran(calls)


def test_merge_pr_missing_build_worktree_still_refuses_checks_failed(tmp_path, monkeypatch):
    rc, calls, _ = _run_pr_merge(tmp_path, monkeypatch, seed={}, setup_kw={"worktree_registered": False})
    assert isinstance(rc, ji.Refusal) and rc.code == "checks-failed"


def test_merge_pr_failing_checks_still_refuse_when_reuse_does_not_hold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    script[("/bin/bash", "-lc")] = _completed(1, "boom")
    _seed_reusable_review(tmp_path, db, review_head="9" * 40)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        with pytest.raises(ji.Refusal) as exc:
            jaxflow_merge.cmd_merge(_MergeArgs(target="staging", checks="pnpm test"), run=fake_run,
                              post=_never_post, env=_merge_env(), now=_fixed_now,
                              allowlist_root=tmp_path.parent)
    assert exc.value.code == "checks-failed"
    assert not any(c[:3] == ["gh", "pr", "merge"] for c in calls)


@pytest.mark.parametrize("recheck", [False, True])
def test_merge_pr_already_merged_recovery_audits_checks_resumed(tmp_path, monkeypatch, capsys, recheck):
    # Review Focus 4: nothing runs and nothing is reused on the recovery arm; --recheck is inert.
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db)
    script[("gh", "pr", "view")] = _completed(0, json.dumps(_GH_PR_VIEW_MERGED))
    _seed_reusable_review(tmp_path, db)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        rc = jaxflow_merge.cmd_merge(
            _MergeArgs(target="staging", checks="pnpm test", recheck=recheck), run=fake_run,
            post=_ledger_post(db, events), env=_merge_env(), now=_fixed_now,
            allowlist_root=tmp_path.parent)
    assert rc == jaxflow_common.OK
    assert not _bash_ran(calls)
    assert events[-1]["payload"]["checks"] == {"mode": "resumed"}
    assert not any(line.startswith("checks:") for line in capsys.readouterr().out.splitlines())


def test_merge_pr_release_head_never_consults_the_reuse_lookup(tmp_path, monkeypatch):
    # D3: a release/* head has no build worktree and no diff review; it always runs.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jaxflow_merge, "_merge_checks_reuse",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("reuse consulted")))
    db = tmp_path / "jaxos.db"
    script, _ = _pr_merge_setup(tmp_path, db, branch="release/x", target="main",
                                worktree_registered=False)
    fake_run, calls = _merge_runner(tmp_path, agents=_DUAL_PR_AGENTS, script=script)
    inner = _pr_stateful_run(fake_run, calls, json.loads(script[("gh", "pr", "view")].stdout))
    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        rc = jaxflow_merge.cmd_merge(
            _MergeArgs(branch="release/x", target="main", checks="pnpm test"), run=inner,
            post=_ledger_post(db, events), env=_merge_env(), now=_fixed_now,
            allowlist_root=tmp_path.parent)
    assert rc == jaxflow_common.OK
    assert _bash_ran(calls)
    assert events[-1]["payload"]["checks"] == {"mode": "run"}


def test_merge_subparser_accepts_recheck_and_defaults_it_off():
    base = ["merge", "feat/x", "--sha", "a" * 40, "--phase", "P", "--checks", "true",
            "--target", "main"]
    assert jaxflow_cli.parse_args(base).recheck is False
    assert jaxflow_cli.parse_args(base + ["--recheck"]).recheck is True


def test_merge_still_requires_checks_even_with_recheck():
    with pytest.raises(SystemExit):
        jaxflow_cli.parse_args(["merge", "feat/x", "--sha", "a" * 40, "--phase", "P",
                            "--target", "main", "--recheck"])


def _run_local_ff(tmp_path, monkeypatch, *, seed=None, make_worktree=True, dirty=None, bash=None,
                  review_target_moved=False, **args_kw):
    """Local-preset merge where the target has not moved (`merge-base --is-ancestor` says yes).
    Returns (rc_or_Refusal, calls, events). `dirty` is the `git status --porcelain` text the
    BUILD worktree reports (the control repo's own preflight status stays clean)."""
    monkeypatch.chdir(tmp_path)
    sha = _REUSE_SHA
    db = tmp_path / "jaxos.db"
    _fresh_db(db).close()
    worktree = _worktree_path(tmp_path)
    if make_worktree:
        worktree.mkdir(parents=True)
    if seed is not None:
        _seed_reusable_review(tmp_path, db, **seed)
    overrides = {("git", "merge-base", "--is-ancestor"):
                 _completed(1 if review_target_moved else 0, "")}
    if bash is not None:
        overrides[("/bin/bash", "-lc")] = bash
    base, calls = _switch_aware_runner(tmp_path, sha, overrides=overrides)

    def run(argv, cwd=None):
        if (dirty and argv[:3] == ["git", "status", "--porcelain"] and cwd is not None
                and Path(cwd) == worktree):
            return _completed(0, dirty)
        return base(argv, cwd=cwd)

    events = []
    with monkeypatch.context() as m:
        m.setattr(jr, "DB_PATH", db)
        try:
            rc = jaxflow_merge.cmd_merge(
                _MergeArgs(sha=sha, checks="pnpm test", **args_kw), run=run,
                post=lambda e: events.append(e) or {"ok": True}, env=_merge_env(),
                now=_fixed_now, allowlist_root=tmp_path.parent)
        except ji.Refusal as exc:
            rc = exc
    return rc, calls, events


def test_merge_local_fast_forward_reuses_the_review_evidence_and_prints_why(tmp_path, monkeypatch, capsys):
    rc, calls, events = _run_local_ff(tmp_path, monkeypatch, seed={})
    assert rc == jaxflow_common.OK
    flat = [" ".join(c) for c in calls]
    assert not any(f.startswith("/bin/bash -lc") for f in flat), "no checks call on reuse"
    assert not any(f.startswith("git diff --quiet") for f in flat)
    assert any(f.startswith("git commit") for f in flat), "the merge commit must still be made"
    assert capsys.readouterr().out.strip().splitlines()[-2:] == [
        f"merged {'c' * 40} pushed origin/main",
        f"checks: reused from review {'e' * 12} at {'a' * 12}",
    ]
    assert events[-1]["payload"]["checks"] == {
        "mode": "reused", "source_review_run_id": "e" * 12, "head_sha": _REUSE_SHA}


@pytest.mark.parametrize("kw", [
    pytest.param({"seed": None}, id="no-diff-review"),
    pytest.param({"seed": {"review_head": "9" * 40}}, id="head-moved-since-review"),
    pytest.param({"seed": {"verify": "pnpm other"}}, id="checks-differ-from-recorded-verify"),
    pytest.param({"seed": {"tests": None}}, id="evidence-missing"),
    pytest.param({"seed": {"manifest": "symlink"}}, id="manifest-symlink-escape"),
    pytest.param({"seed": {"tests": _tests_txt(("pnpm test", 1), ("pnpm build", 0))}}, id="masked-verify"),
    pytest.param({"seed": {}, "dirty": " M src/a.ts\n"}, id="dirty-tracked"),
    pytest.param({"seed": {}, "dirty": "?? scratch.txt\n"}, id="dirty-untracked"),
    pytest.param({"seed": {}, "recheck": True}, id="recheck"),
    pytest.param({"seed": {}, "review_target_moved": True}, id="target-moved"),
])
def test_merge_local_fast_forward_runs_the_checks_when_reuse_does_not_hold(tmp_path, monkeypatch, capsys, kw):
    rc, calls, events = _run_local_ff(tmp_path, monkeypatch, **kw)
    assert rc == jaxflow_common.OK, "a failed reuse lookup must never refuse the merge"
    assert any(" ".join(c).startswith("/bin/bash -lc pnpm test") for c in calls)
    assert events[-1]["payload"]["checks"] == {"mode": "run"}
    assert capsys.readouterr().out.strip().splitlines()[-1] == "checks: pnpm test exit 0"


def test_merge_local_fast_forward_runs_the_checks_when_the_build_worktree_is_missing(tmp_path, monkeypatch):
    # Review Focus 3: the local path has no build-worktree prerequisite; a missing one only
    # disables reuse. (`_seed_reusable_review` would create the directory, so seed nothing.)
    rc, calls, events = _run_local_ff(tmp_path, monkeypatch, seed=None, make_worktree=False)
    assert rc == jaxflow_common.OK
    assert any(" ".join(c).startswith("/bin/bash -lc") for c in calls)
    assert events[-1]["payload"]["checks"] == {"mode": "run"}


def test_merge_local_fast_forward_after_a_post_review_commit_runs_and_refuses_on_failure(tmp_path, monkeypatch):
    # Review Focus 2 / D4 acceptance: a branch with a commit after its last diff review no
    # longer merges untested.
    rc, calls, events = _run_local_ff(
        tmp_path, monkeypatch, seed={"review_head": "9" * 40}, bash=_completed(1, "boom"))
    assert isinstance(rc, ji.Refusal) and rc.code == "checks-failed"
    flat = [" ".join(c) for c in calls]
    assert any(f.startswith("git merge --abort") for f in flat), "the merge must be aborted"
    assert not any(f.startswith("git commit") for f in flat)
    assert events == [], "nothing may be audited for a refused merge"


def test_main_routes_merge_and_maps_a_refusal_to_exit_2(monkeypatch, capsys):
    seen = {}

    def fake_cmd_merge(args, **kwargs):
        seen["branch"] = args.branch
        return jaxflow_common.OK

    monkeypatch.setattr(jaxflow_merge, "cmd_merge", fake_cmd_merge)
    argv = ["merge", "feat/x", "--sha", "a" * 40, "--phase", "P", "--checks", "true",
            "--target", "main"]
    assert jaxflow_cli.main(argv) == jaxflow_common.OK
    assert seen["branch"] == "feat/x"

    def refusing(args, **kwargs):
        exc = ji.Refusal("preset-unsupported")
        exc.hint = "hint: PR preset"
        raise exc

    monkeypatch.setattr(jaxflow_merge, "cmd_merge", refusing)
    assert jaxflow_cli.main(argv) == jaxflow_common.REFUSED
    err = capsys.readouterr().err
    assert "preset-unsupported" in err and "hint: PR preset" in err
