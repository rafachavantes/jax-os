import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import jax_init as ji

def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)

def _init_repo(root: Path, branch="main"):
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "t@t.test")
    _git(root, "config", "user.name", "t")
    if branch != "main":
        _git(root, "checkout", "-b", branch)

def _commit_file(root, name, content="x\n"):
    (root / name).write_text(content, encoding="utf-8")
    _git(root, "add", name)
    _git(root, "commit", "-m", f"add {name}")

def _run_real(argv, cwd=None):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "jax-init-test",
        "GIT_AUTHOR_EMAIL": "jax-init-test@example.invalid",
        "GIT_COMMITTER_NAME": "jax-init-test",
        "GIT_COMMITTER_EMAIL": "jax-init-test@example.invalid",
    }
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False, env=env)

def _fixed_now():
    return datetime(2026, 9, 6, 10, 0, 0, tzinfo=timezone(timedelta(hours=-3)))

def _argv(path, name="Demo", preset="single-branch", stack="other", threat_model="internal-single-user", remote=False, dry_run=False):
    a = ["--path", str(path), "--name", name, "--preset", preset, "--stack", stack, "--threat-model", threat_model]
    if remote:
        a.append("--remote")
    if dry_run:
        a.append("--dry-run")
    return a

def _main(argv, **kwargs):
    kwargs.setdefault("env", {})  # never leak this session's own CLAUDECODE/CODEX_THREAD_ID
    return ji.main(argv, **kwargs)

class FakeRun:
    def __init__(self):
        self.calls = []
        self.cwds = []
        self.fail = set()
    def __call__(self, argv, cwd=None):
        self.calls.append(list(argv))
        self.cwds.append(cwd)
        joined = " ".join(argv)
        if any(f in joined for f in self.fail):
            return SimpleNamespace(returncode=1, stdout="", stderr="fail")
        if argv[:3] == ["git", "rev-parse", "--show-toplevel"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="not a repo")
        if argv[:2] == ["git", "symbolic-ref"]:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        if argv[:3] == ["git", "rev-parse", "--verify"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

def test_allowlist_escape_outside_root_and_symlinked_child(tmp_path):
    root = tmp_path / "repos"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    fake = FakeRun()
    out = _main(_argv(outside / "proj"), run=fake, allowlist_root=root)
    assert out == ji.REFUSED
    evil = root / "evil-link"
    evil.symlink_to(outside, target_is_directory=True)
    out2 = _main(_argv(evil), run=fake, allowlist_root=root)
    assert out2 == ji.REFUSED
    assert not any(outside.iterdir())

def test_non_empty_non_git_refusal(tmp_path, capsys):
    root = tmp_path / "repos"
    target = root / "proj"
    target.mkdir(parents=True)
    (target / "stray.txt").write_text("x", encoding="utf-8")
    fake = FakeRun()
    out = _main(_argv(target), run=fake, allowlist_root=root)
    assert out == ji.REFUSED
    payload = json.loads(capsys.readouterr().out)
    assert payload["refused"] == "path-not-empty"

def test_nested_path_refusal(tmp_path, capsys):
    root = tmp_path / "repos"
    outer = root / "outer"
    _init_repo(outer)
    _commit_file(outer, "README")
    nested = outer / "sub" / "inner"
    out = _main(_argv(nested), run=_run_real, allowlist_root=root)
    assert out == ji.REFUSED
    payload = json.loads(capsys.readouterr().out)
    assert payload["refused"] == "path-nested-in-repo"
    assert not nested.exists()

@pytest.mark.parametrize("bad_name", ["bad\nname", "", "a\x7fb", "a\x85b"])
def test_invalid_name_rejected(tmp_path, capsys, bad_name):
    root = tmp_path / "repos"
    out = _main(_argv(root / "x", name=bad_name), run=FakeRun(), allowlist_root=root)
    assert out == ji.REFUSED
    assert json.loads(capsys.readouterr().out)["refused"] == "invalid-name"

def test_detached_head_and_local_failure(tmp_path, capsys):
    root = tmp_path / "repos"
    adopted = root / "adopted"
    _init_repo(adopted)
    _commit_file(adopted, "README")
    _git(adopted, "checkout", "--detach")
    out2 = _main(_argv(adopted), run=_run_real, allowlist_root=root)
    assert out2 == ji.REFUSED
    assert json.loads(capsys.readouterr().out)["refused"] == "detached-head"
    fake = FakeRun()
    fake.fail.add("git init")
    out3 = _main(_argv(root / "y"), run=fake, allowlist_root=root)
    payload3 = json.loads(capsys.readouterr().out)
    assert out3 == ji.REFUSED
    assert payload3["refused"] == "command-failed: git init"

def test_json_schema_has_exact_keys_ok_refused_and_dry_run(tmp_path, capsys):
    root = tmp_path / "repos"
    keys = {"path", "created", "kept", "ran", "refused", "head"}
    _main(_argv(root / "ok-case"), run=_run_real, allowlist_root=root)
    assert set(json.loads(capsys.readouterr().out).keys()) == keys
    target = root / "occupied"
    target.mkdir(parents=True)
    (target / "stray.txt").write_text("x", encoding="utf-8")
    _main(_argv(target), run=FakeRun(), allowlist_root=root)
    assert set(json.loads(capsys.readouterr().out).keys()) == keys
    fake = FakeRun()
    out = _main(_argv(root / "dry-case", dry_run=True), run=fake, allowlist_root=root)
    assert out == ji.OK
    payload = json.loads(capsys.readouterr().out)
    assert set(payload.keys()) == keys
    assert payload["head"] is None
    assert any(item.startswith("would-run: git init") for item in payload["ran"])
    assert not (root / "dry-case").exists()

# ---- Task 2 ----

def test_fresh_bootstrap_creates_full_baseline(tmp_path, capsys):
    root = tmp_path / "repos"
    target = root / "demo"
    out = _main(_argv(target, name="Demo App"), run=_run_real, allowlist_root=root, now=_fixed_now)
    assert out == ji.OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["refused"] is None
    branch = subprocess.run(["git", "symbolic-ref", "--short", "HEAD"], cwd=target, capture_output=True, text=True)
    assert branch.stdout.strip() == "main"
    gitignore = (target / ".gitignore").read_text()
    for entry in [".local/", ".jax-os/", "AGENTS.md", "CLAUDE.md", ".env*"]:
        assert entry in gitignore.splitlines()
    hooks_path = subprocess.run(
        ["git", "config", "core.hooksPath"], cwd=target, capture_output=True, text=True
    ).stdout.strip()
    assert hooks_path == ".githooks"
    assert (target / ".githooks" / "commit-msg").read_bytes() == ji.HOOK_SOURCE.read_bytes()
    assert os.access(target / ".githooks" / "commit-msg", os.X_OK)
    assert (target / ".local" / "handoff.md").read_text() == ""
    assert (target / "CLAUDE.md").read_text() == "@AGENTS.md\n"
    log = subprocess.run(["git", "log", "--oneline"], cwd=target, capture_output=True, text=True).stdout
    assert log.strip().count("\n") == 0
    assert "chore: bootstrap" in log
    status = (target / ".jax-os" / "status.md").read_text()
    assert status.startswith("---\n")
    assert "project: Demo App" in status
    assert "builder: tech-lead" in status
    assert "branch: main" in status
    assert "gate:" not in status
    assert "## Now\nInitialised by jax-init; first spec pending.\n" in status
    updated_line = [l for l in status.splitlines() if l.startswith("updated:")][0]
    datetime.fromisoformat(updated_line.split(": ", 1)[1])
    for rel in ji.BASELINE_DIRS:
        assert (target / rel).is_dir()
    assert set(payload["created"]) >= {
        ".gitignore", ".githooks/commit-msg", ".local/handoff.md", "CLAUDE.md", ".jax-os/status.md",
    }

def test_adopting_existing_repo_and_second_run_is_create_only(tmp_path, capsys):
    root = tmp_path / "repos"
    target = root / "adopted"
    _init_repo(target)
    (target / ".gitignore").write_text("dist/\n", encoding="utf-8")
    (target / "CLAUDE.md").write_text("custom\n", encoding="utf-8")
    _git(target, "add", ".gitignore", "CLAUDE.md")
    _git(target, "commit", "-m", "init")
    pre_gitignore = (target / ".gitignore").read_bytes()
    out = _main(_argv(target, name="Adopted"), run=_run_real, allowlist_root=root, now=_fixed_now)
    assert out == ji.OK
    payload = json.loads(capsys.readouterr().out)
    assert "CLAUDE.md" in payload["kept"]
    # .gitignore already existed but was missing the base entries, so the
    # merge rule reports it "created" (appended), never "kept" (§2.3).
    assert ".gitignore" in payload["created"]
    gitignore_bytes = (target / ".gitignore").read_bytes()
    assert gitignore_bytes.startswith(pre_gitignore)
    gitignore_lines = gitignore_bytes.decode("utf-8").splitlines()
    for entry in [".local/", ".jax-os/", "AGENTS.md", "CLAUDE.md", ".env*"]:
        assert entry in gitignore_lines
    assert (target / "CLAUDE.md").read_text() == "custom\n"
    log = subprocess.run(["git", "log", "--oneline"], cwd=target, capture_output=True, text=True).stdout
    assert log.strip().count("\n") == 0
    root2 = tmp_path / "repos2"
    target2 = root2 / "fresh"
    _main(_argv(target2, name="Fresh"), run=_run_real, allowlist_root=root2, now=_fixed_now)
    capsys.readouterr()
    tree_a = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=target2, capture_output=True, text=True).stdout
    baseline_files = [".gitignore", ".githooks/commit-msg", ".local/handoff.md", "CLAUDE.md", ".jax-os/status.md"]
    before = {rel: ((target2 / rel).read_bytes(), (target2 / rel).stat().st_mode) for rel in baseline_files}
    out2 = _main(_argv(target2, name="Fresh"), run=_run_real, allowlist_root=root2, now=_fixed_now)
    assert out2 == ji.OK
    payload2 = json.loads(capsys.readouterr().out)
    assert payload2["created"] == []
    assert set(payload2["kept"]) >= {
        ".gitignore", ".githooks/commit-msg", ".local/handoff.md", "CLAUDE.md", ".jax-os/status.md",
        *ji.BASELINE_DIRS,
    }
    tree_b = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=target2, capture_output=True, text=True).stdout
    assert tree_a == tree_b
    after = {rel: ((target2 / rel).read_bytes(), (target2 / rel).stat().st_mode) for rel in baseline_files}
    assert after == before

def test_per_stack_gitignore(tmp_path, capsys):
    root = tmp_path / "repos"
    cases = {
        "nextjs": ["node_modules/", ".next/"],
        "python": [".venv/", "__pycache__/"],
        "bubble": [],
        "other": [],
    }
    for stack, expected in cases.items():
        target = root / f"proj-{stack}"
        _main(_argv(target, stack=stack), run=_run_real, allowlist_root=root, now=_fixed_now)
        capsys.readouterr()
        lines = (target / ".gitignore").read_text().splitlines()
        for entry in expected:
            assert entry in lines
        other_stack_only = (set(cases["nextjs"]) | set(cases["python"])) - set(expected)
        for entry in other_stack_only:
            assert entry not in lines

def test_gitignore_dir_instead_of_file_refuses(tmp_path, capsys):
    root = tmp_path / "repos"
    target = root / "proj"
    _init_repo(target)
    _commit_file(target, "README")
    (target / ".gitignore").mkdir()
    out = _main(_argv(target), run=_run_real, allowlist_root=root, now=_fixed_now)
    assert out == ji.REFUSED
    captured = capsys.readouterr()
    assert captured.out.strip().count("\n") == 0
    payload = json.loads(captured.out)
    assert payload["refused"] == "command-failed: gitignore"

def test_child_symlink_escape(tmp_path, capsys):
    root = tmp_path / "repos"
    sibling = root / "sibling"
    sibling.mkdir(parents=True)
    target = root / "proj"
    target.mkdir(parents=True)
    (target / ".local").symlink_to(sibling, target_is_directory=True)
    out = _main(_argv(target), run=_run_real, allowlist_root=root, now=_fixed_now)
    payload = json.loads(capsys.readouterr().out)
    assert out == ji.REFUSED
    assert payload["refused"] == "path-outside-allowlist"
    assert not any(sibling.iterdir())

def test_detect_builder_reads_env_markers():
    assert ji._detect_builder({}) == "tech-lead"
    assert ji._detect_builder({"CLAUDECODE": "1"}) == "claude-code"
    assert ji._detect_builder({"CODEX_THREAD_ID": "x"}) == "codex"

def test_contained_rejects_sibling_prefix_and_accepts_self_and_descendant(tmp_path):
    root = tmp_path / "repos"
    root.mkdir()
    (root / "x").mkdir()
    sibling = tmp_path / "reposXYZ"
    sibling.mkdir()
    assert ji._contained(root / "x", root) is True
    assert ji._contained(root, root) is True  # equal paths: contained
    assert ji._contained(sibling, root) is False  # string-prefix trap, not a real subpath
    assert ji._contained(tmp_path, root) is False  # parent, not descendant

def test_iso8601_matches_stdlib_isoformat_seconds():
    # `_iso8601` used to hand-splice a colon into strftime's %z; this proves
    # `dt.isoformat(timespec="seconds")` gives byte-identical output for every
    # aware datetime this codebase actually feeds it (UTC, negative offset, and
    # one with microseconds -- both drop to whole seconds the same way).
    for dt in (
        _fixed_now(),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc),
    ):
        assert ji._iso8601(dt) == dt.isoformat(timespec="seconds")

# ---- Task 3 ----

def test_head_reports_real_sha_after_bootstrap_commit(tmp_path, capsys):
    root = tmp_path / "repos"
    target = root / "demo"
    _main(_argv(target), run=_run_real, allowlist_root=root, now=_fixed_now)
    payload = json.loads(capsys.readouterr().out)
    assert payload["head"] and len(payload["head"]) == 40
    real_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=target, capture_output=True, text=True).stdout.strip()
    assert payload["head"] == real_head

def test_secret_guard_is_case_insensitive():
    assert ji._is_secret_path(".ENV")
    assert ji._is_secret_path("foo.PEM")
    assert ji._is_secret_path("id_RSA")

def test_secret_guard_allows_public_env_templates():
    for name in (".env.example", "sub/.env.sample", ".ENV.Template"):
        assert not ji._is_secret_path(name)
    for name in (".env", ".env.local", ".env.production", ".env.example.local"):
        assert ji._is_secret_path(name)


def test_secret_guard_blocks_id_rsa_but_env_is_gitignored(tmp_path):
    root = tmp_path / "repos"
    blocked = root / "blocked"
    _init_repo(blocked)
    (blocked / ".gitignore").write_text(".env*\n", encoding="utf-8")
    (blocked / "id_rsa").write_text("secret", encoding="utf-8")
    state = {"plan": [], "kept": []}
    with pytest.raises(ji.Refusal) as exc:
        ji._run_bootstrap_sequence(blocked, _run_real, "single-branch", False, False, state)
    assert exc.value.code == "secret-detected: id_rsa"
    log = subprocess.run(["git", "log", "--oneline"], cwd=blocked, capture_output=True, text=True).stdout
    assert log.strip() == ""
    ok = root / "ok"
    _init_repo(ok)
    (ok / ".gitignore").write_text(".env*\n", encoding="utf-8")
    (ok / ".env").write_text("SECRET=1", encoding="utf-8")
    (ok / "README").write_text("hi", encoding="utf-8")
    state2 = {"plan": [], "kept": []}
    ji._run_bootstrap_sequence(ok, _run_real, "single-branch", False, False, state2)
    staged = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"], cwd=ok, capture_output=True, text=True
    ).stdout
    assert ".env" not in staged.split()
    assert "README" in staged.split()

@pytest.mark.parametrize("preset,staging", [
    ("dual-branch", True), ("dual-branch-pr", True),
    ("single-branch", False), ("single-branch-pr", False),
])
def test_staging_branch_created_for_dual_branch_but_not_single_branch(tmp_path, preset, staging):
    root = tmp_path / "repos"
    fresh = root / "fresh"
    _init_repo(fresh)
    (fresh / "README").write_text("x", encoding="utf-8")
    state = {"plan": [], "kept": []}
    ji._run_bootstrap_sequence(fresh, _run_real, preset, False, False, state)
    branches = subprocess.run(["git", "branch"], cwd=fresh, capture_output=True, text=True).stdout
    assert ("staging" in branches) == staging
    assert (("run", "git branch staging") in state["plan"]) == staging
    # F7: an adopted repo that already has commit history gets the same staging decision --
    # the zero-commit gate only skips the commit step, not the staging step (spec §2.4 step 5).
    adopted = root / "adopted"
    _init_repo(adopted)
    _commit_file(adopted, "README")
    state2 = {"plan": [], "kept": []}
    ji._run_bootstrap_sequence(adopted, _run_real, preset, False, False, state2)
    branches2 = subprocess.run(["git", "branch"], cwd=adopted, capture_output=True, text=True).stdout
    assert ("staging" in branches2) == staging
    log = subprocess.run(["git", "log", "--oneline"], cwd=adopted, capture_output=True, text=True).stdout
    assert log.strip().count("\n") == 0  # no new bootstrap commit was added

def test_dry_run_is_inert(tmp_path, capsys):
    root = tmp_path / "repos"
    target = root / "fresh"
    fake = FakeRun()
    out = _main(_argv(target, preset="dual-branch-pr", dry_run=True), run=fake, allowlist_root=root, now=_fixed_now)
    assert out == ji.OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["head"] is None
    assert not target.exists()
    # Every git subcommand jax_init.py issues outside an `if not dry_run:`
    # guard is read-only (rg '\["git"' shows only rev-parse/symbolic-ref
    # reachable unguarded); init/add/commit/branch/config/push/gh are all
    # gated behind `if not dry_run:` in the source, so this allowlist is
    # equivalent to asserting none of those ever fire.
    read_only_git_subcommands = {"rev-parse", "symbolic-ref"}
    assert fake.calls
    for call in fake.calls:
        assert call[0] == "git", call
        assert call[1] in read_only_git_subcommands, call
    assert any(c.startswith("would-create") for c in payload["created"])
    assert any(c.startswith("would-run") for c in payload["ran"])


def test_dry_run_single_branch_pr_plans_no_staging_steps(tmp_path, capsys):
    root = tmp_path / "repos"
    out = _main(_argv(root / "fresh", preset="single-branch-pr", remote=True, dry_run=True),
                run=FakeRun(), allowlist_root=root, now=_fixed_now)
    assert out == ji.OK
    payload = json.loads(capsys.readouterr().out)
    assert any(c.startswith("would-run") for c in payload["ran"])
    assert not any("staging" in c for c in payload["ran"])

# ---- Task 4 ----

class GhFakeRun:
    """Real git (so repo state is genuine and inspectable); gh/push faked
    (never touches the network, per invariant I4 and the fixed decision that
    gh is never real in tests)."""
    def __init__(self):
        self.calls = []
        self.fail = set()
    def __call__(self, argv, cwd=None):
        self.calls.append(list(argv))
        if argv[0] == "gh" or argv[:2] == ["git", "push"]:
            joined = " ".join(argv)
            if any(f in joined for f in self.fail):
                return SimpleNamespace(returncode=1, stdout="", stderr="fail")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return _run_real(argv, cwd=cwd)

@pytest.mark.parametrize("preset", ["dual-branch", "dual-branch-pr"])
def test_remote_sequencing_pushes_staging_for_the_dual_pair(tmp_path, capsys, preset):
    root = tmp_path / "repos"
    proj = f"{preset}-proj"
    target = root / proj
    fake = FakeRun()
    out = _main(_argv(target, preset=preset, remote=True), run=fake, allowlist_root=root, now=_fixed_now)
    assert out == ji.OK
    canonical = str(target.resolve())
    joined = [" ".join(c) for c in fake.calls]
    def first_index(prefix):
        return next(i for i, c in enumerate(joined) if c.startswith(prefix))
    i_auth = first_index("gh auth status")
    i_create = first_index("gh repo create")
    i_push_main = first_index("git push -u origin main")
    i_push_staging = first_index("git push -u origin staging")
    i_edit = first_index("gh repo edit")
    i_branch_staging = first_index("git branch staging")
    # git branch staging is unconditional bootstrap-sequence step 5 (§2.4), created before any
    # --remote/gh call is ever issued; gh auth is checked (§2.4 preflight) before gh repo create.
    assert i_branch_staging < i_auth < i_create < i_push_main < i_push_staging < i_edit
    exact_calls = [
        (i_auth, ["gh", "auth", "status"]),
        (i_create, ["gh", "repo", "create", proj, "--private", "--source", canonical, "--remote", "origin"]),
        (i_push_main, ["git", "push", "-u", "origin", "main"]),
        (i_push_staging, ["git", "push", "-u", "origin", "staging"]),
        (i_edit, ["gh", "repo", "edit", "--default-branch", "staging"]),
    ]
    for idx, expected_argv in exact_calls:
        assert fake.calls[idx] == expected_argv, fake.calls[idx]
        assert fake.cwds[idx] == canonical, fake.cwds[idx]


@pytest.mark.parametrize("preset", ["bubble-buildprint", "single-branch", "single-branch-pr"])
def test_remote_sequencing_pushes_no_staging_outside_the_dual_pair(tmp_path, preset):
    root = tmp_path / "repos"
    fake = FakeRun()
    out = _main(_argv(root / "other-proj", preset=preset, remote=True),
                run=fake, allowlist_root=root, now=_fixed_now)
    assert out == ji.OK
    assert any(c == ["git", "push", "-u", "origin", "main"] for c in fake.calls)
    assert not any(c[:2] == ["git", "push"] and "staging" in c for c in fake.calls)
    assert not any(c[:3] == ["gh", "repo", "edit"] for c in fake.calls)

def test_remote_failure_keeps_local_commit(tmp_path, capsys):
    root = tmp_path / "repos"
    target = root / "proj"
    fake = GhFakeRun()
    fake.fail.add("gh repo create")
    out = _main(_argv(target, preset="dual-branch-pr", remote=True), run=fake, allowlist_root=root, now=_fixed_now)
    payload = json.loads(capsys.readouterr().out)
    assert out == ji.REFUSED
    assert payload["refused"] == "remote-failed: gh repo create"
    # the local commit and its sha survive a remote failure (§2.6: head is
    # null only when there truly are no commits)
    assert payload["head"] and len(payload["head"]) == 40
    log = subprocess.run(["git", "log", "--oneline"], cwd=target, capture_output=True, text=True).stdout
    assert "chore: bootstrap" in log
    pushes = [c for c in fake.calls if c[:2] == ["git", "push"]]
    assert pushes == []

# ---- threat model ----

def test_threat_model_flag_renders_into_agents_template_and_parses():
    import jaxflow as jf

    template_path = Path(__file__).resolve().parents[1] / "workflow" / "templates" / "AGENTS-template.md"
    template = template_path.read_text(encoding="utf-8")
    rendered = ji.render_threat_model(template, "public-app")
    assert jf.parse_threat_model(rendered) == "public-app"

def test_threat_model_flag_is_required(tmp_path):
    argv = ["--path", str(tmp_path / "x"), "--name", "X", "--preset", "single-branch", "--stack", "other"]
    with pytest.raises(SystemExit):
        ji.parse_args(argv)


def test_an_old_preset_name_is_rejected_with_the_five_new_names(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        ji.parse_args(_argv(tmp_path / "x", preset="greenfield"))  # old-name-ok
    assert exc.value.code == 2
    err = capsys.readouterr().err
    for name in ("single-branch", "single-branch-pr", "dual-branch", "dual-branch-pr", "bubble-buildprint"):
        assert name in err

def test_remote_on_adopted_repo_requires_main_branch(tmp_path, capsys):
    root = tmp_path / "repos"
    adopted = root / "adopted"
    _init_repo(adopted, branch="feat/x")
    _commit_file(adopted, "README")
    fake = GhFakeRun()
    out = _main(_argv(adopted, preset="single-branch", remote=True), run=fake, allowlist_root=root, now=_fixed_now)
    payload = json.loads(capsys.readouterr().out)
    assert out == ji.REFUSED
    assert payload["refused"] == "remote-failed: branch-not-main"
    assert not any(c[0] == "gh" for c in fake.calls)


def test_remote_refuses_github_integration_disabled_before_any_gh_or_push_call(tmp_path, monkeypatch, capsys):
    # MOA-502 Decision 2: the setting wins over the flag — --remote was explicitly passed,
    # but the disabled integration still refuses before any gh/push call. (main() swallows
    # the Refusal into its own REFUSED payload, the existing convention in this file.)
    import general_settings
    monkeypatch.setattr(general_settings, "read_settings",
                         lambda: {"ok": True, "data": {"integrations": {"github": False}}})
    root = tmp_path / "repos"
    target = root / "proj"
    fake = GhFakeRun()
    out = _main(_argv(target, preset="dual-branch-pr", remote=True), run=fake, allowlist_root=root, now=_fixed_now)
    payload = json.loads(capsys.readouterr().out)
    assert out == ji.REFUSED
    assert payload["refused"] == "github-integration-disabled"
    assert not any(c[0] == "gh" or c[:2] == ["git", "push"] for c in fake.calls)

# ---- allowlist root default (spec: reposRoot consumer) ----

import general_settings


@pytest.fixture(autouse=True)
def _github_on_by_default(monkeypatch):
    # MOA-502 Decision 2: the --remote gate reads integrations.github live through
    # general_settings.read_settings(). Pre-existing tests exercise the real (non-injected)
    # gh path, so default it ON here by delegating to the real reader (reposRoot and every
    # other field keep their real values — the reload tests below depend on that); the
    # gate-specific test re-patches read_settings after this fixture, which wins.
    real = general_settings.read_settings

    def forced(*args, **kwargs):
        r = real(*args, **kwargs)
        if r.get("ok"):
            return {**r, "data": {**r["data"],
                                  "integrations": {**r["data"]["integrations"], "github": True}}}
        return r

    monkeypatch.setattr(general_settings, "read_settings", forced)


def _reload_jax_init_with_home(monkeypatch, home):
    """Reload `ji` under `JAXOS_HOME=home` to exercise its import-time settings read, then
    restore the module namespace exactly. A bare reload leaves the module with NEW class
    objects (Refusal, ...), while every other module's `from jax_init import ...` still
    points at the old ones -- breaking 103 later tests' `except ji.Refusal` identity checks
    when the whole `scripts/` suite runs in one process."""
    monkeypatch.setenv("JAXOS_HOME", str(home))
    import importlib
    saved = dict(ji.__dict__)
    importlib.reload(ji)
    return saved


def _restore_jax_init(saved):
    ji.__dict__.clear()
    ji.__dict__.update(saved)


def test_allowlist_root_default_reads_settings(tmp_path, monkeypatch):
    settings_home = tmp_path / "jaxos-home"
    settings_home.mkdir()
    (settings_home / "settings.json").write_text('{"reposRoot": "/tmp/custom-repos"}', encoding="utf-8")
    saved = _reload_jax_init_with_home(monkeypatch, settings_home)
    try:
        assert ji.ALLOWLIST_ROOT_DEFAULT == Path("/tmp/custom-repos")
    finally:
        _restore_jax_init(saved)  # restore module state for every later test


def test_allowlist_root_default_falls_back_with_no_settings(tmp_path, monkeypatch):
    saved = _reload_jax_init_with_home(monkeypatch, tmp_path / "empty-jaxos-home")
    try:
        assert ji.ALLOWLIST_ROOT_DEFAULT == Path.home() / "repos"
    finally:
        _restore_jax_init(saved)
