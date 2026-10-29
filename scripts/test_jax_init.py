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
        # Fixtures must never read the operator's user/system config or hooks.
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
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
        if argv[:4] == ["gh", "api", "user", "--jq"]:
            return SimpleNamespace(returncode=0, stdout="acme\n", stderr="")
        if argv[:4] == ["git", "rev-parse", "--git-path", "hooks"]:
            return SimpleNamespace(returncode=0, stdout=str(Path(cwd) / ".git/hooks") + "\n", stderr="")
        if argv[:3] == ["git", "config", "--get"]:
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
    # Adoption never edits the tracked .gitignore; local artifacts go to info/exclude.
    assert (target / ".gitignore").read_bytes() == pre_gitignore
    exclude = subprocess.run(
        ["git", "rev-parse", "--git-path", "info/exclude"], cwd=target,
        capture_output=True, text=True).stdout.strip()
    exclude_lines = (target / exclude).read_text(encoding="utf-8").splitlines()
    for entry in ("/.local/", "/.jax-os/", "/AGENTS.md"):
        assert entry in exclude_lines
    assert "/CLAUDE.md" not in exclude_lines   # tracked, so not local-only
    assert ".env*" not in exclude_lines
    assert (target / "CLAUDE.md").read_text() == "custom\n"
    log = subprocess.run(["git", "log", "--oneline"], cwd=target, capture_output=True, text=True).stdout
    assert log.strip().count("\n") == 0
    assert not (target / ".githooks").exists()
    _git(target, "add", "-A")
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=target,
                            capture_output=True, text=True).stdout
    assert staged == ""
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
    # The second run is adoption-shaped (history exists): it adds the shared local
    # exclusions once, then stabilizes.
    assert payload2["created"] == [".git/info/exclude"]
    exclude_after_first = (target2 / ".git/info/exclude").read_bytes()
    out3 = _main(_argv(target2, name="Fresh"), run=_run_real, allowlist_root=root2, now=_fixed_now)
    assert out3 == ji.OK
    payload3 = json.loads(capsys.readouterr().out)
    assert payload3["created"] == []
    assert (target2 / ".git/info/exclude").read_bytes() == exclude_after_first
    assert set(payload2["kept"]) >= {
        ".githooks/commit-msg", ".local/handoff.md", "CLAUDE.md", ".jax-os/status.md",
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

def test_valid_commit_subject_accepts_company_and_default_subjects():
    for subject in ("chore: bootstrap", "DRA-2408: add operator", "feat: x (merge feat/1)"):
        assert ji._valid_commit_subject(subject)


@pytest.mark.parametrize("subject", [
    "", " ", " padded", "padded ", "two\nlines", "tab\there", "bell\x07",
    "bom\ufeff", "Co-Authored-By: someone", "co-authored-by: someone", 42, None,
])
def test_valid_commit_subject_rejects_bad_subjects(subject):
    assert not ji._valid_commit_subject(subject)


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
    # Dry-run may inspect inherited config, but must never change it.
    read_only_git_subcommands = {"rev-parse", "symbolic-ref"}
    assert fake.calls
    for call in fake.calls:
        assert call[0] == "git", call
        assert (call[1] in read_only_git_subcommands
                or call == ["git", "config", "--get", "core.hooksPath"]), call
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
            if argv[:4] == ["gh", "api", "user", "--jq"]:
                return SimpleNamespace(returncode=0, stdout="acme\n", stderr="")
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
    i_account = first_index("gh api user")
    i_create = first_index("gh repo create")
    i_push_main = first_index("git push -u origin main")
    i_push_staging = first_index("git push -u origin staging")
    i_edit = first_index("gh repo edit")
    i_branch_staging = first_index("git branch staging")
    # The resolved account is checked BEFORE any bootstrap write, and the create call names
    # that exact account.
    assert i_account < i_create < i_push_main < i_push_staging < i_edit
    assert i_branch_staging < i_create
    exact_calls = [
        (i_auth, ["gh", "auth", "status"]),
        (i_account, ["gh", "api", "user", "--jq", ".login"]),
        (i_create, ["gh", "repo", "create", f"acme/{proj}", "--private", "--source", canonical, "--remote", "origin"]),
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

def test_remote_on_adopted_history_without_origin_refuses_before_writes(tmp_path, capsys):
    root = tmp_path / "repos"
    adopted = root / "adopted"
    _init_repo(adopted, branch="feat/x")
    _commit_file(adopted, "README")
    before = subprocess.run(["git", "status", "--porcelain"], cwd=adopted,
                            capture_output=True, text=True).stdout
    fake = GhFakeRun()
    out = _main(_argv(adopted, preset="single-branch", remote=True), run=fake, allowlist_root=root, now=_fixed_now)
    payload = json.loads(capsys.readouterr().out)
    assert out == ji.REFUSED
    assert payload["refused"] == "remote-failed: origin-missing"
    assert not any(c[0] == "gh" for c in fake.calls)
    assert subprocess.run(["git", "status", "--porcelain"], cwd=adopted,
                          capture_output=True, text=True).stdout == before
    assert not (adopted / ".local").exists()


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

# ---- adoption: non-destructive invariants (Task 4) ----

def _snapshot(repo):
    git = lambda *args: subprocess.run(["git", *args], cwd=repo, capture_output=True,
                                       text=True, env=_run_real.__globals__.get("os").environ | {
                                           "GIT_CONFIG_GLOBAL": os.devnull,
                                           "GIT_CONFIG_SYSTEM": os.devnull,
                                           "GIT_CONFIG_NOSYSTEM": "1",
                                       }).stdout
    return {
        "head": git("rev-parse", "HEAD"),
        "refs": git("for-each-ref", "--format=%(refname) %(objectname)"),
        "index": git("ls-files", "--stage"),
        "remotes": git("remote", "-v"),
        "status": git("status", "--porcelain", "--untracked-files=no"),
    }


def test_adoption_keeps_tracked_gitignore_and_git_state(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    _commit_file(repo, ".gitignore", "vendor/\n")
    _git(repo, "remote", "add", "origin", "git@github.com:company/app.git")
    before = _snapshot(repo)
    before_ignore = (repo / ".gitignore").read_bytes()
    assert _main(_argv(repo), run=_run_real, allowlist_root=root) == ji.OK
    capsys.readouterr()
    assert (repo / ".gitignore").read_bytes() == before_ignore
    assert _snapshot(repo) == before
    assert not (repo / ".githooks").exists()
    _git(repo, "add", "-A")
    assert subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=repo,
                          capture_output=True, text=True).stdout == ""


def test_adoption_never_creates_branches_or_a_commit(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo, branch="develop")
    _commit_file(repo, "README")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout
    out = _main(_argv(repo, preset="dual-branch"), run=_run_real, allowlist_root=root)
    assert out == ji.OK
    capsys.readouterr()
    branches = subprocess.run(["git", "branch", "--format=%(refname:short)"], cwd=repo,
                              capture_output=True, text=True).stdout.split()
    assert branches == ["develop"]
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout == head
    assert not (repo / ".gitignore").exists() or (repo / ".gitignore").read_text() == ""


def test_adoption_preserves_existing_origin_even_with_remote_flag(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    _commit_file(repo, ".gitignore", "vendor/\n")
    _git(repo, "remote", "add", "origin", "git@github.com:company/app.git")
    fake = GhFakeRun()
    out = _main(_argv(repo, preset="single-branch-pr", remote=True), run=fake,
                allowlist_root=root)
    assert out == ji.OK
    payload = json.loads(capsys.readouterr().out)
    assert any("origin: git@github.com:company/app.git" in item for item in payload["kept"])
    assert not any(c[0] == "gh" for c in fake.calls)
    origin = subprocess.run(["git", "remote", "get-url", "origin"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    assert origin == "git@github.com:company/app.git"


def test_adoption_pr_preset_requires_a_github_origin(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    _commit_file(repo, "README")
    _git(repo, "remote", "add", "origin", "git@gitlab.com:company/app.git")
    before = _snapshot(repo)
    out = _main(_argv(repo, preset="single-branch-pr"), run=_run_real, allowlist_root=root)
    assert out == ji.REFUSED
    assert json.loads(capsys.readouterr().out)["refused"] == "remote-failed: origin-not-github"
    assert _snapshot(repo) == before
    assert not (repo / ".local").exists()


def test_adoption_pr_preset_without_an_origin_refuses_before_writes(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    _commit_file(repo, "README")
    before = _snapshot(repo)
    out = _main(_argv(repo, preset="dual-branch-pr"), run=_run_real, allowlist_root=root)
    assert out == ji.REFUSED
    assert json.loads(capsys.readouterr().out)["refused"] == "remote-failed: origin-missing"
    assert _snapshot(repo) == before
    assert not (repo / ".local").exists()


def test_adoption_ignore_negation_refuses_before_baseline_writes(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    _commit_file(repo, ".gitignore", "vendor/\n!/.local/\n")
    exclude = repo / ".git" / "info" / "exclude"
    before = exclude.read_bytes()
    out = _main(_argv(repo), run=_run_real, allowlist_root=root)
    assert out == ji.REFUSED
    assert json.loads(capsys.readouterr().out)["refused"].startswith("ignore-conflict:")
    assert exclude.read_bytes() == before
    assert not (repo / ".local").exists()


def test_adoption_dry_run_is_inert(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    _commit_file(repo, "README")
    exclude = repo / ".git" / "info" / "exclude"
    before = exclude.read_bytes()
    out = _main(_argv(repo, dry_run=True), run=_run_real, allowlist_root=root)
    assert out == ji.OK
    capsys.readouterr()
    assert exclude.read_bytes() == before
    assert not (repo / ".local").exists()
    assert not (repo / ".jax-os").exists()


def test_adoption_artifact_collision_refuses_before_writes(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    (repo / ".local").write_text("not a dir\n", encoding="utf-8")
    _git(repo, "add", ".local")
    _git(repo, "commit", "-m", "collision")
    before = _snapshot(repo)
    out = _main(_argv(repo), run=_run_real, allowlist_root=root)
    assert out == ji.REFUSED
    assert json.loads(capsys.readouterr().out)["refused"]
    assert _snapshot(repo) == before


def test_adoption_installs_the_shared_hook_and_linked_worktrees_strip_trailers(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    _commit_file(repo, "README")
    assert _main(_argv(repo), run=_run_real, allowlist_root=root) == ji.OK
    capsys.readouterr()
    assert subprocess.run(["git", "config", "core.hooksPath"], cwd=repo,
                          capture_output=True, text=True).returncode != 0
    assert (repo / ".git" / "hooks" / "commit-msg").read_bytes() == ji.HOOK_SOURCE.read_bytes()
    worktree = root / "company-linked"
    _git(repo, "worktree", "add", "-b", "feat/linked", str(worktree))
    _git(worktree, "commit", "--allow-empty", "-m", "feat: linked work",
         "-m", "Co-Authored-By: Someone <s@example.invalid>")
    body = subprocess.run(["git", "log", "-1", "--format=%B"], cwd=worktree,
                          capture_output=True, text=True).stdout
    assert "co-authored-by" not in body.lower()


def test_adoption_preserves_a_configured_hooks_path(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    _commit_file(repo, "README")
    _git(repo, "config", "core.hooksPath", ".company-hooks")
    out = _main(_argv(repo), run=_run_real, allowlist_root=root)
    assert out == ji.OK
    payload = json.loads(capsys.readouterr().out)
    assert any("hook: preserved" in item for item in payload["kept"])
    assert not (repo / ".git" / "hooks" / "commit-msg").exists()
    assert subprocess.run(["git", "config", "core.hooksPath"], cwd=repo,
                          capture_output=True, text=True).stdout.strip() == ".company-hooks"


def test_adoption_preserves_non_sample_default_hooks(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    _commit_file(repo, "README")
    custom = repo / ".git" / "hooks" / "pre-commit"
    custom.write_text("#!/bin/sh\n", encoding="utf-8")
    custom.chmod(0o755)
    out = _main(_argv(repo), run=_run_real, allowlist_root=root)
    assert out == ji.OK
    payload = json.loads(capsys.readouterr().out)
    assert any("hook: preserved" in item for item in payload["kept"])
    assert not (repo / ".git" / "hooks" / "commit-msg").exists()


def test_adoption_preserves_an_existing_githooks_directory(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    _commit_file(repo, "README")
    hooks = repo / ".githooks"
    hooks.mkdir()
    (hooks / "pre-commit").write_text("#!/bin/sh\n", encoding="utf-8")
    out = _main(_argv(repo), run=_run_real, allowlist_root=root)
    assert out == ji.OK
    payload = json.loads(capsys.readouterr().out)
    assert any("hook: preserved" in item for item in payload["kept"])
    assert not (hooks / "commit-msg").exists()
    assert subprocess.run(["git", "config", "core.hooksPath"], cwd=repo,
                          capture_output=True, text=True).returncode != 0


def test_adoption_hooks_preserve_installs_nothing(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    _commit_file(repo, "README")
    argv = _argv(repo) + ["--hooks", "preserve"]
    out = _main(argv, run=_run_real, allowlist_root=root)
    assert out == ji.OK
    payload = json.loads(capsys.readouterr().out)
    assert any("hook: preserved" in item for item in payload["kept"])
    assert not (repo / ".git" / "hooks" / "commit-msg").exists()
    assert not (repo / ".githooks").exists()


def test_fresh_bootstrap_subject_is_passed_as_a_git_argument(tmp_path, capsys):
    root = tmp_path / "repos"
    target = root / "fresh"
    argv = _argv(target) + ["--initial-commit-subject", "DRA-2408: bootstrap the app"]
    out = _main(argv, run=_run_real, allowlist_root=root)
    assert out == ji.OK
    capsys.readouterr()
    log = subprocess.run(["git", "log", "--format=%s"], cwd=target,
                         capture_output=True, text=True).stdout.strip()
    assert log == "DRA-2408: bootstrap the app"


@pytest.mark.parametrize("subject", ["", " padded", "two\nlines", "Co-Authored-By: x"])
def test_fresh_bootstrap_subject_rejects_bad_values_before_writes(tmp_path, capsys, subject):
    root = tmp_path / "repos"
    target = root / "fresh"
    argv = _argv(target, dry_run=True) + ["--initial-commit-subject", subject]
    out = _main(argv, run=_run_real, allowlist_root=root)
    assert out == ji.REFUSED
    assert json.loads(capsys.readouterr().out)["refused"] == "invalid-commit-subject"
    assert not target.exists()


def test_adoption_root_scoped_excludes_do_not_hide_nested_company_docs(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    (repo / "services").mkdir()
    (repo / "services" / "AGENTS.md").write_text("nested rules\n", encoding="utf-8")
    _git(repo, "add", "services/AGENTS.md")
    _git(repo, "commit", "-m", "nested docs")
    assert _main(_argv(repo), run=_run_real, allowlist_root=root) == ji.OK
    capsys.readouterr()
    probe = subprocess.run(["git", "check-ignore", "-q", "services/AGENTS.md"], cwd=repo)
    assert probe.returncode == 1


def test_adoption_shared_excludes_cover_a_linked_worktree(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    _commit_file(repo, "README")
    assert _main(_argv(repo), run=_run_real, allowlist_root=root) == ji.OK
    capsys.readouterr()
    worktree = root / "company-linked"
    _git(repo, "worktree", "add", "-b", "feat/linked", str(worktree))
    for rel in (".local/scratch/probe", ".jax-os/status.md", "AGENTS.md"):
        assert subprocess.run(["git", "check-ignore", "-q", rel], cwd=worktree).returncode == 0, rel


def test_fresh_remote_pre_push_failure_stops_without_bypass(tmp_path, capsys):
    root = tmp_path / "repos"
    target = root / "proj"
    fake = GhFakeRun()
    fake.fail.add("git push")
    out = _main(_argv(target, preset="single-branch-pr", remote=True), run=fake,
                allowlist_root=root, now=_fixed_now)
    assert out == ji.REFUSED
    assert json.loads(capsys.readouterr().out)["refused"] == "remote-failed: git push main"
    log = subprocess.run(["git", "log", "--format=%s"], cwd=target,
                         capture_output=True, text=True).stdout.strip()
    assert log == "chore: bootstrap"


def test_adoption_discovery_never_executes_hooks(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    _commit_file(repo, "README")
    marker = repo / "hook-ran"
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(f"#!/bin/sh\\ntouch {marker}\\n", encoding="utf-8")
    hook.chmod(0o755)
    out = _main(_argv(repo), run=_run_real, allowlist_root=root)
    assert out == ji.OK
    capsys.readouterr()
    assert not marker.exists()


def test_adoption_preserves_a_configured_missing_hook_path(tmp_path, capsys):
    root = tmp_path / "repos"
    repo = root / "company"
    _init_repo(repo)
    _commit_file(repo, "README")
    _git(repo, "config", "core.hooksPath", ".missing-hooks")
    out = _main(_argv(repo), run=_run_real, allowlist_root=root)
    assert out == ji.OK
    payload = json.loads(capsys.readouterr().out)
    assert any("hook: preserved" in item for item in payload["kept"])
    assert not (repo / ".git" / "hooks" / "commit-msg").exists()
    assert subprocess.run(["git", "config", "core.hooksPath"], cwd=repo,
                          capture_output=True, text=True).stdout.strip() == ".missing-hooks"


@pytest.mark.parametrize("redirect", ["file", "directory"])
def test_adoption_refuses_redirected_exclude_metadata(tmp_path, monkeypatch, redirect):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "company"
    _init_repo(repo)
    _commit_file(repo, "README")
    metadata = repo / ".git"
    config = metadata / "config"
    before = config.read_bytes()
    if redirect == "file":
        exclude = metadata / "info" / "exclude"
        exclude.unlink()
        exclude.symlink_to(config)
    else:
        info = metadata / "info"
        original = metadata / "original-info"
        info.rename(original)
        info.symlink_to(original, target_is_directory=True)
    with pytest.raises(ji.Refusal, match="path-outside-allowlist"):
        ji._exclude_destination(_run_real, repo, tmp_path)
    assert config.read_bytes() == before
    assert not (repo / ".local").exists()


def test_dormant_jax_hook_is_preserved_not_reported_as_effective(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "company"
    _init_repo(repo)
    hooks = repo / ".githooks"
    hooks.mkdir()
    (hooks / "commit-msg").write_bytes(ji.HOOK_SOURCE.read_bytes())
    action, label = ji._hook_plan(_run_real, repo, tmp_path, "auto")
    assert action == "preserve"
    assert "existing .githooks" in label
    assert not (repo / ".git" / "hooks" / "commit-msg").exists()
    assert _run_real(["git", "config", "--get", "core.hooksPath"], cwd=repo).returncode == 1


@pytest.mark.parametrize("has_metadata", [False, True])
@pytest.mark.parametrize("mode", ["preserve", "auto"])
def test_fresh_hook_preview_respects_preservation(tmp_path, monkeypatch, capsys, mode, has_metadata):
    global_config = tmp_path / "gitconfig"
    global_config.write_text(
        f"[core]\n\thooksPath = {tmp_path / 'company-hooks'}\n" if mode == "auto" else "",
        encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    def run(argv, cwd=None):
        return subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              check=False, env=os.environ.copy())
    repo = tmp_path / "fresh"
    if has_metadata:
        _init_repo(repo)
    assert _main(_argv(repo, dry_run=True) + ["--hooks", mode], run=run,
                 allowlist_root=tmp_path) == ji.OK
    payload = json.loads(capsys.readouterr().out)
    assert not any(".githooks" in item for item in payload["created"])
    assert not any("core.hooksPath" in item for item in payload["ran"])
    assert any("hook: preserved" in item for item in payload["kept"])
    assert repo.exists() == has_metadata


def test_adoption_validates_hook_destination_before_appending_excludes(tmp_path, capsys):
    repo = tmp_path / "company"
    _init_repo(repo)
    _commit_file(repo, "README")
    metadata = repo / ".git"
    hooks = metadata / "hooks"
    hooks.rename(metadata / "original-hooks")
    foreign = tmp_path / "foreign-hooks"
    foreign.mkdir()
    hooks.symlink_to(foreign, target_is_directory=True)
    exclude = metadata / "info" / "exclude"
    before = exclude.read_bytes()
    assert _main(_argv(repo), run=_run_real, allowlist_root=tmp_path) == ji.REFUSED
    assert json.loads(capsys.readouterr().out)["refused"] == "path-outside-allowlist"
    assert exclude.read_bytes() == before
    assert not (repo / ".local").exists()


def test_adoption_creates_missing_local_exclude_directory(tmp_path, capsys):
    repo = tmp_path / "company"
    _init_repo(repo)
    _commit_file(repo, "README")
    info = repo / ".git" / "info"
    (info / "exclude").unlink()
    info.rmdir()
    assert _main(_argv(repo), run=_run_real, allowlist_root=tmp_path) == ji.OK
    capsys.readouterr()
    assert "/.local/" in (info / "exclude").read_text(encoding="utf-8")
