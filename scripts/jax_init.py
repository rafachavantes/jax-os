#!/usr/bin/env python3
"""jax-init -- deterministic project bootstrap baseline."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path

from jaxflow_hook import run_command
import general_settings

OK = 0
REFUSED = 2
PRESETS = ("single-branch", "single-branch-pr", "dual-branch", "dual-branch-pr", "bubble-buildprint")
STAGING_PRESETS = ("dual-branch", "dual-branch-pr")  # the only presets with a `staging` branch
_settings_result = general_settings.read_settings()
ALLOWLIST_ROOT_DEFAULT = (
    Path(_settings_result["data"]["reposRoot"]) if _settings_result["ok"] else Path.home() / "repos"
)
HOOK_SOURCE = Path(__file__).resolve().parents[1] / ".githooks" / "commit-msg"
GITIGNORE_BASE = [".local/", ".jax-os/", "AGENTS.md", "CLAUDE.md", ".env*"]
STACK_GITIGNORE = {
    "nextjs": ["node_modules/", ".next/"],
    "python": [".venv/", "__pycache__/"],
    "bubble": [],
    "other": [],
}
BASELINE_DIRS = [
    ".local/docs/specs",
    ".local/docs/plans",
    ".local/reports",
    ".local/scratch",
    ".jax-os",
]
RESERVED_TOP_LEVEL = {".local", ".jax-os"}
SECRET_SUFFIXES = (".pem", ".key")
SECRET_NAME_PARTS = ("credentials", "secret")
PUBLIC_ENV_TEMPLATES = (".env.example", ".env.sample", ".env.template")
# anchored to the two supported remote forms exactly --
# `git@github.com:owner/repo(.git)` or `https://github.com/owner/repo(.git)` -- so a
# lookalike host (`evilgithub.com`, which contains "github.com" as a bare substring) can
# never match. Shared with jaxflow_merge (which imports it back under `_GH_REMOTE_RE`).
_GH_REMOTE_RE = re.compile(r"^(?:git@github\.com:|https://github\.com/)([^/\s]+)/([^/\s]+?)(?:\.git)?/?$")
# Same mode keys as scripts/jaxflow.py's `_THREAT_MODEL_LINES` -- duplicated instead of
# imported because jaxflow.py already imports from this module, and importing back
# would create a cycle.
THREAT_MODEL_MODES = ("internal-single-user", "public-app")

class Refusal(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code

def _contained(path, root):
    return path.resolve().is_relative_to(root.resolve())

def canonicalize_target(raw_path, allowlist_root):
    canonical = Path(raw_path).resolve()
    if not _contained(canonical, allowlist_root):
        raise Refusal("path-outside-allowlist")
    return canonical

def _nearest_existing_ancestor(path):
    for ancestor in (path, *path.parents):
        if ancestor.exists():
            return ancestor
    raise Refusal("path-outside-allowlist")

def _is_empty_for_bootstrap(path):
    # ponytail: spec case A -- a target holding only .local/ and/or .jax-os/
    # (Rafa's pre-existing briefs under .local/docs/) still counts as empty.
    return not any(child.name not in RESERVED_TOP_LEVEL for child in path.iterdir())

def _probe(run, argv, cwd, step):
    try:
        return run(argv, cwd=cwd)
    except OSError:
        raise Refusal(f"command-failed: {step}")

def classify_target(canonical, run):
    exists = canonical.exists()
    probe = canonical if exists else _nearest_existing_ancestor(canonical)
    result = _probe(run, ["git", "rev-parse", "--show-toplevel"], str(probe), "git rev-parse --show-toplevel")
    toplevel = None
    if result.returncode == 0 and result.stdout.strip():
        toplevel = Path(result.stdout.strip()).resolve()
    if exists and toplevel is not None and toplevel == canonical:
        return "C"
    is_empty = (not exists) or _is_empty_for_bootstrap(canonical)
    if exists and not is_empty:
        return "D"
    if toplevel is not None:
        return "B"
    return "A"

def _validate_name(name):
    if not name or "\n" in name or any(unicodedata.category(ch) == "Cc" for ch in name):
        raise Refusal("invalid-name")

def _valid_commit_subject(text):
    """True for a single-line, trimmed, UTF-8-encodable subject with no control
    characters, BOM, or case-insensitive Co-Authored-By text. Shared by the bootstrap
    commit, explicit phase subjects, and PR-title delivery."""
    if not isinstance(text, str) or text != text.strip() or not text:
        return False
    try:
        text.encode("utf-8")
    except UnicodeError:
        return False
    if any(unicodedata.category(ch) == "Cc" for ch in text) or "\ufeff" in text:
        return False
    return "co-authored-by" not in text.lower()

def _run_ok(run, argv, cwd, step, prefix="command-failed"):
    try:
        result = run(argv, cwd=cwd)
    except OSError:
        raise Refusal(f"{prefix}: {step}")
    if result.returncode != 0:
        raise Refusal(f"{prefix}: {step}")
    return result

def _record(state, kind, label):
    state["plan"].append((kind, label))

def _render_plan(plan, dry_run):
    created, ran = [], []
    for kind, label in plan:
        text = f"would-{kind}: {label}" if dry_run else label
        (created if kind == "create" else ran).append(text)
    return created, ran

def _setup_repo(canonical, case, run, dry_run, state):
    if case == "C":
        return
    if not dry_run:
        try:
            canonical.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise Refusal("command-failed: mkdir")
        _run_ok(run, ["git", "init", "-b", "main"], str(canonical), "git init")
    _record(state, "run", "git init -b main")

def _read_branch(run, canonical):
    # symbolic-ref (not rev-parse --abbrev-ref) so an unborn branch (fresh
    # `git init`, zero commits) still resolves; a detached HEAD makes this
    # fail (nonzero) instead of returning a commit sha, which is exactly
    # the "HEAD" sentinel the detached-head check below looks for.
    result = _probe(run, ["git", "symbolic-ref", "--short", "-q", "HEAD"], str(canonical), "git symbolic-ref HEAD")
    if result.returncode != 0:
        return "HEAD"
    return result.stdout.strip()

def _resolve_dest(canonical, allowlist_root, relative):
    dest = (canonical / relative).resolve()
    if not (_contained(dest, canonical) and _contained(dest, allowlist_root)):
        raise Refusal("path-outside-allowlist")
    parent = dest.parent.resolve()
    if not (_contained(parent, canonical) and _contained(parent, allowlist_root)):
        raise Refusal("path-outside-allowlist")
    return dest

def _write_new(path, data: bytes, mode=0o644):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)

def _bootstrap_gitignore(canonical, allowlist_root, stack, dry_run, state):
    dest = _resolve_dest(canonical, allowlist_root, ".gitignore")
    entries = GITIGNORE_BASE + STACK_GITIGNORE[stack]
    if dest.exists():
        if not dest.is_file():
            raise Refusal("command-failed: gitignore")
        try:
            existing_lines = dest.read_text(encoding="utf-8").splitlines()
        except OSError:
            raise Refusal("command-failed: gitignore")
        missing = [e for e in entries if e not in existing_lines]
        if not missing:
            state["kept"].append(".gitignore")
            return
        if not dry_run:
            try:
                text = dest.read_text(encoding="utf-8")
            except OSError:
                raise Refusal("command-failed: gitignore")
            prefix = "" if (text == "" or text.endswith("\n")) else "\n"
            try:
                with dest.open("a", encoding="utf-8") as fh:
                    fh.write(prefix + "".join(e + "\n" for e in missing))
            except OSError:
                raise Refusal("command-failed: gitignore")
        _record(state, "create", ".gitignore")
    else:
        if not dry_run:
            try:
                _write_new(dest, "".join(e + "\n" for e in entries).encode("utf-8"))
            except OSError:
                raise Refusal("command-failed: gitignore")
        _record(state, "create", ".gitignore")

def _bootstrap_dirs(canonical, allowlist_root, dry_run, state):
    for rel in BASELINE_DIRS:
        dest = _resolve_dest(canonical, allowlist_root, rel)
        if dest.exists():
            state["kept"].append(rel)
            continue
        if not dry_run:
            try:
                dest.mkdir(parents=True, exist_ok=True)
            except OSError:
                raise Refusal(f"command-failed: mkdir {rel}")
        _record(state, "create", rel)

def _create_only(dest, label, data, state, dry_run):
    if dest.exists():
        state["kept"].append(label)
        return
    if not dry_run:
        try:
            _write_new(dest, data() if callable(data) else data)
        except OSError:
            raise Refusal(f"command-failed: {label}")
    _record(state, "create", label)

def _git_metadata_path(run, canonical, argv, step):
    """Resolve a path reported by git relative to `canonical` (the same cwd the command
    ran in). Callers apply their own containment checks to the result."""
    result = _run_ok(run, argv, str(canonical), step)
    raw = result.stdout.strip()
    if not raw:
        raise Refusal("command-failed: " + step)
    found = Path(raw)
    if not found.is_absolute():
        found = canonical / found
    return found.resolve()

def _exclude_destination(run, canonical, allowlist_root):
    common = _git_metadata_path(
        run, canonical, ["git", "rev-parse", "--git-common-dir"], "git rev-parse --git-common-dir")
    if not _contained(common, allowlist_root):
        raise Refusal("path-outside-allowlist")
    expected = common / "info" / "exclude"
    if expected.is_symlink() or expected.parent.is_symlink():
        raise Refusal("path-outside-allowlist")
    exclude = _git_metadata_path(
        run, canonical, ["git", "rev-parse", "--git-path", "info/exclude"],
        "git rev-parse --git-path info/exclude")
    if exclude != expected or not (_contained(exclude, common) and _contained(exclude, allowlist_root)):
        raise Refusal("path-outside-allowlist")
    if not _contained(exclude.parent, common) or not _contained(exclude.parent, allowlist_root):
        raise Refusal("path-outside-allowlist")
    return common, exclude

def _is_tracked(run, canonical, rel):
    return run(["git", "ls-files", "--error-unmatch", "--", rel], cwd=str(canonical)).returncode == 0

def _adoption_ignore_entries(run, canonical):
    entries = ["/.local/", "/.jax-os/"]
    if not _is_tracked(run, canonical, "AGENTS.md"):
        entries.append("/AGENTS.md")
    if not _is_tracked(run, canonical, "CLAUDE.md"):
        entries.append("/CLAUDE.md")
    return entries

def _read_text_or_empty(path):
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except OSError:
        raise Refusal("command-failed: excludes")

def _append_missing_entries(text, entries):
    lines = text.splitlines()
    missing = [entry for entry in entries if entry not in lines]
    if not missing:
        return None
    prefix = "" if text == "" or text.endswith("\n") else "\n"
    return text + prefix + "".join(entry + "\n" for entry in missing)

def _git_config_value(run, canonical, key):
    result = run(["git", "config", "--get", key], cwd=str(canonical))
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None

def _prospective_ignores(run, canonical, exclude_text, planned):
    """Use Git's own ignore evaluation in throwaway metadata: current info/exclude bytes
    plus the planned entries, the source's effective excludesFile/ignoreCase, and the
    target worktree. No source index/HEAD/ref or .local write happens."""
    with tempfile.TemporaryDirectory() as raw:
        meta = Path(raw) / "meta"
        _run_ok(run, ["git", "init", "--bare", "--template=", str(meta)], str(canonical),
                "git init temporary metadata")
        info = meta / "info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "exclude").write_text(exclude_text, encoding="utf-8")
        base = ["git", f"--git-dir={meta}", f"--work-tree={canonical}", "-c", "core.bare=false"]
        excludes_file = _git_config_value(run, canonical, "core.excludesFile")
        if excludes_file:
            base += ["-c", f"core.excludesFile={excludes_file}"]
        ignore_case = _git_config_value(run, canonical, "core.ignoreCase")
        if ignore_case:
            base += ["-c", f"core.ignoreCase={ignore_case}"]
        for rel in planned:
            probe = run(base + ["check-ignore", "--no-index", "--quiet", rel], cwd=str(canonical))
            if probe.returncode == 1:
                raise Refusal(f"ignore-conflict: {rel}")
            if probe.returncode != 0:
                raise Refusal("command-failed: check-ignore")

def _validate_adoption_paths(canonical, allowlist_root):
    gitignore = canonical / ".gitignore"
    if gitignore.exists() and not gitignore.is_file():
        raise Refusal("command-failed: gitignore")
    for rel in BASELINE_DIRS:
        dest = _resolve_dest(canonical, allowlist_root, rel)
        if dest.exists() and not dest.is_dir():
            raise Refusal(f"command-failed: {rel}")
    for rel in (".local/handoff.md", "CLAUDE.md", ".jax-os/status.md"):
        dest = _resolve_dest(canonical, allowlist_root, rel)
        if dest.exists() and not dest.is_file():
            raise Refusal(f"command-failed: {rel}")

def _bootstrap_adoption_excludes(canonical, allowlist_root, run, dry_run, state):
    _common, exclude = _exclude_destination(run, canonical, allowlist_root)
    current = _read_text_or_empty(exclude)
    entries = _adoption_ignore_entries(run, canonical)
    updated = _append_missing_entries(current, entries)
    planned_text = current if updated is None else updated
    _prospective_ignores(run, canonical, planned_text, [
        ".local/scratch/init-probe", ".jax-os/status.md",
        *[entry.strip("/") for entry in entries if entry in ("/AGENTS.md", "/CLAUDE.md")],
    ])
    if updated is not None and not dry_run:
        try:
            exclude.parent.mkdir(exist_ok=True)
            with exclude.open("a", encoding="utf-8") as handle:
                handle.write(updated[len(current):])
        except OSError:
            raise Refusal("command-failed: excludes")
    if updated is not None:
        label = (str(exclude.relative_to(canonical)) if exclude.is_relative_to(canonical)
                 else str(exclude))
        _record(state, "create", label)

def _hook_configured(run, canonical):
    value = _git_config_value(run, canonical, "core.hooksPath")
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = canonical / path
    return path

def _non_sample_hooks(directory):
    try:
        entries = list(directory.iterdir())
    except OSError:
        return False
    return any(entry.is_file() and not entry.name.endswith(".sample") for entry in entries)

def _hook_plan(run, canonical, allowlist_root, hooks_mode):
    """`("preserve"|"already"|"install", label)` for the commit-msg hook. Never executes a
    hook; a configured but missing/unreadable path is preserved, not replaced."""
    if hooks_mode == "preserve":
        return "preserve", "hook: preserved (--hooks preserve)"
    configured = _hook_configured(run, canonical)
    default_dir = _git_metadata_path(
        run, canonical, ["git", "rev-parse", "--git-path", "hooks"], "git rev-parse --git-path hooks")
    traditional = canonical / ".githooks"
    if configured is not None:
        target = configured / "commit-msg"
        try:
            if target.is_file() and target.read_bytes() == HOOK_SOURCE.read_bytes():
                label = (str(target.relative_to(canonical))
                         if target.is_relative_to(canonical) else "hook: already installed")
                return "already", label
        except OSError:
            return "preserve", f"hook: preserved (unreadable {configured})"
        return "preserve", f"hook: preserved (core.hooksPath {configured})"
    candidate = default_dir / "commit-msg"
    try:
        if candidate.is_file() and candidate.read_bytes() == HOOK_SOURCE.read_bytes():
            label = (str(candidate.relative_to(canonical))
                     if candidate.is_relative_to(canonical) else "hook: already installed")
            return "already", label
    except OSError:
        pass
    if _non_sample_hooks(default_dir):
        return "preserve", f"hook: preserved (existing hooks in {default_dir})"
    if _non_sample_hooks(traditional):
        return "preserve", "hook: preserved (existing .githooks hooks)"
    return "install", ""

def _commit_msg_destination(canonical, allowlist_root, run, target):
    if target is None:
        dest = _resolve_dest(canonical, allowlist_root, ".githooks/commit-msg")
    else:
        dest = target
        common = _git_metadata_path(
            run, canonical, ["git", "rev-parse", "--git-common-dir"],
            "git rev-parse --git-common-dir")
        if not (_contained(dest, common) and _contained(dest, allowlist_root)):
            raise Refusal("path-outside-allowlist")
        if not (dest.parent == common / "hooks"):
            raise Refusal("path-outside-allowlist")
    return dest

def _install_commit_msg_hook(canonical, allowlist_root, run, dry_run, state, target):
    dest = _commit_msg_destination(canonical, allowlist_root, run, target)
    if dest.exists():
        state["kept"].append(str(dest.relative_to(canonical)) if dest.is_relative_to(canonical) else "hook: already installed")
        return False
    if not dry_run:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            _write_new(dest, HOOK_SOURCE.read_bytes(), 0o755)
        except OSError:
            raise Refusal("command-failed: githooks")
    label = str(dest.relative_to(canonical)) if dest.is_relative_to(canonical) else "hook: commit-msg"
    _record(state, "create", label)
    return True

def _bootstrap_hooks(canonical, allowlist_root, dry_run, run, state, hooks_mode):
    if hooks_mode == "preserve":
        state["kept"].append("hook: preserved (--hooks preserve)")
        return
    if dry_run and not (canonical / ".git").exists():
        # A dry-run before `git init` has no metadata: inherited routing is all we can
        # know. Installation stays conditional on the post-init revalidation.
        probe_dir = next(path for path in (canonical, *canonical.parents) if path.is_dir())
        configured = _git_config_value(run, probe_dir, "core.hooksPath")
        if configured is not None:
            state["kept"].append(f"hook: preserved (core.hooksPath {configured})")
            return
        _record(state, "create",
                "hook: native commit-msg (conditional on post-init validation and no template-supplied hooks)")
        return
    action, label = _hook_plan(run, canonical, allowlist_root, hooks_mode)
    if action in ("preserve", "already"):
        state["kept"].append(label)
        return
    hooks_dir = _git_metadata_path(
        run, canonical, ["git", "rev-parse", "--git-path", "hooks"],
        "git rev-parse --git-path hooks")
    _install_commit_msg_hook(canonical, allowlist_root, run, dry_run, state,
                             hooks_dir / "commit-msg")

def _do_fresh_remote_account(run, canonical, dry_run, state):
    if not dry_run:
        _run_ok(run, ["gh", "auth", "status"], str(canonical), "gh auth", prefix="remote-failed")
        resolved = _run_ok(run, ["gh", "api", "user", "--jq", ".login"], str(canonical),
                           "gh api user", prefix="remote-failed")
        account = resolved.stdout.strip()
        if not account:
            raise Refusal("remote-failed: account-unresolved")
    else:
        account = "<account>"
    _record(state, "run", "gh api user")
    return account

def _detect_builder(env):
    if env.get("CLAUDECODE"):
        return "claude-code"
    if env.get("CODEX_THREAD_ID"):
        return "codex"
    return "tech-lead"

def _iso8601(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")

def _status_seed(name, branch, builder, dt):
    return (
        "---\n"
        f"project: {name}\n"
        "stage: spec\n"
        f"builder: {builder}\n"
        f"branch: {branch}\n"
        f"updated: {_iso8601(dt)}\n"
        "---\n\n"
        "## Now\n"
        "Initialised by jax-init; first spec pending.\n"
    )

def _bootstrap_status(canonical, allowlist_root, name, branch, env, now, dry_run, state):
    dest = _resolve_dest(canonical, allowlist_root, ".jax-os/status.md")
    data = lambda: _status_seed(name, branch, _detect_builder(env), now()).encode("utf-8")
    _create_only(dest, ".jax-os/status.md", data, state, dry_run)

def _bootstrap_baseline(canonical, allowlist_root, name, stack, branch, env, now, dry_run,
                        state, run, *, hooks_mode="auto", adopted=False):
    if adopted:
        _bootstrap_adoption_excludes(canonical, allowlist_root, run, dry_run, state)
    _bootstrap_hooks(canonical, allowlist_root, dry_run, run, state, hooks_mode)
    if not adopted:
        _bootstrap_gitignore(canonical, allowlist_root, stack, dry_run, state)
    _bootstrap_dirs(canonical, allowlist_root, dry_run, state)
    _create_only(_resolve_dest(canonical, allowlist_root, ".local/handoff.md"), ".local/handoff.md", b"", state, dry_run)
    _create_only(_resolve_dest(canonical, allowlist_root, "CLAUDE.md"), "CLAUDE.md", b"@AGENTS.md\n", state, dry_run)
    _bootstrap_status(canonical, allowlist_root, name, branch, env, now, dry_run, state)

def _probe_ref(run, canonical, ref):
    result = _probe(run, ["git", "rev-parse", "--verify", "-q", ref], str(canonical), f"git rev-parse --verify {ref}")
    return result.returncode == 0

def _is_secret_path(rel_path):
    # Case-insensitive (fixes branch review G1b): `.ENV`, `foo.PEM`, `id_RSA` used to
    # sail past this guard because only the SECRET_NAME_PARTS check below was lowered.
    lowered = Path(rel_path).name.lower()
    # Committed placeholder templates (e.g. .env.example) are public by design.
    if lowered in PUBLIC_ENV_TEMPLATES:
        return False
    if lowered.startswith(".env"):
        return True
    if lowered.endswith(SECRET_SUFFIXES):
        return True
    if lowered.startswith("id_rsa"):
        return True
    return any(part in lowered for part in SECRET_NAME_PARTS)

def _run_bootstrap_sequence(canonical, run, preset, repo_pending, dry_run, state,
                            subject="chore: bootstrap"):
    has_commits = False if repo_pending else _probe_ref(run, canonical, "HEAD")
    if not has_commits:
        if not dry_run:
            _run_ok(run, ["git", "add", "-A"], str(canonical), "git add")
            staged = _run_ok(run, ["git", "diff", "--cached", "--name-only"], str(canonical), "git diff")
            for rel in (line for line in staged.stdout.splitlines() if line):
                if _is_secret_path(rel):
                    raise Refusal(f"secret-detected: {rel}")
            _run_ok(run, ["git", "commit", "-m", subject], str(canonical), "git commit")
        _record(state, "run", "git add -A")
        _record(state, "run", f'git commit -m "{subject}"')
    # F7: staging is ensured whenever the repo has at least one commit -- a
    # fresh bootstrap (has_commits just became true above) or a repo adopted
    # with pre-existing history (has_commits was already true) -- not gated
    # behind the zero-commit check above (spec §2.4 step 5, amended).
    if preset in STAGING_PRESETS:
        has_staging = False if repo_pending else _probe_ref(run, canonical, "staging")
        if not has_staging:
            if not dry_run:
                _run_ok(run, ["git", "branch", "staging"], str(canonical), "git branch staging")
            _record(state, "run", "git branch staging")

def _current_head(run, canonical, dry_run):
    if dry_run:
        return None
    result = _probe(run, ["git", "rev-parse", "HEAD"], str(canonical), "git rev-parse HEAD")
    if result.returncode != 0:
        return None
    return result.stdout.strip()

def _do_remote(canonical, run, preset, dry_run, state, dir_name, account):
    name = Path(dir_name).name
    if not dry_run:
        _run_ok(
            run,
            ["gh", "repo", "create", f"{account}/{name}", "--private", "--source", str(canonical),
             "--remote", "origin"],
            str(canonical),
            "gh repo create",
            prefix="remote-failed",
        )
    _record(state, "run", f"gh repo create {account}/{name} --private")
    if not dry_run:
        _run_ok(run, ["git", "push", "-u", "origin", "main"], str(canonical), "git push main", prefix="remote-failed")
    _record(state, "run", "git push -u origin main")
    if preset in STAGING_PRESETS:
        if not dry_run:
            _run_ok(run, ["git", "push", "-u", "origin", "staging"], str(canonical), "git push staging", prefix="remote-failed")
        _record(state, "run", "git push -u origin staging")
        if not dry_run:
            _run_ok(run, ["gh", "repo", "edit", "--default-branch", "staging"], str(canonical), "gh repo edit", prefix="remote-failed")
        _record(state, "run", "gh repo edit --default-branch staging")

def _emit(path, created, kept, ran, refused, head):
    print(json.dumps({
        "path": path,
        "created": created,
        "kept": kept,
        "ran": ran,
        "refused": refused,
        "head": head,
    }))

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="jax-init")
    parser.add_argument("--path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--preset", required=True, choices=list(PRESETS))
    parser.add_argument("--stack", required=True, choices=["nextjs", "python", "bubble", "other"])
    parser.add_argument("--threat-model", required=True, choices=THREAT_MODEL_MODES)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--hooks", choices=["auto", "preserve"], default="auto")
    parser.add_argument("--initial-commit-subject", default="chore: bootstrap")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)

def main(argv=None, *, run=run_command, allowlist_root=ALLOWLIST_ROOT_DEFAULT, env=None, now=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    env = os.environ if env is None else env
    now = (lambda: datetime.now().astimezone()) if now is None else now
    args = parse_args(argv)
    state = {"plan": [], "kept": []}
    path_str = args.path
    canonical = None
    try:
        _validate_name(args.name)
        if not _valid_commit_subject(args.initial_commit_subject):
            raise Refusal("invalid-commit-subject")
        canonical = canonicalize_target(Path(args.path), allowlist_root)
        path_str = str(canonical)
        case = classify_target(canonical, run)
        if case == "B":
            raise Refusal("path-nested-in-repo")
        if case == "D":
            raise Refusal("path-not-empty")
        repo_pending = args.dry_run and case == "A"
        adopted = False
        origin = None
        if not repo_pending and case == "C":
            adopted = _probe_ref(run, canonical, "HEAD")
        if adopted:
            # Every adoption refusal happens BEFORE any setup/baseline mutation.
            origin_result = run(["git", "remote", "get-url", "origin"], cwd=str(canonical))
            if origin_result.returncode == 0 and origin_result.stdout.strip():
                origin = origin_result.stdout.strip()
            if args.remote and origin is None:
                raise Refusal("remote-failed: origin-missing")
            if args.preset in ("single-branch-pr", "dual-branch-pr"):
                if origin is None:
                    raise Refusal("remote-failed: origin-missing")
                if not _GH_REMOTE_RE.match(origin):
                    raise Refusal("remote-failed: origin-not-github")
            _validate_adoption_paths(canonical, allowlist_root)
            hook_action, _label = _hook_plan(run, canonical, allowlist_root, args.hooks)
            if hook_action == "install":
                hooks_dir = _git_metadata_path(
                    run, canonical, ["git", "rev-parse", "--git-path", "hooks"],
                    "git rev-parse --git-path hooks")
                _commit_msg_destination(canonical, allowlist_root, run, hooks_dir / "commit-msg")
        _setup_repo(canonical, case, run, args.dry_run, state)
        branch = "main" if repo_pending else _read_branch(run, canonical)
        if case == "C" and branch == "HEAD":
            raise Refusal("detached-head")
        if args.remote and case == "C" and branch != "main" and not adopted:
            raise Refusal("remote-failed: branch-not-main")
        account = None
        if args.remote and not adopted:
            settings = general_settings.read_settings()
            if not (settings.get("ok") and settings["data"]["integrations"]["github"]):
                # The setting wins over the flag (spec Decision 2) — --remote was explicitly
                # passed, but a disabled integration still refuses before any gh/push call
                # or bootstrap write.
                raise Refusal("github-integration-disabled")
            account = _do_fresh_remote_account(run, canonical, args.dry_run, state)
        _bootstrap_baseline(canonical, allowlist_root, args.name, args.stack, branch, env, now,
                            args.dry_run, state, run, hooks_mode=args.hooks, adopted=adopted)
        if adopted:
            if origin:
                state["kept"].append(f"origin: {origin}")
        else:
            _run_bootstrap_sequence(canonical, run, args.preset, repo_pending, args.dry_run,
                                    state, subject=args.initial_commit_subject)
        if args.remote and not adopted:
            _do_remote(canonical, run, args.preset, args.dry_run, state, path_str, account)
        head = _current_head(run, canonical, args.dry_run)
    except Refusal as exc:
        created, ran = _render_plan(state["plan"], args.dry_run)
        head = None
        if canonical is not None:
            try:
                head = _current_head(run, canonical, args.dry_run)
            except Refusal:
                head = None
        print(f"refused: {exc.code}", file=sys.stderr)
        _emit(path_str, created, state["kept"], ran, exc.code, head)
        return REFUSED
    created, ran = _render_plan(state["plan"], args.dry_run)
    _emit(path_str, created, state["kept"], ran, None, head)
    return OK
if __name__ == "__main__":
    raise SystemExit(main())
