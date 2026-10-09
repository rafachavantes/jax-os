#!/usr/bin/env python3
"""Mechanical checks for the OSS docs (MOA-500).

    python3 scripts/check_oss_docs.py               # check the real tree
    python3 scripts/check_oss_docs.py --self-test   # fixture-driven self-test

Named `check_oss_docs.py`, not `test_*.py`, so pytest discovery never collects it.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOC_FILES = (
    "README.md",
    "INSTALL.md",
    "docs/ARCHITECTURE.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "LICENSE",
)
DOC_GLOBS = (
    "workflow/contracts/*.md",
    "workflow/templates/*.md",
    "workflow/skills/*/SKILL.md",
)
ENV_KEY_RE = re.compile(r"^([A-Z][A-Z0-9_]*)=", re.MULTILINE)
ENV_BLOCK_RE = re.compile(r"<!-- env-keys:start -->(.*?)<!-- env-keys:end -->", re.DOTALL)
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SCHEME_RE = re.compile(r"^[a-z]+://")
SCREENSHOT_PREFIX = "docs/screenshots/"


def doc_paths(root: Path) -> list[Path]:
    paths = [root / name for name in DOC_FILES]
    for pattern in DOC_GLOBS:
        paths.extend(sorted(root.glob(pattern)))
    return paths


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def check_no_private_paths(root: Path) -> list[str]:
    """(a) No /home/rafa anywhere, no .local/ in README.md.

    Contracts legitimately name `.local/` as the run's own scratch/report path
    (`<repo>/.local/reports/...`); what a clone-reader must never be pointed at
    is README's `.local/` docs directory or any absolute `/home/rafa` path.
    """
    failures = []
    for path in doc_paths(root):
        if not path.exists():
            continue
        for lineno, line in enumerate(_lines(path), 1):
            if "/home/rafa" in line:
                failures.append(f"{path.relative_to(root)}:{lineno}: contains '/home/rafa'")
    readme = root / "README.md"
    if readme.exists():
        for lineno, line in enumerate(_lines(readme), 1):
            if ".local/" in line:
                failures.append(f"README.md:{lineno}: contains '.local/'")
    return failures


def check_readme_no_agent_docs(root: Path) -> list[str]:
    """(b) README.md mentions neither AGENTS.md nor CLAUDE.md."""
    readme = root / "README.md"
    if not readme.exists():
        return ["README.md: missing"]
    failures = []
    for lineno, line in enumerate(_lines(readme), 1):
        for needle in ("AGENTS.md", "CLAUDE.md"):
            if needle in line:
                failures.append(f"README.md:{lineno}: contains '{needle}'")
    return failures


def check_env_keys(root: Path) -> list[str]:
    """(c) INSTALL.md's env-keys block matches .env.example exactly."""
    install = root / "INSTALL.md"
    example = root / ".env.example"
    if not install.exists():
        return ["INSTALL.md: missing"]
    if not example.exists():
        return [".env.example: missing"]
    match = ENV_BLOCK_RE.search(install.read_text(encoding="utf-8"))
    if match is None:
        return ["INSTALL.md: env-keys block markers not found"]
    install_keys = {line.strip() for line in match.group(1).splitlines() if line.strip()}
    example_keys = set(ENV_KEY_RE.findall(example.read_text(encoding="utf-8")))
    failures = [f"INSTALL.md: env-keys block is missing {key}" for key in sorted(example_keys - install_keys)]
    failures += [f".env.example: missing {key} (named in INSTALL.md's env-keys block)" for key in sorted(install_keys - example_keys)]
    return failures


def check_relative_links(root: Path) -> list[str]:
    """(d) Every relative markdown link resolves (docs/screenshots/ excluded)."""
    failures = []
    for path in doc_paths(root):
        if not path.exists():
            continue
        for lineno, line in enumerate(_lines(path), 1):
            for raw in LINK_RE.findall(line):
                target = raw.strip()
                if target.startswith("<") and target.endswith(">"):
                    target = target[1:-1]
                target = target.split(" ", 1)[0].split("#", 1)[0]
                if not target or SCHEME_RE.match(target) or target.startswith(SCREENSHOT_PREFIX):
                    continue
                if not (path.parent / target).exists() and not (root / target).exists():
                    failures.append(f"{path.relative_to(root)}:{lineno}: unresolved link '{target}'")
    return failures


CHECKS = (
    ("(a) no /home/rafa, no .local/ in README.md", check_no_private_paths),
    ("(b) no AGENTS.md/CLAUDE.md in README.md", check_readme_no_agent_docs),
    ("(c) INSTALL.md env-keys block matches .env.example", check_env_keys),
    ("(d) relative markdown links resolve", check_relative_links),
)


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        root = base / "a-pass"
        (root / "workflow" / "contracts").mkdir(parents=True)
        (root / "README.md").write_text("# clean\nNo private paths here.\n")
        (root / "workflow" / "contracts" / "builder-contract.md").write_text(
            "Reports land in `<repo>/.local/reports/<run_id>.md`.\n"
        )
        assert check_no_private_paths(root) == [], "check (a) pass fixture failed"

        root = base / "a-fail"
        root.mkdir()
        (root / "README.md").write_text("Clone into /home/rafa/repos/jax-os.\n")
        assert check_no_private_paths(root), "check (a) fail fixture passed"

        root = base / "b-pass"
        root.mkdir()
        (root / "README.md").write_text("# Jax OS\nSee docs/ARCHITECTURE.md for the rules.\n")
        assert check_readme_no_agent_docs(root) == [], "check (b) pass fixture failed"

        root = base / "b-fail"
        root.mkdir()
        (root / "README.md").write_text("See AGENTS.md for more.\n")
        assert check_readme_no_agent_docs(root), "check (b) fail fixture passed"

        root = base / "c-pass"
        root.mkdir()
        (root / "INSTALL.md").write_text(
            "<!-- env-keys:start -->\nALPHA\nBETA\n<!-- env-keys:end -->\n"
        )
        (root / ".env.example").write_text("ALPHA=1\nBETA=2\n")
        assert check_env_keys(root) == [], "check (c) pass fixture failed"

        root = base / "c-fail"
        root.mkdir()
        (root / "INSTALL.md").write_text(
            "<!-- env-keys:start -->\nALPHA\n<!-- env-keys:end -->\n"
        )
        (root / ".env.example").write_text("ALPHA=1\nBETA=2\n")
        assert check_env_keys(root), "check (c) fail fixture passed"

        root = base / "d-pass"
        (root / "docs").mkdir(parents=True)
        (root / "docs" / "ARCHITECTURE.md").write_text("# Architecture\n")
        (root / "README.md").write_text(
            "[Architecture](docs/ARCHITECTURE.md)\n"
            "![shot](docs/screenshots/mission-control.png)\n"
        )
        assert check_relative_links(root) == [], "check (d) pass fixture failed"

        root = base / "d-fail"
        root.mkdir()
        (root / "README.md").write_text("[missing](docs/NOPE.md)\n")
        assert check_relative_links(root), "check (d) fail fixture passed"

    print("self-test ok")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    failed = False
    for name, check in CHECKS:
        failures = check(ROOT)
        if failures:
            failed = True
            print(f"FAIL: {name}")
            for failure in failures:
                print(f"  {failure}")
        else:
            print(f"ok: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
