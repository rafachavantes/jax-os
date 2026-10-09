#!/usr/bin/env python3
"""JAXOS_HOME / API base URL resolution -- process env, read once per language, NOT
file-backed config (general_settings.py is that). See src/server/env.ts for the TS twin;
the two MUST agree on every default. MOA-497 Build A. Zero local imports on purpose: every
consumer in scripts/ imports this module, so it must never import any of them back."""
from __future__ import annotations

import os
from pathlib import Path


def jaxos_home() -> Path:
    override = os.environ.get("JAXOS_HOME")
    return Path(override) if override else Path.home() / ".jax-os"


def api_base_url() -> str:
    return f"http://127.0.0.1:{os.environ.get('PORT', '3100')}"
