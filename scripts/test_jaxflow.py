#!/usr/bin/env python3
"""Stdlib-only tests for jaxflow.py (slice a)."""
import contextlib
import io
import json
import multiprocessing
import os
import re
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import time
import urllib.error
from subprocess import CompletedProcess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

import general_settings
import jaxflow
import jaxflow_cli
import jaxflow_worker
import jaxflow_build
import jaxflow_review
import jaxflow_merge
import jaxflow_workerkit
import jaxflow_common
import uuid
import shlex
import jaxflow_hook
import jaxflow_run as jr
import jaxflow_settings as jset
import jax_init as ji
from testkit import (  # noqa: F401
    FakeBuilderPopen,
    FakePopen,
    FakeTmux,
    _ALL_AGENTS_ON,
    _CAPTURED_THREAD,
    _DUAL_PR_AGENTS,
    _E2EBuilderPopen,
    _MergeArgs,
    _STATUS_TEMPLATE,
    _TEST_CLAUDE_SESSION_ID,
    _agents,
    _agents_setting,
    _assert_no_reservation,
    _build_args,
    _capture_builder_popen,
    _checkpoint_path,
    _completed,
    _diff_args,
    _fixed_now,
    _forbidden_tmux,
    _fresh_db,
    _git,
    _init_repo,
    _init_separate_git_dir_repo,
    _init_worktree,
    _insert,
    _install_settings,
    _integration_on_by_default,
    _isolate_agent_settings,
    _isolate_callbacks,
    _ledger_post,
    _managed_worker_repo,
    _merge_env,
    _merge_runner,
    _native_config,
    _never_post,
    _plan_file,
    _pr_open_args,
    _probe_dispatch_build,
    _restore_signal_handlers,
    _review_args,
    _rewrite_json,
    _run_builder_worker_test,
    _run_diff_worker_and_capture_prompt,
    _run_real,
    _run_with_tmux,
    _run_with_tmux_and_log,
    _seed_finished_build,
    _seed_resumable,
    _spec_file,
    _stub_launch_paths,
    _switch_aware_runner,
    _worktree_path,
    _write_manifest_for_builder_worker,
    _write_manifest_for_diff_worker,
    _write_manifest_for_worker,
    _write_plan,
    _write_status_md,
)



