#!/usr/bin/env python3
"""Managed agent-settings consumer (MOA-473 Part 1)."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import stat
import unicodedata
from pathlib import Path

from jax_init import Refusal
import jaxflow_env as jenv

SETTINGS_CAP = 128 * 1024
SETTINGS_PATH = jenv.jaxos_home() / "agent-settings.json"
_MODEL_DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "workflow" / "fixtures" / "model-defaults-v1.json"
_MODEL_DEFAULTS = json.loads(_MODEL_DEFAULTS_PATH.read_text(encoding="utf-8"))
# Caller-routing logic (opposite runtime), not duplicated data -- stays hand-written, times the
# fixture's bare reviewer models (spec: "One source of model defaults").
_OPPOSITE_RUNTIME = {"claude": "codex", "codex": "claude", "jaxos": "codex"}
# MOA-504 D8: when the opposite reviewer agent is off, review on the OTHER reviewer runtime
# (a review is already a fresh sealed session, so the same agent is still a cold reviewer).
_FALLBACK_RUNTIME = {"claude": "claude", "codex": "codex", "jaxos": "claude"}


def reviewer_fallback(caller, runtime):
    """'<opposite> off' when `runtime` is not the caller's opposite reviewer runtime, else None.
    Pure: also derives the fallback of a finished run from its ledger row (caller, runtime)."""
    opposite = _OPPOSITE_RUNTIME.get(caller)
    return None if opposite is None or runtime == opposite else f"{opposite} off"
REVIEWER_DEFAULTS = {
    caller: {
        "runtime": opposite,
        "model": _MODEL_DEFAULTS["reviewers"][opposite]["model"],
        "effort": _MODEL_DEFAULTS["reviewers"][opposite]["effort"],
    }
    for caller, opposite in _OPPOSITE_RUNTIME.items()
}
REVIEWER_EFFORTS = {
    "claude": frozenset({"low", "medium", "high", "xhigh", "max"}),
    "codex": frozenset({"minimal", "low", "medium", "high", "xhigh"}),
}
ROOT_KEYS = ("schema_version", "revision", "reviewers", "builders", "source_revisions")
REVIEWER_FIELD_KEYS = ("model", "effort")
PROFILE_KEYS = ("connection", "model", "effort", "credential", "routing")
SOURCE_FILES = ("opencode.json", "opencode.jsonc")
ROUTING_KEYS = frozenset({
    "sort", "allow_fallbacks", "preferred_min_throughput", "preferred_max_latency",
    "max_price", "only", "ignore", "quantizations",
})
PERCENTILES = frozenset({"p50", "p75", "p90", "p99"})
# Wire order from OpenRouter's provider.quantizations enum; also the UI checkbox order.
QUANTIZATIONS = (
    "int4", "int8", "fp4", "mxfp4", "nvfp4", "fp6", "fp8", "mxfp8", "fp16", "bf16", "fp32", "unknown",
)
QUANTIZATION_SET = frozenset(QUANTIZATIONS)
NATIVE_ENV = frozenset({
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "XAI_API_KEY",
    "DEEPSEEK_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY",
    "GROQ_API_KEY", "MISTRAL_API_KEY", "TOGETHER_API_KEY", "CEREBRAS_API_KEY",
    "COHERE_API_KEY", "FIREWORKS_API_KEY", "PERPLEXITY_API_KEY",
})
CONNECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
EFFORT_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{32}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
JAX_ENV_RE = re.compile(r"^JAX_PROVIDER_[A-Z0-9_]+_API_KEY$")
OPENROUTER_NPM = "@openrouter/ai-sdk-provider"
SOURCE_CAP = 1024 * 1024
CONFIG_STDOUT_CAP = 2 * 1024 * 1024
LOCK_PATH = jenv.jaxos_home() / "agent-settings.lock"
ENV_PATH = jenv.jaxos_home() / ".env"
SOURCE_PATHS = {
    "opencode.json": Path.home() / ".config" / "opencode" / "opencode.json",
    "opencode.jsonc": Path.home() / ".config" / "opencode" / "opencode.jsonc",
}


def _malformed():
    raise Refusal("agent-settings-malformed")


def _exact_keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        _malformed()
    return value


def _is_control_or_space(char):
    return char.isspace() or unicodedata.category(char).startswith("C")


def _require_model(value):
    if type(value) is not str or not value or value.startswith("-"):
        _malformed()
    if any(_is_control_or_space(ch) for ch in value) or len(value.encode("utf-8")) > 256:
        _malformed()


def _finite(value, *, positive=False, nonnegative=False):
    if type(value) not in (int, float):
        return False
    if value != value or value in (float("inf"), float("-inf")):
        return False
    if positive and not (value > 0):
        return False
    if nonnegative and value < 0:
        return False
    return True


def _percentile(value):
    if type(value) is not dict or set(value) - PERCENTILES or len(value) != 1:
        _malformed()
    number = next(iter(value.values()))
    if not _finite(number, positive=True):
        _malformed()


def _price(value):
    if type(value) is not dict or not value or set(value) - {"prompt", "completion"}:
        _malformed()
    for amount in value.values():
        if not _finite(amount, nonnegative=True):
            _malformed()


def _endpoint_ids(value):
    if type(value) is not list or not value or len(value) > 128:
        _malformed()
    seen = set()
    for item in value:
        if type(item) is not str or not item or len(item) > 256:
            _malformed()
        if any(_is_control_or_space(ch) for ch in item) or item in seen:
            _malformed()
        seen.add(item)
    return seen


def _quantizations(value):
    if type(value) is not list or not value:
        _malformed()
    seen = set()
    for item in value:
        if type(item) is not str or item not in QUANTIZATION_SET or item in seen:
            _malformed()
        seen.add(item)


def _routing(value):
    if value is None:
        return
    if type(value) is not dict:
        _malformed()
    keys = set(value)
    if not {"sort", "allow_fallbacks"} <= keys or keys - ROUTING_KEYS:
        _malformed()
    if value["sort"] != "price" or type(value["allow_fallbacks"]) is not bool:
        _malformed()
    if "preferred_min_throughput" in value:
        _percentile(value["preferred_min_throughput"])
    if "preferred_max_latency" in value:
        _percentile(value["preferred_max_latency"])
    if "max_price" in value:
        _price(value["max_price"])
    if "quantizations" in value:
        _quantizations(value["quantizations"])
    only = _endpoint_ids(value["only"]) if "only" in value else set()
    ignore = _endpoint_ids(value["ignore"]) if "ignore" in value else set()
    if only & ignore:
        _malformed()


def _credential(value):
    if type(value) is not dict or "kind" not in value:
        _malformed()
    if value["kind"] == "native":
        if set(value) != {"kind"}:
            _malformed()
        return
    if value["kind"] != "env" or set(value) != {"kind", "env"}:
        _malformed()
    env = value["env"]
    if type(env) is not str:
        _malformed()
    if env in NATIVE_ENV or JAX_ENV_RE.fullmatch(env):
        return
    _malformed()


def _profile(value):
    _exact_keys(value, PROFILE_KEYS)
    if type(value["connection"]) is not str or not CONNECTION_RE.fullmatch(value["connection"]):
        _malformed()
    _require_model(value["model"])
    effort = value["effort"]
    if effort is not None and (type(effort) is not str or not EFFORT_RE.fullmatch(effort)):
        _malformed()
    _credential(value["credential"])
    _routing(value["routing"])


def _reviewer(value, runtime):
    _exact_keys(value, REVIEWER_FIELD_KEYS)
    _require_model(value["model"])
    effort = value["effort"]
    if type(effort) is not str or effort not in REVIEWER_EFFORTS[runtime]:
        _malformed()


def _reject_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            _malformed()
        out[key] = value
    return out


def _reject_constant(_name):
    _malformed()


def decode_settings(raw):
    if type(raw) is not bytes:
        _malformed()
    if len(raw) > SETTINGS_CAP:
        raise Refusal("agent-settings-too-large")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _malformed()
    try:
        parsed = json.loads(text, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
    except Refusal:
        raise
    except (json.JSONDecodeError, ValueError, TypeError, RecursionError):
        _malformed()
    _exact_keys(parsed, ROOT_KEYS)
    if type(parsed["schema_version"]) is not int or parsed["schema_version"] != 1:
        _malformed()
    if type(parsed["revision"]) is not str or not REVISION_RE.fullmatch(parsed["revision"]):
        _malformed()
    reviewers = parsed["reviewers"]
    _exact_keys(reviewers, ("claude", "codex"))
    _reviewer(reviewers["claude"], "claude")
    _reviewer(reviewers["codex"], "codex")
    builders = parsed["builders"]
    _exact_keys(builders, ("default", "fallback"))
    _profile(builders["default"])
    _profile(builders["fallback"])
    sources = parsed["source_revisions"]
    _exact_keys(sources, SOURCE_FILES)
    for token in sources.values():
        if token != "absent" and (type(token) is not str or not SHA256_RE.fullmatch(token)):
            _malformed()
    return parsed


def _reject_symlink_parents(path, *, allow_missing=False):
    current = path.parent
    while True:
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not allow_missing:
                raise Refusal("agent-settings-permissions") from None
            if current.parent == current:
                return
            current = current.parent
            continue
        except OSError as exc:
            raise Refusal("agent-settings-permissions") from exc
        if stat.S_ISLNK(info.st_mode):
            raise Refusal("agent-settings-symlink")
        if current.parent == current:
            return
        current = current.parent


def read_settings(path=None):
    target = SETTINGS_PATH if path is None else Path(path)
    try:
        info = target.lstat()
    except FileNotFoundError:
        _reject_symlink_parents(target, allow_missing=True)
        return None
    except OSError as exc:
        raise Refusal("agent-settings-permissions") from exc
    if stat.S_ISLNK(info.st_mode):
        raise Refusal("agent-settings-symlink")
    if not stat.S_ISREG(info.st_mode):
        raise Refusal("agent-settings-not-file")
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise Refusal("agent-settings-permissions")
    _reject_symlink_parents(target)
    try:
        fd = os.open(str(target), os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise Refusal("agent-settings-permissions") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o600:
            raise Refusal("agent-settings-permissions")
        raw = os.read(fd, SETTINGS_CAP + 1)
    finally:
        os.close(fd)
    if len(raw) > SETTINGS_CAP:
        raise Refusal("agent-settings-too-large")
    return decode_settings(raw)


def _validate_reviewer_override(runtime, *, model, effort):
    if model is not None:
        try:
            _require_model(model)
        except Refusal:
            raise Refusal("reviewer-model-invalid") from None
    if effort is not None and (type(effort) is not str or effort not in REVIEWER_EFFORTS[runtime]):
        raise Refusal("reviewer-effort-invalid")


def select_reviewer(settings, caller, *, model, effort, agents):
    if caller not in REVIEWER_DEFAULTS:
        _malformed()
    runtime = _OPPOSITE_RUNTIME[caller]
    if not agents[runtime]:
        runtime = _FALLBACK_RUNTIME[caller]
        if not agents[runtime]:
            exc = Refusal("no-reviewer-agent")
            exc.hint = "hint: a review needs claude or codex: turn one on in /settings"
            raise exc
    _validate_reviewer_override(runtime, model=model, effort=effort)
    configured = _MODEL_DEFAULTS["reviewers"][runtime] if settings is None else settings["reviewers"][runtime]
    return {
        "runtime": runtime,
        "model": configured["model"] if model is None else model,
        "effort": configured["effort"] if effort is None else effort,
        "fallback": reviewer_fallback(caller, runtime),
    }


def select_builder(settings, *, fallback, builder, model, effort):
    managed = bool(fallback) or builder == "opencode-builder"
    if settings is None:
        if managed:
            raise Refusal("agent-settings-uninitialized")
        return None
    if builder not in (None, "opencode-builder") or model is not None or effort is not None:
        raise Refusal("agent-settings-override")
    return {
        "runtime": "opencode-builder",
        "profile_name": "fallback" if fallback else "default",
    }


def _conflict():
    raise Refusal("agent-profile-conflict")


def _routing_override(container):
    if type(container) is not dict:
        _conflict()
    nested = None
    options = container.get("options")
    if options is not None:
        if type(options) is not dict:
            _conflict()
        if "provider" in options:
            nested = options["provider"]
            if type(nested) is not dict:
                _conflict()
    direct = None
    if "provider" in container:
        direct = container["provider"]
        if type(direct) is not dict:
            _conflict()
    if nested is None:
        return direct
    if direct is None or direct == nested:
        return nested
    _conflict()


def _require_routing(saved, observed, *, required):
    if observed is None:
        if required and saved is not None:
            _conflict()
        if required and saved is None:
            return
        return
    if saved is None or observed != saved:
        _conflict()


def resolve_profile(settings, profile_name, *, effective_config):
    if profile_name not in ("default", "fallback") or type(effective_config) is not dict:
        _conflict()
    profile = settings["builders"][profile_name]
    connection = profile["connection"]
    alias = f"jaxflow-builder-{profile_name}"
    providers = effective_config.get("provider")
    if type(providers) is not dict or connection not in providers:
        _conflict()
    native_provider = providers[connection]
    if type(native_provider) is not dict:
        _conflict()
    npm = native_provider.get("npm")
    routing = profile["routing"]
    if routing is None:
        if npm == OPENROUTER_NPM:
            _conflict()
    elif npm != OPENROUTER_NPM:
        _conflict()
    models = native_provider.get("models")
    if type(models) is not dict or alias not in models:
        _conflict()
    native_model = models[alias]
    if type(native_model) is not dict or native_model.get("id") != profile["model"]:
        _conflict()
    _require_routing(routing, _routing_override(native_model), required=True)
    effort = profile["effort"]
    variants = native_model.get("variants")
    if effort is not None:
        if type(variants) is not dict or effort not in variants:
            _conflict()
        variant = variants[effort]
        if type(variant) is not dict:
            _conflict()
        if npm == OPENROUTER_NPM:
            reasoning = variant.get("reasoning")
            if type(reasoning) is not dict or reasoning.get("effort") != effort:
                _conflict()
        _require_routing(routing, _routing_override(variant), required=False)
    agent = effective_config.get("agent")
    build_agent = None
    if agent is not None:
        if type(agent) is not dict:
            _conflict()
        build_agent = agent.get("build")
        if build_agent is not None and type(build_agent) is not dict:
            _conflict()
    if type(build_agent) is dict:
        _require_routing(routing, _routing_override(build_agent), required=False)
        inherited_variant = build_agent.get("variant")
        if effort is None:
            if inherited_variant is not None:
                _conflict()
        elif inherited_variant is not None and inherited_variant != effort:
            _conflict()
    return {
        "profile_name": profile_name,
        "settings_revision": settings["revision"],
        "model": f"{connection}/{profile['model']}",
        "runtime_model": f"{connection}/{alias}",
        "effort": effort,
        "credential": profile["credential"],
    }


def _source_token(path):
    target = Path(path)
    try:
        info = target.lstat()
    except FileNotFoundError:
        return "absent"
    except OSError as exc:
        raise Refusal("agent-settings-permissions") from exc
    if stat.S_ISLNK(info.st_mode):
        raise Refusal("agent-settings-symlink")
    if not stat.S_ISREG(info.st_mode):
        raise Refusal("agent-settings-not-file")
    try:
        fd = os.open(str(target), os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise Refusal("agent-settings-permissions") from exc
    try:
        raw = os.read(fd, SOURCE_CAP + 1)
    finally:
        os.close(fd)
    if len(raw) > SOURCE_CAP:
        raise Refusal("agent-settings-too-large")
    return hashlib.sha256(raw).hexdigest()


def require_source_revisions(settings, source_paths):
    expected = settings["source_revisions"]
    for name in SOURCE_FILES:
        if _source_token(source_paths[name]) != expected[name]:
            raise Refusal("agent-settings-source-changed")


def _open_directory_nofollow(path, *, create=False):
    target = Path(path)
    if not target.is_absolute():
        target = Path(os.path.abspath(str(target)))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        names = target.parts[1:]
        for i, name in enumerate(names):
            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                if not create:
                    raise Refusal("agent-settings-permissions") from None
                parent = os.fstat(fd)
                if parent.st_uid != os.getuid() or not stat.S_ISDIR(parent.st_mode):
                    raise Refusal("agent-settings-permissions") from None
                try:
                    os.mkdir(name, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise Refusal("agent-settings-permissions") from exc
                try:
                    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                except OSError as exc:
                    raise Refusal("agent-settings-permissions") from exc
            except OSError as exc:
                raise Refusal("agent-settings-permissions") from exc
            if stat.S_ISLNK(info.st_mode):
                raise Refusal("agent-settings-symlink")
            if not stat.S_ISDIR(info.st_mode):
                raise Refusal("agent-settings-permissions")
            try:
                nxt = os.open(name, flags, dir_fd=fd)
            except OSError as exc:
                raise Refusal("agent-settings-permissions") from exc
            os.close(fd)
            fd = nxt
            st = os.fstat(fd)
            if not stat.S_ISDIR(st.st_mode):
                raise Refusal("agent-settings-permissions")
            if i == len(names) - 1 and st.st_uid != os.getuid():
                raise Refusal("agent-settings-permissions")
        return fd
    except Exception:
        os.close(fd)
        raise


def _require_same_directory(parent_fd, path):
    named_fd = _open_directory_nofollow(path, create=False)
    try:
        left = os.fstat(parent_fd)
        right = os.fstat(named_fd)
        if (left.st_ino, left.st_dev) != (right.st_ino, right.st_dev):
            raise Refusal("agent-settings-permissions")
    finally:
        os.close(named_fd)


def _write_all(fd, data):
    view = memoryview(data)
    while len(view):
        n = os.write(fd, view)
        if n == 0:
            raise Refusal("agent-settings-permissions")
        view = view[n:]


def _write_new(parent_fd, name, data, mode):
    """Create `name` under `parent_fd` exclusively (O_EXCL|O_NOFOLLOW), write `data`,
    fchmod to `mode`, fsync, close. On any failure: close, best-effort unlink, re-raise."""
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
    try:
        os.fchmod(fd, mode)
        _write_all(fd, data)
        os.fsync(fd)
    except Exception:
        os.close(fd)
        try:
            os.unlink(name, dir_fd=parent_fd)
        except OSError:
            pass
        raise
    os.close(fd)


@contextlib.contextmanager
def configuration_claim(lock_path, *, create=False):
    target = Path(lock_path)
    parent_fd = _open_directory_nofollow(target.parent, create=create)
    fd = None
    try:
        opened = os.fstat(parent_fd)
        try:
            named = os.stat(str(target.parent), follow_symlinks=False)
        except OSError as exc:
            raise Refusal("agent-settings-permissions") from exc
        if (
            not stat.S_ISDIR(named.st_mode)
            or named.st_uid != os.getuid()
            or (named.st_ino, named.st_dev) != (opened.st_ino, opened.st_dev)
        ):
            raise Refusal("agent-settings-permissions")
        try:
            existing = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        except OSError as exc:
            raise Refusal("agent-settings-permissions") from exc
        if existing is not None:
            if stat.S_ISLNK(existing.st_mode):
                raise Refusal("agent-settings-symlink")
            if not stat.S_ISREG(existing.st_mode) or existing.st_uid != os.getuid():
                raise Refusal("agent-settings-permissions")
        try:
            fd = os.open(
                target.name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd,
            )
        except OSError as exc:
            raise Refusal("agent-settings-permissions") from exc
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
            raise Refusal("agent-settings-permissions")
        try:
            named_lock = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise Refusal("agent-settings-permissions") from exc
        if (named_lock.st_ino, named_lock.st_dev) != (st.st_ino, st.st_dev):
            raise Refusal("agent-settings-permissions")
        os.fchmod(fd, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refusal("agent-settings-permissions") from exc
        named_fd = _open_directory_nofollow(target.parent, create=False)
        try:
            left = os.fstat(parent_fd)
            right = os.fstat(named_fd)
            if (left.st_ino, left.st_dev) != (right.st_ino, right.st_dev):
                raise Refusal("agent-settings-permissions")
            try:
                named_lock = os.stat(target.name, dir_fd=named_fd, follow_symlinks=False)
            except OSError as exc:
                raise Refusal("agent-settings-permissions") from exc
            held = os.fstat(fd)
            if (named_lock.st_ino, named_lock.st_dev) != (held.st_ino, held.st_dev):
                raise Refusal("agent-settings-permissions")
        finally:
            os.close(named_fd)
        yield
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


def _unescape_double(value):
    out = []
    i = 0
    escapes = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}
    while i < len(value):
        if value[i] == "\\" and i + 1 < len(value):
            out.append(escapes.get(value[i + 1], value[i + 1]))
            i += 2
            continue
        out.append(value[i])
        i += 1
    return "".join(out)


def _unquote_env(value):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        inner = value[1:-1]
        if value[0] == '"':
            return _unescape_double(inner)
        return inner
    return value


def _selected_env_value(raw, selected):
    found = None
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key != selected:
            continue
        if found is not None:
            raise Refusal("agent-settings-malformed")
        found = _unquote_env(value)
    if found is None or found == "":
        raise Refusal("agent-settings-malformed")
    return found


def builder_environment(profile, inherited, *, env_path):
    out = dict(inherited)
    out.pop("BWS_ACCESS_TOKEN", None)
    credential = profile["credential"]
    if credential["kind"] != "env":
        return out
    target = Path(env_path)
    try:
        info = target.lstat()
    except OSError as exc:
        raise Refusal("agent-settings-permissions") from exc
    if stat.S_ISLNK(info.st_mode):
        raise Refusal("agent-settings-symlink")
    if not stat.S_ISREG(info.st_mode):
        raise Refusal("agent-settings-not-file")
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise Refusal("agent-settings-permissions")
    try:
        fd = os.open(str(target), os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise Refusal("agent-settings-permissions") from exc
    try:
        raw = os.read(fd, SETTINGS_CAP + 1)
    finally:
        os.close(fd)
    if len(raw) > SETTINGS_CAP:
        raise Refusal("agent-settings-too-large")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Refusal("agent-settings-malformed") from exc
    out[credential["env"]] = _selected_env_value(text, credential["env"])
    return out


def read_effective_opencode_config(*, run, env, repo):
    try:
        result = run(
            ["opencode", "debug", "config", "--pure"],
            cwd=str(repo),
            env=env,
        )
    except TypeError:
        result = run(["opencode", "debug", "config", "--pure"], cwd=str(repo))
    if getattr(result, "returncode", 1) != 0:
        _conflict()
    stdout = result.stdout
    if type(stdout) is bytes:
        if len(stdout) > CONFIG_STDOUT_CAP:
            _conflict()
        try:
            text = stdout.decode("utf-8")
        except UnicodeDecodeError:
            _conflict()
    elif type(stdout) is str:
        if len(stdout.encode("utf-8")) > CONFIG_STDOUT_CAP:
            _conflict()
        text = stdout
    else:
        _conflict()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        _conflict()
    if type(parsed) is not dict:
        _conflict()
    return parsed
