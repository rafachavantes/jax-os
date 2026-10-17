#!/usr/bin/env python3
"""jaxflow -- entry shim (lean spec P2). The code lives in jaxflow_{common,workerkit,merge,review,
build,worker,cli}.py; this path stays because hooks, `bin/jaxflow`, `~/.local/bin/jaxflow-*`, the
spawned worker (`SCRIPT_PATH --run-worker`) and `import jaxflow` consumers (workflow_poll,
jaxflow_settings_io) use it. Re-exported names are SNAPSHOTS: tests patch the DEFINING module."""
from __future__ import annotations

from jax_init import ALLOWLIST_ROOT_DEFAULT, _contained  # noqa: F401
from jaxflow_common import (  # noqa: F401
    _RUN_ID_RE,
    _delete_spool,
    _is_strict_descendant,
    _manifest_dir,
    _open_ro,
    _post_event,
    _spool_path,
    _write_spool,
)
from jaxflow_workerkit import (  # noqa: F401
    _persist_resume_checkpoint,
    _send_callback,
)
from jaxflow_worker import (  # noqa: F401
    _pure_config_run,
)
from jaxflow_cli import (  # noqa: F401
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
