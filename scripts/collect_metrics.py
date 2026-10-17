#!/usr/bin/env python3
"""jax-os metrics collector — runs once per systemd user timer tick (1 min).

Samples host + services into ~/.jax-os/jaxos.db (metrics table), aggregates
completed hours into metrics_hourly, prunes (raw >7d, hourly >90d), and keeps
a daily jaxos.db.bak. Stdlib only, plus the local jaxflow_env resolver. The
Next.js app OWNS the schema (migrations via PRAGMA user_version); this script
exits silently until user_version >= 2.
"""
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jaxflow_env as jenv
import general_settings

DB_PATH = jenv.jaxos_home() / "jaxos.db"
BACKUP_PATH = jenv.jaxos_home() / "jaxos.db.bak"


def _monitored_units_from_settings(result):
    """{ok:false} (malformed/unreadable settings.json) samples nothing and lets `main()` log
    one stderr line -- never a crash (spec: monitoredUnits consumer). {ok:true} already
    defaults an absent monitoredUnits/user/system independently (foundation reader), so this
    function only ever branches on `ok`."""
    if not result["ok"]:
        return [], []
    units = result["data"]["monitoredUnits"]
    return units["user"], units["system"]

RAW_KEEP_DAYS = 7
HOURLY_KEEP_DAYS = 90

# ---- pure parse functions (tested by test_collect_metrics.py) ----

def parse_proc_stat(text):
    """First 'cpu ' line -> dict with total and idle jiffies. idle = idle + iowait (pinned)."""
    fields = [int(x) for x in text.splitlines()[0].split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    return {"total": sum(fields), "idle": idle}

def cpu_pct_from(a, b):
    dt = b["total"] - a["total"]
    if dt <= 0:
        return None
    return 100.0 * (dt - (b["idle"] - a["idle"])) / dt

def parse_meminfo(text):
    kv = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            kv[parts[0][:-1]] = int(parts[1])  # kB
    total = kv["MemTotal"] // 1024
    used = total - kv["MemAvailable"] // 1024
    swap_used = (kv.get("SwapTotal", 0) - kv.get("SwapFree", 0)) // 1024
    return {"mem_total_mb": total, "mem_used_mb": used, "swap_used_mb": swap_used}

def parse_loadavg(text):
    parts = text.split()
    return float(parts[0]), float(parts[1]), float(parts[2])

# ---- probes (each fails independently -> None fields, sample still lands) ----

def probe_cpu():
    with open("/proc/stat") as f:
        a = parse_proc_stat(f.read())
    time.sleep(1)
    with open("/proc/stat") as f:
        b = parse_proc_stat(f.read())
    return cpu_pct_from(a, b)

def probe_disk():
    st = os.statvfs("/")
    used_b = (st.f_blocks - st.f_bfree) * st.f_frsize
    avail_b = st.f_bavail * st.f_frsize
    # df-parity: total for the % denominator is used+avail (root-reserved excluded);
    # we store used and used+avail so the UI's used/total matches df's %.
    return round(used_b / 1e9, 2), round((used_b + avail_b) / 1e9, 2)

def is_active(unit, user):
    cmd = ["systemctl"] + (["--user"] if user else []) + ["is-active", unit]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        return out if out in ("active", "inactive", "failed") else "unknown"
    except Exception:
        return "unknown"

def probe_services(user_units, system_units):
    units = {}
    for u in user_units:
        units[u] = is_active(u, user=True)
    for u in system_units:
        units[u] = is_active(u, user=False)
    containers = []
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.State}}"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        for line in out.splitlines():
            name, _, state = line.partition("\t")
            if name:
                containers.append({"name": name, "state": state})
    except Exception:
        pass  # docker down -> empty list; the docker.service status says why
    return {"units": units, "containers": containers}

# ---- main ----

def collect(now=None, *, user_units=(), system_units=()):
    sample = {"ts": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%S.000Z")}
    try:
        sample["cpu_pct"] = probe_cpu()
    except Exception:
        sample["cpu_pct"] = None
    try:
        with open("/proc/meminfo") as f:
            sample.update(parse_meminfo(f.read()))
    except Exception:
        sample.update({"mem_total_mb": None, "mem_used_mb": None, "swap_used_mb": None})
    try:
        used, total = probe_disk()
        sample["disk_used_gb"], sample["disk_total_gb"] = used, total
    except Exception:
        sample["disk_used_gb"] = sample["disk_total_gb"] = None
    try:
        with open("/proc/loadavg") as f:
            sample["load1"], sample["load5"], sample["load15"] = parse_loadavg(f.read())
    except Exception:
        sample["load1"] = sample["load5"] = sample["load15"] = None
    try:
        with open("/proc/uptime") as f:
            sample["uptime_s"] = int(float(f.read().split()[0]))
    except Exception:
        sample["uptime_s"] = None
    sample["services"] = json.dumps(probe_services(user_units, system_units))
    return sample

AGGREGATE_SQL = """
INSERT OR REPLACE INTO metrics_hourly
  (hour, cpu_pct_avg, cpu_pct_max, mem_used_mb_avg, mem_used_mb_max, load5_max, disk_used_gb_last, samples)
SELECT substr(ts, 1, 13) AS h,
       avg(cpu_pct), max(cpu_pct),
       CAST(avg(mem_used_mb) AS INTEGER), max(mem_used_mb),
       max(load5),
       (SELECT m2.disk_used_gb FROM metrics m2
         WHERE substr(m2.ts, 1, 13) = substr(m.ts, 1, 13)
         ORDER BY m2.ts DESC LIMIT 1),
       count(*)
FROM metrics m
WHERE substr(ts, 1, 13) < ?
  AND substr(ts, 1, 13) NOT IN (SELECT hour FROM metrics_hourly)
GROUP BY substr(ts, 1, 13)
"""

def main():
    if not DB_PATH.exists():
        return 0  # app has never run — nothing to write into
    db = sqlite3.connect(DB_PATH, timeout=5)
    try:
        if db.execute("PRAGMA user_version").fetchone()[0] < 2:
            return 0  # schema not migrated yet — the app owns migrations
        user_units, system_units = _monitored_units_from_settings(general_settings.read_settings())
        if not user_units and not system_units:
            print("monitoredUnits: settings.json absent/malformed/empty — monitoring nothing this tick", file=sys.stderr)
        now = datetime.now(timezone.utc)
        sample = collect(now, user_units=user_units, system_units=system_units)
        cur_hour = now.strftime("%Y-%m-%dT%H")
        raw_cutoff = (now - timedelta(days=RAW_KEEP_DAYS)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        hourly_cutoff = (now - timedelta(days=HOURLY_KEEP_DAYS)).strftime("%Y-%m-%dT%H")
        with db:  # single transaction: insert + aggregate + prune
            db.execute(
                """INSERT INTO metrics (ts, cpu_pct, mem_used_mb, mem_total_mb, swap_used_mb,
                     disk_used_gb, disk_total_gb, load1, load5, load15, uptime_s, services)
                   VALUES (:ts, :cpu_pct, :mem_used_mb, :mem_total_mb, :swap_used_mb,
                     :disk_used_gb, :disk_total_gb, :load1, :load5, :load15, :uptime_s, :services)""",
                sample,
            )
            db.execute(AGGREGATE_SQL, (cur_hour,))
            db.execute("DELETE FROM metrics WHERE ts < ?", (raw_cutoff,))
            db.execute("DELETE FROM metrics_hourly WHERE hour < ?", (hourly_cutoff,))
        backup_if_due(db)
        return 0
    finally:
        db.close()

def backup_if_due(db):
    try:
        if BACKUP_PATH.exists() and time.time() - BACKUP_PATH.stat().st_mtime < 86400:
            return
        tmp = BACKUP_PATH.with_suffix(".bak.tmp")
        dst = sqlite3.connect(tmp)
        with dst:
            db.backup(dst)
        dst.close()
        tmp.replace(BACKUP_PATH)  # atomic
    except Exception as e:
        print(f"backup failed: {e}", file=sys.stderr)  # non-fatal

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"collect-metrics failed: {e}", file=sys.stderr)
        sys.exit(1)
