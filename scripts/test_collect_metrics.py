#!/usr/bin/env python3
"""Stdlib-only tests for the pure parse functions. Run: python3 scripts/test_collect_metrics.py"""
import collect_metrics as cm

STAT_A = "cpu  100 0 50 800 50 0 0 0 0 0\nintr 0\n"
STAT_B = "cpu  140 0 70 860 70 0 0 0 0 0\nintr 0\n"

MEMINFO = """MemTotal:       15988000 kB
MemFree:          880000 kB
MemAvailable:    5157000 kB
SwapTotal:             0 kB
SwapFree:              0 kB
"""

def test_cpu():
    a, b = cm.parse_proc_stat(STAT_A), cm.parse_proc_stat(STAT_B)
    # idle = idle + iowait (pinned): A idle=850 total=1000; B idle=930 total=1140
    # delta total=140, delta idle=80 -> busy 60/140
    assert abs(cm.cpu_pct_from(a, b) - 42.857) < 0.01, cm.cpu_pct_from(a, b)

def test_meminfo():
    m = cm.parse_meminfo(MEMINFO)
    assert m["mem_total_mb"] == 15613  # 15988000 kB // 1024
    assert m["mem_used_mb"] == 15613 - 5036  # total - available
    assert m["swap_used_mb"] == 0

def test_loadavg():
    l1, l5, l15 = cm.parse_loadavg("1.76 1.17 0.89 3/2154 12345\n")
    assert (l1, l5, l15) == (1.76, 1.17, 0.89)

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all tests passed")

# ---- monitoredUnits from settings (spec: monitoredUnits consumer) ----

import json


def test_probe_services_samples_exactly_the_given_lists(monkeypatch):
    monkeypatch.setattr(cm, "is_active", lambda unit, user: "active")
    result = cm.probe_services(["a.service"], ["b.service"])
    assert set(result["units"].keys()) == {"a.service", "b.service"}


def test_probe_services_with_empty_lists_samples_nothing(monkeypatch):
    monkeypatch.setattr(cm, "is_active", lambda unit, user: "active")
    result = cm.probe_services([], [])
    assert result["units"] == {}


def test_collect_threads_monitored_units_into_services(monkeypatch):
    monkeypatch.setattr(cm, "is_active", lambda unit, user: "active")
    sample = cm.collect(user_units=["u.service"], system_units=["s.service"])
    services = json.loads(sample["services"])
    assert set(services["units"].keys()) == {"u.service", "s.service"}


def test_monitored_units_from_settings_defaults_to_empty_on_not_ok():
    assert cm._monitored_units_from_settings({"ok": False, "error": "settings-malformed"}) == ([], [])


def test_monitored_units_from_settings_reads_configured_lists():
    result = {"ok": True, "data": {"monitoredUnits": {"user": ["a"], "system": ["b"]}}}
    assert cm._monitored_units_from_settings(result) == (["a"], ["b"])
