#!/usr/bin/env python3
"""
llama_stats.py — llama-server configuration and runtime stats

Usage:
  python llama_stats.py              — show server defaults + slot states
  python llama_stats.py --raw        — dump full /props JSON
  python llama_stats.py --live       — poll slots at 50ms, capture active params
  python llama_stats.py --slots-raw  — dump raw /slots JSON once
"""

import json
import sys
import time
import urllib.request
import urllib.error

HOST = "http://127.0.0.1:8080"
RAW       = "--raw"       in sys.argv
LIVE      = "--live"      in sys.argv
SLOTS_RAW = "--slots-raw" in sys.argv

PARAM_LABELS = {
    "n_predict":          "num_predict    (max_tokens)",
    "max_tokens":         "max_tokens",
    "seed":               "seed",
    "temperature":        "temperature",
    "top_k":              "top_k",
    "top_p":              "top_p",
    "min_p":              "min_p",
    "typical_p":          "typical_p",
    "repeat_last_n":      "repeat_last_n",
    "repeat_penalty":     "repeat_penalty",
    "presence_penalty":   "presence_penalty",
    "frequency_penalty":  "frequency_penalty",
    "mirostat":           "mirostat",
    "mirostat_tau":       "mirostat_tau",
    "mirostat_eta":       "mirostat_eta",
    "n_keep":             "num_keep",
    "dry_multiplier":     "dry_multiplier",
    "samplers":           "sampler pipeline",
    "reasoning_format":   "reasoning_format",
    "chat_format":        "chat_format",
}


def fetch(path: str):
    try:
        with urllib.request.urlopen(HOST + path, timeout=5) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        print(f"\n  ✗ Cannot reach llama-server at {HOST}: {e}")
        sys.exit(1)


def sep(title: str = "") -> None:
    print(f"\n{'─' * 56}")
    if title:
        print(f"  {title}")
        print(f"{'─' * 56}")


def pv(label: str, val, changed: bool = False) -> None:
    if isinstance(val, list):
        val = " → ".join(str(v) for v in val) if val else "(none)"
    tag = "  ◀ OVERRIDE" if changed else ""
    print(f"  {label:<44} {val}{tag}")


def extract_params(slot: dict) -> dict:
    """Params are directly on the slot as slot['params']."""
    return slot.get("params", {})


def print_slot_params(slot: dict, defaults: dict) -> None:
    sid   = slot.get("id", "?")
    ctx   = slot.get("n_ctx", "?")
    task  = slot.get("id_task", "?")
    nxt   = slot.get("next_token", [{}])
    if isinstance(nxt, list):
        nxt = nxt[0] if nxt else {}
    n_gen = nxt.get("n_decoded", 0)
    n_rem = nxt.get("n_remain", "?")

    print(f"\n  ┌─ slot {sid}  [processing]  task={task}  "
          f"ctx={ctx}  tokens_out={n_gen}  remaining={n_rem}")

    params = extract_params(slot)
    if not params:
        print("  │  (no params in slot data)")
        print("  └─")
        return

    for key, label in PARAM_LABELS.items():
        val = params.get(key)
        if val is None:
            continue
        default_val = defaults.get(key)
        changed = default_val is not None and val != default_val
        tag = "  ◀ OVERRIDE" if changed else ""
        if isinstance(val, list):
            val = " → ".join(str(v) for v in val) if val else "(none)"
        print(f"  │  {label:<42} {val}{tag}")
    print("  └─")


def main():
    health = fetch("/health")
    props  = fetch("/props")

    # ── raw dumps ─────────────────────────────────────────
    if RAW:
        print("\n── Raw /props ──")
        print(json.dumps(props, indent=2))
        return

    if SLOTS_RAW:
        print("\n── Raw /slots ──")
        print(json.dumps(fetch("/slots"), indent=2))
        return

    dgs      = props.get("default_generation_settings", {})
    defaults = dgs.get("params", {})

    # ── header ────────────────────────────────────────────
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║              llama-server  live stats                  ║")
    print("╚════════════════════════════════════════════════════════╝")
    print("  ⚠  /props shows SERVER DEFAULTS only — never changes.")
    print("     Per-request overrides appear in slots while active.")

    sep("Server")
    pv("status",      health.get("status", "?"))
    pv("build",       props.get("build_info", "?"))
    pv("model",       props.get("model_alias", "?"))
    pv("total_slots", props.get("total_slots", "?"))

    sep("Server defaults  (from /props)")
    pv("n_ctx  (context per slot)", dgs.get("n_ctx", "?"))
    for key, label in PARAM_LABELS.items():
        val = defaults.get(key)
        if val is not None:
            pv(label, val)

    sep("Current slot states")
    slots = fetch("/slots")
    for s in slots:
        sid   = s.get("id", "?")
        state = "processing" if s.get("is_processing", False) else "idle"
        ctx   = s.get("n_ctx", "?")
        print(f"  slot {sid}  [{state}]  n_ctx={ctx}")
    print()

    if not LIVE:
        print("  Tip: run with --live to catch per-request params in real time.")
        print("       run with --slots-raw to inspect raw slot JSON structure.")
        return

    # ── live mode: 50ms poll ──────────────────────────────
    sep("LIVE MODE  (50ms poll — send a request now)")
    print("  ◀ OVERRIDE marks values that differ from server defaults.")
    print("  Press Ctrl+C to stop.\n")

    last_seen_tasks: set = set()
    snapshots_caught = 0

    try:
        while True:
            time.sleep(0.05)   # 50ms
            slots = fetch("/slots")
            for s in slots:
                if not s.get("is_processing", False):
                    continue
                task_id = s.get("id_task", s.get("id"))
                if task_id in last_seen_tasks:
                    continue
                last_seen_tasks.add(task_id)
                snapshots_caught += 1
                ts = time.strftime("%H:%M:%S")
                print(f"\n  ═══ [{ts}] caught active slot "
                      f"(snapshot #{snapshots_caught}) ═══")
                print_slot_params(s, defaults)
                if len(last_seen_tasks) > 200:
                    last_seen_tasks.clear()

    except KeyboardInterrupt:
        print(f"\n\n  Stopped. Caught {snapshots_caught} active slot snapshot(s).")


if __name__ == "__main__":
    main()
