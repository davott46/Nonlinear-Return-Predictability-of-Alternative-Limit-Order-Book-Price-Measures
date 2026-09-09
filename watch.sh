#!/usr/bin/env bash
# Watch a run directory: finished (stock, target) units per stage, read off the checkpoint files.
# Usage: ./watch.sh model_outputs/runs/<name>        (./watch.sh <run_dir> --once prints the status once)
RUN="${1:?usage: $0 <run_dir> [--once]}"
if [ "$2" != "--once" ]; then
    exec watch -n 10 "$0" "$RUN" --once
fi

python3 - "$RUN" <<'EOF'
import json, os, sys
run = sys.argv[1]
m = json.load(open(f"{run}/manifest.json"))
n_t = len(m["target_cols"])
print(f'{m["run_id"]}  status: {m["status"]}  targets: {n_t}')
stages = [("trials", m.get("tune_symbols", m["symbols"]))] + [(f"partial/{leg}", m["symbols"]) for leg in m.get("legs", [])]
for stage, symbols in stages:
    counts = {s: len([f for f in os.listdir(f"{run}/{stage}/{s}") if f.endswith(".parquet")])
              if os.path.isdir(f"{run}/{stage}/{s}") else 0 for s in symbols}
    done = sum(c == n_t for c in counts.values())
    busy = "  ".join(f"{s} {c}/{n_t}" for s, c in counts.items() if 0 < c < n_t)
    print(f"{stage:<16}{done:>3}/{len(symbols)} stocks done   {busy}")
EOF
[ -r /sys/fs/cgroup/memory.max ] && echo "mem: $(awk '$1=="anon"{printf "%.1fGi", $2/1073741824}' /sys/fs/cgroup/memory.stat) anon / $(numfmt --to=iec < /sys/fs/cgroup/memory.max) limit"
