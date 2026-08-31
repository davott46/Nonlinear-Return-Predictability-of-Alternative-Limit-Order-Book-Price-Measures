#!/bin/bash
# Watch progress in a directory: finished parquets + file count per subdir.
# Usage: ./watch.sh <dir> [total]
#   e.g. ./watch.sh data/processed 126, ./watch.sh model_outputs/XGBoost/runs/<name>/trials
export P="${1:?usage: $0 <dir> [total]}" T="${2:+/$2}"
watch -n 10 '
echo "done: $(ls "$P"/*.parquet 2>/dev/null | xargs -rn1 basename | tr "\n" " ")";
for d in "$P"/*/; do [ -d "$d" ] && echo "$(basename "$d"): $(ls "$d" | wc -l)$T"; done'
