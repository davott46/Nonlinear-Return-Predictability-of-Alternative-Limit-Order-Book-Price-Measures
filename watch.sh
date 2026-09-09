# Watch progress in a directory: finished parquets + file count per subdir.
# Usage: ./watch.sh <dir> [total]
export P="${1:?usage: $0 <dir> [total]}" T="${2:+/$2}"
watch -n 10 '
echo "done: $(ls "$P"/*.parquet 2>/dev/null | xargs -rn1 basename | tr "\n" " ")";
echo "mem: $(awk "\$1==\"anon\"{printf \"%.1fGi\", \$2/1073741824}" /sys/fs/cgroup/memory.stat) anon / $(numfmt --to=iec < /sys/fs/cgroup/memory.max) limit";
for d in "$P"/*/; do [ -d "$d" ] && echo "$(basename "$d"): $(ls "$d" | wc -l)$T"; done'
