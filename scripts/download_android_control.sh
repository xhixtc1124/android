#!/usr/bin/env bash
# Download Google's official AndroidControl dataset (Apache 2.0, ~50 GB) into data/raw/android_control.
# Safe to re-run: finished files are skipped, half-finished ones resume.
#
#   bash scripts/download_android_control.sh          # everything (20 shards + split files)
#   bash scripts/download_android_control.sh 0 1      # only shards 0 and 1
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
OUT="$ROOT/data/raw/android_control"
BASE="https://storage.googleapis.com/gresearch/android_control"
API="https://storage.googleapis.com/storage/v1/b/gresearch/o/android_control%2F"
PARALLEL=4
mkdir -p "$OUT"

# Size Google reports for a file, so we can tell finished downloads from partial ones.
remote_size() {
  curl -sS --fail -m 30 "${API}$1?fields=size" | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["size"])'
}

fetch() {
  local name="$1" want have
  want="$(remote_size "$name")"
  have="$(stat -f%z "$OUT/$name" 2>/dev/null || echo 0)"
  if [[ "$have" == "$want" ]]; then
    echo "ok    $name"
    return 0
  fi
  curl -sS --fail --retry 5 -C - -o "$OUT/$name" "$BASE/$name"
  have="$(stat -f%z "$OUT/$name")"
  if [[ "$have" != "$want" ]]; then
    echo "SIZE MISMATCH $name: have $have, want $want" >&2
    return 1
  fi
  echo "done  $name"
}
export -f fetch remote_size
export OUT BASE API

fetch splits.json
fetch test_subsplits.json

if [[ $# -gt 0 ]]; then shards=("$@"); else shards=($(seq 0 19)); fi
printf 'android_control-%05d-of-00020\n' "${shards[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'fetch "$1"' _ {}
echo "All requested files are in $OUT"
