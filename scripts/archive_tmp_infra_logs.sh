#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${SRC_DIR:-/tmp/skyrl-logs}"
DEST_DIR="${DEST_DIR:-./tmp/archived_infra_logs}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"

mkdir -p "$DEST_DIR"

copy_once() {
  local found=0
  shopt -s nullglob
  for src in "$SRC_DIR"/infra-*.log; do
    found=1
    cp -p "$src" "$DEST_DIR"/
  done
  shopt -u nullglob

  if [[ "$found" -eq 0 ]]; then
    printf '%s no infra logs found under %s\n' "$(date '+%F %T')" "$SRC_DIR"
  else
    printf '%s copied infra logs from %s to %s\n' "$(date '+%F %T')" "$SRC_DIR" "$DEST_DIR"
  fi
}

while true; do
  copy_once
  sleep "$INTERVAL_SECONDS"
done
