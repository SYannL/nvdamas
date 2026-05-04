#!/usr/bin/env bash
# Sync nvdamas dev repo -> container-mounted MCMA-main-recover directory.
# Checks for changes in DEST that would be overwritten before proceeding.
set -euo pipefail

SRC=/bigdata/xenial/nvdamas
DEST=/bigdata/xenial/MCMA-main-recover/MCMA-main/nvdamas

SYNC_DIRS=(mas scripts tasks)
SYNC_FILES=(requirements-memrl.txt)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

if [ ! -d "$DEST" ]; then
    echo "[sync] ERROR: DEST not found: $DEST"
    exit 1
fi

# --- Safety check: find files in DEST that differ from SRC ---
echo "[sync] Checking for changes in DEST that would be overwritten..."
CONFLICTS=()
for d in "${SYNC_DIRS[@]}"; do
    while IFS= read -r line; do
        # diff -rq output: "Files A and B differ" or "Only in DEST/..."
        if [[ "$line" == Only\ in\ "$DEST/$d"* ]]; then
            # File exists only in DEST (would be deleted by --delete)
            CONFLICTS+=("$line")
        elif [[ "$line" == Files\ "$DEST/$d"* ]]; then
            # File differs; check if DEST version matches git HEAD in SRC
            dest_file="${line#Files }"
            dest_file="${dest_file% and *}"
            CONFLICTS+=("$line")
        fi
    done < <(diff -rq \
        --exclude="__pycache__" --exclude="*.pyc" \
        "$SRC/$d" "$DEST/$d" 2>/dev/null || true)
done
for f in "${SYNC_FILES[@]}"; do
    if [ -f "$DEST/$f" ] && ! diff -q "$SRC/$f" "$DEST/$f" &>/dev/null; then
        CONFLICTS+=("Files $DEST/$f and $SRC/$f differ")
    fi
done

if [ ${#CONFLICTS[@]} -gt 0 ]; then
    echo ""
    echo "[sync] WARNING: The following DEST files differ from SRC and will be overwritten:"
    for c in "${CONFLICTS[@]}"; do
        echo "  $c"
    done
    echo ""
    if [ $DRY_RUN -eq 1 ]; then
        echo "[sync] Dry-run mode: no files written."
        exit 0
    fi
    read -r -p "[sync] Proceed and overwrite? [y/N] " ans
    if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
        echo "[sync] Aborted."
        exit 1
    fi
else
    echo "[sync] No conflicts in DEST. Proceeding."
    if [ $DRY_RUN -eq 1 ]; then
        echo "[sync] Dry-run mode: no files written."
        exit 0
    fi
fi

# --- Sync ---
for d in "${SYNC_DIRS[@]}"; do
    rsync -a --delete \
        --exclude="__pycache__" --exclude="*.pyc" --exclude=".git" \
        "$SRC/$d/" "$DEST/$d/"
done
for f in "${SYNC_FILES[@]}"; do
    cp "$SRC/$f" "$DEST/$f"
done

echo "[sync] Done."
