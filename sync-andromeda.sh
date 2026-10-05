#!/bin/sh
# Copy this project to the SSH host configured as fl-andromeda.
set -eu

REMOTE_HOST=fl-andromeda
REMOTE_DIR=/home/andromeda/hackathonCalpe2026_POC
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: sh sync-andromeda.sh [--dry-run]

Synchronize this project to:
  fl-andromeda:/home/andromeda/hackathonCalpe2026_POC/

Includes source files, data, documentation and uncommitted changes.
Skips Git, local environments, caches, .env files and experiment outputs.
Remote-only files are preserved. Existing project files are overwritten.
No packages are installed and no application is started or restarted.

  --dry-run  Preview transfers without creating or changing remote files.
  --help     Show this help.
EOF
}

case $# in
    0) ;;
    1)
        case $1 in
            --dry-run) DRY_RUN=1 ;;
            --help|-h) usage; exit 0 ;;
            *) usage >&2; exit 2 ;;
        esac
        ;;
    *) usage >&2; exit 2 ;;
esac

for dependency in ssh rsync; do
    if ! command -v "$dependency" >/dev/null 2>&1; then
        printf 'Error: required local command is missing: %s\n' "$dependency" >&2
        exit 1
    fi
done

if [ ! -f "$SCRIPT_DIR/pyproject.toml" ] || [ ! -f "$SCRIPT_DIR/dashboard.py" ]; then
    printf '%s\n' 'Error: keep this script in the project root.' >&2
    exit 1
fi

printf 'Source: %s/\nDestination: %s:%s/\n' "$SCRIPT_DIR" "$REMOTE_HOST" "$REMOTE_DIR"
printf '%s\n' 'Checking SSH access and remote rsync…'
ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    "$REMOTE_HOST" 'command -v rsync >/dev/null 2>&1 || { echo "Error: install rsync on the server first." >&2; exit 1; }'

if [ "$DRY_RUN" -eq 0 ]; then
    ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR' && test -w '$REMOTE_DIR'"
fi

# Avoid copying macOS ownership/permissions to Linux; preserve times and links.
# No --delete: server-only configuration, results and environments survive.
set -- -rltz --itemize-changes --human-readable --stats --partial \
    --partial-dir=.rsync-partial \
    --exclude=.git/ \
    --exclude=.venv/ \
    --exclude=venv/ \
    --exclude=.cache/ \
    --exclude=.ruff_cache/ \
    --exclude=.pytest_cache/ \
    --exclude=__pycache__/ \
    --exclude='*.py[cod]' \
    --exclude=.DS_Store \
    --include=.env.example \
    --exclude=.env \
    --exclude='.env.*' \
    --exclude=/runs/ \
    --exclude=/gemma_sae_contrastive_runs_ab/ \
    --exclude=/gemma_sae_common_feature_runs/ \
    -e 'ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3'

if [ "$DRY_RUN" -eq 1 ]; then
    set -- "$@" --dry-run
fi

rsync "$@" "$SCRIPT_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

if [ "$DRY_RUN" -eq 1 ]; then
    printf '%s\n' 'Preview complete. No remote files were changed.'
else
    printf '%s\n' 'Synchronization completed successfully.'
fi
