#!/usr/bin/env bash

set -euo pipefail

COMMIT_MSG="${1:-}"

if [[ -z "$COMMIT_MSG" ]]; then
    echo "Usage: $0 \"Your commit message\""
    exit 1
fi

# Find repo root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "Error: Not inside a git repository."
    exit 1
}

echo "Repo root: $REPO_ROOT"
echo

echo "Checking git status..."
git -C "$REPO_ROOT" status

echo
echo "Adding all tracked/untracked changes from repo root..."
git -C "$REPO_ROOT" add -A

echo
echo "Committing changes..."
if git -C "$REPO_ROOT" commit -m "$COMMIT_MSG"; then
    BRANCH=$(git -C "$REPO_ROOT" branch --show-current)

    echo
    echo "Pushing to origin/$BRANCH ..."
    git -C "$REPO_ROOT" push origin "$BRANCH"

    echo
    echo "Done."
else
    echo "Nothing to commit."
fi
