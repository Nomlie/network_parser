#!/usr/bin/env bash

set -euo pipefail

# Usage:
# ./git_commit.sh "Your commit message"

COMMIT_MSG="${1:-}"

if [[ -z "$COMMIT_MSG" ]]; then
    echo "Usage: $0 \"Your commit message\""
    exit 1
fi

# Check if we're inside a git repo
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: This is not a git repository."
    exit 1
fi

echo "Checking git status..."
git status

echo
echo "Adding all changes..."
git add .

echo
echo "Committing changes..."
git commit -m "$COMMIT_MSG" || {
    echo "Nothing to commit or commit failed."
    exit 1
}

# Get current branch name
BRANCH=$(git branch --show-current)

if [[ -z "$BRANCH" ]]; then
    echo "Error: Could not determine current branch."
    exit 1
fi

echo
echo "Pushing to origin/$BRANCH ..."
git push origin "$BRANCH"

echo
echo "Done."