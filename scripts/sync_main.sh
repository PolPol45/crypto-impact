#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[sync] repo: $REPO_ROOT"
git fetch origin
git checkout main
git pull --ff-only origin main

echo -n "[sync] HEAD: "
git rev-parse --short HEAD
echo "[sync] status:"
git status -sb
