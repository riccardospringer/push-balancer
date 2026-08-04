#!/usr/bin/env bash
set -euo pipefail

TARGET_REPO="${1:-.}"
cd "$TARGET_REPO"

if [[ "$(git rev-parse --is-bare-repository 2>/dev/null || echo false)" == "true" ]]; then
  echo "Surface scan requires a non-bare working tree." >&2
  exit 1
fi

PATTERN='(push-frontend\.bildcms\.de|push-balancer\.onrender\.com|\.as-infra\.de|/next/nmt/|@axelspringer\.com|@bild\.de|next-aws-secret-manager|next-aws-ssm-parameter-store|145\.243\.0\.0/16|91\.220\.134\.0/24|PUSH_SYNC_SECRET=[^[:space:]]+)'

if git grep -nE "$PATTERN" -- . \
  ':(exclude)scripts/check-public-surface.sh' \
  ':(exclude)scripts/check-public-history.sh' \
  ':(exclude)scripts/rewrite-public-history.sh'
then
  echo "Potential public-release blockers remain in tracked files." >&2
  exit 1
fi

echo "No obvious public-release blockers found in tracked files."
