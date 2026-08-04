#!/usr/bin/env bash
set -euo pipefail

TARGET_REPO="${1:-.}"
cd "$TARGET_REPO"

ARTIFACT_PATTERN='(^|/)(push-snapshot\.json|\.ml_lgbm\.joblib|\.sklearn_gbrt\.joblib|\.gbrt_model\.json|\.tuning_state\.json)$'
CONTENT_PATTERN='(push-frontend\.bildcms\.de|push-balancer\.onrender\.com|\.as-infra\.de|/next/nmt/|@axelspringer\.com|@bild\.de|next-aws-secret-manager|next-aws-ssm-parameter-store|145\.243\.0\.0/16|91\.220\.134\.0/24|PUSH_SYNC_SECRET=[^[:space:]]+)'

if git rev-list --objects --all | rg "$ARTIFACT_PATTERN"; then
  echo "Blocked public-release artifacts are still present in git history." >&2
  exit 1
fi

matches_file="$(mktemp)"
while IFS= read -r rev; do
  git grep -nIE "$CONTENT_PATTERN" "$rev" -- \
    ':(exclude)scripts/check-public-surface.sh' \
    ':(exclude)scripts/check-public-history.sh' \
    ':(exclude)scripts/rewrite-public-history.sh' \
    || true
done < <(git rev-list --all) > "$matches_file"

if [[ -s "$matches_file" ]]; then
  cat "$matches_file"
  rm -f "$matches_file"
  echo "Blocked public-release content is still present in reachable git history." >&2
  exit 1
fi

rm -f "$matches_file"

echo "No obvious public-release blockers found in reachable git history."
