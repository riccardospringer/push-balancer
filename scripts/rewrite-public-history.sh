#!/usr/bin/env bash
set -euo pipefail

if [[ "$(git rev-parse --is-bare-repository 2>/dev/null || echo false)" != "true" ]]; then
  echo "Run this script inside a bare mirror clone created via git clone --mirror." >&2
  exit 1
fi

export FILTER_BRANCH_SQUELCH_WARNING=1

git filter-branch --force --tree-filter '
  rm -f push-snapshot.json .ml_lgbm.joblib .sklearn_gbrt.joblib .gbrt_model.json .tuning_state.json

  git grep -Il -E "push-frontend\.bildcms\.de|push-balancer\.onrender\.com|\.as-infra\.de|/next/nmt/|@axelspringer\.com|@bild\.de|next-aws-secret-manager|next-aws-ssm-parameter-store|145\.243\.0\.0/16|91\.220\.134\.0/24|PUSH_SYNC_SECRET=[^[:space:]]+" -- . \
    ":(exclude)scripts/check-public-surface.sh" \
    ":(exclude)scripts/check-public-history.sh" \
    ":(exclude)scripts/rewrite-public-history.sh" \
    > .public-release-files || true
  while IFS= read -r file; do
    [ -n "$file" ] || continue
    perl -0pi -e "
      s/push-frontend\.bildcms\.de/internal-push-api.example.invalid/g;
      s/push-balancer\.onrender\.com/deployment.example.invalid/g;
      s/[A-Za-z0-9.-]+\.as-infra\.de/push-balancer.example.invalid/g;
      s/operations\@axelspringer\.com/operations\@example.invalid/g;
      s/[A-Za-z0-9._%+-]+\@axelspringer\.com/contact\@example.invalid/g;
      s/[A-Za-z0-9._%+-]+\@bild\.de/contact\@example.invalid/g;
      s#next-aws-secret-manager#example-secret-manager#g;
      s#next-aws-ssm-parameter-store#example-parameter-store#g;
      s#/next/nmt/#/example/project/#g;
      s#127\.0\.0\.1/32,::1/128,145\.243\.0\.0/16,91\.220\.134\.0/24#127.0.0.1/32,::1/128#g;
      s#145\.243\.0\.0/16,91\.220\.134\.0/24##g;
      s#145\.243\.0\.0/16##g;
      s#91\.220\.134\.0/24##g;
      s/^PUSH_SYNC_SECRET=.*/PUSH_SYNC_SECRET=/mg;
      s/^PUSH_API_BASE=.*/PUSH_API_BASE=/mg;
    " "$file"
  done < .public-release-files
  rm -f .public-release-files
' -- --all

rm -rf refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive
