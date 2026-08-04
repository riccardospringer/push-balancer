# Public Release Checklist

Use this checklist before switching the repository visibility from private to public.

## 1. Clean the current tree

- Keep runtime secrets only in deployment systems, never in tracked files.
- Keep startup seed data outside the repository and reference it only via `PUSH_SNAPSHOT_PATH`.
- Verify the current checkout with:

```bash
bash scripts/check-public-surface.sh .
```

## 2. Rewrite the reachable git history

Create a disposable mirror clone and rewrite that clone, not your daily working copy:

```bash
git clone --mirror <repo-url> /tmp/push-balancer.public.git
cd /tmp/push-balancer.public.git
bash /path/to/repo/scripts/rewrite-public-history.sh
bash /path/to/repo/scripts/check-public-history.sh .
```

The rewrite removes blocked binary artifacts and sanitizes historical text examples that should not remain in a public repository.

## 3. Keep only the branches you actually want to expose

- Archive or back up internal feature branches before publishing.
- Delete obsolete remote branches from the cleaned mirror before the final mirror push if they are not meant to stay visible in the public repository.

## 4. Rotate secrets if they were ever used

- `PUSH_SYNC_SECRET`
- `ADMIN_API_KEY`
- Adobe credentials
- Any deployment-only API tokens or package registry tokens

If a value may have been used outside local development, rotate it before or immediately after the public switch.

## 5. Final verification

Run these checks on the cleaned mirror or the final public branch:

```bash
bash scripts/check-public-surface.sh .
bash scripts/check-public-history.sh .
python3 -m pytest tests/test_api.py -k 'InternalAccessControl or PushApiBaseCandidates or PredictionFeedbackValidation'
```

## 6. Push the cleaned result

Only after the checks are green:

```bash
git push --force --mirror origin
```
