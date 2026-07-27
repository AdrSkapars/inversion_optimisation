#!/usr/bin/env bash
# Back up FULL G-PAIR hyperparam run data (runs_new/<beh>/<model>/gpair_*) from both GPU boxes to
# local, then commit+push. Mirrors push_all_runs.sh but targets the runs_new gpair_* config dirs.
# A config is COMPLETE once round_<ROUNDS>/judgment.json exists (default ROUNDS=7). READ-ONLY on the
# boxes (tar/scp only). Incremental: skips configs already synced locally. Run from repo ROOT.
set -u
ROUNDS="${1:-7}"
KEY=/c/Users/t75879as/.ssh/id_rsa
HOST=root@82.79.85.125
REMOTE=/workspace/inversion_optimisation/experiments/bloom/runs_new
LOCAL=experiments/bloom/runs_new
mkdir -p "$LOCAL"

for port in 42562 42766; do
  SSH="ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i $KEY"
  # completed configs: gpair_* dirs whose final round has a judgment
  cfgs=$($SSH "$HOST" "cd $REMOTE 2>/dev/null && ls */*/gpair_*/round_$ROUNDS/judgment.json 2>/dev/null | sed 's#/round_$ROUNDS/judgment.json##'" 2>/dev/null)
  for cfg in $cfgs; do
    [ -f "$LOCAL/$cfg/round_$ROUNDS/judgment.json" ] && continue   # already synced
    echo "== syncing $cfg (port $port) =="
    $SSH "$HOST" "cd $REMOTE && tar czf /tmp/gpaircfg.tgz '$cfg'" 2>/dev/null
    scp -P "$port" -o StrictHostKeyChecking=no -i "$KEY" "$HOST:/tmp/gpaircfg.tgz" /tmp/gpaircfg.tgz 2>/dev/null \
      && tar xzf /tmp/gpaircfg.tgz -C "$LOCAL/" && echo "   ok"
  done
done

git add "$LOCAL" experiments/bloom/gpair_results_running.md 2>&1 | tail -1
if git diff --cached --quiet; then
  echo "no new gpair data to commit"
else
  git commit -q -m "gpair: full run data for newly-completed configs ($(date -u +%Y-%m-%dT%H:%MZ))" \
    && git push origin HEAD 2>&1 | tail -1 && echo "pushed $(git rev-parse --short HEAD)"
fi
