#!/usr/bin/env bash
# Back up the G-PAIR FINAL run dirs (runs_final/<beh>/<model>/gpair_t3_sfull) from both boxes to
# local + commit+push. push_all_runs.sh SKIPS these because the parent WILT cell (cell.json) is
# already local, so this targets the gpair_t3_sfull subdirs directly. A config is COMPLETE once
# gpair_t3_sfull/round_<ROUNDS>/judgment.json exists (default ROUNDS=7). READ-ONLY on the boxes.
# Run from repo ROOT.
set -u
ROUNDS="${1:-7}"
KEY=/c/Users/t75879as/.ssh/id_rsa
HOST=root@82.79.85.125
REMOTE=/workspace/inversion_optimisation/experiments/bloom/runs_final
LOCAL=experiments/bloom/runs_final
mkdir -p "$LOCAL"

for port in 42562 42766; do
  SSH="ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i $KEY"
  cfgs=$($SSH "$HOST" "cd $REMOTE 2>/dev/null && ls */*/gpair_t3_sfull/round_$ROUNDS/judgment.json 2>/dev/null | sed 's#/round_$ROUNDS/judgment.json##'" 2>/dev/null)
  for cfg in $cfgs; do
    [ -f "$LOCAL/$cfg/round_$ROUNDS/judgment.json" ] && continue
    echo "== syncing $cfg (port $port) =="
    $SSH "$HOST" "cd $REMOTE && tar czf /tmp/gpf.tgz '$cfg'" 2>/dev/null
    scp -P "$port" -o StrictHostKeyChecking=no -i "$KEY" "$HOST:/tmp/gpf.tgz" /tmp/gpf.tgz 2>/dev/null \
      && tar xzf /tmp/gpf.tgz -C "$LOCAL/" && echo "   ok"
  done
done

git add "$LOCAL" experiments/bloom/gpair_final_results.md 2>&1 | tail -1
if git diff --cached --quiet; then
  echo "no new gpair-final data to commit"
else
  git commit -q -m "gpair-final: run data for newly-completed cells ($(date -u +%Y-%m-%dT%H:%MZ))" \
    && git push origin HEAD 2>&1 | tail -1 && echo "pushed $(git rev-parse --short HEAD)"
fi
