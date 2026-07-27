#!/usr/bin/env bash
# Back up the COMBO (jail+refinement) run dirs (runs_new/<beh>/<model>/combo_jailrefine) from both
# boxes to local + commit+push. A cell is COMPLETE once combo_jailrefine/round_<ROUNDS>/judgment.json
# exists (default ROUNDS=7). READ-ONLY on boxes. Run from repo ROOT.
set -u
ROUNDS="${1:-7}"
KEY=/c/Users/t75879as/.ssh/id_rsa
HOST=root@82.79.85.125
REMOTE=/workspace/inversion_optimisation/experiments/bloom/runs_new
LOCAL=experiments/bloom/runs_new
mkdir -p "$LOCAL"
for port in 42562 42766; do
  SSH="ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i $KEY"
  cfgs=$($SSH "$HOST" "cd $REMOTE 2>/dev/null && ls */*/combo_jailrefine/round_$ROUNDS/judgment.json 2>/dev/null | sed 's#/round_$ROUNDS/judgment.json##'" 2>/dev/null)
  for cfg in $cfgs; do
    [ -f "$LOCAL/$cfg/round_$ROUNDS/judgment.json" ] && continue
    echo "== syncing $cfg (port $port) =="
    $SSH "$HOST" "cd $REMOTE && tar czf /tmp/combo.tgz '$cfg'" 2>/dev/null
    scp -P "$port" -o StrictHostKeyChecking=no -i "$KEY" "$HOST:/tmp/combo.tgz" /tmp/combo.tgz 2>/dev/null \
      && tar xzf /tmp/combo.tgz -C "$LOCAL/" && echo "   ok"
  done
done
git add "$LOCAL" experiments/bloom/combo_results.md 2>&1 | tail -1
if git diff --cached --quiet; then
  echo "no new combo data to commit"
else
  git commit -q -m "combo: jail+refine run data ($(date -u +%Y-%m-%dT%H:%MZ))" \
    && git push origin HEAD 2>&1 | tail -1 && echo "pushed $(git rev-parse --short HEAD)"
fi
