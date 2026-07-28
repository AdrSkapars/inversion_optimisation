#!/usr/bin/env bash
# Back up COMBO (jail+refinement) run dirs from both boxes to local + commit+push. Covers:
#   runs_new/<beh>/<model>/combo            (Stage A/B, 15 scen; beta_<b> subdirs)
#   runs_new/<beh>/<model>/combo_jailrefine (the earlier exploratory run)
#   runs_final/<beh>/<model>/combo          (Stage C, 100 scen)
# Idempotent: tars each combo* config dir on the box and overwrites local (data is small); git
# commits only if something changed. READ-ONLY on boxes. Run from repo ROOT.
set -u
KEY=/c/Users/t75879as/.ssh/id_rsa
HOST=root@82.79.85.125
RBASE=/workspace/inversion_optimisation/experiments/bloom
LBASE=experiments/bloom
for port in 42562 42766; do
  SSH="ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i $KEY"
  for root in runs_new runs_final; do
    cfgs=$($SSH "$HOST" "cd $RBASE/$root 2>/dev/null && ls -d */*/combo */*/combo_jailrefine 2>/dev/null" 2>/dev/null)
    for cfg in $cfgs; do
      [ -z "$cfg" ] && continue
      $SSH "$HOST" "cd $RBASE/$root && tar czf /tmp/cb.tgz '$cfg'" 2>/dev/null || continue
      scp -P "$port" -o StrictHostKeyChecking=no -i "$KEY" "$HOST:/tmp/cb.tgz" /tmp/cb.tgz 2>/dev/null || continue
      mkdir -p "$LBASE/$root"
      tar xzf /tmp/cb.tgz -C "$LBASE/$root/" && echo "  synced $root/$cfg (port $port)"
    done
  done
done
git add "$LBASE/runs_new" "$LBASE/runs_final" "$LBASE/combo_results.md" 2>&1 | tail -1
if git diff --cached --quiet; then
  echo "no new combo data to commit"
else
  git commit -q -m "combo: run data ($(date -u +%Y-%m-%dT%H:%MZ))" \
    && git push origin HEAD 2>&1 | tail -1 && echo "pushed $(git rev-parse --short HEAD)"
fi
