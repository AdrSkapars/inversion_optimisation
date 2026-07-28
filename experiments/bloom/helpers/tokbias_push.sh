#!/usr/bin/env bash
# Back up TokenBias run dirs from both boxes to local + commit+push. Covers all three stages:
#   runs_pilot/<beh>/<model>/tokbias_pilot*   (explore)
#   runs_new/<beh>/<model>/tokbias            (sweep, 15 scen)
#   runs_final/<beh>/<model>/tokbias          (final, 100 scen)
# Idempotent: tars each tokbias* config dir on the box and overwrites local (data is small);
# git commits only if something actually changed. READ-ONLY on boxes. Run from repo ROOT.
set -u
KEY=/c/Users/t75879as/.ssh/id_rsa
HOST=root@82.79.85.125
RBASE=/workspace/inversion_optimisation/experiments/bloom
LBASE=experiments/bloom
for port in 42562 42766; do
  SSH="ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i $KEY"
  for root in runs_pilot runs_new runs_final; do
    cfgs=$($SSH "$HOST" "cd $RBASE/$root 2>/dev/null && ls -d */*/tokbias* 2>/dev/null" 2>/dev/null)
    for cfg in $cfgs; do
      [ -z "$cfg" ] && continue
      $SSH "$HOST" "cd $RBASE/$root && tar czf /tmp/tb.tgz '$cfg'" 2>/dev/null || continue
      scp -P "$port" -o StrictHostKeyChecking=no -i "$KEY" "$HOST:/tmp/tb.tgz" /tmp/tb.tgz 2>/dev/null || continue
      mkdir -p "$LBASE/$root"
      tar xzf /tmp/tb.tgz -C "$LBASE/$root/" && echo "  synced $root/$cfg (port $port)"
    done
  done
done
git add "$LBASE/runs_pilot" "$LBASE/runs_new" "$LBASE/runs_final" \
        "$LBASE/tokbias_results.md" "$LBASE/tokbias_state.json" 2>&1 | tail -1
if git diff --cached --quiet; then
  echo "no new tokbias data to commit"
else
  git commit -q -m "tokbias: run data ($(date -u +%Y-%m-%dT%H:%MZ))" \
    && git push origin HEAD 2>&1 | tail -1 && echo "pushed $(git rev-parse --short HEAD)"
fi
