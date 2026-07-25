#!/usr/bin/env bash
# Sync FULL run data (transcripts/judgments/rollouts/banks) for every COMPLETED cell from both
# GPU boxes to local, then commit+push to git. tar/scp are READ-ONLY on the boxes -> safe to run
# while the drivers are producing (never touches/locks the live scripts). Incremental: only cells
# whose cell.json isn't already local are transferred. Single-writer (local) -> no git conflicts.
# Run from the repo ROOT (dir containing experiments/). Local has no rsync, hence tar+scp.
set -u
KEY=/c/Users/t75879as/.ssh/id_rsa
HOST=root@82.79.85.125
REMOTE=/workspace/inversion_optimisation/experiments/bloom/runs_final
LOCAL=experiments/bloom/runs_final
mkdir -p "$LOCAL"
new=0

for port in 42562 42766; do
  SSH="ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i $KEY"
  # a cell is COMPLETE once final_run writes its cell.json
  cells=$($SSH "$HOST" "cd $REMOTE 2>/dev/null && ls */*/cell.json 2>/dev/null | sed 's#/cell.json##'" 2>/dev/null)
  for cell in $cells; do
    [ -f "$LOCAL/$cell/cell.json" ] && continue   # already synced locally
    echo "== syncing $cell (port $port) =="
    $SSH "$HOST" "cd $REMOTE && tar czf /tmp/wiltcell.tgz '$cell'" 2>/dev/null
    scp -P "$port" -o StrictHostKeyChecking=no -i "$KEY" "$HOST:/tmp/wiltcell.tgz" /tmp/wiltcell.tgz 2>/dev/null \
      && tar xzf /tmp/wiltcell.tgz -C "$LOCAL/" && new=1 && echo "   ok"
  done
done

# also refresh the shared banks (cheap) in case any changed
git add "$LOCAL" experiments/bloom/final_results_running.md 2>&1 | tail -1
if git diff --cached --quiet; then
  echo "no new run data to commit"
else
  git commit -q -m "wilt: full run data for newly-completed cells ($(date -u +%Y-%m-%dT%H:%MZ))" \
    && git push origin HEAD 2>&1 | tail -1 && echo "pushed $(git rev-parse --short HEAD)"
fi
