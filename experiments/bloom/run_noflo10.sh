#!/usr/bin/env bash
# Corruption ON, NO floor (target_floor=0), 10 rounds, 1 turn, input resampled.
# Upper bracket / ceiling for the best-across-10 comparison: pairs with the
# corruption-off (lower bracket) and the floored runs (0.84). Expect ~1.0.
# Assumes HF_HOME/HF_TOKEN/PYTORCH_CUDA_ALLOC_CONF exported by caller.
# Resumable; kills leaked vLLM GPU procs (fuser) around the run.
set -u
cd /workspace/inversion_optimisation || exit 1
git config user.email 'adr.skapars@gmail.com' >/dev/null 2>&1
git config user.name 'AdrSkapars'             >/dev/null 2>&1
git config pull.rebase false                  >/dev/null 2>&1

kill_gpu () {
  fuser -k /dev/nvidia* 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 8
}

G=runs_16/p10_noflo_10round
echo "================ RUN corruption-ON no-floor 10-round -> $G ================"
kill_gpu
if [ -f "experiments/bloom/$G/round_10/judgment.json" ]; then
  echo "  skip pipeline (round_10 judgment exists)"
else
  BLOOM_FOLDER="$G" BLOOM_CORRUPTION_ENABLED=1 BLOOM_TARGET_FLOOR=0 BLOOM_MAX_TURNS=1 \
    BLOOM_NUM_ROUNDS=10 BLOOM_FREEZE_INPUT=0 BLOOM_SKIP_FINISHED=0 \
    uv run python experiments/bloom/bloom_beast_tree_corrupt.py
  kill_gpu
fi
for i in 1 2 3; do
  git add "experiments/bloom/$G" && git commit -q -m "runs_16 p10 corruption-ON no-floor 10-round best-across (ceiling)" \
    && git pull --no-edit -q && git push -q && break || sleep 5
done
echo "NOFLO10DONE_$(date +%s)"
