#!/usr/bin/env bash
# Corruption ON with beta=1 (weak corruption), target_floor=1e-5, 10 rounds, 1 turn,
# input resampled. Tests the beta x floor hypothesis: does weaker corruption (more
# target-plausible) recover elicitation under the floor better than beta=5 did?
# Compare to the beta=5 @1e-5 10-round run. Assumes HF_HOME/HF_TOKEN/PYTORCH_CUDA_ALLOC_CONF
# exported by caller. Resumable; kills leaked vLLM GPU procs (fuser) around the run.
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

G=runs_16/p10_beta1_tf1e5_10round
echo "================ RUN beta=1 floor=1e-5 10-round -> $G ================"
kill_gpu
if [ -f "experiments/bloom/$G/round_10/judgment.json" ]; then
  echo "  skip pipeline (round_10 judgment exists)"
else
  BLOOM_FOLDER="$G" BLOOM_CORRUPTION_ENABLED=1 BLOOM_BETA=1 BLOOM_TARGET_FLOOR=1e-5 BLOOM_MAX_TURNS=1 \
    BLOOM_NUM_ROUNDS=10 BLOOM_FREEZE_INPUT=0 BLOOM_SKIP_FINISHED=0 \
    uv run python experiments/bloom/bloom_beast_tree_corrupt.py
  kill_gpu
fi
for i in 1 2 3; do
  git add "experiments/bloom/$G" && git commit -q -m "runs_16 p10 beta=1 target_floor=1e-5 10-round best-across" \
    && git pull --no-edit -q && git push -q && break || sleep 5
done
echo "BETA1DONE_$(date +%s)"
