#!/usr/bin/env bash
# Self-corruption test: 1 round x 3 turns, corruption ON (beta=5, floor 1e-5), but the
# CORRUPTOR model is the TARGET model itself (Qwen3-4B) instead of the abliterated jail
# model. Direct comparison vs corr_3turn_f1e5 (jail-corruptor, same config) = 6.32/0.52.
# Scenarios FROZEN from baseline_10round/round_1 (same as corr_3turn). target_floor 1e-5
# is now the cfg default too. Assumes HF_HOME/HF_TOKEN/PYTORCH_CUDA_ALLOC_CONF exported.
set -u
cd /workspace/inversion_optimisation || exit 1
git config user.email 'adr.skapars@gmail.com' >/dev/null 2>&1
git config user.name  'AdrSkapars'            >/dev/null 2>&1
git config pull.rebase false                  >/dev/null 2>&1

REF=experiments/bloom/runs_16/baseline_10round/round_1
FOLDER=runs_16/corr_selfcorrupt_1r3t

kill_gpu () {
  fuser -k /dev/nvidia* 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 8
}

echo "================ RUN $FOLDER (self-corruption: corruptor=target Qwen3-4B, beta=5 floor=1e-5, 1r3t) ================"
kill_gpu
if [ -f "experiments/bloom/$FOLDER/round_1/judgment.json" ]; then
  echo "  skip (round_1 judgment exists)"
else
  mkdir -p "experiments/bloom/$FOLDER/round_1"
  cp "$REF/understanding.json" "$REF/ideation.json" "experiments/bloom/$FOLDER/round_1/"
  BLOOM_FOLDER="$FOLDER" BLOOM_CORRUPTION_ENABLED=1 BLOOM_BETA=5 BLOOM_TARGET_FLOOR=1e-5 \
    BLOOM_CORRUPT_MODEL="local/Qwen/Qwen3-4B" \
    BLOOM_MAX_TURNS=3 BLOOM_NUM_ROUNDS=1 \
    BLOOM_FREEZE_INPUT=0 BLOOM_SKIP_FINISHED=0 \
    uv run python experiments/bloom/bloom_beast_tree_corrupt.py
  kill_gpu
fi
for i in 1 2 3; do
  git add "experiments/bloom/$FOLDER" && git commit -q -m "self-corruption 1r3t (corruptor=target Qwen3-4B vs jail; floor 1e-5)" \
    && git pull --no-edit -q && git push -q && break || sleep 5
done
echo "SELFCORRUPTDONE_$(date +%s)"
