#!/usr/bin/env bash
# Clean io_search 3x3, 1 round x 3 turns (no jail / no input_search / no corruption).
# This is the "without corruption" arm. The "with corruption" arm is produced post-hoc
# from THIS run's transcripts by experiments/bloom/posthoc_corrupt_rejudge.py (no second
# io_search). Scenarios FROZEN from baseline_10round/round_1 (same as the 3t/10rd pair).
# Assumes HF_HOME/HF_TOKEN/PYTORCH_CUDA_ALLOC_CONF exported by caller. Resumable.
set -u
cd /workspace/inversion_optimisation || exit 1
git config user.email 'adr.skapars@gmail.com' >/dev/null 2>&1
git config user.name  'AdrSkapars'            >/dev/null 2>&1
git config pull.rebase false                  >/dev/null 2>&1

REF=experiments/bloom/runs_16/baseline_10round/round_1
FOLDER=runs_16/io3x3_clean_1r3t

kill_gpu () {
  fuser -k /dev/nvidia* 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 8
}

echo "================ RUN $FOLDER (io_search 3x3, corruption OFF, 1r3t) ================"
kill_gpu
if [ -f "experiments/bloom/$FOLDER/round_1/judgment.json" ]; then
  echo "  skip (round_1 judgment exists)"
else
  mkdir -p "experiments/bloom/$FOLDER/round_1"
  cp "$REF/understanding.json" "$REF/ideation.json" "experiments/bloom/$FOLDER/round_1/"
  BLOOM_FOLDER="$FOLDER" BLOOM_IO_SEARCH=1 BLOOM_IO_BEAMS=3 BLOOM_IO_CANDS=3 \
    BLOOM_CORRUPTION_ENABLED=0 BLOOM_MAX_TURNS=3 BLOOM_NUM_ROUNDS=1 \
    BLOOM_FREEZE_INPUT=0 BLOOM_SKIP_FINISHED=0 \
    uv run python experiments/bloom/bloom_beast_tree_corrupt.py
  kill_gpu
fi
for i in 1 2 3; do
  git add "experiments/bloom/$FOLDER" && git commit -q -m "io3x3 clean 1r3t (without-corruption arm; post-hoc corruption to follow)" \
    && git pull --no-edit -q && git push -q && break || sleep 5
done
echo "IO3X3DONE_$(date +%s)"
