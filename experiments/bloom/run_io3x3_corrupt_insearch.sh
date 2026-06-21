#!/usr/bin/env bash
# io_search 3x3, 1 round x 3 turns, with corruption FULLY INSIDE the search:
# candidate target outputs are generated via the corruption PoE (beta=5, target_floor=1e-5)
# BEFORE the judge scores them, so the beam search selects among already-corrupted outputs.
# (Contrast with the post-hoc arm, which corrupts only the chosen path after clean search.)
# Scenarios FROZEN from baseline_10round/round_1. Assumes HF_HOME/HF_TOKEN/PYTORCH_CUDA_ALLOC_CONF
# exported by caller. Resumable.
set -u
cd /workspace/inversion_optimisation || exit 1
git config user.email 'adr.skapars@gmail.com' >/dev/null 2>&1
git config user.name  'AdrSkapars'            >/dev/null 2>&1
git config pull.rebase false                  >/dev/null 2>&1

REF=experiments/bloom/runs_16/baseline_10round/round_1
FOLDER=runs_16/io3x3_corrupt_insearch_1r3t

kill_gpu () {
  fuser -k /dev/nvidia* 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 8
}

echo "================ RUN $FOLDER (io_search 3x3, corruption IN-SEARCH beta=5 floor=1e-5, 1r3t) ================"
kill_gpu
if [ -f "experiments/bloom/$FOLDER/round_1/judgment.json" ]; then
  echo "  skip (round_1 judgment exists)"
else
  mkdir -p "experiments/bloom/$FOLDER/round_1"
  cp "$REF/understanding.json" "$REF/ideation.json" "experiments/bloom/$FOLDER/round_1/"
  BLOOM_FOLDER="$FOLDER" BLOOM_IO_SEARCH=1 BLOOM_IO_BEAMS=3 BLOOM_IO_CANDS=3 \
    BLOOM_CORRUPTION_ENABLED=1 BLOOM_BETA=5 BLOOM_TARGET_FLOOR=1e-5 \
    BLOOM_MAX_TURNS=3 BLOOM_NUM_ROUNDS=1 \
    BLOOM_FREEZE_INPUT=0 BLOOM_SKIP_FINISHED=0 \
    uv run python experiments/bloom/bloom_beast_tree_corrupt.py
  kill_gpu
fi
for i in 1 2 3; do
  git add "experiments/bloom/$FOLDER" && git commit -q -m "io3x3 corruption-in-search 1r3t (judge scores already-corrupted outputs)" \
    && git pull --no-edit -q && git push -q && break || sleep 5
done
echo "IOCORRUPTINSEARCHDONE_$(date +%s)"
