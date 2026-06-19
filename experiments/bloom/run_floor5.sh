#!/usr/bin/env bash
# Redo the 3-turn and 10-round floored-corruption tests at target_floor=1e-5
# (closer to the original 6.49e-5). Assumes HF_HOME/HF_TOKEN/PYTORCH_CUDA_ALLOC_CONF
# exported by caller. Resumable + kills leaked vLLM GPU procs (fuser) between runs.
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

push () {
  for i in 1 2 3; do
    git add "experiments/bloom/$1" && git commit -q -m "$2" && git pull --no-edit -q && git push -q && break || sleep 5
  done
}

# 1) 3-turn @ 1e-5 (single iteration, 3 turns)
F=runs_16/tfsweep/f5_3t
echo "================ RUN 3-turn floor=1e-5 -> $F ================"
kill_gpu
if [ -f "experiments/bloom/$F/round_1/judgment.json" ]; then echo "  skip pipeline (done)"; else
  BLOOM_FOLDER="$F" BLOOM_TARGET_FLOOR=1e-5 BLOOM_MAX_TURNS=3 BLOOM_NUM_ROUNDS=1 \
    uv run python experiments/bloom/bloom_beast_tree_corrupt.py
  kill_gpu
fi
if [ ! -f "experiments/bloom/$F/round_1/score_tokens.json" ]; then
  .venv/bin/python experiments/bloom/score_tokens.py "experiments/bloom/$F/round_1" || true
  kill_gpu
fi
push "$F" "tfsweep f5_3t (target_floor=1e-5 turns=3)"
echo "================ DONE $F ================"

# 2) 10-round @ 1e-5 (1 turn, 10 rounds, input resampled, all scenarios)
G=runs_16/p10_tf1e5_10round
echo "================ RUN 10-round floor=1e-5 -> $G ================"
kill_gpu
if [ -f "experiments/bloom/$G/round_10/judgment.json" ]; then echo "  skip pipeline (done)"; else
  BLOOM_FOLDER="$G" BLOOM_TARGET_FLOOR=1e-5 BLOOM_MAX_TURNS=1 BLOOM_NUM_ROUNDS=10 \
    BLOOM_FREEZE_INPUT=0 BLOOM_SKIP_FINISHED=0 \
    uv run python experiments/bloom/bloom_beast_tree_corrupt.py
  kill_gpu
fi
push "$G" "p10 target_floor=1e-5 10-round best-across"
echo "================ DONE $G ================"

echo "FLOOR5DONE_$(date +%s)"
