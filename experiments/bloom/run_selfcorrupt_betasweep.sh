#!/usr/bin/env bash
# Self-corruption beta sweep, 1 turn x 1 round, floor 1e-5, corruptor = TARGET (Qwen3-4B).
# Tests whether higher beta compensates for the target's softer corruption logits vs the
# abliterated model. beta=5 is the baseline comparison point. Scenarios FROZEN from
# baseline_10round/round_1 (same as all other corruption runs). Assumes
# HF_HOME/HF_TOKEN/PYTORCH_CUDA_ALLOC_CONF exported. Resumable.
set -u
cd /workspace/inversion_optimisation || exit 1
git config user.email 'adr.skapars@gmail.com' >/dev/null 2>&1
git config user.name  'AdrSkapars'            >/dev/null 2>&1
git config pull.rebase false                  >/dev/null 2>&1

REF=experiments/bloom/runs_16/baseline_10round/round_1

kill_gpu () {
  fuser -k /dev/nvidia* 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 8
}

run () {  # beta
  local beta="$1" folder="runs_16/selfcorr_1t_b$1"
  echo "================ RUN $folder (self-corrupt 1t1r, beta=$beta, floor=1e-5) ================"
  kill_gpu
  if [ -f "experiments/bloom/$folder/round_1/judgment.json" ]; then
    echo "  skip (judgment exists)"
  else
    mkdir -p "experiments/bloom/$folder/round_1"
    cp "$REF/understanding.json" "$REF/ideation.json" "experiments/bloom/$folder/round_1/"
    BLOOM_FOLDER="$folder" BLOOM_CORRUPTION_ENABLED=1 BLOOM_BETA="$beta" BLOOM_TARGET_FLOOR=1e-5 \
      BLOOM_CORRUPT_MODEL="local/Qwen/Qwen3-4B" \
      BLOOM_MAX_TURNS=1 BLOOM_NUM_ROUNDS=1 BLOOM_FREEZE_INPUT=0 BLOOM_SKIP_FINISHED=0 \
      uv run python experiments/bloom/bloom_beast_tree_corrupt.py
    kill_gpu
  fi
  for i in 1 2 3; do
    git add "experiments/bloom/$folder" && git commit -q -m "self-corrupt beta sweep $folder (1t1r, corruptor=target)" \
      && git pull --no-edit -q && git push -q && break || sleep 5
  done
  echo "================ DONE $folder ================"
}

for b in 5 10 20 40; do run "$b"; done
echo "SELFCORRSWEEPDONE_$(date +%s)"
