#!/usr/bin/env bash
# Fills the empty cell of the rounds×turns matrix: 10 rounds × 3 turns.
# Two runs, sequential (each uses the whole box):
#   A) corruption ON  (beta=5, target_floor=1e-5)
#   B) corruption OFF
# Scenarios FROZEN from baseline_10round/round_1 (same as corr_3turn / off_10round /
# io3x3), so best-across is directly comparable across the matrix. Expensive (~9-10h
# for the corruption-on arm). Assumes HF_HOME/HF_TOKEN/PYTORCH_CUDA_ALLOC_CONF exported.
# Resumable; kills leaked vLLM GPU procs (fuser) between runs.
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

run () {  # folder corr beta floor
  local folder="$1" corr="$2" beta="$3" floor="$4"
  echo "================ RUN $folder (corr=$corr beta=$beta floor=$floor turns=3 rounds=10) ================"
  kill_gpu
  if [ -f "experiments/bloom/$folder/round_10/judgment.json" ]; then
    echo "  skip (round_10 judgment exists)"
  else
    mkdir -p "experiments/bloom/$folder/round_1"
    cp "$REF/understanding.json" "$REF/ideation.json" "experiments/bloom/$folder/round_1/"
    BLOOM_FOLDER="$folder" BLOOM_CORRUPTION_ENABLED="$corr" BLOOM_BETA="$beta" \
      BLOOM_TARGET_FLOOR="$floor" BLOOM_MAX_TURNS=3 BLOOM_NUM_ROUNDS=10 \
      BLOOM_FREEZE_INPUT=0 BLOOM_SKIP_FINISHED=0 \
      uv run python experiments/bloom/bloom_beast_tree_corrupt.py
    kill_gpu
  fi
  for i in 1 2 3; do
    git add "experiments/bloom/$folder" && git commit -q -m "10round x 3turn $folder (corr=$corr, frozen scenarios)" \
      && git pull --no-edit -q && git push -q && break || sleep 5
  done
  echo "================ DONE $folder ================"
}

run runs_16/corr_10r3t_f1e5 1 5 1e-5
run runs_16/off_10r3t       0 5 0
echo "SEQ10R3TDONE_$(date +%s)"
