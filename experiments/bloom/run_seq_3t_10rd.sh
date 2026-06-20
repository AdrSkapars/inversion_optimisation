#!/usr/bin/env bash
# Overnight sequential pair (run one after another, each uses the whole box):
#   A) 3-turn,  corruption ON  (beta=5, target_floor=1e-5),  1 round
#   B) 10-round, corruption OFF, 1 turn
# Scenarios FROZEN from the same REF as the table-3 redo so B is comparable to redo_off.
# Assumes HF_HOME/HF_TOKEN/PYTORCH_CUDA_ALLOC_CONF exported by caller. Resumable.
set -u
cd /workspace/inversion_optimisation || exit 1
git config user.email "adr.skapars@gmail.com" >/dev/null 2>&1
git config user.name  "AdrSkapars"            >/dev/null 2>&1
git config pull.rebase false                  >/dev/null 2>&1

REF=experiments/bloom/runs_16/baseline_10round/round_1

kill_gpu () {
  fuser -k /dev/nvidia* 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 8
}

run () {  # folder corr beta floor turns rounds last_round
  local folder="$1" corr="$2" beta="$3" floor="$4" turns="$5" rounds="$6" last="$7"
  echo "================ RUN $folder (corr=$corr beta=$beta floor=$floor turns=$turns rounds=$rounds) ================"
  kill_gpu
  if [ -f "experiments/bloom/$folder/round_${last}/judgment.json" ]; then
    echo "  skip (round_${last} judgment exists)"
  else
    mkdir -p "experiments/bloom/$folder/round_1"
    cp "$REF/understanding.json" "$REF/ideation.json" "experiments/bloom/$folder/round_1/"
    BLOOM_FOLDER="$folder" BLOOM_CORRUPTION_ENABLED="$corr" BLOOM_BETA="$beta" \
      BLOOM_TARGET_FLOOR="$floor" BLOOM_MAX_TURNS="$turns" BLOOM_NUM_ROUNDS="$rounds" \
      BLOOM_FREEZE_INPUT=0 BLOOM_SKIP_FINISHED=0 \
      uv run python experiments/bloom/bloom_beast_tree_corrupt.py
    kill_gpu
  fi
  for i in 1 2 3; do
    git add "experiments/bloom/$folder" && git commit -q -m "overnight $folder (3t/10rd pair, frozen scenarios)" \
      && git pull --no-edit -q && git push -q && break || sleep 5
  done
  echo "================ DONE $folder ================"
}

run runs_16/corr_3turn_f1e5 1 5 1e-5 3 1  1
run runs_16/off_10round_b   0 5 0     1 10 10
echo "SEQ3T10RDDONE_$(date +%s)"
