#!/usr/bin/env bash
# Controlled Table-3 redo with the scenario-collapse bug FIXED.
# Two runs: corruption-off and corruption-ON beta=5 @ target_floor=1e-5.
# Scenarios FROZEN: each run's round_1 is seeded with the SAME understanding.json +
# ideation.json (the 25 scenarios), so both share identical scenarios; round-1 inputs
# are generated fresh per run. 10 rounds, freeze_input=False (now correct: rounds 2+
# keep each scenario's own description). Assumes HF_HOME/HF_TOKEN/PYTORCH_CUDA_ALLOC_CONF
# exported by caller. Resumable; kills leaked vLLM GPU procs (fuser) between runs.
set -u
cd /workspace/inversion_optimisation || exit 1
git config user.email 'adr.skapars@gmail.com' >/dev/null 2>&1
git config user.name 'AdrSkapars'             >/dev/null 2>&1
git config pull.rebase false                  >/dev/null 2>&1

REF=experiments/bloom/runs_16/baseline_10round/round_1   # source of the frozen 25 scenarios

kill_gpu () {
  fuser -k /dev/nvidia* 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 8
}

run () {  # folder corruption_enabled beta target_floor
  local folder="$1" corr="$2" beta="$3" floor="$4"
  echo "================ RUN $folder (corr=$corr beta=$beta floor=$floor) ================"
  kill_gpu
  if [ -f "experiments/bloom/$folder/round_10/judgment.json" ]; then
    echo "  skip (round_10 judgment exists)"
  else
    mkdir -p "experiments/bloom/$folder/round_1"
    cp "$REF/understanding.json" "$REF/ideation.json" "experiments/bloom/$folder/round_1/"
    BLOOM_FOLDER="$folder" BLOOM_CORRUPTION_ENABLED="$corr" BLOOM_BETA="$beta" \
      BLOOM_TARGET_FLOOR="$floor" BLOOM_MAX_TURNS=1 BLOOM_NUM_ROUNDS=10 \
      BLOOM_FREEZE_INPUT=0 BLOOM_SKIP_FINISHED=0 \
      uv run python experiments/bloom/bloom_beast_tree_corrupt.py
    kill_gpu
  fi
  for i in 1 2 3; do
    git add "experiments/bloom/$folder" && git commit -q -m "table3 redo $folder (scenario-collapse fixed, frozen scenarios)" \
      && git pull --no-edit -q && git push -q && break || sleep 5
  done
  echo "================ DONE $folder ================"
}

run runs_16/redo_off    0 5 0
run runs_16/redo_b5_1e5 1 5 1e-5
echo "TABLE3REDODONE_$(date +%s)"
