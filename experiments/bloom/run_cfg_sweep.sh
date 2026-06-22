#!/usr/bin/env bash
# CFG-steered self-corruption, 1 turn x 1 round, beta=6, floor 1e-5, corruptor = TARGET.
# c_eff = (1+gamma)*c - gamma*neutral  (CFG on the target between the p10 corrupt prompt and
# a neutral rewrite prompt), then PoE'd with the normal target dist. Sweeps gamma in {1,2,3}.
# gamma=0 baseline is selfcorr_1t_b6 (2.92/0.12). Scenarios FROZEN from baseline_10round.
# Assumes HF_HOME/HF_TOKEN/PYTORCH_CUDA_ALLOC_CONF exported. Resumable.
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

run () {  # gamma
  local g="$1" folder="runs_16/selfcorr_cfg_1t_b6_g$1"
  echo "================ RUN $folder (self-corrupt CFG, beta=6 gamma=$g floor=1e-5, 1t1r) ================"
  kill_gpu
  if [ -f "experiments/bloom/$folder/round_1/judgment.json" ]; then
    echo "  skip (judgment exists)"
  else
    mkdir -p "experiments/bloom/$folder/round_1"
    cp "$REF/understanding.json" "$REF/ideation.json" "experiments/bloom/$folder/round_1/"
    BLOOM_FOLDER="$folder" BLOOM_CORRUPTION_ENABLED=1 BLOOM_BETA=6 BLOOM_TARGET_FLOOR=1e-5 \
      BLOOM_CORRUPT_MODEL="local/Qwen/Qwen3-4B" BLOOM_CFG_GAMMA="$g" \
      BLOOM_MAX_TURNS=1 BLOOM_NUM_ROUNDS=1 BLOOM_FREEZE_INPUT=0 BLOOM_SKIP_FINISHED=0 \
      uv run python experiments/bloom/bloom_beast_tree_corrupt.py
    kill_gpu
  fi
  for i in 1 2 3; do
    git add "experiments/bloom/$folder" && git commit -q -m "CFG self-corrupt sweep $folder (beta6, gamma=$g, 1t1r)" \
      && git pull --no-edit -q && git push -q && break || sleep 5
  done
  echo "================ DONE $folder ================"
}

for g in 1 2 3; do run "$g"; done
echo "CFGSWEEPDONE_$(date +%s)"
