#!/usr/bin/env bash
# Overnight CFG follow-ups (run after the 1-turn target CFG sweep). All beta=6, floor 1e-5,
# scenarios frozen from baseline_10round. CFG = c_eff=(1+g)c - g*neutral on the corruptor.
#   A) 3-turn SELF-corruption (corruptor=target Qwen3-4B), CFG gamma=1
#   B) 1-turn JAIL-corruption (corruptor=abliterated, the default), CFG gamma in {1,2}
# (C = conditional 3-turn jail CFG is launched separately after B is evaluated.)
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

run () {  # folder corrupt_model(""=default abliterated) beta gamma turns last_round
  local folder="$1" cm="$2" beta="$3" g="$4" turns="$5" last="$6"
  echo "================ RUN $folder (cm='${cm:-abliterated(default)}' beta=$beta gamma=$g turns=$turns floor=1e-5) ================"
  kill_gpu
  if [ -f "experiments/bloom/$folder/round_${last}/judgment.json" ]; then
    echo "  skip (judgment exists)"
  else
    mkdir -p "experiments/bloom/$folder/round_1"
    cp "$REF/understanding.json" "$REF/ideation.json" "experiments/bloom/$folder/round_1/"
    local CM_ENV=""
    [ -n "$cm" ] && CM_ENV="BLOOM_CORRUPT_MODEL=$cm"
    env BLOOM_FOLDER="$folder" BLOOM_CORRUPTION_ENABLED=1 BLOOM_BETA="$beta" BLOOM_TARGET_FLOOR=1e-5 \
      $CM_ENV BLOOM_CFG_GAMMA="$g" BLOOM_MAX_TURNS="$turns" BLOOM_NUM_ROUNDS=1 \
      BLOOM_FREEZE_INPUT=0 BLOOM_SKIP_FINISHED=0 \
      uv run python experiments/bloom/bloom_corrupt.py
    kill_gpu
  fi
  for i in 1 2 3; do
    git add "experiments/bloom/$folder" && git commit -q -m "overnight CFG $folder (beta=$beta gamma=$g, cm=${cm:-abliterated})" \
      && git pull --no-edit -q && git push -q && break || sleep 5
  done
  echo "================ DONE $folder ================"
}

# A) 3-turn SELF CFG gamma=1, beta=6 (target's optimum)
run runs_16/selfcorr_cfg_3t_b6_g1 "local/Qwen/Qwen3-4B" 6 1 3 1
# B) 1-turn JAIL CFG gamma 1 and 2, beta=5 (abliterated's optimum; default corruptor)
run runs_16/jail_cfg_1t_b5_g1 "" 5 1 1 1
run runs_16/jail_cfg_1t_b5_g2 "" 5 2 1 1
echo "OVERNIGHTCFGDONE_$(date +%s)"
