#!/usr/bin/env bash
# Autonomous target_floor sweep for ON corruption (p10, 1 iteration).
# Sweep target_floor in 1e-2..1e-6 at 1 turn, then re-run 1e-3 at 3 turns.
# Assumes HF_HOME, HF_TOKEN, PYTORCH_CUDA_ALLOC_CONF are already exported by the caller
# (so no secrets live in this committed file). Commits+pushes results after each run.
set -u
cd /workspace/inversion_optimisation || exit 1
git config user.email 'adr.skapars@gmail.com' >/dev/null 2>&1
git config user.name 'AdrSkapars'             >/dev/null 2>&1
git config pull.rebase false                  >/dev/null 2>&1

run_one () {
  local folder="$1"; local floor="$2"; local turns="$3"
  echo "================ RUN floor=${floor} turns=${turns} folder=${folder} ================"
  BLOOM_FOLDER="${folder}" BLOOM_TARGET_FLOOR="${floor}" BLOOM_MAX_TURNS="${turns}" BLOOM_NUM_ROUNDS=1 \
    uv run python experiments/bloom/bloom_beast_tree_corrupt.py
  echo "---- token scoring (turn 1) ${folder} ----"
  .venv/bin/python experiments/bloom/score_tokens.py "experiments/bloom/${folder}/round_1" || true
  for i in 1 2 3; do
    git add "experiments/bloom/${folder}" && git commit -q -m "tfsweep ${folder} (target_floor=${floor} turns=${turns})" \
      && git pull --no-edit -q && git push -q && break || sleep 5
  done
  echo "================ DONE ${folder} ================"
}

for X in 2 3 4 5 6; do
  run_one "runs_16/tfsweep/f${X}" "1e-${X}" 1
done
# multi-turn test: does 3 turns recover elicitation at the 1e-3 floor?
run_one "runs_16/tfsweep/f3_3t" "1e-3" 3

echo "SWEEPDONE_$(date +%s)"
