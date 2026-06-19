#!/usr/bin/env bash
# Autonomous target_floor sweep for ON corruption (p10, 1 iteration).
# Sweep target_floor in 1e-2..1e-6 at 1 turn, then re-run 1e-3 at 3 turns.
# Assumes HF_HOME, HF_TOKEN, PYTORCH_CUDA_ALLOC_CONF are already exported by the caller
# (so no secrets live in this committed file). Commits+pushes results after each run.
#
# RESUMABLE: skips the pipeline if round_1/judgment.json exists and skips token
# scoring if round_1/score_tokens.json exists. Kills orphaned GPU procs (leaked
# vLLM EngineCore workers) between every stage so the next run gets a clean GPU.
set -u
cd /workspace/inversion_optimisation || exit 1
git config user.email 'adr.skapars@gmail.com' >/dev/null 2>&1
git config user.name 'AdrSkapars'             >/dev/null 2>&1
git config pull.rebase false                  >/dev/null 2>&1

kill_gpu () {
  # fuser reliably kills procs holding the GPU device (the nvidia-smi compute-apps
  # query under-reports pids inside this container, so leaked vLLM workers survived).
  fuser -k /dev/nvidia* 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 8
}

run_one () {
  local folder="$1"; local floor="$2"; local turns="$3"
  local rd="experiments/bloom/${folder}/round_1"
  echo "================ RUN floor=${floor} turns=${turns} folder=${folder} ================"
  kill_gpu
  if [ -f "${rd}/judgment.json" ]; then
    echo "  skip pipeline (judgment.json exists)"
  else
    BLOOM_FOLDER="${folder}" BLOOM_TARGET_FLOOR="${floor}" BLOOM_MAX_TURNS="${turns}" BLOOM_NUM_ROUNDS=1 \
      uv run python experiments/bloom/bloom_beast_tree_corrupt.py
    kill_gpu
  fi
  if [ -f "${rd}/score_tokens.json" ]; then
    echo "  skip token scoring (score_tokens.json exists)"
  else
    echo "---- token scoring (turn 1) ${folder} ----"
    .venv/bin/python experiments/bloom/score_tokens.py "${rd}" || true
    kill_gpu
  fi
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

kill_gpu
echo "SWEEPDONE_$(date +%s)"
