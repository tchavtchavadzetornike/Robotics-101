#!/usr/bin/env bash
# Run a heavy command gently, and stop it if the Mac gets too hot.
#
#   humanoid/tell-the-robot/light.sh uv run python humanoid/tell-the-robot/spikes/world_eval.py 5
#
# Gentle: low priority (nice 15), THREADS (default 4) threads for numpy, PyTorch and
# OpenMP, model weights from the local cache only (HF_HUB_OFFLINE).
#
# Watchdog, every CHECK seconds (default 3):
#   - macOS thermal state (pmset -g therm): any thermal or performance warning, or a
#     CPU speed limit below 100%, means the Mac is throttling from heat -> stop now
#   - the job's CPU use (the whole process tree: dora starts its nodes in their own
#     process groups, so the tree is followed through parent ids): above MAX_CPU percent
#     (default 600 = 6 of 12 cores) for SUSTAINED seconds (default 60) in a row -> stop
# Every 30 s the log also gets the 5 busiest processes of the job.
# Samples go to LOG (default /tmp/light-<time>.log), a short line every 30 s, a summary
# at the end. Exit code 99 means the watchdog stopped the job.
set -uo pipefail
threads="${THREADS:-4}"
check="${CHECK:-3}"
max_cpu="${MAX_CPU:-600}"
sustained="${SUSTAINED:-60}"
log="${LOG:-/tmp/light-$(date +%Y%m%d-%H%M%S).log}"
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" VECLIB_MAXIMUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads" TORCH_NUM_THREADS="$threads" HF_HUB_OFFLINE=1

nice -n 15 "$@" &
job=$!
stopped=0

tree() {  # the job and every process started from it, at any depth
  ps -Ao pid=,ppid= | awk -v root="$job" '
    { parent[$1] = $2 }
    END {
      for (p in parent) {
        q = p
        while (q != "" && q != 0 && q != 1) {
          if (q == root) { print p; break }
          q = parent[q]
        }
      }
    }'
}

stop_job() {
  echo "[light] STOPPING: $1" | tee -a "$log" >&2
  local pids
  pids=$(tree)
  kill -TERM $pids 2>/dev/null
  sleep 3
  pids=$(tree)
  [[ -n "$pids" ]] && kill -KILL $pids 2>/dev/null
  stopped=1
}
trap 'stop_job "interrupted"; exit 130' INT TERM HUP

thermal_problem() {
  local out
  out=$(pmset -g therm 2>/dev/null)
  if echo "$out" | grep -qE "CPU_Speed_Limit[[:space:]]*=[[:space:]]*([0-9]|[1-9][0-9])$"; then
    echo "$out" | tr '\n' ' '
    return
  fi
  if echo "$out" | grep -qi "thermal warning level" &&
    ! echo "$out" | grep -qi "No thermal warning level has been recorded"; then
    echo "$out" | tr '\n' ' '
    return
  fi
  if echo "$out" | grep -qi "performance warning level" &&
    ! echo "$out" | grep -qi "No performance warning level has been recorded"; then
    echo "$out" | tr '\n' ' '
  fi
}

peak=0
hot_for=0
elapsed=0
echo "[light] job $job, threads $threads, stop above ${max_cpu}% CPU for ${sustained}s or on any thermal warning, log $log" >&2
while kill -0 "$job" 2>/dev/null; do
  sleep "$check"
  elapsed=$((elapsed + check))
  pids=" $(tree | tr '\n' ' ') "
  cpu=$(ps -Ao pid=,pcpu= | awk -v list="$pids" 'index(list, " " $1 " ") { s += $2 } END { printf "%d", s }')
  load=$(sysctl -n vm.loadavg | awk '{print $2}')
  therm=$(thermal_problem)
  ((cpu > peak)) && peak=$cpu
  echo "$(date +%T) cpu ${cpu}% load ${load} thermal ${therm:-ok}" >>"$log"
  if ((elapsed % 30 < check)); then
    echo "[light] $(date +%T) job CPU ${cpu}%, load ${load}, thermal ${therm:-ok}" >&2
    ps -Ao pid=,pcpu=,command= | awk -v list="$pids" 'index(list, " " $1 " ")' | sort -k2 -nr | head -5 |
      awk '{ cmd = ""; for (i = 3; i <= NF && i <= 8; i++) cmd = cmd " " $i; printf "    %5s%%  %s\n", $2, cmd }' >>"$log"
  fi
  if [[ -n "$therm" ]]; then
    stop_job "macOS reports heat: $therm"
    break
  fi
  if ((cpu > max_cpu)); then
    hot_for=$((hot_for + check))
    if ((hot_for >= sustained)); then
      stop_job "job used more than ${max_cpu}% CPU for ${hot_for}s"
      break
    fi
  else
    hot_for=0
  fi
done
wait "$job" 2>/dev/null
code=$?
if ((stopped)); then
  echo "[light] stopped after ${elapsed}s, peak job CPU ${peak}%" >&2
  exit 99
fi
echo "[light] done after ${elapsed}s, peak job CPU ${peak}%, no heat warning" >&2
exit "$code"
