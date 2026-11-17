#!/usr/bin/env bash
# ./run.sh                     show what can be run
# ./run.sh <name>              run it, Ctrl-C to stop
# ./run.sh <name> --stop-after 20s
# ./run.sh <name> --raw        show dora's full log instead of the cleaned one
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

folder_of() {
  case "$1" in
    first) echo official/python-dataflow ;;
    first-change) echo steps/00-first-change ;;
    yolo) echo official/python-yolo-detection ;;
    fake-camera) echo steps/01-fake-camera ;;
    robust) echo steps/02-robust ;;
    hello) echo extras/examples/01-hello-graph ;;
    camera) echo extras/examples/02-camera-pipeline ;;
    types) echo extras/examples/03-types ;;
    modules) echo extras/examples/04-modules ;;
    services) echo extras/examples/05-services-actions ;;
    faults) echo extras/examples/06-fault-tolerance ;;
    mobile-robot) echo extras/simulations/mobile-robot ;;
    arm) echo extras/simulations/arm ;;
    arm-nodes) echo humanoid/02-dora-nodes ;;
    g1-pick-place) echo humanoid/03-g1-pick-place ;;
    g1-crowd) echo humanoid/04-g1-crowd ;;
    g1-skills) echo humanoid/03-g1-pick-place ;;
    g1-agent) echo humanoid/03-g1-pick-place ;;
    *) echo "" ;;
  esac
}

usage() {
  echo "usage: ./run.sh <name> [--stop-after 20s] [--raw]"
  echo
  echo "  the lessons, in order:"
  echo "    first          official example: sender -> transformer -> receiver"
  echo "    first-change   the same graph plus one new node you wrote"
  echo "    yolo           official example: webcam -> YOLO -> window (needs a webcam)"
  echo "    fake-camera    the YOLO example with a fake camera, no webcam needed"
  echo "    robust         fake-camera plus restarts and dropping old frames"
  echo
  echo "  extras:"
  echo "    hello camera types modules services faults mobile-robot arm"
  echo
  echo "  humanoid track:"
  echo "    arm-nodes      lesson 2: the MuJoCo arm as 7 dora nodes (3D window + Rerun charts)"
  echo "    g1-pick-place  lesson 3: the Unitree G1 walks, picks up a bottle, puts it on another table"
  echo "    g1-crowd       lesson 4: the Unitree G1 crosses a square full of walking people"
  echo "    g1-agent       tell the robot in words: Claude plans with the skills (needs ANTHROPIC_API_KEY);"
  echo "                   then talk in a second terminal: cd humanoid/03-g1-pick-place && uv run --no-sync python nodes/chat.py"
  echo "    g1-skills      tell the robot: skills from a script, ROBOT_SCRIPT=\"look_around; pick red can; place mark\" SEED=3"
}

die() { printf '\033[0;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
command -v uv >/dev/null || die "uv is not installed, see https://docs.astral.sh/uv/"
if ! command -v dora >/dev/null; then
  echo "installing the dora command line tool (dora-rs-cli 1.1.0)"
  uv tool install dora-rs-cli==1.1.0
fi

name="${1:-}"
[[ $# -gt 0 ]] && shift
dir="$(folder_of "$name")"
[[ -z "$dir" ]] && { usage; exit 0; }

case "$name" in
  yolo|fake-camera|robust) uv sync --quiet --extra vision ;;
  arm-nodes|g1-crowd) uv sync --quiet --extra sim ;;
  g1-pick-place|g1-skills|g1-agent) uv sync --quiet --extra sim --extra detect --extra agent ;;
  *) uv sync --quiet ;;
esac

raw=0
args=()
for arg in "$@"; do
  if [[ "$arg" == "--raw" ]]; then raw=1; else args+=("$arg"); fi
done

cd "$dir"
if [[ "$name" == "g1-pick-place" || "$name" == "g1-skills" || "$name" == "g1-agent" ]]; then
  # the room: robot/scene.xml, or with SEED=<n> a random room with more objects (tell-the-robot phase 3)
  uv run --no-sync python -m g1.scenes "${SEED:-}"
fi
if [[ "$name" == "types" ]]; then
  dora validate dataflow.yml
  dora validate broken.yml
  dora validate --strict-types broken.yml || echo "exit code $?: --strict-types turned the warning into a failure"
  exit 0
fi

case "$dir" in
  official/*|steps/*) args=(--uv ${args[@]+"${args[@]}"}) ;;
esac
flow=dataflow.yml
[[ "$name" == "g1-skills" ]] && flow=skills.yml
[[ "$name" == "g1-agent" ]] && flow=agent.yml
export VIEWER="${VIEWER:-rerun}"
if [[ $raw -eq 1 ]]; then
  exec dora run "$flow" ${args[@]+"${args[@]}"}
fi
dora run "$flow" ${args[@]+"${args[@]}"} 2>&1 | uv run --no-sync python "$ROOT/tools/pretty-logs.py"
