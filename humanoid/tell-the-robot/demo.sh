#!/usr/bin/env bash
# Run a live demo with windows (MuJoCo + Rerun), gently: the heaviest thing in this project.
#
#   humanoid/tell-the-robot/demo.sh g1-agent --stop-after 120s
#   humanoid/tell-the-robot/demo.sh g1-skills
#
# Same watchdog as light.sh, stricter: the whole demo may use at most 3 cores (MAX_CPU 300)
# for 20 s in a row, then it stops. The windows draw at 30 frames per second, the head camera
# renders 10 images per second, the dashboard logs 2 images per second (the defaults in
# nodes/sim.py and nodes/dashboard.py). The user's Mac got hot with live windows before macOS
# raised any warning, so the CPU cap matters more here than the thermal check.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
export MAX_CPU="${MAX_CPU:-300}" SUSTAINED="${SUSTAINED:-20}" CHECK="${CHECK:-3}"
cd "$root"
exec "$here/light.sh" ./run.sh "$@"
