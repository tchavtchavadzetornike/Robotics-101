"""Phase 4: things go wrong on purpose. Every skill must fail clearly and leave the robot safe.

  1. the bottle is pulled away just as the hand closes   -> pick fails: closed on nothing
  2. the bottle is pulled out of the hand while carrying -> place fails: fell out of my hand
  3. pick an object in the middle of table A             -> refused: can't reach it, how deep
  4. pick an id the memory doesn't have                  -> refused: look around first
After each: the robot stands, its arm is down, and the next skill still works.

    humanoid/tell-the-robot/light.sh uv run python humanoid/tell-the-robot/spikes/skills_failures.py
"""

import sys
from pathlib import Path

LESSON = Path(__file__).resolve().parents[2] / "03-g1-pick-place"
sys.path.insert(0, str(LESSON / "tests"))
sys.path.insert(0, str(LESSON))

from harness import Bench  # noqa: E402
from g1 import robot, skills  # noqa: E402

import numpy as np  # noqa: E402


def arm_down(bench):
    q, _ = bench.sim.joint_state()
    return float(np.abs(q[robot.RIGHT_ARM] - robot.RIGHT_ARM_WALK).max()) < 0.15


def report(name, result, bench, expect):
    good = (not result.ok) and expect in result.message and bench.sim.data.qpos[2] > 0.7 and arm_down(bench)
    print(f"{'PASS' if good else 'FAIL'}  {name}: {result.message}  (standing={bench.sim.data.qpos[2] > 0.7}, arm down={arm_down(bench)})", flush=True)
    return good


results = []

bench = Bench()
bench.run(skills.LookAround(), verbose=False)
bottle = bench.find("blue", "bottle")
r = bench.run(skills.Pick(bottle), verbose=False, on_step=lambda label, b: label == "close the hand" and b.knock_away("bottle"))
results.append(report("pulled away while closing", r, bench, "closed on nothing"))

bench = Bench()
bench.run(skills.LookAround(), verbose=False)
bottle = bench.find("blue", "bottle")
bench.run(skills.Pick(bottle), verbose=False)
r = bench.run(skills.Place("mark"), verbose=False, on_step=lambda label, b: label.startswith("carry") and b.knock_away("bottle"))
results.append(report("pulled out while carrying", r, bench, "fell out of my hand"))
again = bench.run(skills.LookAround(), verbose=False)
print(f"      and after that, look_around still works: {again.ok}: {again.message}")

bench = Bench(seed=0)
bench.run(skills.LookAround(), verbose=False)
middle = [o for o in bench.robot.objects if 1.5 < o["position"][0] < 1.7 and abs(o["position"][1]) < 0.2]
if middle:
    r = bench.run(skills.Pick(middle[0]["id"]), verbose=False)
    results.append(report("object in the middle of table A", r, bench, "can't reach"))

r = bench.run(skills.Pick(99), verbose=False)
results.append(report("an id nobody has seen", r, bench, "look around first"))

print(f"\n{sum(results)}/{len(results)} pass")
