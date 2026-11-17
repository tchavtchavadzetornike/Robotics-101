"""Phase 4 evaluation: pick and place every kind of object in random rooms, checked against the truth.

Per room: look around. Then for each kind (bottle, can, box), the first remembered
object of that kind: if reach.py says it can't be reached, that is an honest
refusal (counted apart). Else pick it and place it next to an object on the
other table. Checked against the simulator: the object really stands within
NEAR of the spot place() chose, upright.

    humanoid/tell-the-robot/light.sh uv run python humanoid/tell-the-robot/spikes/skills_eval.py [rooms] [first seed] [pause s]
"""

import os
import sys
import time as wall
from pathlib import Path

import numpy as np

LESSON = Path(__file__).resolve().parents[2] / "03-g1-pick-place"
sys.path.insert(0, str(LESSON / "tests"))
sys.path.insert(0, str(LESSON))

from harness import Bench  # noqa: E402
from g1 import reach, scenes, skills  # noqa: E402

NEAR = 0.03  # m
UPRIGHT = 0.95  # cos of the tilt: the object's axis within 18 degrees of vertical


def true_object(bench, remembered):
    """The scene object a remembered one is, by position."""
    return min(bench.scene.objects, key=lambda o: np.hypot(*(bench.truth(o.name)[:2] - np.array(remembered["position"][:2]))))


def upright(bench, name):
    return bench.sim.data.xmat[bench.sim.model.body(name).id].reshape(3, 3)[2, 2] > UPRIGHT


def watcher(bench):
    """on_step hook: print every object that really moved since the last step, and in which step."""
    last = {o.name: bench.truth(o.name) for o in bench.scene.objects}
    previous = {"label": "start"}

    def watch(label, b):
        for name, before in list(last.items()):
            now = b.truth(name)
            shift = np.hypot(*(now[:2] - before[:2]))
            if shift > 0.02 and name != watch.moving:
                base, _ = b.sim.base_pose()
                print(f"      MOVED {b.scene.object(name).label} {100 * shift:.0f} cm during '{previous['label']}' (robot at {base[:2].round(2).tolist()})", flush=True)
            last[name] = now
        previous["label"] = label

    watch.moving = None  # the object the robot is moving on purpose
    return watch


def room(seed):
    bench = Bench(seed)
    watch = watcher(bench)
    print(f"\n== room {seed}: " + ", ".join(o.label for o in bench.scene.objects), flush=True)
    look = bench.run(skills.LookAround(), verbose=False)
    print(f"   look_around: {look.message}", flush=True)
    rows = []
    for kind in ("bottle", "can", "box"):
        bench.observe()
        candidates = [o for o in bench.robot.objects if o["kind"] == kind]
        if not candidates:
            continue
        obj = candidates[0]
        truth = true_object(bench, obj)
        options, reason = reach.grasp_options(obj["position"], bench.robot.others(but=obj["id"]))
        if not options:
            rows.append((kind, "refused", reason))
            print(f"   {truth.label}: refused, {reason}", flush=True)
            continue
        here = reach.table_of(obj["position"])
        anchors = [o for o in bench.robot.objects if o["id"] != obj["id"] and reach.table_of(o["position"]) != here]
        target = ("next_to", anchors[0]["id"]) if anchors else ("table_b" if here == "table_a" else "table_a")
        watch.moving = truth.name
        pick = bench.run(skills.Pick(obj["id"]), verbose=False, on_step=watch)
        if not pick.ok:
            rows.append((kind, "pick failed", pick.message))
            print(f"   {truth.label}: PICK FAILED: {pick.message}", flush=True)
            continue
        before = {o.name: bench.truth(o.name) for o in bench.scene.objects if o.name != truth.name}
        other_table = "table_b" if here == "table_a" else "table_a"
        for target in (target, other_table, here):  # if one place is impossible, the next
            place_skill = skills.Place(target)
            place = bench.run(place_skill, verbose=False, on_step=watch)
            if place.ok or "can't place" not in place.message:
                break
            print(f"   (can't place at {target}: {place.message})", flush=True)
        if not place.ok:
            rows.append((kind, "place failed", place.message))
            where = ""
            if hasattr(place_skill, "option"):
                miss = np.hypot(*(bench.truth(truth.name)[:2] - place_skill.option.spot))
                where = f" | TRUTH: {100 * miss:.1f} cm from the spot, upright={upright(bench, truth.name)}, at {bench.truth(truth.name).round(2).tolist()}"
            print(f"   {truth.label}: PLACE FAILED: {place.message}{where}", flush=True)
            continue
        error = np.hypot(*(bench.truth(truth.name)[:2] - place_skill.option.spot))
        pushed = [f"{bench.scene.object(n).label} {100 * np.hypot(*(bench.truth(n)[:2] - p[:2])):.0f} cm" for n, p in before.items() if np.hypot(*(bench.truth(n)[:2] - p[:2])) > 0.02]
        if pushed:
            print(f"      TRUTH: really pushed: {', '.join(pushed)}", flush=True)
        good = error < NEAR and upright(bench, truth.name)
        rows.append((kind, "ok" if good else "wrong", f"{100 * error:.1f} cm"))
        print(
            f"   {truth.label}: placed on {place_skill.option.table} from side {place_skill.option.side}, "
            f"{place.message}: {'OK' if good else 'WRONG'} {100 * error:.1f} cm from the spot, upright={upright(bench, truth.name)}",
            flush=True,
        )
    return rows


def main(count, first, pause):
    all_rows = []
    seeds = [int(s) for s in os.environ["SEEDS"].split(",")] if os.environ.get("SEEDS") else range(first, first + count)
    for n, seed in enumerate(seeds):
        if n:
            wall.sleep(pause)
        start = wall.perf_counter()
        all_rows += room(seed)
        print(f"   ({wall.perf_counter() - start:.0f} s)", flush=True)
    counts = {}
    for _, outcome, _ in all_rows:
        counts[outcome] = counts.get(outcome, 0) + 1
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    for kind in ("bottle", "can", "box"):
        mine = [o for k, o, _ in all_rows if k == kind]
        print(f"  {kind}: {mine.count('ok')} ok of {len([o for o in mine if o != 'refused'])} attempted, {mine.count('refused')} refused")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 5, int(a[1]) if len(a) > 1 else 0, float(a[2]) if len(a) > 2 else 15.0)
