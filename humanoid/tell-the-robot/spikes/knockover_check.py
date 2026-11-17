"""Agent case 7 without the model: put two objects on one table, watch the first one.

Room 6: look around, blue bottle to table B, yellow box to table B. Prints every
object that really moved and during which step (skills_eval.watcher), the spots
place() chose, and where the bottle ends up.

    humanoid/tell-the-robot/light.sh uv run python humanoid/tell-the-robot/spikes/knockover_check.py [seed] [table]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skills_eval import upright, watcher  # noqa: E402
from harness import Bench  # noqa: E402
from g1 import skills  # noqa: E402


def touches(bench, step_label):
    """Print the first robot body that touches each still object, per step."""
    sim = bench.sim
    model = sim.model
    names = {model.body(o.name).id: o.label for o in bench.scene.objects}
    seen = set()
    step = sim.step

    def checked_step():
        step()
        for contact in sim.data.contact[: sim.data.ncon]:
            a, b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
            for mine, other in ((a, b), (b, a)):
                if mine in names and other not in names and model.body(other).name not in ("world", "table_a", "table_b") and model.body_rootid[other] != 0:
                    key = (names[mine], model.body(other).name, step_label["label"])
                    if key not in seen and names[mine] != step_label.get("held"):
                        seen.add(key)
                        print(f"      TOUCH {key[0]} by {key[1]} during '{key[2]}'", flush=True)

    sim.step = checked_step


def main(seed, table):
    bench = Bench(seed)
    step_label = {"label": "start"}
    touches(bench, step_label)
    seen_by_watch = watcher(bench)

    def watch(label, b):
        step_label["label"] = label
        seen_by_watch(label, b)

    watch.moving = None
    bench.run(skills.LookAround(), verbose=False, on_step=watch)
    names = {}
    for color, kind in (("blue", "bottle"), ("yellow", "box")):
        names[color] = next(o.name for o in bench.scene.objects if o.color == color and o.kind == kind)
        target = bench.find(color, kind)
        print(f"{color} {kind}: memory #{target}", flush=True)
        watch.moving = seen_by_watch.moving = step_label["held"] = bench.scene.object(names[color]).label
        seen_by_watch.moving = names[color]
        print("  pick:", bench.run(skills.Pick(target), verbose=False, on_step=watch).message, flush=True)
        place = skills.Place(table)
        result = bench.run(place, verbose=False, on_step=watch)
        option = place.option
        print(f"  place at {option.spot.round(3).tolist()} from the {option.side} side: {result.message}", flush=True)
        seen_by_watch.moving = step_label["held"] = None
    bottle = names["blue"]
    p = bench.truth(bottle)
    print(f"blue bottle ends at {p.round(3).tolist()}, upright={upright(bench, bottle)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6, sys.argv[2] if len(sys.argv) > 2 else "table_b")
