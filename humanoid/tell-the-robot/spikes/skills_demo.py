"""Phase 4: run a short script of skills headless and print what happens.

    humanoid/tell-the-robot/light.sh uv run python humanoid/tell-the-robot/spikes/skills_demo.py [seed]

Without a seed: the original room, look around, pick the blue bottle, put it on the mark.
"""

import sys
from pathlib import Path

import numpy as np

LESSON = Path(__file__).resolve().parents[2] / "03-g1-pick-place"
sys.path.insert(0, str(LESSON / "tests"))
sys.path.insert(0, str(LESSON))

from harness import Bench  # noqa: E402
from g1 import scenes, skills  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
bench = Bench(seed)
print(bench.run(skills.LookAround()).message)
bottle = bench.find("blue", "bottle")
result = bench.run(skills.Pick(bottle))
if result.ok:
    result = bench.run(skills.Place("mark"))
    true = bench.truth("bottle")
    print(f"truth: the bottle stands {100 * np.hypot(*(true[:2] - np.array(scenes.PLACE_MARK))):.1f} cm from the mark, at height {true[2]:.3f} m")
