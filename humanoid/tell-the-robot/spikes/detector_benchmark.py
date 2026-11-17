"""Phase 3 spike: YOLO-World against OWLv2 on our own rendered scenes.

For each random scene, the head camera looks from 4 places a robot would stand.
The truth comes from MuJoCo's segmentation image: every object with at least
MIN_PIXELS visible pixels is one true box. Each detector gets the prompts
"bottle", "can", "box". A detection counts when it overlaps a true box of the
same kind with IoU >= 0.5.

    uv run python humanoid/tell-the-robot/spikes/detector_benchmark.py [scenes] [yolo|owl|both] [names|shapes]

First run downloads the weights (yolov8s-worldv2.pt, its CLIP text model,
google/owlv2-base-patch16-ensemble).
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LESSON = HERE.parents[1] / "03-g1-pick-place"
sys.path.insert(0, str(LESSON))

from g1 import scenes  # noqa: E402
from g1.camera import Camera  # noqa: E402
from g1.simulator import Simulator  # noqa: E402

KINDS = ["bottle", "can", "box"]
# What the detector is asked for, and which kind each prompt means. Several prompts
# can mean the same kind: plain shapes look more like "a cylinder" than "a can".
PROMPT_SETS = {
    "names": {"bottle": "bottle", "can": "can", "box": "box"},
    "shapes": {"bottle": "bottle", "can": "can", "cylinder": "can", "box": "box", "cube": "box", "block": "box"},
}
PROMPTS = PROMPT_SETS["names"]
MIN_PIXELS = 150
VIEWS = {  # where the robot stands: (x, y, heading)
    "start": (0.0, 0.0, 0.0),
    "at_table_a": (1.12, 0.0, 0.0),
    "at_table_b": (1.0, -1.15, -np.pi / 2),
    "middle": (0.5, -0.6, -0.6),
}
THRESHOLDS = np.round(np.arange(0.05, 0.55, 0.05), 2)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def truth_boxes(seg, model, scene):
    boxes = []
    for obj in scene.objects:
        mask = seg == model.body(obj.name).id
        if mask.sum() < MIN_PIXELS:
            continue
        v, u = np.nonzero(mask)
        boxes.append((obj.kind, (u.min(), v.min(), u.max() + 1, v.max() + 1), obj.label))
    return boxes


class YoloWorld:
    name = "YOLO-World v2 s"

    def __init__(self):
        from ultralytics import YOLOWorld

        self.model = YOLOWorld("yolov8s-worldv2.pt")
        self.prompts = list(PROMPTS)
        self.model.set_classes(self.prompts)

    def __call__(self, rgb):
        result = self.model.predict(rgb[..., ::-1], conf=0.05, device="mps", verbose=False)[0]
        boxes = result.boxes
        return [
            (PROMPTS[self.prompts[int(c)]], tuple(b), float(s))
            for b, c, s in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy())
        ]


class Owl:
    name = "OWLv2 base"

    def __init__(self):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        name = "google/owlv2-base-patch16-ensemble"
        self.processor = Owlv2Processor.from_pretrained(name)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = Owlv2ForObjectDetection.from_pretrained(name).to(self.device).eval()
        self.prompts = list(PROMPTS)
        self.texts = [[f"a photo of a {k}" for k in self.prompts]]

    def __call__(self, rgb):
        from PIL import Image

        image = Image.fromarray(rgb)
        inputs = self.processor(text=self.texts, images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        side = max(rgb.shape[:2])  # OWLv2 pads the image to a square
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=0.05, target_sizes=self.torch.tensor([[side, side]])
        )[0]
        return [
            (PROMPTS[self.prompts[int(l)]], tuple(b), float(s))
            for b, l, s in zip(results["boxes"].cpu().numpy(), results["labels"].cpu().numpy(), results["scores"].cpu().numpy())
        ]


def score(detections, truths, threshold, any_kind=False):
    kept = sorted([d for d in detections if d[2] >= threshold], key=lambda d: -d[2])
    matched = set()
    hits = 0
    ious = []
    for kind, box, _ in kept:
        best, best_iou = None, 0.5
        for i, (true_kind, true_box, _) in enumerate(truths):
            if i in matched or (true_kind != kind and not any_kind):
                continue
            overlap = iou(box, true_box)
            if overlap >= best_iou:
                best, best_iou = i, overlap
        if best is not None:
            matched.add(best)
            hits += 1
            ious.append(best_iou)
    return hits, len(kept), len(truths), ious


def main(scene_count, which="both"):
    robot_dir = LESSON / "robot"
    frames = []
    for seed in range(scene_count):
        scene = scenes.make_scene(seed)
        path = str(scenes.write_scene(scene, robot_dir))
        camera = Camera(path, "head_cam", 640, 480)
        for view, start in VIEWS.items():
            sim = Simulator(path, start=start)
            camera.set_state(sim.world_state())
            truths = truth_boxes(camera.segmentation(), camera.model, scene)
            frames.append((seed, view, camera.color(), truths))
    print(f"{len(frames)} images, {sum(len(f[3]) for f in frames)} visible objects\n")

    chosen = {"yolo": (YoloWorld,), "owl": (Owl,), "both": (YoloWorld, Owl)}[which]
    for detector_class in chosen:
        detector = detector_class()
        detector(frames[0][2])  # warm up
        times, results = [], []
        for _, view, rgb, truths in frames:
            start = time.perf_counter()
            detections = detector(rgb)
            times.append(time.perf_counter() - start)
            results.append((view, detections, truths))
        print(f"{detector.name}: {1000 * np.median(times):.0f} ms median per image")
        print("  threshold  precision  recall   F1   mean IoU")
        best = None
        for threshold in THRESHOLDS:
            hits = kept = total = 0
            ious = []
            for _, detections, truths in results:
                h, k, t, i = score(detections, truths, threshold)
                hits, kept, total, ious = hits + h, kept + k, total + t, ious + i
            precision = hits / kept if kept else 0.0
            recall = hits / total if total else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            print(f"  {threshold:9.2f}  {precision:9.2f}  {recall:6.2f}  {f1:5.2f}  {np.mean(ious) if ious else 0:6.2f}")
            if best is None or f1 > best[1]:
                best = (threshold, f1)
        threshold = best[0]
        print(f"  best threshold {threshold}: recall per view", end="")
        for view in VIEWS:
            hits = total = 0
            for v, detections, truths in results:
                if v == view:
                    h, _, t, _ = score(detections, truths, threshold)
                    hits, total = hits + h, total + t
            print(f"  {view} {hits}/{total}", end="")
        print()
        hits = total = 0
        confusion = {k: {j: 0 for j in KINDS + ["missed"]} for k in KINDS}
        for _, detections, truths in results:
            h, _, t, _ = score(detections, truths, threshold, any_kind=True)
            hits, total = hits + h, total + t
            kept = [d for d in detections if d[2] >= threshold]
            for true_kind, true_box, _ in truths:
                best = max(kept, key=lambda d: iou(d[1], true_box), default=None)
                if best is None or iou(best[1], true_box) < 0.5:
                    confusion[true_kind]["missed"] += 1
                else:
                    confusion[true_kind][best[0]] += 1
        print(f"  found at all (any kind, IoU >= 0.5): {hits}/{total} = {hits / total:.2f}")
        print("  true kind -> predicted: " + "  ".join(f"{k}: {v}" for k, v in confusion.items()))
        print()


if __name__ == "__main__":
    if len(sys.argv) > 3:
        PROMPTS = PROMPT_SETS[sys.argv[3]]
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10, sys.argv[2] if len(sys.argv) > 2 else "both")
