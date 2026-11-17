"""Where the objects are in an image: OWLv2, an open-vocabulary detector.

Open-vocabulary: the things to find are given as text, not trained in. OWLv2
compares every image patch with the text prompts (a CLIP-style model: images
and text in one space) and returns boxes with a score per prompt.

On our plain, untextured shapes it finds objects best when asked for shapes
("cylinder", "cube") and names together, but its kind label is unreliable
(every bottle is a "cylinder"). So it is used here only for *where*: all
prompts together, overlapping boxes merged, kind and color come later
(g1/objects.py). Measured in tell-the-robot/PHASE-3.md.

The weights (google/owlv2-base-patch16-ensemble, ~600 MB) come from Hugging Face
on first use and stay in ~/.cache/huggingface.
"""

import os

import numpy as np

MODEL = "google/owlv2-base-patch16-ensemble"
PROMPTS = ["bottle", "can", "cylinder", "box", "cube", "block"]
THRESHOLD = 0.15  # 0.2 missed objects half hidden behind others (PHASE-3.md)
SAME_OBJECT_IOU = 0.4  # boxes overlapping this much show the same object


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_overlapping(boxes, scores, same=SAME_OBJECT_IOU):
    """Non-maximum suppression: keep the best box, drop every box that overlaps it, repeat."""
    order = np.argsort(scores)[::-1]
    kept = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) < same for j in kept):
            kept.append(i)
    return [boxes[i] for i in kept], [scores[i] for i in kept]


class Detector:
    def __init__(self, prompts=PROMPTS, threshold=THRESHOLD):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", torch.get_num_threads())))
        self.threshold = threshold
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained(MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL).to(self.device).eval()
        self.texts = [[f"a photo of a {p}" for p in prompts]]

    def __call__(self, rgb):
        """uint8 RGB image -> (boxes [(u1, v1, u2, v2)], scores), one box per object."""
        from PIL import Image

        inputs = self.processor(text=self.texts, images=Image.fromarray(rgb), return_tensors="pt")
        inputs = inputs.to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        side = max(rgb.shape[:2])  # OWLv2 pads the image to a square
        result = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=self.threshold, target_sizes=[(side, side)]
        )[0]
        boxes = [tuple(b) for b in result["boxes"].cpu().numpy()]
        scores = list(result["scores"].cpu().numpy())
        return merge_overlapping(boxes, scores)
