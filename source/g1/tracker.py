"""From lidar scans to people: where each person is and how they move.

Three standard steps (the same idea as the leg and people trackers in ROS):
  1. Cluster   scan points that lie close together belong to one thing.
  2. Detect    a small cluster is a person; its center is the person's spot.
  3. Track     match this scan's people to the last scan's people, and
               estimate each one's velocity from how they moved.

Everything is in the world frame (the square), so the robot's own walking
doesn't look like the people moving.
"""

import numpy as np

from .config import PERSON_RADIUS, SCAN_DT

# ---- clustering and detection ----
CLUSTER_GAP = 0.25  # m: neighbours on the scan closer than this are one thing
MAX_WIDTH = 0.8  # m: anything wider is not one person
MIN_POINTS = 2  # one point alone could be noise
# The lidar only sees the near side of a body. For a round body of radius R the
# middle of the visible arc lies 2R/pi = 0.64 R in front of the true center,
# so we push the detection that far back. R = 0.22 m gives 0.14 m.
CENTER_OFFSET = 2 * PERSON_RADIUS / np.pi

# ---- tracking (an "alpha-beta filter": the simplest tracking filter) ----
GATE = 0.6  # m: a detection farther than this from a track is someone else
# How much to trust a new measurement against the prediction:
ALPHA = 0.6  # for the position (1 = believe the lidar fully)
BETA = 0.3  # for the velocity (small = smooth, but slow to notice turns)
CONFIRM_HITS = 3  # seen in 3 scans (0.3 s) before we believe it is a person
MAX_MISSES = 8  # not seen for 8 scans (0.8 s, hidden behind someone): forget


def scan_points(ranges, pose):
    """Scan distances -> points (x, y) in the world. pose: robot (x, y, heading)."""
    angles = pose[2] + np.arange(len(ranges)) * 2 * np.pi / len(ranges)
    seen = np.isfinite(ranges)
    points = np.stack(
        [
            pose[0] + ranges[seen] * np.cos(angles[seen]),
            pose[1] + ranges[seen] * np.sin(angles[seen]),
        ],
        axis=1,
    )
    return points, np.flatnonzero(seen)


def detect_people(ranges, pose):
    """The centers (x, y) of everything in the scan that looks like a person."""
    points, ray = scan_points(ranges, pose)
    if len(points) == 0:
        return np.zeros((0, 2))
    # split into clusters wherever two neighbouring points are far apart,
    # or rays in between saw nothing
    gap = np.linalg.norm(np.diff(points, axis=0), axis=1)
    new_cluster = (gap > CLUSTER_GAP) | (np.diff(ray) > 2)
    clusters = np.split(points, np.flatnonzero(new_cluster) + 1)
    # the scan is a circle: the last cluster may continue into the first
    wraps = ray[0] + len(ranges) - ray[-1] <= 2
    if len(clusters) > 1 and wraps:
        if np.linalg.norm(points[0] - points[-1]) <= CLUSTER_GAP:
            clusters[0] = np.concatenate([clusters.pop(), clusters[0]])

    people = []
    for cluster in clusters:
        width = np.linalg.norm(cluster[0] - cluster[-1])
        if len(cluster) < MIN_POINTS or width > MAX_WIDTH:
            continue
        center = cluster.mean(axis=0)
        away = center - np.asarray(pose[:2])
        people.append(center + CENTER_OFFSET * away / np.linalg.norm(away))
    return np.array(people).reshape(-1, 2)


class Track:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2)
        self.hits = 1
        self.misses = 0


class Tracker:
    def __init__(self):
        self.tracks = []

    def update(self, ranges, pose):
        """Take one scan. Returns the confirmed people as rows (x, y, vx, vy)."""
        detections = detect_people(ranges, pose)
        for track in self.tracks:  # predict: everyone keeps walking as before
            track.position += track.velocity * SCAN_DT

        # match each track to the nearest free detection, closest pairs first
        free = set(range(len(detections)))
        pairs = sorted(
            (np.linalg.norm(track.position - detections[d]), t, d) for t, track in enumerate(self.tracks) for d in free
        )
        matched = set()
        for distance, t, d in pairs:
            if distance > GATE:
                break
            if t in matched or d not in free:
                continue
            matched.add(t)
            free.discard(d)
            track = self.tracks[t]
            error = detections[d] - track.position  # measurement minus prediction
            track.position += ALPHA * error
            track.velocity += BETA * error / SCAN_DT
            track.hits += 1
            track.misses = 0

        for t, track in enumerate(self.tracks):
            if t not in matched:
                track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= MAX_MISSES]
        self.tracks += [Track(detections[d]) for d in free]  # new people

        confirmed = [t for t in self.tracks if t.hits >= CONFIRM_HITS]
        return np.array([[*t.position, *t.velocity] for t in confirmed], dtype=float).reshape(-1, 4)
