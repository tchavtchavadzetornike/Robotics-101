"""Walking around tables instead of through them: the shortest free path on the floor.

Three classic steps (every textbook on motion planning starts here):

  1. Configuration space: treat the robot as a point and grow every table by
     the robot's radius. A point outside the grown tables is a robot that
     touches nothing.
  2. Visibility graph: the shortest path around rectangles bends only at their
     corners. Nodes: start, goal, and the corners of the grown tables (pushed
     out a little). Edges: every pair of nodes the robot can walk between in
     a straight line without entering a grown table.
  3. Dijkstra: the shortest way through that graph.

The robot has to get close to a table to grasp from it, closer than the grown
table allows. So a walk to a table goes in two parts: plan() to an approach
point outside the grown table, then straight in (skills.py does that).
"""

import heapq

import numpy as np

from g1 import scenes

ROBOT_RADIUS = 0.3  # m: shoulders, elbows held forward, and the swing of the feet
CORNER_OUT = 0.02  # m: corners are pushed out so a path along an edge doesn't count as inside


def obstacles(tables=None, grow=ROBOT_RADIUS):
    """Grown tables: [(x_min, y_min, x_max, y_max)]."""
    tables = scenes.TABLES if tables is None else tables
    return [(cx - hx - grow, cy - hy - grow, cx + hx + grow, cy + hy + grow) for (cx, cy), (hx, hy) in tables.values()]


def inside(point, box):
    return box[0] < point[0] < box[2] and box[1] < point[1] < box[3]


def crosses(a, b, box):
    """Does the segment a -> b pass through the open rectangle? (Liang-Barsky clipping)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-d[0], a[0] - box[0]),
        (d[0], box[2] - a[0]),
        (-d[1], a[1] - box[1]),
        (d[1], box[3] - a[1]),
    ):
        if abs(p) < 1e-12:
            if q <= 0:
                return False  # parallel to this side and outside it
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 >= t1:
            return False
    return t1 - t0 > 1e-9


def free(a, b, boxes):
    return not any(crosses(a, b, box) for box in boxes)


def plan(start, goal, boxes=None):
    """Shortest list of floor points from start to goal (both included), or None.

    start and goal must be outside every grown table.
    """
    boxes = obstacles() if boxes is None else boxes
    start, goal = np.asarray(start[:2], float), np.asarray(goal[:2], float)
    nodes = [start, goal]
    for x0, y0, x1, y1 in boxes:
        e = CORNER_OUT
        for corner in ((x0 - e, y0 - e), (x0 - e, y1 + e), (x1 + e, y0 - e), (x1 + e, y1 + e)):
            if not any(inside(corner, b) for b in boxes):
                nodes.append(np.array(corner))
    best = {0: 0.0}
    previous = {}
    queue = [(0.0, 0)]
    while queue:
        cost, i = heapq.heappop(queue)
        if i == 1:
            break
        if cost > best.get(i, np.inf):
            continue
        for j, node in enumerate(nodes):
            if j == i or not free(nodes[i], node, boxes):
                continue
            new = cost + np.linalg.norm(node - nodes[i])
            if new < best.get(j, np.inf):
                best[j] = new
                previous[j] = i
                heapq.heappush(queue, (new, j))
    if 1 not in previous and not free(start, goal, boxes):
        return None
    path, i = [goal], 1
    while i in previous:
        i = previous[i]
        path.append(nodes[i])
    if len(path) == 1:
        path.append(start)
    return path[::-1]


def length(path):
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:])))
