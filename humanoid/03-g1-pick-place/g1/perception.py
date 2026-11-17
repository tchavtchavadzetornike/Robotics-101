"""Find the bottle in the head camera's images and say where it is in the room.

Four steps, each a function below:

  1. color_mask      which pixels have the bottle's color (hue, saturation, brightness)
  2. back_project    pixel + depth -> 3D point in the camera frame (the pinhole model, backwards)
  3. to_world        camera frame -> room frame, with the camera's pose
  4. cylinder_center the visible surface -> the bottle's center

Color alone is not enough: the floor tiles have the same hue as the bottle.
So step 3 is followed by a height check (MIN_HEIGHT): objects stand on tables,
the floor is at 0. Real systems do the same, color or a detector first, then
geometry to throw out what can't be right.

The pinhole model, forwards (g1/camera.py) and backwards (here):

    u = fx * X / Z + cx           X = (u - cx) * Z / fx
    v = fy * Y / Z + cy           Y = (v - cy) * Z / fy

Z is the depth image's value at (u, v): the distance along the viewing axis.
Pixel (u, v) covers u to u + 1, so its center is u + 0.5.
"""

import numpy as np

BOTTLE_HUE = (190.0, 225.0)  # degrees: blue, including the light blue top
MIN_SATURATION = 0.7  # the bottle is a strong blue, the floor tiles are greyer
MIN_BRIGHTNESS = 0.4
MIN_HEIGHT = 0.5  # m: everything lower is floor
MIN_POINTS = 30  # fewer bottle points than this: not seen
EDGE = 2  # pixels: bottle pixels this close to the top edge mean its top may be cut off


def hsv(rgb):
    """uint8 RGB image -> hue (degrees), saturation (0-1), brightness (0-1)."""
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    high = a.max(axis=-1)
    low = a.min(axis=-1)
    chroma = high - low
    safe = np.where(chroma == 0, 1.0, chroma)
    hue = np.select(
        [chroma == 0, high == r, high == g],
        [0.0, ((g - b) / safe) % 6, (b - r) / safe + 2],
        (r - g) / safe + 4,
    )
    saturation = np.where(high == 0, 0.0, chroma / np.where(high == 0, 1.0, high))
    return hue * 60.0, saturation, high


def color_mask(rgb, hue_range=BOTTLE_HUE, min_saturation=MIN_SATURATION, min_brightness=MIN_BRIGHTNESS):
    """True where a pixel has the wanted color."""
    hue, saturation, brightness = hsv(rgb)
    return (
        (hue >= hue_range[0])
        & (hue <= hue_range[1])
        & (saturation >= min_saturation)
        & (brightness >= min_brightness)
    )


def back_project(u, v, depth, fx, fy, cx, cy):
    """Pixels (u, v) with their depth -> (N, 3) points in the camera frame (x right, y down, z ahead)."""
    z = np.asarray(depth, dtype=float)
    x = (np.asarray(u) + 0.5 - cx) * z / fx
    y = (np.asarray(v) + 0.5 - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def to_world(points, position, rotation):
    """(N, 3) camera-frame points -> room frame. rotation, position: from Camera.pose()."""
    return points @ rotation.T + position


def cylinder_center(points, camera_position, radius, iterations=10):
    """Center of an upright cylinder (x, y) from points on its visible side.

    The visible points lie on a circle of the known radius around the axis.
    Find the circle center c that makes every |p - c| as close to the radius as
    possible (least squares, Gauss-Newton). Start one radius behind the points,
    away from the camera, where the center has to be.
    """
    xy = points[:, :2]
    mean = xy.mean(axis=0)
    away = mean - np.asarray(camera_position)[:2]
    center = mean + radius * away / np.linalg.norm(away)
    for _ in range(iterations):
        offset = xy - center
        distance = np.linalg.norm(offset, axis=1)
        distance = np.where(distance < 1e-9, 1e-9, distance)
        residual = distance - radius  # how far each point is from the circle
        jacobian = -offset / distance[:, None]  # how the residual changes when the center moves
        step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        center = center + step
        if np.linalg.norm(step) < 1e-6:
            break
    return center


def find_bottle(rgb, depth, fx, fy, cx, cy, cam_position, cam_rotation, radius, half_height):
    """The bottle's center in the room, and how many of its points were seen.

    Returns (center, points) or (None, points) if fewer than MIN_POINTS were seen.
    The height comes from the top: the highest bottle points are its top rim,
    the center is half a bottle lower. If the bottle touches the top edge of the
    image, its top may be outside the picture: then the height is NaN (unknown).
    """
    v, u = np.nonzero(color_mask(rgb))
    z = depth[v, u]
    camera_points = back_project(u, v, z, fx, fy, cx, cy)
    world = to_world(camera_points, cam_position, cam_rotation)
    world = world[world[:, 2] > MIN_HEIGHT]
    if len(world) < MIN_POINTS:
        return None, len(world)
    xy = cylinder_center(world, cam_position, radius)
    top_cut_off = v.min() < EDGE
    height = np.nan if top_cut_off else np.percentile(world[:, 2], 98) - half_height
    return np.array([xy[0], xy[1], height]), len(world)


def keep_height(seen, last):
    """Use the last known height when this frame couldn't measure it (NaN).

    Returns the filled-in position, or None if no height was ever measured.
    """
    if not np.isnan(seen[2]):
        return seen
    if last is None:
        return None
    return np.array([seen[0], seen[1], last[2]])
