"""Numbers that several parts of the robot software have to agree on."""

SCAN_DT = 0.1  # s between lidar scans (10 Hz); the tracker, the safety layer and the sim node all rely on it
CONTROL_DT = 0.02  # s: the sim node's control period (g1.isaac: 200 Hz physics / 4)
PERSON_RADIUS = 0.22  # m: a person seen from above, about 44 cm shoulder to shoulder (area.crowd.RADIUS)
