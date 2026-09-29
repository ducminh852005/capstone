import numpy as np

frames = [391, 396, 406, 411, 416, 421, 426, 431, 436, 441, 446, 451, 456, 461, 466, 476]
pts = [
    [1340, 498],
    [1433, 542],
    [1514, 599],
    [1534, 619],
    [1559, 645],
    [1565, 636],
    [1575, 629],
    [1576, 629],
    [1578, 634],
    [1579, 631],
    [1579, 631],
    [1576, 629],
    [1576, 634],
    [1578, 634],
    [1581, 634],
    [1541, 586]
]
flight = []
for i in range(len(frames)):
    flight.append((frames[i], pts[i], False))

from core.umpire import RallyUmpire
umpire = RallyUmpire(H_inv=np.eye(3), match_type="singles", frame_size=(1080, 1920), rest_frames=2)
print("Testing landing_point...")
i = umpire.landing_point(flight)
print("Result:", i)
