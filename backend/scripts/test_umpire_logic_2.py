import numpy as np
from core.umpire import RallyUmpire

frames = [2376, 2381, 2386, 2391, 2396, 2401]
pts = [
    [791, 510],
    [774, 552],
    [756, 629],
    [745, 677],
    [737, 700],
    [741, 690]
]
flight = []
for i in range(len(frames)):
    flight.append((frames[i], pts[i], False))

umpire = RallyUmpire(H_inv=np.eye(3), match_type="singles", frame_size=(1080, 1920), rest_frames=3)
print("Testing landing_point...")
i = umpire.landing_point(flight)
print("Result:", i)
