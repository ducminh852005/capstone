import numpy as np
from core.umpire import RallyUmpire
from core import court_model

H, H_inv = court_model.load_calibration()

lines = """Frame 7418: Shuttle detected at (1314, 434)
Frame 7426: Shuttle detected at (1304, 500)"""

flight = []
for line in lines.split('\n'):
    parts = line.split(' ')
    f_idx = int(parts[1][:-1])
    x = int(parts[5][1:-1])
    y = int(parts[6][:-1])
    flight.append((f_idx, [x, y], False))

# We need at least 3 points for extrapolation logic
flight = [(7410, [1324, 380], False)] + flight

umpire = RallyUmpire(H_inv=H_inv, match_type="singles", frame_size=(1080, 1920), rest_frames=3)
umpire.net_top_y = None
print("Testing _judge_flight...")
call = umpire._judge_flight(flight)
print("_judge_flight:", call)
