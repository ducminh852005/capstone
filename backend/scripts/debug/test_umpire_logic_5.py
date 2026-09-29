import numpy as np
from core.umpire import RallyUmpire
from core import court_model

H, H_inv = court_model.load_calibration()

lines = """Frame 1411: Shuttle detected at (1541, 388)
Frame 1416: Shuttle detected at (1549, 432)
Frame 1421: Shuttle detected at (1549, 461)
Frame 1426: Shuttle detected at (1518, 493)
Frame 1431: Shuttle detected at (1607, 488)
Frame 1436: Shuttle detected at (1647, 500)
Frame 1441: Shuttle detected at (1510, 510)
Frame 1446: Shuttle detected at (1518, 513)
Frame 1451: Shuttle detected at (1724, 564)
Frame 1466: Shuttle detected at (1792, 658)
Frame 1471: Shuttle detected at (1792, 650)
Frame 1476: Shuttle detected at (1785, 641)
Frame 1481: Shuttle detected at (1782, 636)
Frame 1486: Shuttle detected at (1785, 643)
Frame 1491: Shuttle detected at (1787, 656)
Frame 1496: Shuttle detected at (1784, 662)
Frame 1501: Shuttle detected at (1785, 660)
Frame 1511: Shuttle detected at (1787, 662)
Frame 1521: Shuttle detected at (1789, 663)
Frame 1526: Shuttle detected at (1787, 665)"""

flight = []
for line in lines.split('\n'):
    parts = line.split(' ')
    f_idx = int(parts[1][:-1])
    x = int(parts[5][1:-1])
    y = int(parts[6][:-1])
    flight.append((f_idx, [x, y], False))

umpire = RallyUmpire(H_inv=H_inv, match_type="singles", frame_size=(1080, 1920), rest_frames=3)
umpire.net_top_y = None
print("Testing landing_point...")
idx = umpire.landing_point(flight)
print("landing_point:", idx)
if idx is not None:
    print("Frame:", flight[idx][0])
    print("_judge_flight:", umpire._judge_flight(flight))
