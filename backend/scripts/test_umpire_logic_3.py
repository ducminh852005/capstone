import numpy as np
from core.umpire import RallyUmpire
from core import court_model

H, H_inv = court_model.load_calibration()

lines = """Frame 991: Shuttle detected at (1322, 253)
Frame 996: Shuttle detected at (1401, 226)
Frame 1001: Shuttle detected at (1446, 219)
Frame 1006: Shuttle detected at (1499, 213)
Frame 1011: Shuttle detected at (1530, 218)
Frame 1016: Shuttle detected at (1573, 230)
Frame 1021: Shuttle detected at (1597, 243)
Frame 1026: Shuttle detected at (1628, 268)
Frame 1031: Shuttle detected at (1645, 289)
Frame 1036: Shuttle detected at (1671, 324)
Frame 1041: Shuttle detected at (1687, 351)
Frame 1046: Shuttle detected at (1707, 390)
Frame 1051: Shuttle detected at (1719, 422)
Frame 1081: Shuttle detected at (1774, 640)
Frame 1086: Shuttle detected at (1774, 631)
Frame 1091: Shuttle detected at (1774, 626)
Frame 1096: Shuttle detected at (1774, 623)
Frame 1101: Shuttle detected at (1777, 629)
Frame 1106: Shuttle detected at (1779, 646)
Frame 1111: Shuttle detected at (1779, 653)
Frame 1116: Shuttle detected at (1787, 650)
Frame 1121: Shuttle detected at (1789, 650)
Frame 1126: Shuttle detected at (1792, 651)"""

flight = []
for line in lines.split('\n'):
    parts = line.split(' ')
    f_idx = int(parts[1][:-1])
    x = int(parts[5][1:-1])
    y = int(parts[6][:-1])
    flight.append((f_idx, [x, y], False))

umpire = RallyUmpire(H_inv=H_inv, match_type="singles", frame_size=(1080, 1920), rest_frames=3)
umpire.net_top_y = None  # to test landing logic only
print("Testing landing_point...")
idx = umpire.landing_point(flight)
print("landing_point:", idx)
if idx is not None:
    print("Frame:", flight[idx][0])
    print("_judge_flight:", umpire._judge_flight(flight))
