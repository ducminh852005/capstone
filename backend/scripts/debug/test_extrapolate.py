import numpy as np
from core import court_model

H, H_inv = court_model.load_calibration()
# court bounds in meters
floor_region = (-3.0, 16.4, -3.0, 8.18)

def extrapolate_drop(img_pt, vx, vy, max_steps=50):
    cur_pt = np.array(img_pt, dtype=float)
    for step in range(1, max_steps + 1):
        cur_pt[0] += vx
        cur_pt[1] += vy
        
        world = court_model.img_to_world([cur_pt], H_inv)[0]
        if court_model.in_region(world, floor_region):
            return step, cur_pt, world
    return None

# Flight 7467: Frame 7418 (1314, 434) to 7426 (1304, 500)
dx = 1304 - 1314
dy = 500 - 434
dt = 8
vx = dx / dt
vy = dy / dt

print(f"Extrapolating from (1304, 500) with vx={vx}, vy={vy}...")
result = extrapolate_drop([1304, 500], vx, vy)
if result:
    step, pt, w = result
    print(f"Hit floor after {step} frames at image {pt}, world {w}")
else:
    print("Never hit floor")
    
# Flight 5915: 5834 (1095, 518) to 5842 (1060, 616)
dx = 1060 - 1095
dy = 616 - 518
vx = dx / 8
vy = dy / 8
print(f"\nExtrapolating 5842: from (1060, 616) with vx={vx}, vy={vy}...")
result = extrapolate_drop([1060, 616], vx, vy)
if result:
    step, pt, w = result
    print(f"Hit floor after {step} frames at image {pt}, world {w}")
else:
    print("Never hit floor")
