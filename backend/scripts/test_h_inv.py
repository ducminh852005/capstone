import numpy as np
import cv2
from core import court_model

H, H_inv = court_model.load_calibration()
img_pt = [1514, 599]
world = court_model.img_to_world([img_pt], H_inv)[0]
print("World coords:", world)
floor_region = (-3.0, 16.4, -3.0, 8.18)
in_reg = court_model.in_region(world, floor_region)
print("In region:", in_reg)
