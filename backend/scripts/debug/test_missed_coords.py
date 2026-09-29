import numpy as np
from core import court_model

H, H_inv = court_model.load_calibration()
print("1116:", court_model.img_to_world([[1787, 650]], H_inv)[0])
print("1526:", court_model.img_to_world([[1787, 665]], H_inv)[0])
