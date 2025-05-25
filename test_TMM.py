"""
this code is used only to test the TMM.py module
"""

from TMM import *
import torch

kx = torch.linspace(0.01,8,48)
ky = torch.linspace(0.01,0.1,3)

sim = TMM_2D(wavelength = 0.405, kx = kx, ky = ky)
sim.add_ref_layer()
sim.add_layer(er_layer = 1.00, mur_layer= 1, thickness = 1)
sim.add_layer(er_layer = 1.00, mur_layer= 1, thickness = 1.4)
sim.add_layer(er_layer = 1.00, mur_layer= 1, thickness = 0.5)
sim.add_trs_layer(er_trs = 10.00, mur_trs = 1)
sim.solve_TMM(RT = True)
# B_dk, B_uk = sim.find_resist_layer_param(resist_layer = 1)
# B_dk2, B_uk2 = sim.find_resist_layer_param(resist_layer = 2)
# B_dk3, B_uk3 = sim.find_resist_layer_param(resist_layer = 3)

# print(B_dk, B_uk)
# print(B_uk, B_uk2, B_uk3)
