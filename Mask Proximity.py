import torch
from TMM import *
import matplotlib.pyplot as plt
import skfmm

step = 1024
period = 6
dx = period - 2.15
airgap = 0
resist_thickness = 1.4

fx = torch.linspace(-step / period / 2, step / period / 2, step, dtype=torch.complex64)
kx = fx * torch.pi * 2
z = torch.linspace(0.001, resist_thickness, 144, dtype=torch.float32);
phi = 10
dr = z * torch.tan(torch.tensor(phi / 180 * torch.pi))

"""
Calculation for Mercury h-line (405 nm)
"""

k0 = 2 * torch.pi / 0.405
kx_trim = kx[torch.abs(kx) <= k0]
Fm = dx / period * torch.sinc(0.5 * dx * kx_trim / torch.pi)
I_coh = []

FK_RSI = torch.ones_like(kx_trim)
FK_RSII = k0 / torch.sqrt(k0 ** 2 - kx_trim ** 2) * FK_RSI
FK_KIR = 0.5 * (FK_RSI + FK_RSII)
c_inc = Fm * FK_KIR

B_dk = []
B_uk = []
for _k in kx_trim:
    sim = TMM(wavelength=0.405, kx=_k)
    sim.add_ref_layer()
    sim.add_layer(er_layer=1, thickness=airgap)
    sim.add_layer(er_layer=1.70 ** 2, thickness=resist_thickness)
    sim.add_layer(er_layer=1.48 ** 2, thickness=0.5)
    sim.add_trs_layer(er_trs=6.67309475 - 13.6503061 * 1j)
    sim.solve_TMM()
    B_d, B_u = sim.find_resist_layer_param(resist_layer=2)
    B_dk.append(B_d)
    B_uk.append(B_u)

B_dk = torch.stack(B_dk)
B_uk = torch.stack(B_uk)

Psi_z = []
for _z in z:
    S_z = B_dk * torch.exp(1j * _z * torch.sqrt(1.70 ** 2 * k0 ** 2 - kx_trim ** 2))
    + B_uk * torch.exp(- 1j * _z * torch.sqrt(1.70 ** 2 * k0 ** 2 - kx_trim ** 2))
    Psi = torch.fft.ifftshift(torch.fft.ifft(c_inc * S_z, n=step, norm='forward'), dim=-1)
    Psi_z.append(Psi)

Psi_z = torch.stack(Psi_z)
I_fft = torch.fft.fftshift(torch.fft.fft(torch.abs(Psi_z) ** 2, norm='ortho'), dim=-1)
FP = 2 * torch.special.bessel_j1(torch.outer(dr, torch.abs(kx))) / torch.outer(dr, torch.abs(kx))
I_pc_h = torch.abs(torch.fft.ifft(I_fft * FP, norm='ortho')) ** 2

"""
Calculation for Mercury i-line (365 nm)
"""

k0 = 2 * torch.pi / 0.365
kx_trim = kx[torch.abs(kx) <= k0]
Fm = dx / period * torch.sinc(0.5 * dx * kx_trim / torch.pi)
I_coh = []

FK_RSI = torch.ones_like(kx_trim)
FK_RSII = k0 / torch.sqrt(k0 ** 2 - kx_trim ** 2) * FK_RSI
FK_KIR = 0.5 * (FK_RSI + FK_RSII)

c_inc = Fm * FK_KIR

B_dk = []
B_uk = []
for _k in kx_trim:
    sim = TMM(wavelength=0.365, kx=_k)
    sim.add_ref_layer()
    sim.add_layer(er_layer=1, mur_layer=1, thickness=airgap)
    sim.add_layer(er_layer=1.70 ** 2, mur_layer=1, thickness=resist_thickness)
    sim.add_layer(er_layer=1.48 ** 2, mur_layer=1, thickness=0.5)
    sim.add_trs_layer(er_trs=6.67309475 - 13.6503061 * 1j, mur_trs=1)
    sim.solve_TMM()
    B_d, B_u = sim.find_resist_layer_param(resist_layer=2)
    B_dk.append(B_d)
    B_uk.append(B_u)

B_dk = torch.stack(B_dk)
B_uk = torch.stack(B_uk)

Psi_z = []
for _z in z:
    S_z = B_dk * torch.exp(1j * _z * torch.sqrt(1.70 ** 2 * k0 ** 2 - kx_trim ** 2)) + B_uk * torch.exp(
        - 1j * _z * torch.sqrt(1.70 ** 2 * k0 ** 2 - kx_trim ** 2))
    Psi = torch.fft.ifftshift(torch.fft.ifft(c_inc * S_z, n=step, norm='forward'), dim=-1)
    Psi_z.append(Psi)
Psi_z = torch.stack(Psi_z)
I_fft = torch.fft.fftshift(torch.fft.fft(torch.abs(Psi_z) ** 2, norm='ortho'), dim=-1)
FP = 2 * torch.special.bessel_j1(torch.outer(dr, torch.abs(kx))) / torch.outer(dr, torch.abs(kx))
I_pc_i = torch.abs(torch.fft.ifft(I_fft * FP, norm='ortho')) ** 2

I_pc = torch.fft.fftshift(0.25 * I_pc_i + 0.75 * I_pc_h, dim = -1)

plt.imshow(I_pc, cmap = 'grey', aspect='auto', vmin=0, extent=[-period / 2, period / 2, resist_thickness, 0]);
plt.xlabel('x position (um)')
plt.ylabel('z position in resist (um)')
plt.title('Standing wave (Normalized) in resist after exposure')
plt.colorbar()
plt.show()

m_r = torch.exp(- 360 * 5.5 * 0.0220 * I_pc)
x = torch.linspace(-period / 2, period / 2, step)
z = torch.linspace(-resist_thickness / 2, resist_thickness / 2, 144)
X, Z = torch.meshgrid(x, z, indexing='xy')

# introducing the 4 parameter mack model:
r_min = 0.1827 / 1000
r_max = r_min * 300
n = 2
mth = 0.021

# Post Exposure Bake Model, using simple Fickian model
Bake_time = 60 # s
D_A = 9.24 * 1e-6
sigma = torch.sqrt(torch.tensor(2 * Bake_time * D_A))
DPSF = torch.exp(-(X ** 2 + Z ** 2) / (2 * sigma ** 2))
F_DPSF = torch.fft.fft2(DPSF, norm='forward')
F_mr = torch.fft.fft2(m_r, norm='forward')
m_rdiff = torch.abs(torch.fft.ifftshift(torch.fft.ifft2(F_DPSF * F_mr, norm='forward')))
m_rdiff /= m_rdiff.max()

plt.imshow(m_r, aspect='auto', vmin=0, extent=[-period / 2, period / 2, resist_thickness, 0]);
plt.xlabel('x position (um)')
plt.ylabel('z position in resist (um)')
plt.title('PAC conc. (Normalized) in resist after exposure')
plt.colorbar()
plt.show()

plt.imshow(m_rdiff, aspect='auto', vmin=0, extent=[-period / 2, period / 2, resist_thickness, 0]);
plt.xlabel('x position (um)')
plt.ylabel('z position in resist (um)')
plt.title('PAC conc. (Normalized) in resist after PEB')
plt.colorbar()
plt.show()

lat_img = m_rdiff

a = (n + 1) / (n - 1) * (1 - mth) ** n
r_m = r_max * (a + 1) * (1 - lat_img) ** n / (a + (1 - lat_img) ** n) + r_min
cur_dev_rate = torch.clip(r_m, r_min, r_max)

neg_layer = -1 * torch.ones_like(lat_img[0, :])
lat_img = torch.vstack([lat_img, neg_layer])
cur_dev_rate = torch.vstack([neg_layer, cur_dev_rate])
dx = period / step
dz = resist_thickness / 144
time_resist_fmm = skfmm.travel_time(lat_img, cur_dev_rate, dx=[dz, dx], periodic=[False, True])
time_resist_fmm = torch.as_tensor(time_resist_fmm[1:, :])
time_resist_fmm = torch.rot90(time_resist_fmm, k=2)

fig, ax = plt.subplots(1, 1)
ax.set_title('Resist profile computed with fast marching algorithm')
ax.imshow(time_resist_fmm, extent=[-period / 2, period / 2, resist_thickness, 0])
ax.contour(X, Z + resist_thickness / 2, time_resist_fmm, levels=[60, 120], colors=['r', 'b'])

ax.set_xlabel('x position [um]')
ax.set_ylabel('Resist height [um]')
plt.show()