import torch


class TMM:
    def __init__(self, wavelength, kx):
        if torch.cuda.is_available == True:
            self.torch_device = torch.device('cuda')
        else:
            self.torch_device = torch.device('cpu')

        self.dtype = torch.complex64

        self.wavelength = torch.as_tensor(wavelength, dtype=self.dtype, device=self.torch_device)
        self.k0 = torch.pi * 2 / self.wavelength
        self.kx_inc = kx / self.k0
        self.kz_inc = torch.sqrt(1 - self.kx_inc ** 2)

        # initialize the device scattering matrix
        S12_global = torch.eye(2, dtype=self.dtype, device=self.torch_device)
        S11_global = torch.zeros_like(S12_global, dtype=self.dtype, device=self.torch_device)

        self.S_global = torch.stack([S11_global, S12_global, S12_global, S11_global])

        # initialize list of matrix to be stored in the forward pass
        self.V_store = []
        self.kz_store = []
        self.S_global_store = []
        self.layer_count = 0
        self.layer_store = torch.tensor(0, dtype=torch.float64, device=self.torch_device).unsqueeze(0)
        self.er_layer = []
        self.mur_layer = []

    def add_ref_layer(self, er_ref=1.0, mur_ref=1.0):
        self.er_ref = torch.tensor(er_ref, dtype=self.dtype, device=self.torch_device)
        self.mur_ref = torch.tensor(mur_ref, dtype=self.dtype, device=self.torch_device)
        self.n_inc = torch.sqrt(self.er_ref * self.mur_ref)

        # calculating the k vector of the incoming wave
        self.k_inc = torch.tensor([self.kx_inc, 0.0, self.kz_inc], device=self.torch_device)
        self.kz_ref = torch.sqrt(self.er_ref * self.mur_ref - self.kx_inc ** 2)
        self.initialize_gap_medium()
        # solve for the reflection region scattering matrix
        self.S_global = self.solve_ref_region_S_matrix(self.S_global)

    def initialize_gap_medium(self):
        # initialize the gap material matrices
        self.Q_g = torch.vstack(
            [torch.tensor([0.0, 1.0 - self.kx_inc ** 2], dtype=self.dtype, device=self.torch_device),
             torch.tensor([-1.0, 0.0], dtype=self.dtype, device=self.torch_device)])

        self.V_g = -1j * self.Q_g / self.kz_inc

    def add_layer(self, er_layer=1.0, mur_layer=1.0, thickness=0.0):
        thickness = torch.tensor(thickness, dtype=torch.float32, device=self.torch_device).unsqueeze(0)
        self.layer_count += 1
        self.er_layer.append(er_layer)
        self.mur_layer.append(mur_layer)
        self.S_global = self.solve_layer_S_matrix(thickness, self.S_global, er_layer, mur_layer)
        self.layer_store = torch.cat([self.layer_store, thickness])

    def add_trs_layer(self, er_trs=1.0, mur_trs=1.0):
        self.er_trs = torch.as_tensor(er_trs, dtype=self.dtype, device=self.torch_device)
        self.mur_trs = torch.as_tensor(mur_trs, dtype=self.dtype, device=self.torch_device)
        self.kz_trs = torch.sqrt(self.er_trs * self.mur_trs - self.kx_inc ** 2)
        self.S_global = self.solve_trs_region_S_matrix(self.S_global)

    def solve_layer_S_matrix(self, layer_thickness, S_global, er_layer, mur_layer):
        kz_i = torch.sqrt(mur_layer * er_layer - self.kx_inc ** 2)
        Q_i = 1 / mur_layer * torch.vstack(
            [torch.tensor([0, er_layer * mur_layer - self.kx_inc ** 2]), torch.tensor([- mur_layer * er_layer, 0])])

        V_i = - 1j * Q_i / kz_i
        X_i = torch.exp(1j * kz_i * self.k0 * layer_thickness)

        S11, S12 = self.get_scattering_matrix(V_i, X_i)

        S_layer = torch.stack([S11, S12, S12, S11])
        S_global = self.calc_Redheffer_star(S_global, S_layer)

        self.V_store.append(V_i)
        self.kz_store.append(kz_i)
        self.S_global_store.append(S_global)
        return S_global

    def get_scattering_matrix(self, V_layer, X_layer):
        I = torch.eye(2, dtype=self.dtype, device=self.torch_device)
        A_i = I + torch.linalg.solve(V_layer, self.V_g)
        B_i = I - torch.linalg.solve(V_layer, self.V_g)

        D = A_i - X_layer * B_i @ torch.linalg.solve(A_i, B_i) * X_layer
        S11 = torch.linalg.solve(D, (X_layer * B_i * X_layer - B_i))
        S12 = torch.linalg.solve(D, X_layer * (A_i - B_i @ torch.linalg.solve(A_i, B_i)))
        return S11, S12

    def calc_Redheffer_star(self, SA, SB):
        S11_A, S12_A, S21_A, S22_A = SA
        S11_B, S12_B, S21_B, S22_B = SB

        I = torch.eye(2, dtype=self.dtype, device=self.torch_device)

        D = I - S11_B @ S22_A
        F = I - S22_A @ S11_B

        S11_AB = S11_A + S12_A @ torch.linalg.solve(D, S11_B) @ S21_A
        S12_AB = S12_A @ torch.linalg.solve(D, S12_B)
        S21_AB = S21_B @ torch.linalg.solve(F, S21_A)
        S22_AB = S22_B + S21_B @ torch.linalg.solve(D, S22_A) @ S12_B

        S_AB = torch.stack([S11_AB, S12_AB, S21_AB, S22_AB])
        return S_AB

    def solve_ref_region_S_matrix(self, S_global):

        Q_ref = 1 / self.mur_ref * torch.vstack([torch.tensor([0, self.er_ref * self.mur_ref - self.kx_inc ** 2]),
                                                 torch.tensor([-self.er_ref * self.mur_ref, 0])])
        V_ref = 1j * Q_ref / self.kz_ref

        I_ref = torch.eye(2, dtype=self.dtype, device=self.torch_device)
        O_ref = torch.zeros_like(I_ref)
        S11_ref = O_ref
        S12_ref = I_ref

        S_ref = torch.stack([S11_ref, S12_ref, S12_ref, S11_ref])
        S_global = self.calc_Redheffer_star(S_ref, S_global)

        self.V_store.append(V_ref)
        self.kz_store.append(self.kz_ref)
        self.S_global_store.append(S_global)
        return S_global

    def solve_trs_region_S_matrix(self, S_global):
        Q_trs = 1 / self.mur_trs * torch.vstack([torch.tensor([0, self.er_trs * self.mur_trs - self.kx_inc ** 2]),
                                                 torch.tensor([-self.er_trs * self.mur_trs, 0])])
        V_trs = -1j * Q_trs / self.kz_trs

        I = torch.eye(2, dtype=self.dtype, device=self.torch_device)
        A_trs = I + torch.linalg.solve(self.V_g, V_trs)
        B_trs = I - torch.linalg.solve(self.V_g, V_trs)

        S11_trs = torch.linalg.solve(A_trs, B_trs, left=False)
        S12_trs = 0.5 * (A_trs - B_trs @ torch.linalg.solve(A_trs, B_trs))
        S21_trs = 2 * torch.linalg.inv(A_trs)
        S22_trs = - torch.linalg.solve(A_trs, B_trs)

        S_trs = torch.stack([S11_trs, S12_trs, S21_trs, S22_trs])
        S_global = self.calc_Redheffer_star(S_global, S_trs)

        self.V_store.append(V_trs)
        self.kz_store.append(self.kz_trs)
        self.S_global_store.append(S_global)
        return S_global

    def solve_TMM(self):
        a_TE = torch.tensor([0, 1, 0], dtype=self.dtype, device=self.torch_device)

        a_TM = torch.linalg.cross(self.k_inc, a_TE)
        a_TM = a_TM / torch.linalg.norm(a_TM)

        self.P = 0.0 * a_TE + 1.0 * a_TM
        self.P = self.P / torch.linalg.norm(self.P)

        self.c_src = self.P[:2]
        self.c_ref = self.S_global[0] @ self.c_src
        self.c_trs = self.S_global[2] @ self.c_src

        self.Ez_ref = - self.kx_inc * self.c_ref[0] / self.kz_ref
        self.Ez_trs = - self.kx_inc * self.c_trs[0] / self.kz_trs

        R = torch.linalg.norm(torch.tensor([self.c_ref[0], self.c_ref[1], self.Ez_ref])) ** 2
        T = torch.linalg.norm(torch.tensor([self.c_trs[0], self.c_trs[1], self.Ez_trs])) ** 2 * torch.real(self.kz_trs / self.mur_trs) / torch.real(self.kz_inc)
        return R, T
    def find_resist_layer_param(self, resist_layer = 1):
        I = torch.eye(2, dtype=self.dtype, device=self.torch_device)

        c_ln = torch.linalg.solve(self.S_global_store[resist_layer - 1][1],
                                  self.c_ref - self.S_global_store[resist_layer - 1][0] @ self.c_src)
        c_lp = self.S_global_store[resist_layer - 1][2] @ self.c_src + self.S_global_store[resist_layer - 1][
            3] @ c_ln

        B = torch.vstack([torch.hstack([I, I]), torch.hstack([-self.V_g, self.V_g])])
        V_res = self.V_store[resist_layer]
        A = torch.vstack([torch.hstack([I, I]), torch.hstack([-V_res, V_res])])

        c_res = torch.linalg.solve(A, B) @ torch.hstack([c_lp, c_ln])
        c_ip, c_in = torch.split(c_res, 2)

        kz_layer = self.kz_store[resist_layer]
        Ez_up = - c_ip[0] * self.kx_inc / kz_layer
        Ez_down = - c_in[0] * self.kx_inc / kz_layer
        B_down = torch.tensor([c_ip[0], c_ip[1], Ez_up])
        B_up = torch.tensor([c_in[0], c_in[1], Ez_down])
        return torch.dot(self.P,B_down), torch.dot(self.P,B_up)
