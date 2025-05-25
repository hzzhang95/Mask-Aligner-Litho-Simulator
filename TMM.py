import torch


class TMM_1D(object):
    def __init__(self, wavelength, kx):
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.complex64

        self.gap_medium_index = torch.randn(1).to(self.torch_device) * 1e-6 + 1
        self.wavelength = torch.tensor(wavelength, dtype=self.dtype, device=self.torch_device)
        self.k0 = torch.pi * 2 / self.wavelength
        self.kx_inc = kx.to(self.torch_device) / self.k0
        self.kx_length = self.kx_inc.shape[0]
        self.v0 = torch.zeros_like(self.kx_inc)
        self.v1 = torch.ones_like(self.kx_inc)
        self.kz_inc = torch.sqrt(1 - self.kx_inc ** 2)
        # initialize the identity matrix
        self.I = torch.eye(2, dtype=self.dtype, device=self.torch_device).repeat(self.kx_length, 1, 1)

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
        self.k_inc = torch.vstack([self.kx_inc, self.v0, self.kz_inc])
        self.kz_ref = torch.sqrt(self.er_ref * self.mur_ref - self.kx_inc ** 2)
        self.initialize_gap_medium()
        # solve for the reflection region scattering matrix
        self.solve_ref_region_s_matrix()

    def initialize_gap_medium(self):
        # initialize the gap material matrices
        self.Q_g = torch.vstack(
            [torch.vstack([self.v0, self.gap_medium_index - self.kx_inc ** 2]),
             torch.vstack([-self.v1 * self.gap_medium_index, self.v0])]).permute(1, 0).view(-1, 2, 2)
        self.V_g = -1j * self.Q_g / self.kz_inc.unsqueeze(-1).unsqueeze(-1)

    def add_layer(self, er_layer=1.0, mur_layer=1.0, thickness=0.0):
        thickness = torch.tensor(thickness, dtype=torch.float32, device=self.torch_device)
        self.layer_count += 1
        self.er_layer.append(er_layer)
        self.mur_layer.append(mur_layer)
        self.solve_layer_s_matrix(thickness, er_layer, mur_layer)
        self.layer_store = torch.cat([self.layer_store, thickness.unsqueeze(-1)])

    def add_trs_layer(self, er_trs=1.0, mur_trs=1.0):
        self.er_trs = torch.as_tensor(er_trs, dtype=self.dtype, device=self.torch_device)
        self.mur_trs = torch.as_tensor(mur_trs, dtype=self.dtype, device=self.torch_device)
        self.solve_trs_region_s_matrix()

    def solve_layer_s_matrix(self, layer_thickness, er_layer, mur_layer):
        kz_i = torch.sqrt(mur_layer * er_layer - self.kx_inc ** 2)
        Q_i = 1 / mur_layer * torch.stack(
            [self.v0, er_layer * mur_layer - self.kx_inc ** 2, -mur_layer * er_layer * self.v1, self.v0]).permute(1,
                                                                                                                  0).view(
            -1, 2, 2)
        V_i = - 1j * Q_i / kz_i.unsqueeze(-1).unsqueeze(-1)
        X_i = torch.exp(1j * kz_i * self.k0 * layer_thickness)
        S11, S12 = self.get_scattering_matrix(V_i, X_i)
        S_layer = torch.stack([S11, S12, S12, S11], dim=0)
        self.S_global = self.calc_Redheffer_star(self.S_global, S_layer)
        self.V_store.append(V_i)
        self.kz_store.append(kz_i)
        self.S_global_store.append(self.S_global)

    def get_scattering_matrix(self, V_layer, X_layer):
        A_i = self.I + torch.linalg.solve(V_layer, self.V_g)
        B_i = self.I - torch.linalg.solve(V_layer, self.V_g)
        D = A_i - B_i @ torch.linalg.solve(A_i, B_i) * X_layer.unsqueeze(-1).unsqueeze(-1) ** 2
        S11 = torch.linalg.solve(D, (B_i * X_layer.unsqueeze(-1).unsqueeze(-1) ** 2 - B_i))
        S12 = torch.linalg.solve(D, X_layer.unsqueeze(-1).unsqueeze(-1) * (A_i - B_i @ torch.linalg.solve(A_i, B_i)))
        return S11, S12

    def calc_Redheffer_star(self, SA, SB):
        S11_A, S12_A, S21_A, S22_A = SA
        S11_B, S12_B, S21_B, S22_B = SB

        D = self.I - S11_B @ S22_A
        F = self.I - S22_A @ S11_B

        S11_AB = S11_A + S12_A @ torch.linalg.solve(D, S11_B) @ S21_A
        S12_AB = S12_A @ torch.linalg.solve(D, S12_B)
        S21_AB = S21_B @ torch.linalg.solve(F, S21_A)
        S22_AB = S22_B + S21_B @ torch.linalg.solve(D, S22_A) @ S12_B

        S_AB = torch.stack([S11_AB, S12_AB, S21_AB, S22_AB], dim=0)
        return S_AB

    def solve_ref_region_s_matrix(self):
        Q_ref = torch.stack([self.v0, self.er_ref * self.mur_ref - self.kx_inc ** 2,
                             -self.er_ref * self.mur_ref * self.v1, self.v0]).permute(1, 0).view(-1, 2, 2)
        V_ref = 1j * Q_ref / self.kz_ref.unsqueeze(-1).unsqueeze(-1)

        I_ref = torch.eye(2, dtype=self.dtype, device=self.torch_device).repeat(self.kx_length, 1, 1)
        O_ref = torch.zeros_like(I_ref)
        S11_ref = O_ref
        S12_ref = I_ref

        S_ref = torch.stack([S11_ref, S12_ref, S12_ref, S11_ref])
        self.S_global = self.calc_Redheffer_star(S_ref, self.S_global)
        self.V_store.append(V_ref)
        self.kz_store.append(self.kz_ref)
        self.S_global_store.append(self.S_global)

    def solve_trs_region_s_matrix(self):
        self.kz_trs = torch.sqrt(self.er_trs * self.mur_trs - self.kx_inc ** 2)

        Q_trs = torch.stack([self.v0, self.er_trs * self.mur_trs - self.kx_inc ** 2,
                             -self.er_trs * self.mur_trs * self.v1, self.v0]).permute(1, 0).view(-1, 2, 2)

        V_trs = -1j * Q_trs / self.kz_trs.unsqueeze(-1).unsqueeze(-1)

        A_trs = self.I + torch.linalg.solve(self.V_g, V_trs)
        B_trs = self.I - torch.linalg.solve(self.V_g, V_trs)

        S11_trs = torch.linalg.solve(A_trs, B_trs, left=False)
        S12_trs = 0.5 * (A_trs - B_trs @ torch.linalg.solve(A_trs, B_trs))
        S21_trs = 2 * torch.linalg.inv(A_trs)
        S22_trs = - torch.linalg.solve(A_trs, B_trs)

        S_trs = torch.stack([S11_trs, S12_trs, S21_trs, S22_trs])
        self.S_global = self.calc_Redheffer_star(self.S_global, S_trs)

        self.V_store.append(V_trs)
        self.kz_store.append(self.kz_trs)
        self.S_global_store.append(self.S_global)

    def solve_TMM(self, RT=False):
        self.TE = torch.tensor([0, 1, 0], dtype=self.dtype, device=self.torch_device).repeat(self.kx_length, 1)
        self.TM = torch.linalg.cross(self.k_inc.permute(1, 0), self.TE)

        self.TE_src = self.TE[:, :2].unsqueeze(-1)
        self.TE_ref = self.S_global[0] @ self.TE_src
        self.TE_trs = self.S_global[2] @ self.TE_src

        self.TM_src = self.TM[:, :2].unsqueeze(-1)
        self.TM_ref = self.S_global[0] @ self.TM_src
        self.TM_trs = self.S_global[2] @ self.TM_src

        self.TE_Ezref = - self.kx_inc.unsqueeze(-1).unsqueeze(-1) * self.TE_ref[:, 0, :].unsqueeze(
            -1) / self.kz_ref.unsqueeze(-1).unsqueeze(-1)
        self.TE_Eztrs = - self.kx_inc.unsqueeze(-1).unsqueeze(-1) * self.TE_trs[:, 0, :].unsqueeze(
            -1) / self.kz_trs.unsqueeze(-1).unsqueeze(-1)
        self.TM_Ezref = - self.kx_inc.unsqueeze(-1).unsqueeze(-1) * self.TM_ref[:, 0, :].unsqueeze(
            -1) / self.kz_ref.unsqueeze(-1).unsqueeze(-1)
        self.TM_Eztrs = - self.kx_inc.unsqueeze(-1).unsqueeze(-1) * self.TM_trs[:, 0, :].unsqueeze(
            -1) / self.kz_trs.unsqueeze(-1).unsqueeze(-1)

        if RT:
            R = 0.5 * (torch.linalg.norm(
                torch.hstack([self.TE_ref[:, 0].unsqueeze(-1), self.TE_ref[:, 1].unsqueeze(-1), self.TE_Ezref]),
                dim=1) ** 2 + torch.linalg.norm(
                torch.hstack([self.TM_ref[:, 0].unsqueeze(-1), self.TM_ref[:, 1].unsqueeze(-1), self.TM_Ezref]),
                dim=1) ** 2)
            T = 0.5 * (torch.linalg.norm(
                torch.hstack([self.TE_trs[:, 0].unsqueeze(-1), self.TE_trs[:, 1].unsqueeze(-1), self.TE_Eztrs]),
                dim=1) ** 2 + torch.linalg.norm(
                torch.hstack([self.TM_trs[:, 0].unsqueeze(-1), self.TM_trs[:, 1].unsqueeze(-1), self.TM_Eztrs]),
                dim=1) ** 2) * (torch.real(self.kz_trs / self.mur_trs) / torch.real(self.kz_inc)).unsqueeze(-1)
            print(R + T)
            return R, T

    def find_resist_layer_param(self, resist_layer=1):
        " For TE Mode"
        TE_ln = torch.linalg.solve(self.S_global_store[resist_layer - 1][1],
                                   self.TE_ref - self.S_global_store[resist_layer - 1][0] @ self.TE_src)
        TE_lp = self.S_global_store[resist_layer - 1][2] @ self.TE_src + self.S_global_store[resist_layer - 1][
            3] @ TE_ln

        " For TM Mode"
        TM_ln = torch.linalg.solve(self.S_global_store[resist_layer - 1][1],
                                   self.TM_ref - self.S_global_store[resist_layer - 1][0] @ self.TM_src)
        TM_lp = self.S_global_store[resist_layer - 1][2] @ self.TE_src + self.S_global_store[resist_layer - 1][
            3] @ TM_ln

        V_res = self.V_store[resist_layer]
        B = torch.cat([torch.hstack([self.I, self.I]), torch.hstack([-self.V_g, self.V_g])], dim=-1)
        A = torch.cat([torch.hstack([self.I, self.I]), torch.hstack([-V_res, V_res])], dim=-1)

        TE_res = torch.linalg.solve(A, B) @ torch.hstack([TE_lp, TE_ln])
        TM_res = torch.linalg.solve(A, B) @ torch.hstack([TM_lp, TM_ln])
        TE_ip, TE_in = torch.split(TE_res, 2, dim=1)
        TM_ip, TM_in = torch.split(TM_res, 2, dim=1)

        kz_layer = self.kz_store[resist_layer].unsqueeze(-1).unsqueeze(-1)
        kx_inc = self.kx_inc.unsqueeze(-1).unsqueeze(-1)
        TE_Ezup = - TE_ip[:, 0].unsqueeze(-1) * kx_inc / kz_layer
        TM_Ezup = - TM_ip[:, 0].unsqueeze(-1) * kx_inc / kz_layer
        TE_Ezdown = - TE_in[:, 0].unsqueeze(-1) * kx_inc / kz_layer
        TM_Ezdown = - TM_in[:, 0].unsqueeze(-1) * kx_inc / kz_layer
        TE_down = torch.stack([TE_ip[:, 0].unsqueeze(-1), TE_ip[:, 1].unsqueeze(-1), TE_Ezup], dim=1)
        TM_down = torch.stack([TM_ip[:, 0].unsqueeze(-1), TM_ip[:, 1].unsqueeze(-1), TM_Ezup], dim=1)
        TE_up = torch.stack([TE_in[:, 0].unsqueeze(-1), TE_in[:, 1].unsqueeze(-1), TE_Ezdown], dim=1)
        TM_up = torch.stack([TM_in[:, 0].unsqueeze(-1), TM_in[:, 1].unsqueeze(-1), TM_Ezdown], dim=1)

        " Average is taken between TE and TM mode "
        B_down = 0.5 * torch.einsum('ij,ij->i', self.TE, TE_down.squeeze()) + 0.5 * torch.einsum('ij,ij->i', self.TM,
                                                                                                 TM_down.squeeze())
        B_up = 0.5 * torch.einsum('ij,ij->i', self.TE, TE_up.squeeze()) + 0.5 * torch.einsum('ij,ij->i', self.TM,
                                                                                             TM_up.squeeze())
        return B_down, B_up


class TMM_2D(TMM_1D):
    def __init__(self, wavelength, kx, ky):
        super().__init__(wavelength, kx)
        self.ky_inc = ky.to(self.torch_device) / self.k0
        self.ky_length = self.ky_inc.shape[0]
        self.kx_inc, self.ky_inc = torch.meshgrid((self.kx_inc, self.ky_inc), indexing='ij')
        self.kz_inc = torch.sqrt(1 - self.kx_inc ** 2 - self.ky_inc ** 2)

        self.I = torch.eye(2, dtype=self.dtype, device=self.torch_device).repeat(self.kx_length, self.ky_length, 1, 1)
        self.O = torch.zeros_like(self.I)

    def initialize_gap_medium(self):
        # initialize the gap material matrices for 2D case
        self.Q_g = torch.stack(
            [torch.stack([self.kx_inc * self.ky_inc, self.gap_medium_index - self.kx_inc ** 2], dim=-1),
             torch.stack([self.ky_inc ** 2 - self.gap_medium_index, -self.kx_inc * self.ky_inc], dim=-1)], dim=-1)
        self.V_g = -1j * self.Q_g / self.kz_inc.unsqueeze(-1).unsqueeze(-1)

    def get_scattering_matrix(self, V_layer, X_layer):
        A_i = self.I + torch.linalg.solve(V_layer, self.V_g)
        B_i = self.I - torch.linalg.solve(V_layer, self.V_g)
        D = A_i - B_i @ torch.linalg.solve(A_i, B_i) * X_layer.unsqueeze(-1).unsqueeze(-1) ** 2
        S11 = torch.linalg.solve(D, (B_i * X_layer.unsqueeze(-1).unsqueeze(-1) ** 2 - B_i))
        S12 = torch.linalg.solve(D, X_layer.unsqueeze(-1).unsqueeze(-1) * (A_i - B_i @ torch.linalg.solve(A_i, B_i)))
        return S11, S12

    def add_ref_layer(self, er_ref=1.0, mur_ref=1.0):
        self.er_ref = torch.tensor(er_ref, dtype=self.dtype, device=self.torch_device)
        self.mur_ref = torch.tensor(mur_ref, dtype=self.dtype, device=self.torch_device)
        self.n_inc = torch.sqrt(self.er_ref * self.mur_ref)

        # calculating the k vector of the incoming wave
        self.k_inc = torch.stack([self.kx_inc, self.ky_inc, self.kz_inc], dim=-1)
        self.kz_ref = torch.sqrt(self.er_ref * self.mur_ref - self.kx_inc ** 2 - self.ky_inc ** 2)
        self.initialize_gap_medium()
        # solve for the reflection region scattering matrix
        self.solve_ref_region_s_matrix()

    def solve_ref_region_s_matrix(self):
        Q_ref = 1 / self.mur_ref * torch.stack(
            [torch.stack([self.kx_inc * self.ky_inc, self.er_ref * self.mur_ref - self.kx_inc ** 2], dim=-1),
             torch.stack([self.ky_inc ** 2 - self.er_ref * self.mur_ref, -self.kx_inc * self.ky_inc], dim=-1)], dim=-1)
        V_ref = 1j * Q_ref / self.kz_ref.unsqueeze(-1).unsqueeze(-1)

        S11_ref = self.O
        S12_ref = self.I
        S_ref = torch.stack([S11_ref, S12_ref, S12_ref, S11_ref])
        self.S_global = self.calc_Redheffer_star(S_ref, self.S_global)

        self.V_store.append(V_ref)
        self.kz_store.append(self.kz_ref)
        self.S_global_store.append(self.S_global)

    def solve_layer_s_matrix(self, layer_thickness, er_layer, mur_layer):
        kz_i = torch.sqrt(mur_layer * er_layer - self.kx_inc ** 2 - self.ky_inc ** 2)

        Q_i = 1 / mur_layer * torch.stack(
            [self.kx_inc * self.ky_inc, er_layer * mur_layer - self.kx_inc ** 2,
             self.ky_inc ** 2 - mur_layer * er_layer, - self.kx_inc * self.ky_inc]).permute(1, 2, 0).view(
            self.kx_length, self.ky_length, 2, 2)

        V_i = - 1j * Q_i / kz_i.unsqueeze(-1).unsqueeze(-1)
        X_i = torch.exp(1j * kz_i * self.k0 * layer_thickness)

        S11, S12 = self.get_scattering_matrix(V_i, X_i)

        S_layer = torch.stack([S11, S12, S12, S11])
        self.S_global = self.calc_Redheffer_star(self.S_global, S_layer)

        self.V_store.append(V_i)
        self.kz_store.append(kz_i)
        self.S_global_store.append(self.S_global)

    def solve_trs_region_s_matrix(self):
        self.kz_trs = torch.sqrt(self.er_trs * self.mur_trs - self.kx_inc ** 2 - self.ky_inc ** 2)

        Q_trs = 1 / self.mur_trs * torch.stack(
            [self.kx_inc * self.ky_inc, self.er_trs * self.mur_trs - self.kx_inc ** 2,
             self.ky_inc ** 2 - self.er_trs * self.mur_trs, - self.kx_inc * self.ky_inc]).permute(1, 2, 0).view(
            self.kx_length, self.ky_length, 2, 2)

        V_trs = -1j * Q_trs / self.kz_trs.unsqueeze(-1).unsqueeze(-1)

        A_trs = self.I + torch.linalg.solve(self.V_g, V_trs)
        B_trs = self.I - torch.linalg.solve(self.V_g, V_trs)

        S11_trs = torch.linalg.solve(A_trs, B_trs, left=False)
        S12_trs = 0.5 * (A_trs - B_trs @ torch.linalg.solve(A_trs, B_trs))
        S21_trs = 2 * torch.linalg.inv(A_trs)
        S22_trs = - torch.linalg.solve(A_trs, B_trs)

        S_trs = torch.stack([S11_trs, S12_trs, S21_trs, S22_trs])
        self.S_global = self.calc_Redheffer_star(self.S_global, S_trs)

        self.V_store.append(V_trs)
        self.kz_store.append(self.kz_trs)
        self.S_global_store.append(self.S_global)

    def calc_Redheffer_star(self, SA, SB):
        S11_A, S12_A, S21_A, S22_A = SA
        S11_B, S12_B, S21_B, S22_B = SB

        D = self.I - S11_B @ S22_A
        F = self.I - S22_A @ S11_B

        S11_AB = S11_A + S12_A @ torch.linalg.solve(D, S11_B) @ S21_A
        S12_AB = S12_A @ torch.linalg.solve(D, S12_B)
        S21_AB = S21_B @ torch.linalg.solve(F, S21_A)
        S22_AB = S22_B + S21_B @ torch.linalg.solve(D, S22_A) @ S12_B

        S_AB = torch.stack([S11_AB, S12_AB, S21_AB, S22_AB])
        return S_AB

    def solve_TMM(self, RT=False):
        self.TE = torch.tensor([0, 1, 0], dtype=self.dtype, device=self.torch_device).repeat(self.kx_length,
                                                                                             self.ky_length, 1)
        self.TM = torch.linalg.cross(self.k_inc, self.TE)

        self.TE_src = self.TE[:, :, :2].unsqueeze(-1)
        self.TE_ref = self.S_global[0] @ self.TE_src
        self.TE_trs = self.S_global[2] @ self.TE_src

        self.TM_src = self.TM[:, :, :2].unsqueeze(-1)
        self.TM_ref = self.S_global[0] @ self.TM_src
        self.TM_trs = self.S_global[2] @ self.TM_src

        self.TE_Ezref = - self.kx_inc.unsqueeze(-1).unsqueeze(-1) * self.TE_ref[:, :, 0, :].unsqueeze(
            -1) / self.kz_ref.unsqueeze(-1).unsqueeze(-1)
        self.TE_Eztrs = - self.kx_inc.unsqueeze(-1).unsqueeze(-1) * self.TE_trs[:, :, 0, :].unsqueeze(
            -1) / self.kz_trs.unsqueeze(-1).unsqueeze(-1)
        self.TM_Ezref = - self.kx_inc.unsqueeze(-1).unsqueeze(-1) * self.TM_ref[:, :, 0, :].unsqueeze(
            -1) / self.kz_ref.unsqueeze(-1).unsqueeze(-1)
        self.TM_Eztrs = - self.kx_inc.unsqueeze(-1).unsqueeze(-1) * self.TM_trs[:, :, 0, :].unsqueeze(
            -1) / self.kz_trs.unsqueeze(-1).unsqueeze(-1)

        if RT:
            R = 0.5 * (torch.linalg.norm(torch.stack(
                [self.TE_ref[:, :, 0, :].unsqueeze(-1), self.TE_ref[:, :, 1, :].unsqueeze(-1), self.TE_Ezref],
                dim=2).squeeze(), dim=2) ** 2 + torch.linalg.norm(torch.stack(
                [self.TM_ref[:, :, 0, :].unsqueeze(-1), self.TM_ref[:, :, 1, :].unsqueeze(-1), self.TM_Ezref],
                dim=2).squeeze(), dim=2) ** 2)
            T = 0.5 * (torch.linalg.norm(torch.stack(
                [self.TE_trs[:, :, 0, :].unsqueeze(-1), self.TE_trs[:, :, 1, :].unsqueeze(-1), self.TE_Eztrs],
                dim=2).squeeze(), dim=2) ** 2 + torch.linalg.norm(torch.stack(
                [self.TM_trs[:, :, 0, :].unsqueeze(-1), self.TM_trs[:, :, 1, :].unsqueeze(-1), self.TM_Eztrs],
                dim=2).squeeze(), dim=2) ** 2)
                 # * (torch.real(self.kz_trs / self.mur_trs) / torch.real(self.kz_inc)).unsqueeze(-1))
            return print(R + T)

    def find_resist_layer_param(self, resist_layer=1):
        I = torch.eye(2, dtype=self.dtype, device=self.torch_device).repeat(self.kx_length, 1, 1)

        " For TE Mode"
        TE_ln = torch.linalg.solve(self.S_global_store[resist_layer - 1][1],
                                   self.TE_ref - self.S_global_store[resist_layer - 1][0] @ self.TE_src)
        TE_lp = self.S_global_store[resist_layer - 1][2] @ self.TE_src + self.S_global_store[resist_layer - 1][
            3] @ TE_ln

        " For TM Mode"
        TM_ln = torch.linalg.solve(self.S_global_store[resist_layer - 1][1],
                                   self.TM_ref - self.S_global_store[resist_layer - 1][0] @ self.TM_src)
        TM_lp = self.S_global_store[resist_layer - 1][2] @ self.TE_src + self.S_global_store[resist_layer - 1][
            3] @ TM_ln

        V_res = self.V_store[resist_layer]
        B = torch.cat([torch.hstack([I, I]), torch.hstack([-self.V_g, self.V_g])], dim=-1)
        A = torch.cat([torch.hstack([I, I]), torch.hstack([-V_res, V_res])], dim=-1)

        TE_res = torch.linalg.solve(A, B) @ torch.hstack([TE_lp, TE_ln])
        TM_res = torch.linalg.solve(A, B) @ torch.hstack([TM_lp, TM_ln])
        TE_ip, TE_in = torch.split(TE_res, 2, dim=1)
        TM_ip, TM_in = torch.split(TM_res, 2, dim=1)

        kz_layer = self.kz_store[resist_layer].unsqueeze(-1).unsqueeze(-1)
        kx_inc = self.kx_inc.unsqueeze(-1).unsqueeze(-1)
        TE_Ezup = - TE_ip[:, 0].unsqueeze(-1) * kx_inc / kz_layer
        TM_Ezup = - TM_ip[:, 0].unsqueeze(-1) * kx_inc / kz_layer
        TE_Ezdown = - TE_in[:, 0].unsqueeze(-1) * kx_inc / kz_layer
        TM_Ezdown = - TM_in[:, 0].unsqueeze(-1) * kx_inc / kz_layer
        TE_down = torch.stack([TE_ip[:, 0].unsqueeze(-1), TE_ip[:, 1].unsqueeze(-1), TE_Ezup], dim=1)
        TM_down = torch.stack([TM_ip[:, 0].unsqueeze(-1), TM_ip[:, 1].unsqueeze(-1), TM_Ezup], dim=1)
        TE_up = torch.stack([TE_in[:, 0].unsqueeze(-1), TE_in[:, 1].unsqueeze(-1), TE_Ezdown], dim=1)
        TM_up = torch.stack([TM_in[:, 0].unsqueeze(-1), TM_in[:, 1].unsqueeze(-1), TM_Ezdown], dim=1)

        " Average is taken between TE and TM mode "
        B_down = 0.5 * torch.einsum('ij,ij->i', self.TE, TE_down.squeeze()) + 0.5 * torch.einsum('ij,ij->i', self.TM,
                                                                                                 TM_down.squeeze())
        B_up = 0.5 * torch.einsum('ij,ij->i', self.TE, TE_up.squeeze()) + 0.5 * torch.einsum('ij,ij->i', self.TM,
                                                                                             TM_up.squeeze())
        return B_down, B_up
