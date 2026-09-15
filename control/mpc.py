import numpy as np
import scipy.sparse as sp
import osqp
from typing import Tuple, List, Optional
from .kinematics import KinematicBicycleModel


class AckermannMPC:
    def __init__(
        self,
        wheelbase: float = 0.50,
        horizon: int = 20,
        dt: float = 0.05,
        v_ref: float = 4.0,
        road_width: float = 3.5,
        safety_margin: float = 0.35,
        max_steer: float = 0.55,
        max_steer_rate: float = 0.8,
        max_accel: float = 2.5,
        max_decel: float = 3.5,
        max_speed: float = 7.0,
    ):
        self.model = KinematicBicycleModel(wheelbase=wheelbase, max_steer=max_steer, max_accel=max_accel)
        self.N = int(horizon)
        self.dt = float(dt)
        self.v_ref = float(v_ref)
        self.road_width = float(road_width)
        self.safety_margin = float(safety_margin)
        self.max_ey = (self.road_width / 2.0) - self.safety_margin

        self.max_steer = float(max_steer)
        self.max_steer_rate = float(max_steer_rate)
        self.max_accel = float(max_accel)
        self.max_decel = float(max_decel)
        self.max_speed = float(max_speed)

        self.w_ey = 12.0
        self.w_epsi = 8.0
        self.w_v = 4.0
        self.w_steer = 3.0
        self.w_accel = 1.0
        self.w_dsteer = 15.0
        self.w_daccel = 2.0

        self.solver = None
        self.prev_steer = 0.0
        self.prev_accel = 0.0

        self.nz = 3
        self.nu = 2
        self.n_step = self.nz + self.nu
        self.n_vars = self.N * self.n_step

    def solve(
        self,
        current_ey: float,
        current_epsi: float,
        current_v: float,
        future_curvatures: List[float],
        target_v: Optional[float] = None
    ) -> Tuple[float, float, np.ndarray]:
        v_ref = float(target_v) if target_v is not None else self.v_ref
        z0 = np.array([current_ey, current_epsi, current_v], dtype=np.float64)

        curvs = list(future_curvatures)
        while len(curvs) < self.N:
            curvs.append(curvs[-1] if len(curvs) > 0 else 0.0)

        P_diag = np.zeros(self.n_vars, dtype=np.float64)
        q_vec = np.zeros(self.n_vars, dtype=np.float64)

        P_triplets_r = []
        P_triplets_c = []
        P_triplets_v = []

        for k in range(self.N):
            idx_z = k * self.n_step
            idx_u = idx_z + self.nz

            P_diag[idx_z + 0] += 2.0 * self.w_ey
            P_diag[idx_z + 1] += 2.0 * self.w_epsi
            P_diag[idx_z + 2] += 2.0 * self.w_v
            q_vec[idx_z + 2] += -2.0 * self.w_v * v_ref

            P_diag[idx_u + 0] += 2.0 * self.w_steer
            P_diag[idx_u + 1] += 2.0 * self.w_accel

            if k == 0:
                P_diag[idx_u + 0] += 2.0 * self.w_dsteer
                q_vec[idx_u + 0] += -2.0 * self.w_dsteer * self.prev_steer
                P_diag[idx_u + 1] += 2.0 * self.w_daccel
                q_vec[idx_u + 1] += -2.0 * self.w_daccel * self.prev_accel
            else:
                idx_u_prev = (k - 1) * self.n_step + self.nz
                P_diag[idx_u + 0] += 2.0 * self.w_dsteer
                P_diag[idx_u_prev + 0] += 2.0 * self.w_dsteer
                P_triplets_r.append(idx_u + 0)
                P_triplets_c.append(idx_u_prev + 0)
                P_triplets_v.append(-2.0 * self.w_dsteer)

                P_diag[idx_u + 1] += 2.0 * self.w_daccel
                P_diag[idx_u_prev + 1] += 2.0 * self.w_daccel
                P_triplets_r.append(idx_u + 1)
                P_triplets_c.append(idx_u_prev + 1)
                P_triplets_v.append(-2.0 * self.w_daccel)

        row_indices = list(range(self.n_vars)) + P_triplets_r + P_triplets_c
        col_indices = list(range(self.n_vars)) + P_triplets_c + P_triplets_r
        data_values = list(P_diag) + P_triplets_v + P_triplets_v
        P_sparse = sp.csc_matrix((data_values, (row_indices, col_indices)), shape=(self.n_vars, self.n_vars))

        A_rows = []
        A_cols = []
        A_vals = []
        l_bounds = []
        u_bounds = []

        con_idx = 0

        v_traj = max(0.5, current_v)
        for k in range(self.N):
            kappa_k = curvs[k]
            A_k, B_k, d_k = self.model.get_discrete_frenet_matrices(v_traj, kappa_k, self.dt)

            idx_z_k = k * self.n_step
            idx_u_k = idx_z_k + self.nz

            if k == 0:
                rhs = A_k @ z0 + d_k
                for r in range(self.nz):
                    A_rows.append(con_idx + r)
                    A_cols.append(idx_z_k + r)
                    A_vals.append(1.0)
                    for c in range(self.nu):
                        if B_k[r, c] != 0.0:
                            A_rows.append(con_idx + r)
                            A_cols.append(idx_u_k + c)
                            A_vals.append(-B_k[r, c])
                    l_bounds.append(rhs[r])
                    u_bounds.append(rhs[r])
                con_idx += self.nz
            else:
                idx_z_prev = (k - 1) * self.n_step
                for r in range(self.nz):
                    A_rows.append(con_idx + r)
                    A_cols.append(idx_z_k + r)
                    A_vals.append(1.0)
                    for c in range(self.nz):
                        if A_k[r, c] != 0.0:
                            A_rows.append(con_idx + r)
                            A_cols.append(idx_z_prev + c)
                            A_vals.append(-A_k[r, c])
                    for c in range(self.nu):
                        if B_k[r, c] != 0.0:
                            A_rows.append(con_idx + r)
                            A_cols.append(idx_u_k + c)
                            A_vals.append(-B_k[r, c])
                    l_bounds.append(d_k[r])
                    u_bounds.append(d_k[r])
                con_idx += self.nz

        for k in range(self.N):
            idx_z = k * self.n_step
            idx_u = idx_z + self.nz

            A_rows.append(con_idx)
            A_cols.append(idx_z + 0)
            A_vals.append(1.0)
            l_bounds.append(-self.max_ey)
            u_bounds.append(self.max_ey)
            con_idx += 1

            A_rows.append(con_idx)
            A_cols.append(idx_z + 1)
            A_vals.append(1.0)
            l_bounds.append(-1.5)
            u_bounds.append(1.5)
            con_idx += 1

            A_rows.append(con_idx)
            A_cols.append(idx_z + 2)
            A_vals.append(1.0)
            l_bounds.append(0.0)
            u_bounds.append(self.max_speed)
            con_idx += 1

            A_rows.append(con_idx)
            A_cols.append(idx_u + 0)
            A_vals.append(1.0)
            l_bounds.append(-self.max_steer)
            u_bounds.append(self.max_steer)
            con_idx += 1

            A_rows.append(con_idx)
            A_cols.append(idx_u + 1)
            A_vals.append(1.0)
            l_bounds.append(-self.max_decel)
            u_bounds.append(self.max_accel)
            con_idx += 1

        A_sparse = sp.csc_matrix((A_vals, (A_rows, A_cols)), shape=(con_idx, self.n_vars))
        l_arr = np.array(l_bounds, dtype=np.float64)
        u_arr = np.array(u_bounds, dtype=np.float64)

        solver = osqp.OSQP()
        solver.setup(
            P=P_sparse,
            q=q_vec,
            A=A_sparse,
            l=l_arr,
            u=u_arr,
            verbose=False,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=400,
            warm_starting=True
        )
        res = solver.solve()

        if res.info.status not in ["solved", "solved inaccurate"]:
            steer_cmd = float(np.clip(-1.2 * current_ey - 1.5 * current_epsi, -self.max_steer, self.max_steer))
            accel_cmd = float(np.clip(1.5 * (v_ref - current_v), -self.max_decel, self.max_accel))
            pred_states = np.tile(z0, (self.N, 1))
        else:
            sol = res.x
            steer_cmd = float(np.clip(sol[self.nz + 0], -self.max_steer, self.max_steer))
            accel_cmd = float(np.clip(sol[self.nz + 1], -self.max_decel, self.max_accel))

            pred_states = np.zeros((self.N, self.nz), dtype=np.float64)
            for k in range(self.N):
                pred_states[k] = sol[k * self.n_step : k * self.n_step + self.nz]

        self.prev_steer = steer_cmd
        self.prev_accel = accel_cmd

        return steer_cmd, accel_cmd, pred_states
