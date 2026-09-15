import numpy as np
from typing import Tuple


class KinematicBicycleModel:
    def __init__(self, wheelbase: float = 0.50, max_steer: float = 0.60, max_accel: float = 3.0):
        self.L = float(wheelbase)
        self.max_steer = float(max_steer)
        self.max_accel = float(max_accel)

    def continuous_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        yaw = state[2]
        v = state[3]
        steer = np.clip(control[0], -self.max_steer, self.max_steer)
        accel = np.clip(control[1], -self.max_accel, self.max_accel)

        dx = v * np.cos(yaw)
        dy = v * np.sin(yaw)
        dyaw = (v / self.L) * np.tan(steer)
        dv = accel

        return np.array([dx, dy, dyaw, dv], dtype=np.float64)

    def step_rk4(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.continuous_dynamics(state, control)
        k2 = self.continuous_dynamics(state + 0.5 * dt * k1, control)
        k3 = self.continuous_dynamics(state + 0.5 * dt * k2, control)
        k4 = self.continuous_dynamics(state + dt * k3, control)

        next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        next_state[2] = (next_state[2] + np.pi) % (2 * np.pi) - np.pi
        return next_state

    def get_discrete_frenet_matrices(self, v: float, curvature: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        v_eff = max(0.5, float(v))

        A = np.array([[1.0, v_eff * dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        B = np.array([[0.5 * (v_eff**2) * (dt**2) / self.L, 0.0], [v_eff * dt / self.L, 0.0], [0.0, dt]], dtype=np.float64)

        d = np.array([-0.5 * (v_eff**2) * curvature * (dt**2), -v_eff * curvature * dt, 0.0], dtype=np.float64)

        return A, B, d
