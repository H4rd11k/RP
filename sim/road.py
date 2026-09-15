import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple, List, Dict


def wrap_angle(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


class RoadTrack:
    def __init__(self, waypoints: np.ndarray = None, width: float = 3.5, ds: float = 0.1):
        if waypoints is None:
            waypoints = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 1.0], [35.0, 5.0], [50.0, 12.0], [65.0, 14.0], [80.0, 8.0], [95.0, 0.0], [110.0, -6.0], [125.0, -7.0], [140.0, -3.0], [155.0, 0.0], [175.0, 0.0]])

        self.waypoints = np.array(waypoints, dtype=np.float64)
        self.width = float(width)
        self.half_width = self.width / 2.0
        self.ds = ds

        dists = np.sqrt(np.sum(np.diff(self.waypoints, axis=0) ** 2, axis=1))
        self.s_waypoints = np.concatenate(([0.0], np.cumsum(dists)))
        self.total_length = self.s_waypoints[-1]

        self.spline_x = CubicSpline(self.s_waypoints, self.waypoints[:, 0], bc_type='natural')
        self.spline_y = CubicSpline(self.s_waypoints, self.waypoints[:, 1], bc_type='natural')

        self.spline_dx = self.spline_x.derivative(1)
        self.spline_dy = self.spline_y.derivative(1) 
        self.spline_ddx = self.spline_x.derivative(2) 
        self.spline_ddy = self.spline_y.derivative(2)

        self.s_dense = np.arange(0.0, self.total_length, self.ds)
        self.x_dense = self.spline_x(self.s_dense)
        self.y_dense = self.spline_y(self.s_dense)

        dx = self.spline_dx(self.s_dense)
        dy = self.spline_dy(self.s_dense)
        self.yaw_dense = np.arctan2(dy, dx)

        nx = -np.sin(self.yaw_dense)
        ny = np.cos(self.yaw_dense)
        self.left_bound_x = self.x_dense + self.half_width * nx
        self.left_bound_y = self.y_dense + self.half_width * ny
        self.right_bound_x = self.x_dense - self.half_width * nx
        self.right_bound_y = self.y_dense - self.half_width * ny

    def get_state_at_s(self, s: float) -> Tuple[float, float, float, float]:
        s_clamped = np.clip(s, 0.0, self.total_length)
        x = float(self.spline_x(s_clamped))
        y = float(self.spline_y(s_clamped))
        dx = float(self.spline_dx(s_clamped))
        dy = float(self.spline_dy(s_clamped))
        ddx = float(self.spline_ddx(s_clamped))
        ddy = float(self.spline_ddy(s_clamped))

        yaw = np.arctan2(dy, dx)
        speed_sq = dx * dx + dy * dy
        if speed_sq < 1e-6:
            curvature = 0.0
        else:
            curvature = (dx * ddy - dy * ddx) / (speed_sq ** 1.5)

        return x, y, yaw, curvature

    def find_closest_s(self, x: float, y: float, s_hint: float = None) -> float:
        if s_hint is not None and 0.0 <= s_hint <= self.total_length:
            window_size = 10.0
            idx_center = int(s_hint / self.ds)
            w_idx = int(window_size / self.ds)
            idx_start = max(0, idx_center - w_idx)
            idx_end = min(len(self.s_dense), idx_center + w_idx + 1)
        else:
            idx_start = 0
            idx_end = len(self.s_dense)

        sub_x = self.x_dense[idx_start:idx_end]
        sub_y = self.y_dense[idx_start:idx_end]
        sub_s = self.s_dense[idx_start:idx_end]

        dists_sq = (sub_x - x) ** 2 + (sub_y - y) ** 2
        min_idx = np.argmin(dists_sq)
        s_coarse = sub_s[min_idx]

        delta_fine = np.linspace(-self.ds, self.ds, 21)
        s_candidates = np.clip(s_coarse + delta_fine, 0.0, self.total_length)
        cand_x = self.spline_x(s_candidates)
        cand_y = self.spline_y(s_candidates)
        cand_dists = (cand_x - x) ** 2 + (cand_y - y) ** 2
        best_s = float(s_candidates[np.argmin(cand_dists)])

        return best_s

    def compute_frenet_errors(self, x: float, y: float, yaw: float, s_hint: float = None) -> Tuple[float, float, float, float]:
        s_proj = self.find_closest_s(x, y, s_hint)
        rx, ry, ryaw, curvature = self.get_state_at_s(s_proj)

        dx = x - rx
        dy = y - ry

        e_y = -dx * np.sin(ryaw) + dy * np.cos(ryaw)
        e_psi = wrap_angle(yaw - ryaw)

        return s_proj, e_y, e_psi, curvature

    def is_in_corridor(self, e_y: float = None, ey: float = None, safety_margin: float = 0.2) -> bool:
        val = e_y if e_y is not None else ey
        if val is None:
            raise ValueError("Must provide either e_y or ey")
        return abs(val) <= (self.half_width - safety_margin)
