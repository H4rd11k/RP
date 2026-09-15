import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from comm.protocol import ControllerClient, ControlCommand, VehicleState
from sim.road import RoadTrack
from control.mpc import AckermannMPC


class AckermannControllerNode:
    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        target_speed: float = 4.0,
        horizon: int = 20,
        dt: float = 0.05,
        road_width: float = 3.5,
    ):
        self.endpoint = endpoint
        self.target_speed = float(target_speed)
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.road_width = float(road_width)

        self.track = RoadTrack(width=self.road_width)
        self.mpc = AckermannMPC(
            horizon=self.horizon,
            dt=self.dt,
            v_ref=self.target_speed,
            road_width=self.road_width,
            safety_margin=0.35,
        )
        self.client = None
        self.log_records = []

    def run(self):
        self.client = ControllerClient(endpoint=self.endpoint)

        initial_cmd = ControlCommand(steer=0.0, speed=0.0, accel=0.0)
        state = None
        for attempt in range(15):
            try:
                state = self.client.exchange(initial_cmd)
                break
            except Exception:
                time.sleep(0.5)

        if state is None:
            raise ConnectionError(f"Could not connect to simulator at {self.endpoint}")

        s_hint = 0.0
        step_idx = 0
        current_target_v = self.target_speed

        try:
            while not state.done:
                s_proj, e_y, e_psi, curvature = self.track.compute_frenet_errors(
                    state.x, state.y, state.yaw, s_hint=s_hint
                )
                s_hint = s_proj

                curvatures = []
                for k in range(self.horizon):
                    s_preview = s_proj + k * max(1.0, state.v) * self.dt
                    _, _, _, curv_k = self.track.get_state_at_s(s_preview)
                    curvatures.append(curv_k)

                steer_opt, accel_opt, pred_states = self.mpc.solve(
                    current_ey=e_y,
                    current_epsi=e_psi,
                    current_v=state.v,
                    future_curvatures=curvatures,
                    target_v=self.target_speed,
                )

                current_target_v = np.clip(state.v + accel_opt * self.dt, 0.0, self.target_speed)

                self.log_records.append({
                    "time": state.time,
                    "x": state.x,
                    "y": state.y,
                    "yaw": state.yaw,
                    "v": state.v,
                    "omega": state.omega,
                    "s": s_proj,
                    "e_y": e_y,
                    "e_psi": e_psi,
                    "curvature": curvature,
                    "steer_cmd": steer_opt,
                    "accel_cmd": accel_opt,
                    "target_v": current_target_v,
                })

                cmd = ControlCommand(
                    steer=steer_opt,
                    speed=current_target_v,
                    accel=accel_opt,
                    stop=False,
                )
                state = self.client.exchange(cmd)
                step_idx += 1

        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.client.exchange(ControlCommand(stop=True))
            except Exception:
                pass
            self.client.close()

        self._print_summary()

    def _print_summary(self):
        if not self.log_records:
            return

        e_y_arr = np.array([r["e_y"] for r in self.log_records])
        e_psi_arr = np.array([r["e_psi"] for r in self.log_records])
        v_arr = np.array([r["v"] for r in self.log_records])

        max_ey = np.max(np.abs(e_y_arr))
        rmse_ey = np.sqrt(np.mean(e_y_arr ** 2))
        max_epsi_deg = np.degrees(np.max(np.abs(e_psi_arr)))
        rmse_epsi_deg = np.degrees(np.sqrt(np.mean(e_psi_arr ** 2)))
        mean_v = np.mean(v_arr)
        corridor_violations = np.sum(np.abs(e_y_arr) > (self.road_width / 2.0))

        print("\n" + "=" * 55)
        print("          TRACKING PERFORMANCE SUMMARY")
        print("=" * 55)
        print(f" Total Duration:          {self.log_records[-1]['time']:.2f} s")
        print(f" Total Distance:          {self.log_records[-1]['s']:.2f} m")
        print(f" Average Speed:           {mean_v:.2f} m/s (target: {self.target_speed:.2f} m/s)")
        print(f" Maximum Lateral Error:   {max_ey:.3f} m (corridor half-width: {self.road_width/2:.2f} m)")
        print(f" RMSE Lateral Error:      {rmse_ey:.3f} m")
        print(f" Maximum Heading Error:   {max_epsi_deg:.2f} deg")
        print(f" RMSE Heading Error:      {rmse_epsi_deg:.2f} deg")
        print(f" Corridor Violations:     {corridor_violations} ({'PASSED' if corridor_violations == 0 else 'FAILED'})")
        print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="2D Ackermann MPC Controller Process")
    parser.add_argument("--endpoint", type=str, default="tcp://127.0.0.1:5555", help="ZeroMQ endpoint")
    parser.add_argument("--speed", type=float, default=4.0, help="Reference speed in m/s")
    parser.add_argument("--horizon", type=int, default=20, help="MPC preview horizon")
    parser.add_argument("--dt", type=float, default=0.05, help="Sampling time in seconds")
    parser.add_argument("--width", type=float, default=3.5, help="Road width in meters")
    args = parser.parse_args()

    controller = AckermannControllerNode(
        endpoint=args.endpoint,
        target_speed=args.speed,
        horizon=args.horizon,
        dt=args.dt,
        road_width=args.width,
    )
    controller.run()


if __name__ == "__main__":
    main()
