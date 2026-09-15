import os
import sys
import time
import argparse
import numpy as np
import mujoco
try:
    import mujoco.viewer
    HAS_VIEWER = True
except ImportError:
    HAS_VIEWER = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from comm.protocol import SimulatorServer, VehicleState, ControlCommand
from sim.road import RoadTrack


class AckermannSimulator:
    def __init__(
        self,
        xml_path: str = None,
        endpoint: str = "tcp://127.0.0.1:5555",
        road_width: float = 3.5,
        gui: bool = True,
        max_time: float = 60.0,
    ):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "ackerman_car.xml")
        self.xml_path = xml_path
        self.endpoint = endpoint
        self.gui = gui
        self.max_time = max_time

        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"MuJoCo XML model not found at {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.steer_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "steer_actuator")

        self.wheel_radius = 0.08
        self.track = RoadTrack(width=road_width)
        self.v = 0.0
        self.last_s = 0.0

        self.server = SimulatorServer(endpoint=self.endpoint)
        self._prepare_road_visuals()
        self.mpc_pred_points = None

    def _prepare_road_visuals(self):
        step = max(1, int(1.5 / self.track.ds))
        self.centerline_vis_x = self.track.x_dense[::step]
        self.centerline_vis_y = self.track.y_dense[::step]

        self.left_bound_vis_x = self.track.left_bound_x[::step]
        self.left_bound_vis_y = self.track.left_bound_y[::step]

        self.right_bound_vis_x = self.track.right_bound_x[::step]
        self.right_bound_vis_y = self.track.right_bound_y[::step]

    def get_current_state(self) -> VehicleState:
        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        yaw = float(self.data.qpos[2])
        omega = float(self.data.qvel[2])
        sim_time = float(self.data.time)

        s_curr = self.track.find_closest_s(x, y, s_hint=self.last_s)
        self.last_s = s_curr
        done = bool((sim_time >= self.max_time) or (s_curr >= self.track.total_length - 3.0))

        return VehicleState(
            x=x,
            y=y,
            yaw=yaw,
            v=float(self.v),
            omega=omega,
            time=sim_time,
            done=done,
        )

    def step_vehicle(self, steer: float, target_speed: float, accel: float, dt_sub: float):
        L = 0.50

        self.v = float(np.clip(self.v + accel * dt_sub, 0.0, 10.0))
        steer_clamped = float(np.clip(steer, -0.6, 0.6))
        yaw = float(self.data.qpos[2])

        dx = self.v * np.cos(yaw)
        dy = self.v * np.sin(yaw)
        dyaw = (self.v / L) * np.tan(steer_clamped)

        self.data.qpos[0] += dx * dt_sub
        self.data.qpos[1] += dy * dt_sub
        self.data.qpos[2] = float((yaw + dyaw * dt_sub + np.pi) % (2 * np.pi) - np.pi)
        self.data.qpos[5] = steer_clamped

        if self.steer_act_id >= 0:
            self.data.ctrl[self.steer_act_id] = steer_clamped

        wheel_spin = (self.v / self.wheel_radius) * dt_sub
        self.data.qpos[3] = (self.data.qpos[3] + wheel_spin) % (2 * np.pi)
        self.data.qpos[4] = (self.data.qpos[4] + wheel_spin) % (2 * np.pi)
        self.data.qpos[6] = (self.data.qpos[6] + wheel_spin) % (2 * np.pi)
        self.data.qpos[7] = (self.data.qpos[7] + wheel_spin) % (2 * np.pi)

        self.data.qvel[0] = dx
        self.data.qvel[1] = dy
        self.data.qvel[2] = dyaw
        self.data.time += dt_sub

        mujoco.mj_forward(self.model, self.data)

    def update_viewer_scene(self, viewer):
        if not hasattr(viewer, "user_scn") or viewer.user_scn is None:
            return

        with viewer.lock():
            scn = viewer.user_scn
            max_geom = scn.maxgeom
            scn.ngeom = 0

            mat_identity = np.eye(3).flatten()
            for px, py in zip(self.centerline_vis_x, self.centerline_vis_y):
                if scn.ngeom >= max_geom - 5:
                    break
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.06, 0.0, 0.0]),
                    np.array([px, py, 0.015]),
                    mat_identity,
                    np.array([1.0, 0.85, 0.1, 0.8]),
                )
                scn.ngeom += 1

            for px, py in zip(self.left_bound_vis_x, self.left_bound_vis_y):
                if scn.ngeom >= max_geom - 5:
                    break
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.05, 0.0, 0.0]),
                    np.array([px, py, 0.015]),
                    mat_identity,
                    np.array([0.9, 0.2, 0.2, 0.7]),
                )
                scn.ngeom += 1

            for px, py in zip(self.right_bound_vis_x, self.right_bound_vis_y):
                if scn.ngeom >= max_geom - 5:
                    break
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.05, 0.0, 0.0]),
                    np.array([px, py, 0.015]),
                    mat_identity,
                    np.array([0.9, 0.2, 0.2, 0.7]),
                )
                scn.ngeom += 1

            if self.mpc_pred_points is not None:
                for px, py in self.mpc_pred_points:
                    if scn.ngeom >= max_geom - 1:
                        break
                    mujoco.mjv_initGeom(
                        scn.geoms[scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.09, 0.0, 0.0]),
                        np.array([px, py, 0.05]),
                        mat_identity,
                        np.array([0.1, 1.0, 0.2, 0.95]),
                    )
                    scn.ngeom += 1

    def run(self):
        start_x, start_y, start_yaw, _ = self.track.get_state_at_s(0.0)
        self.data.qpos[0] = start_x
        self.data.qpos[1] = start_y
        self.data.qpos[2] = start_yaw
        self.v = 0.0
        mujoco.mj_forward(self.model, self.data)

        dt_control = 0.05
        substeps = 5
        dt_sub = dt_control / substeps

        if self.gui and HAS_VIEWER:
            try:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    viewer.cam.distance = 12.0
                    viewer.cam.elevation = -45.0
                    viewer.cam.lookat[0] = start_x + 5.0
                    viewer.cam.lookat[1] = start_y
                    viewer.cam.lookat[2] = 0.5

                    while viewer.is_running():
                        t_start = time.perf_counter()

                        try:
                            cmd = self.server.recv_command()
                        except Exception:
                            break

                        if cmd.stop:
                            self.server.send_state(self.get_current_state())
                            break

                        for _ in range(substeps):
                            self.step_vehicle(cmd.steer, cmd.speed, cmd.accel, dt_sub)

                        viewer.cam.lookat[0] = float(self.data.qpos[0])
                        viewer.cam.lookat[1] = float(self.data.qpos[1])
                        self.update_viewer_scene(viewer)
                        viewer.sync()

                        state = self.get_current_state()
                        self.server.send_state(state)

                        if state.done:
                            break

                        t_elapsed = time.perf_counter() - t_start
                        if t_elapsed < dt_control:
                            time.sleep(dt_control - t_elapsed)
            except Exception:
                self.gui = False

        if not self.gui:
            while True:
                try:
                    cmd = self.server.recv_command()
                except Exception:
                    break

                if cmd.stop:
                    self.server.send_state(self.get_current_state())
                    break

                for _ in range(substeps):
                    self.step_vehicle(cmd.steer, cmd.speed, cmd.accel, dt_sub)

                state = self.get_current_state()
                self.server.send_state(state)

                if state.done:
                    break

        self.server.close()


def main():
    parser = argparse.ArgumentParser(description="2D Ackermann MuJoCo Simulator")
    parser.add_argument("--no-gui", action="store_true", help="Run without graphical viewer")
    parser.add_argument("--endpoint", type=str, default="tcp://127.0.0.1:5555", help="ZeroMQ endpoint")
    parser.add_argument("--duration", type=float, default=60.0, help="Maximum duration in seconds")
    args = parser.parse_args()

    sim = AckermannSimulator(
        endpoint=args.endpoint,
        gui=(not args.no_gui),
        max_time=args.duration,
    )
    sim.run()


if __name__ == "__main__":
    main()
