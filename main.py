import os
import sys
import time
import signal
import argparse
import subprocess

root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root_dir)

venv_dir = os.path.join(root_dir, ".venv")
venv_python = os.path.join(venv_dir, "bin", "python")
if os.path.isfile(venv_python) and sys.prefix != venv_dir:
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env["VIRTUAL_ENV"] = venv_dir
    os.execve(venv_python, [venv_python] + sys.argv, clean_env)

os.environ.pop("PYTHONPATH", None)


def run_orchestrator(gui: bool = True, speed: float = 4.0, duration: float = 50.0):
    python_bin = sys.executable

    sim_script = os.path.join(root_dir, "sim", "simulator.py")
    ctrl_script = os.path.join(root_dir, "control", "controller.py")

    sim_cmd = [
        python_bin, sim_script,
        "--endpoint", "tcp://127.0.0.1:5555",
        "--duration", str(duration)
    ]
    if not gui:
        sim_cmd.append("--no-gui")

    ctrl_cmd = [
        python_bin, ctrl_script,
        "--endpoint", "tcp://127.0.0.1:5555",
        "--speed", str(speed),
        "--width", "3.5"
    ]

    def free_port():
        try:
            subprocess.run(["fuser", "-k", "5555/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.3)
        except Exception:
            pass

    free_port()

    procs = []

    def cleanup(signum=None, frame=None):
        for p in procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    p.kill()
        free_port()
        if signum is not None:
            sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    child_env = os.environ.copy()
    child_env.pop("PYTHONPATH", None)

    try:
        sim_proc = subprocess.Popen(sim_cmd, env=child_env)
        procs.append(sim_proc)

        time.sleep(1.2)

        ctrl_proc = subprocess.Popen(ctrl_cmd, env=child_env)
        procs.append(ctrl_proc)

        ctrl_proc.wait()
        sim_proc.wait()

    except KeyboardInterrupt:
        cleanup()
    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description="2D Ackermann MPC Orchestrator")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless mode")
    parser.add_argument("--speed", type=float, default=4.0, help="Target speed in m/s")
    parser.add_argument("--duration", type=float, default=50.0, help="Duration in seconds")
    args = parser.parse_args()

    run_orchestrator(
        gui=(not args.no_gui),
        speed=args.speed,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
