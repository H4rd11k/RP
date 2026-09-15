from dataclasses import dataclass, asdict
import json
import zmq
from typing import Optional, Dict, Any


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    v: float
    omega: float
    time: float
    done: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "yaw": float(self.yaw),
            "v": float(self.v),
            "omega": float(self.omega),
            "time": float(self.time),
            "done": bool(self.done),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VehicleState":
        return cls(
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            yaw=float(d.get("yaw", 0.0)),
            v=float(d.get("v", 0.0)),
            omega=float(d.get("omega", 0.0)),
            time=float(d.get("time", 0.0)),
            done=bool(d.get("done", False)),
        )


@dataclass
class ControlCommand:
    steer: float = 0.0
    speed: float = 0.0
    accel: float = 0.0
    reset: bool = False
    stop: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steer": float(self.steer),
            "speed": float(self.speed),
            "accel": float(self.accel),
            "reset": bool(self.reset),
            "stop": bool(self.stop),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ControlCommand":
        return cls(
            steer=float(d.get("steer", 0.0)),
            speed=float(d.get("speed", 0.0)),
            accel=float(d.get("accel", 0.0)),
            reset=bool(d.get("reset", False)),
            stop=bool(d.get("stop", False)),
        )


class SimulatorServer:
    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555"):
        self.endpoint = endpoint
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.socket.bind(self.endpoint)

    def recv_command(self) -> ControlCommand:
        msg = self.socket.recv_json()
        return ControlCommand.from_dict(msg)

    def send_state(self, state: VehicleState):
        self.socket.send_json(state.to_dict())

    def close(self):
        self.socket.close()
        self.context.term()


class ControllerClient:
    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555"):
        self.endpoint = endpoint
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.socket.connect(self.endpoint)

    def exchange(self, cmd: ControlCommand) -> VehicleState:
        self.socket.send_json(cmd.to_dict())
        msg = self.socket.recv_json()
        return VehicleState.from_dict(msg)

    def close(self):
        self.socket.close()
        self.context.term()
