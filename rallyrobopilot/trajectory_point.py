from __future__ import annotations

from dataclasses import dataclass, field

from rallyrobopilot.car import Car
from ursina.vec2 import Vec2
from ursina.vec3 import Vec3

from rallyrobopilot.frame_input import FrameInput
from rallyrobopilot.sensing_message import SensingSnapshot


@dataclass
class TrajectoryPoint:
    pos: Vec2 = field(default_factory=Vec2)
    angle: float = 0
    speed: float = 0
    inputs: FrameInput = (0, 0, 0, 0)

    @staticmethod
    def from_snapshot(snapshot: SensingSnapshot) -> TrajectoryPoint:
        return TrajectoryPoint(
            Vec3(*snapshot.car_position).xz,
            snapshot.car_angle,
            snapshot.car_speed,
            snapshot.current_controls
        )
    
    @staticmethod
    def from_car(car: Car):
        return TrajectoryPoint(
            car.position.xz,
            car.rotation_y,
            car.speed,
            tuple(car.keys[ctrl] for ctrl in car.controls) # type: ignore
        )
