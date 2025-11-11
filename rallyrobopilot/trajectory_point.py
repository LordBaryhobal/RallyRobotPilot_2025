from __future__ import annotations

from dataclasses import dataclass

from rallyrobopilot.car import Car
from ursina.vec2 import Vec2
from ursina.vec3 import Vec3

from rallyrobopilot.sensing_message import SensingSnapshot


@dataclass
class TrajectoryPoint:
    pos: Vec2
    angle: float
    speed: float

    @staticmethod
    def from_snapshot(snapshot: SensingSnapshot) -> TrajectoryPoint:
        return TrajectoryPoint(
            Vec3(*snapshot.car_position).xz,
            snapshot.car_angle,
            snapshot.car_speed,
        )
    
    @staticmethod
    def from_car(car: Car):
        return TrajectoryPoint(
            car.position.xz,
            car.rotation_y,
            car.speed,
        )
