from __future__ import annotations

from math import cos, radians, sin
from typing import TYPE_CHECKING, Optional

from ursina.vec2 import Vec2
from ursina.vec3 import Vec3

from rallyrobopilot.sensing_message import SensingSnapshot

if TYPE_CHECKING:
    from rallyrobopilot.car import Car


class Checkpoint:
    def __init__(
        self,
        start_pos: Vec3,
        start_dir: Vec3,
        start_speed: float,
        end: Optional[tuple[Vec2, Vec2]] = None,
    ) -> None:
        self.start_pos: Vec3 = start_pos
        self.start_dir: Vec3 = start_dir
        self.start_speed: float = start_speed
        self.end: Optional[tuple[Vec2, Vec2]] = end

    def reset_car(self, car: Car):
        car.reset_position = self.start_pos
        car.reset_orientation = self.start_dir
        car.reset_car()
        car.speed = self.start_speed

    def intersects(self, c: Vec2, d: Vec2) -> bool:
        if self.end is None:
            return False
        a, b = self.end
        return self.ccw(a, c, d) != self.ccw(b, c, d) and self.ccw(a, b, c) != self.ccw(
            a, b, d
        )

    @staticmethod
    def ccw(A: Vec2, B: Vec2, C: Vec2) -> bool:
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

    @staticmethod
    def create(car: Car) -> Checkpoint:
        return Checkpoint(car.position, car.rotation, car.speed)

    @staticmethod
    def from_snapshots(start: SensingSnapshot, end: SensingSnapshot) -> Checkpoint:
        start_pos: Vec3 = Vec3(*start.car_position)
        start_dir: Vec3 = Vec3(0, start.car_angle, 0)
        start_speed: float = start.car_speed
        angle: float = radians(90 - end.car_angle)
        end_dir: Vec2 = Vec2(cos(angle), sin(angle))
        end_pts: tuple[Vec2, Vec2] = Checkpoint.compute_end_pts(
            Vec3(*end.car_position).xz - end_dir * .5,
            end_dir,
            end.raycast_distances[0],
            end.raycast_distances[-1],
        )
        return Checkpoint(start_pos, start_dir, start_speed, end_pts)

    @staticmethod
    def compute_end_pts(
        car_pos: Vec2 | Vec3, car_dir: Vec2 | Vec3, left_dist: float, right_dist: float
    ) -> tuple[Vec2, Vec2]:
        center: Vec2 = car_pos.xz if isinstance(car_pos, Vec3) else car_pos
        n: Vec2 = car_dir.xz if isinstance(car_dir, Vec3) else car_dir
        v: Vec2 = Vec2(n.y, -n.x)
        left: Vec2 = center - left_dist * v
        right: Vec2 = center + right_dist * v
        return (left, right)

    def set_end(self, car: Car):
        distances: list[float] = car.multiray_sensor.collect_sensor_values()
        self.end = self.compute_end_pts(
            car.position, car.forward, distances[0], distances[-1]
        )
    
    def distance(self, pos: Vec2) -> float:
        if self.end is None:
            return 0
        d1: float = (self.end[0] - pos).length()
        d2: float = (self.end[1] - pos).length()
        return (d1 + d2) / 2
