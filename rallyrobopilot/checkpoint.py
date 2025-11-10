from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Optional

from ursina.vec2 import Vec2
from ursina.vec3 import Vec3

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

    def set_end(self, car: Car):
        center: Vec2 = car.position.xz
        n: Vec2 = car.forward.xz
        distances: list[float] = car.multiray_sensor.collect_sensor_values()
        left_dist: float = distances[0]
        right_dist: float = distances[-1]
        v: Vec2 = Vec2(n.y, -n.x)
        left: Vec2 = center - left_dist * v
        right: Vec2 = center + right_dist * v
        self.end = (left, right)
