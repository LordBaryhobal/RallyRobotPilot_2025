from pathlib import Path
from typing import Optional

import pandas as pd
from ursina import Entity, Mesh
from ursina.vec3 import Vec3

from rallyrobopilot.trajectory_point import TrajectoryPoint


class Trajectory(Entity):
    def __init__(self, pts: Optional[list[TrajectoryPoint]] = None, **kwargs):
        if pts is None:
            pts = [TrajectoryPoint(), TrajectoryPoint()]
        super().__init__(model=self.make_mesh(pts), mode="line", **kwargs)
        self.pts: list[TrajectoryPoint] = pts

    @staticmethod
    def make_mesh(pts: list[TrajectoryPoint]) -> Mesh:
        return Mesh(vertices=[Vec3(pt.pos.x, 0, pt.pos.y) for pt in pts], mode="line")

    def update_mesh(self):
        self.model = self.make_mesh(self.pts)

    def add_pt(self, pt: TrajectoryPoint):
        self.pts.append(pt)
        self.update_mesh()

    def clear(self):
        self.pts = []
        self.update_mesh()

    def set_pts(self, pts: list[TrajectoryPoint]):
        self.pts = pts
        self.update_mesh()

    def save(self, path: Path | str):
        data: list[dict] = [
            {"x": pt.pos.x, "y": pt.pos.y, "angle": pt.angle, "speed": pt.speed}
            for pt in self.pts
        ]
        df: pd.DataFrame = pd.DataFrame(data)
        df.to_csv(path)
