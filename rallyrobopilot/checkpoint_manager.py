import json
import os
from pathlib import Path
from typing import Optional
from ursina import Entity, color
from ursina.vec2 import Vec2
from ursina.vec3 import Vec3

from rallyrobopilot.car import Car
from rallyrobopilot.checkpoint import Checkpoint


class CheckpointManager(Entity):
    SAVE_PATH: Path = Path("checkpoints.json")

    def __init__(self):
        super().__init__()
        self.checkpoints: list[Checkpoint] = []
        self.pending: Optional[Checkpoint] = None
        self.entities: list[Entity] = []

    @staticmethod
    def deserialize(data: dict) -> Checkpoint:
        start_pos: Vec3 = Vec3(*data["start"])
        start_dir: Vec3 = Vec3(*data["dir"])
        end: Optional[tuple[Vec2, Vec2]] = None
        if data["end"] is not None:
            end = (
                Vec2(*data["end"][0]),
                Vec2(*data["end"][1]),
            )
        return Checkpoint(start_pos, start_dir, data["speed"], end)

    @staticmethod
    def serialize(cp: Checkpoint) -> dict:
        return {
            "start": [cp.start_pos.x, cp.start_pos.y, cp.start_pos.z],
            "dir": [cp.start_dir.x, cp.start_dir.y, cp.start_dir.z],
            "speed": cp.start_speed,
            "end": (
                [
                    [cp.end[0].x, cp.end[0].y],
                    [cp.end[1].x, cp.end[1].y],
                ]
                if cp.end is not None
                else None
            ),
        }

    def load(self):
        if not os.path.exists(self.SAVE_PATH):
            return

        with open(self.SAVE_PATH, "r") as f:
            checkpoints: list[dict] = json.load(f)

        self.checkpoints = list(map(self.deserialize, checkpoints))
    
    def save(self):
        with open(self.SAVE_PATH, "w") as f:
            json.dump(list(map(self.serialize, self.checkpoints)), f, indent=4)
        print(f"Saved {len(self.checkpoints)} checkpoints")
    
    def add(self, checkpoint: Checkpoint):
        self.checkpoints.append(checkpoint)
        self.save()
        self.add_entity(checkpoint)

    def input(self, key: str):
        if key == "p":
            if self.pending is None:
                print("Saving checkpoint start")
                self.pending = Checkpoint.create(self.car)
            else:
                print("Saving checkpoint end")
                self.pending.set_end(self.car)
                self.add(self.pending)
                self.pending = None
    
    def add_entity(self, checkpoint: Checkpoint):
        if checkpoint.end is None:
            return
        pos2D: Vec2 = (checkpoint.end[0] + checkpoint.end[1]) / 2
        pos: tuple[float, float, float] = (pos2D.x, 1, pos2D.y)
        delta: Vec2 = checkpoint.end[1] - checkpoint.end[0]
        rotation = delta.signed_angle_deg(Vec2(1, 0)) - 90
        entity: Entity = Entity(
            model='cube',
            color=color.rgba(255, 255, 255, 100),
            position=pos,
            rotation=(0, rotation, 0),
            scale=(.2, .2, delta.length())
        )
        entity.enable()
        self.entities.append(entity)
    
    def add_entities(self):
        for cp in self.checkpoints:
            self.add_entity(cp)
