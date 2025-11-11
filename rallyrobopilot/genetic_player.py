from __future__ import annotations
import random

from rallyrobopilot.trajectory_point import TrajectoryPoint
from ursina import Vec3

from rallyrobopilot.car import Car
from rallyrobopilot.checkpoint import Checkpoint

FrameInput = tuple[int, int, int, int]

class GeneticPlayer:
    def __init__(self, id: int, dna: list[FrameInput]):
        self.dna: list[FrameInput] = dna
        self.i: int = 1
        self.id: int = id

        self.evaluation: float = 0.0
        self.evaluated: bool = False

        self.prev_pos: Vec3 = Vec3(0, 0, 0)
        self.reached_end: bool = False
        self.steps_till_gate: int = 0
        self.wall_hits: int = 0
        self.trajectory: list[TrajectoryPoint] = []
    
    def set_evaluation(self, evaluation: float):
        self.evaluation = evaluation
        self.evaluated = True
    
    def infer(self, car: Car, checkpoint: Checkpoint):
        controls: FrameInput = self.get_inputs()
        car.keys["w"] = bool(controls[0])
        car.keys["s"] = bool(controls[1])
        car.keys["a"] = bool(controls[2])
        car.keys["d"] = bool(controls[3])
        self.trajectory.append(TrajectoryPoint.from_car(car))
        
        if checkpoint.intersects(self.prev_pos.xz, car.position.xz):
            self.reached_end = True
            print(f"Player {self.id} reached the end")
        
        if not self.reached_end:
            self.steps_till_gate += 1

        if car.hitting_wall:
            self.wall_hits += 1

        self.prev_pos = car.position
        self.i += 1
    
    def get_inputs(self):
        if self.i >= len(self.dna):
            return (0, 0, 0, 0)
        inputs: FrameInput = self.dna[self.i]
        return inputs

    @staticmethod
    def random(id: int, dna_length: int) -> GeneticPlayer:
        return GeneticPlayer(id, [
            GeneticPlayer.random_frame()
            for _ in range(dna_length)
        ])
    
    @staticmethod
    def random_frame() -> FrameInput:
        return (
            round(random.random()),
            round(random.random()),
            round(random.random()),
            round(random.random()),
        )
