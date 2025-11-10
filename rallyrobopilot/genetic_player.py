from __future__ import annotations
import random

FrameInput = tuple[int, int, int, int]

class GeneticPlayer:
    def __init__(self, id: int, dna: list[FrameInput]):
        self.dna: list[FrameInput] = dna
        self.i: int = 0
        self.id: int = id

        self.evaluation: float = 0.0
        self.evaluated: bool = False
    
    def set_evaluation(self, evaluation: float):
        self.evaluation = evaluation
        self.evaluated = True
    
    def infer(self) -> FrameInput:
        if self.i >= len(self.dna):
            return (0, 0, 0, 0)
        inputs: FrameInput = self.dna[self.i]
        self.i += 1
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
