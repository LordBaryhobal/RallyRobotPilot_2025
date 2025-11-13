from dataclasses import dataclass
from enum import StrEnum
import random

from rallyrobopilot.genetic_player import FrameInput


class FrameConstruction(StrEnum):
    RANDOM_ALL = "random"
    FLIP1 = "flip1"
    FLIP_RAND = "flip_rand"


class FrameSelection(StrEnum):
    INDIVIDUAL = "individual"
    CONSECUTIVE = "consecutive"
    CONSECUTIVE_SAME = "consecutive-same"


@dataclass
class MutationStrategy:
    construction: FrameConstruction
    selection: FrameSelection


    def mutate(self, dna: list[FrameInput], mutation_prob: float, flip_prob: float):
        if self.selection == FrameSelection.INDIVIDUAL:
            for i in range(len(dna)):
                if random.random() < mutation_prob:
                    dna[i] = random_frame(self.construction, dna[i], flip_prob)
        
        else:
            width: int = random.randint(1, 5)
            i: int = random.randint(0, len(dna) - width)

            if self.selection == FrameSelection.CONSECUTIVE:
                for j in range(width):
                    dna[i + j] = random_frame(self.construction, dna[i + j], flip_prob)
            elif self.selection == FrameSelection.CONSECUTIVE_SAME:
                frame: FrameInput = random_frame(self.construction, dna[i], flip_prob)
                for j in range(width):
                    dna[i + j] = frame


def random_frame(strategy: FrameConstruction, current: FrameInput, flip_prob: float) -> FrameInput:
    if strategy == FrameConstruction.RANDOM_ALL:
        return (
            round(random.random()),
            round(random.random()),
            round(random.random()),
            round(random.random()),
        )

    if strategy == FrameConstruction.FLIP1:
        i: int = random.randint(0, 3)
        return tuple(
            1 - current[j] if i == j else current[j]
            for j in range(4)
        ) # type: ignore
    
    if strategy == FrameConstruction.FLIP_RAND:
        return tuple(
            1 - current[i] if random.random() < flip_prob else current[i]
            for i in range(4)
        ) # type: ignore
