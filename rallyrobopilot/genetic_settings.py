from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from rallyrobopilot.mutation_strategy import MutationStrategy


@dataclass
class GeneticSettings:
    pop_size: int
    dna_length: int
    generations: int
    rounds: int
    passthrough_rate: float
    mutation_rate: float
    mutation_prob: float
    mutation_strategy: MutationStrategy
    flip_prob: float

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f)

    @staticmethod
    def load(path: Path) -> GeneticSettings:
        with open(path, "r") as f:
            data: dict = json.load(f)
        return GeneticSettings(**data)

    def copy_with(self, **kwargs) -> GeneticSettings:
        return replace(self, **kwargs)
