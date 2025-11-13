from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class GeneticSettings:
    pop_size: int
    dna_length: int
    generations: int
    rounds: int
    passthrough_rate: float
    mutation_rate: float
    mutation_prob: float

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f)

    @staticmethod
    def load(path: Path) -> GeneticSettings:
        with open(path, "r") as f:
            data: dict = json.load(f)
        return GeneticSettings(**data)
