from dataclasses import dataclass


@dataclass
class GenerationStats:
    gen: int
    pop_size: int
    dna_length: int
    min_eval: float
    mean_eval: float
    median_eval: float
    max_eval: float
    min_steps: int
    mean_steps: float
    median_steps: int
