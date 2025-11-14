import multiprocessing as mp
import random
from itertools import product
from pathlib import Path

from rallyrobopilot.game_launcher import prepare_game_app
from rallyrobopilot.genetic_manager import GeneticManager
from rallyrobopilot.genetic_settings import GeneticSettings
from rallyrobopilot.mutation_strategy import (
    FrameConstruction,
    FrameSelection,
    MutationStrategy,
)
from rallyrobopilot.trajectory import Trajectory
from rallyrobopilot.trajectory_optimizer import TrajectoryOptimizer
from rallyrobopilot.trajectory_point import TrajectoryPoint
from rallyrobopilot.trajectory_segment import TrajectorySegment

RECORD_PATH: str = "record_0.npz"

base_settings: GeneticSettings = GeneticSettings(
    20,
    100,
    20,
    3,
    0.2,
    0.6,
    0.2,
    MutationStrategy(FrameConstruction.RANDOM_ALL, FrameSelection.CONSECUTIVE_SAME),
    0.6,
)
settings_lst: list[GeneticSettings] = [
    base_settings.copy_with(
        mutation_strategy=MutationStrategy(
            FrameConstruction.RANDOM_ALL, FrameSelection.CONSECUTIVE_SAME
        )
    ),
    base_settings.copy_with(
        mutation_strategy=MutationStrategy(
            FrameConstruction.RANDOM_ALL, FrameSelection.CONSECUTIVE
        )
    ),
    base_settings.copy_with(
        mutation_strategy=MutationStrategy(
            FrameConstruction.RANDOM_ALL, FrameSelection.INDIVIDUAL
        )
    ),
    base_settings.copy_with(
        mutation_strategy=MutationStrategy(
            FrameConstruction.FLIP1, FrameSelection.CONSECUTIVE_SAME
        )
    ),
    base_settings.copy_with(
        mutation_strategy=MutationStrategy(
            FrameConstruction.FLIP1, FrameSelection.CONSECUTIVE
        )
    ),
    base_settings.copy_with(
        mutation_strategy=MutationStrategy(
            FrameConstruction.FLIP1, FrameSelection.INDIVIDUAL
        )
    ),
    base_settings.copy_with(
        mutation_strategy=MutationStrategy(
            FrameConstruction.FLIP_RAND, FrameSelection.CONSECUTIVE_SAME
        )
    ),
    base_settings.copy_with(
        mutation_strategy=MutationStrategy(
            FrameConstruction.FLIP_RAND, FrameSelection.CONSECUTIVE
        )
    ),
    base_settings.copy_with(
        mutation_strategy=MutationStrategy(
            FrameConstruction.FLIP_RAND, FrameSelection.INDIVIDUAL
        )
    ),
]


def run_ga(pair: tuple[int, tuple[int, GeneticSettings]]):
    i: int
    seed: int
    settings: GeneticSettings
    seed, (i, settings) = pair

    app, _, track = prepare_game_app("SimpleTrack/track_metadata.json")
    to: TrajectoryOptimizer = TrajectoryOptimizer(RECORD_PATH)
    dir: Path = Path(__file__).parent.parent / "results" / f"conf_{i}" / f"seed_{seed}"
    gm: GeneticManager = GeneticManager(app, track, dir, settings)
    random.seed(seed)
    segment: TrajectorySegment = to.random_segment(settings.dna_length)
    gm.optimize(segment)

    original_trajectory: Trajectory = Trajectory(
        [TrajectoryPoint.from_snapshot(s) for s in segment.snapshots][:-1]
    )
    best_trajectory: Trajectory = gm.best_trajectory
    original_trajectory.save(dir / "trajectory_ref.csv")
    best_trajectory.save(dir / "trajectory_opti.csv")

    quit()


def main():
    n: int = mp.cpu_count() - 1
    seeds: list[int] = [random.randint(0, 0xFFFFFFFF) for _ in range(4)]
    params: product[tuple[int, tuple[int, GeneticSettings]]] = product(
        seeds, enumerate(settings_lst)
    )
    with mp.Pool(n) as pool:
        pool.map(run_ga, params)
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()
