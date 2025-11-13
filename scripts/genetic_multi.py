import multiprocessing as mp
from pathlib import Path

from rallyrobopilot.game_launcher import prepare_game_app
from rallyrobopilot.genetic_manager import GeneticManager
from rallyrobopilot.genetic_settings import GeneticSettings
from rallyrobopilot.mutation_strategy import FrameConstruction, FrameSelection, MutationStrategy
from rallyrobopilot.trajectory_optimizer import TrajectoryOptimizer
from rallyrobopilot.trajectory_segment import TrajectorySegment

RECORD_PATH: str = "record_0.npz"

base_settings: GeneticSettings = GeneticSettings(20, 100, 20, 3, 0.2, 0.6, 0.2, MutationStrategy(FrameConstruction.RANDOM_ALL, FrameSelection.CONSECUTIVE_SAME), 0.6)
settings_lst: list[GeneticSettings] = [
    base_settings.copy_with(mutation_strategy=MutationStrategy(FrameConstruction.RANDOM_ALL, FrameSelection.CONSECUTIVE_SAME)),
    base_settings.copy_with(mutation_strategy=MutationStrategy(FrameConstruction.RANDOM_ALL, FrameSelection.CONSECUTIVE)),
    base_settings.copy_with(mutation_strategy=MutationStrategy(FrameConstruction.RANDOM_ALL, FrameSelection.INDIVIDUAL)),
    base_settings.copy_with(mutation_strategy=MutationStrategy(FrameConstruction.FLIP1, FrameSelection.CONSECUTIVE_SAME)),
    base_settings.copy_with(mutation_strategy=MutationStrategy(FrameConstruction.FLIP1, FrameSelection.CONSECUTIVE)),
    base_settings.copy_with(mutation_strategy=MutationStrategy(FrameConstruction.FLIP1, FrameSelection.INDIVIDUAL)),
    base_settings.copy_with(mutation_strategy=MutationStrategy(FrameConstruction.FLIP_RAND, FrameSelection.CONSECUTIVE_SAME)),
    base_settings.copy_with(mutation_strategy=MutationStrategy(FrameConstruction.FLIP_RAND, FrameSelection.CONSECUTIVE)),
    base_settings.copy_with(mutation_strategy=MutationStrategy(FrameConstruction.FLIP_RAND, FrameSelection.INDIVIDUAL)),
]

def run_ga(pair: tuple[int, GeneticSettings]):
    i: int
    settings: GeneticSettings
    i, settings = pair

    app, _, track = prepare_game_app("SimpleTrack/track_metadata.json")
    to: TrajectoryOptimizer = TrajectoryOptimizer(RECORD_PATH)
    gm: GeneticManager = GeneticManager(
        app, track, Path(__file__).parent.parent / "results" / str(i), settings
    )
    segment: TrajectorySegment = to.random_segment(settings.dna_length)
    gm.optimize(segment)

    quit()

def main():
    n: int = mp.cpu_count() - 1
    with mp.Pool(n) as pool:
        pool.map(run_ga, enumerate(settings_lst))


if __name__ == "__main__":
    main()
