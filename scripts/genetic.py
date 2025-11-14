import argparse
import random
from pathlib import Path
from typing import Optional

from rallyrobopilot.game_launcher import prepare_game_app
from rallyrobopilot.genetic_manager import GeneticManager
from rallyrobopilot.genetic_settings import GeneticSettings
from rallyrobopilot.mutation_strategy import (
    FrameConstruction,
    FrameSelection,
    MutationStrategy,
)
from rallyrobopilot.trajectory_optimizer import TrajectoryOptimizer
from rallyrobopilot.trajectory_segment import TrajectorySegment

plots: bool = False


def main(settings: GeneticSettings, record_path: str, seed: Optional[int]):
    app, _, track = prepare_game_app("SimpleTrack/track_metadata.json")

    to: TrajectoryOptimizer = TrajectoryOptimizer(record_path)
    gm: GeneticManager = GeneticManager(
        app, track, settings
    )
    if seed is not None:
        random.seed(seed)
    segment: TrajectorySegment = to.random_segment(settings.dna_length)
    gm.optimize(segment)
    dir: Path = Path(__file__).parent.parent / "results" / "0"
    gm.save_stats(dir)
    quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--len", help="Number of frames in a segment", type=int, default=100
    )
    parser.add_argument(
        "-g", "--gens", help="Number of generations", type=int, default=30
    )
    parser.add_argument(
        "-r", "--record", help="Reference record file", default="record_0.npz"
    )
    parser.add_argument("-S", "--seed", help="RNG seed", type=int)
    parser.add_argument(
        "-p", "--pop-size", help="Population size", type=int, default=20
    )
    parser.add_argument(
        "-t",
        "--tournament-rounds",
        help="Number of tournament rounds",
        type=int,
        default=3,
    )
    parser.add_argument(
        "-k",
        "--passthrough",
        help="Passthrough rate, i.e. ratio of population passing to the next generation without crossover",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "-m",
        "--mutation-rate",
        help="Mutation rate, i.e. probability of and individual to be mutated",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "-d",
        "--mutation-prob",
        help="Mutation probability, i.e. probability of each gene to be modified",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "-s",
        "--mutation-strategy",
        help="Mutation strategy",
        type=str,
        default="random,consecutive-same",
    )
    parser.add_argument(
        "-f",
        "--flip-prob",
        help="Flip probability",
        type=float,
        default=0.2,
    )
    args = parser.parse_args()
    mut_strat_parts: list[str] = args.mutation_strategy.split(",")
    mutation_strategy = MutationStrategy(
        FrameConstruction(mut_strat_parts[0]), FrameSelection(mut_strat_parts[1])
    )

    settings: GeneticSettings = GeneticSettings(
        args.pop_size,
        args.len,
        args.gens,
        args.tournament_rounds,
        args.passthrough,
        args.mutation_rate,
        args.mutation_prob,
        mutation_strategy,
        args.flip_prob,
    )
    main(settings, args.record, args.seed)
