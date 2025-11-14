import multiprocessing as mp
import random
from pathlib import Path
import time

from rallyrobopilot.game_launcher import prepare_game_app
from rallyrobopilot.genetic_manager import GeneticManager
from rallyrobopilot.genetic_player import GeneticPlayer
from rallyrobopilot.genetic_settings import GeneticSettings
from rallyrobopilot.mutation_strategy import (
    FrameConstruction,
    FrameSelection,
    MutationStrategy,
)
from rallyrobopilot.recorder import Recorder
from rallyrobopilot.trajectory_optimizer import TrajectoryOptimizer
from rallyrobopilot.trajectory_segment import TrajectorySegment

SAMPLE_LEN = 40
OUT_DIR = Path(__file__).parent.parent / "generated"

settings: GeneticSettings = GeneticSettings(
    20,
    SAMPLE_LEN,
    20,
    3,
    0.2,
    0.6,
    0.2,
    MutationStrategy(FrameConstruction.FLIP1, FrameSelection.CONSECUTIVE_SAME),
    0.6,
)

params: list = [
    ("SimpleTrack", "SimpleTrack/track_metadata.json", "record_0.npz", 1),
    # ("SimpleTrack", "SimpleTrack/track_metadata.json", "record_0.npz", 10),
    # ("NotSoSimpleTrack", "NotSoSimpleTrack/track_metadata.json", "record_1.npz", 15),
    # ("SlightlyHarder", "SlightlyHarder/track_metadata.json", "record_2.npz", 20),
]


def run_ga(name: str, track_path: str, record_path: str, seed: int):
    app, car, track = prepare_game_app(track_path, True)
    if car is None:
        raise ValueError()
    car.disable()
    to: TrajectoryOptimizer = TrajectoryOptimizer(record_path)
    gm: GeneticManager = GeneticManager(app, track, settings)
    random.seed(seed)
    segment: TrajectorySegment = to.random_segment(settings.dna_length)
    print("Optimizing")
    t1 = time.time()
    gm.optimize(segment)
    t2 = time.time()
    print(f"Optimized in {t2-t1:.2f}s")

    for c in gm.cars:
        c.disable()
    gm.checkpoint_manager.remove_entities()
    gm.ref_trajectory.disable()
    gm.best_trajectory.disable()
    
    car.enable()
    segment.checkpoint.reset_car(car)
    app.step()
    player: GeneticPlayer = gm.best_player
    player.reset()
    recorder: Recorder = Recorder(car, True, True)
    recorder.recording = True
    print("Recording")
    t1 = time.time()
    for _ in range(settings.dna_length):
        player.infer(car, segment.checkpoint)
        app.step()
    t2 = time.time()
    print(f"Recorded in {t2-t1:.2f}s")

    recorder.save(OUT_DIR / f"{name}.npz")

    quit()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n: int = 1#mp.cpu_count() - 1
    all_params: list[tuple[str, str, str, int]] = []
    for name, track, record, samples in params:
        for i in range(samples):
            all_params.append(
                (f"{name}_{i}", track, record, random.randint(0, 0xFFFFFFFF))
            )

    with mp.Pool(n) as pool:
        pool.starmap(run_ga, all_params)
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()
