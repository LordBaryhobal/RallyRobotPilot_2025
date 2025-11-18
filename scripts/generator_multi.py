import lzma
import multiprocessing as mp
import pickle
import random
from pathlib import Path
import time

from ursina.vec3 import Vec3

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
from rallyrobopilot.trajectory import Trajectory
from rallyrobopilot.trajectory_optimizer import TrajectoryOptimizer
from rallyrobopilot.trajectory_point import TrajectoryPoint
from rallyrobopilot.trajectory_segment import TrajectorySegment

SAMPLE_LEN = 100
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
    ("SimpleTrack", "SimpleTrack/track_metadata.json", "record_0.npz"),
    # ("SimpleTrack", "SimpleTrack/track_metadata.json", "record_0.npz"),
    # ("NotSoSimpleTrack", "NotSoSimpleTrack/track_metadata.json", "record_1.npz"),
    # ("SlightlyHarder", "SlightlyHarder/track_metadata.json", "record_2.npz"),
]


def run_ga(name: str, track_path: str, record_path: str, start: int):
    app, car, track = prepare_game_app(track_path, True)
    if car is None:
        raise ValueError()
    car.disable()
    to: TrajectoryOptimizer = TrajectoryOptimizer(record_path)
    gm: GeneticManager = GeneticManager(app, track, settings)
    segment: TrajectorySegment = to.segment_at(start, settings.dna_length)
    print("Optimizing")
    t1 = time.time()
    gm.optimize(segment)
    t2 = time.time()
    print(f"Optimized in {t2-t1:.2f}s")
    gm.save_stats(OUT_DIR / name / "stats")
    gm.best_trajectory.save(OUT_DIR / name / "stats" / "traj.json")

    if gm.best_player.wall_hits != 0:
        print("Did not find valid trajectory")

    for c in gm.cars:
        c.disable()
    gm.checkpoint_manager.remove_entities()
    gm.ref_trajectory.disable()
    gm.best_trajectory.disable()
    
    car.enable()
    segment.checkpoint.reset_car(car)
    app.step()
    segment.checkpoint.reset_car(car)
    player: GeneticPlayer = gm.best_player
    player.reset()
    recorder: Recorder = Recorder(car, True, True)
    recorder.recording = True
    print("Recording")
    t1 = time.time()
    car.speed = 0
    traj: Trajectory = gm.best_trajectory
    for i in range(settings.dna_length - 1):
        pt: TrajectoryPoint = traj.pts[i]
        car.position = Vec3(pt.pos.x, car.position.y, pt.pos.y)
        car.rotation_y = pt.angle
        #player.infer(car, segment.checkpoint)
        app.step()
    t2 = time.time()
    print(f"Recorded in {t2-t1:.2f}s")

    recorder.save(OUT_DIR / f"{name}.npz")

    quit()


def get_record_length(path: Path | str) -> int:
    with lzma.open(path, "rb") as f:
        return len(pickle.load(f))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n: int = 1#mp.cpu_count() - 1
    all_params: list[tuple[str, str, str, int]] = []
    for name, track, record in params:
        length: int = get_record_length(record)
        for i in range(0, length, settings.dna_length // 2):
            all_params.append(
                (f"{name}_{i}", track, record, i)
            )

    with mp.Pool(n) as pool:
        pool.starmap(run_ga, all_params)
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()
