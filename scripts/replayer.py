from pathlib import Path

from rallyrobopilot.clock import Clock
from rallyrobopilot.game_launcher import prepare_game_app
from rallyrobopilot.genetic_player import GeneticPlayer
from rallyrobopilot.trajectory_optimizer import TrajectoryOptimizer
from rallyrobopilot.trajectory_segment import TrajectorySegment

RECORD_DIR = Path(__file__).parent.parent / "generated"


def main(track_path: str, record_path: Path):
    app, car, track = prepare_game_app(track_path, True)

    to: TrajectoryOptimizer = TrajectoryOptimizer(str(record_path))
    segment: TrajectorySegment = to.full_segment()
    player: GeneticPlayer = GeneticPlayer(0, [pt.inputs for pt in segment.trajectory])
    print(f"Loaded {segment.length} frames")

    clock: Clock = Clock()
    while True:
        segment.checkpoint.reset_car(car)
        player.reset()
        for _ in range(segment.length):
            player.infer(car, segment.checkpoint)
            app.step()
            clock.tick(30)

    quit()


if __name__ == "__main__":
    main("SimpleTrack/track_metadata.json", RECORD_DIR / "SimpleTrack_0.npz")
