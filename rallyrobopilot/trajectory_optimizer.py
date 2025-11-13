import lzma
import pickle
import random

from rallyrobopilot.checkpoint import Checkpoint
from rallyrobopilot.sensing_message import SensingSnapshot
from rallyrobopilot.trajectory_point import TrajectoryPoint
from rallyrobopilot.trajectory_segment import TrajectorySegment


class TrajectoryOptimizer:
    def __init__(self, record_path: str):
        self.record_path: str = record_path
        self.snapshots: list[SensingSnapshot] = []
        self.trajectory: list[TrajectoryPoint] = []
        self.load_record()

    def load_record(self):
        with lzma.open(self.record_path, "rb") as file:
            self.snapshots = pickle.load(file)

        self.trajectory = list(map(TrajectoryPoint.from_snapshot, self.snapshots))

    def random_segment(self, length: int = 100) -> TrajectorySegment:
        start_i: int = random.randint(0, len(self.snapshots) - length)
        end_i: int = start_i + length
        snapshots: list[SensingSnapshot] = self.snapshots[start_i:end_i]
        start: SensingSnapshot = snapshots[0]
        end: SensingSnapshot = snapshots[-1]
        checkpoint: Checkpoint = Checkpoint.from_snapshots(start, end)

        return TrajectorySegment(snapshots, checkpoint, start_i, end_i)
