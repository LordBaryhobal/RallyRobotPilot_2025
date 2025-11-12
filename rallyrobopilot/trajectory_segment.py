from dataclasses import dataclass, field

from rallyrobopilot.genetic_player import FrameInput
from rallyrobopilot.trajectory_point import TrajectoryPoint
from ursina.vec3 import Vec3

from rallyrobopilot.checkpoint import Checkpoint
from rallyrobopilot.sensing_message import SensingSnapshot


@dataclass
class TrajectorySegment:
    snapshots: list[SensingSnapshot] = field(default_factory=list)
    checkpoint: Checkpoint = field(
        default_factory=lambda: Checkpoint(Vec3(), Vec3(), 0, None)
    )
    start_i: int = 0
    end_i: int = 0
    trajectory: list[TrajectoryPoint] = field(init = False)

    def __post_init__(self):
        self.trajectory = [TrajectoryPoint.from_snapshot(s) for s in self.snapshots]

    @property
    def length(self) -> int:
        return self.end_i - self.start_i
    
    def frame_at(self, i: int) -> FrameInput:
        return self.snapshots[i].current_controls
