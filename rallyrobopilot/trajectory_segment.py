from dataclasses import dataclass, field

from rallyrobopilot.genetic_player import FrameInput
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

    @property
    def length(self) -> int:
        return self.end_i - self.start_i
    
    def frame_at(self, i: int) -> FrameInput:
        return self.snapshots[i].current_controls
