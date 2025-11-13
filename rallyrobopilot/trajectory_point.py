from __future__ import annotations
import json
from dataclasses import dataclass, field
import os

from matplotlib.pylab import TYPE_CHECKING
from rallyrobopilot.car import Car
from ursina.vec2 import Vec2
from ursina.vec3 import Vec3
from rallyrobopilot.sensing_message import SensingSnapshot

if TYPE_CHECKING:
    from rallyrobopilot.genetic_player import FrameInput

@dataclass
class TrajectoryPoint:
    pos: Vec2 = field(default_factory=Vec2)
    angle: float = 0
    speed: float = 0
    inputs: FrameInput = (0,0,0,0)


    @staticmethod
    def from_snapshot(snapshot: SensingSnapshot) -> TrajectoryPoint:
        return TrajectoryPoint(
            Vec3(*snapshot.car_position).xz,
            snapshot.car_angle,
            snapshot.car_speed,
            snapshot.current_controls
        )
        
    def to_json(self):
        return {
            "pos":(self.pos.x,self.pos.y),
            "angle":self.angle,
            "speed":self.speed,
            "inputs":self.inputs
        }
    
    @staticmethod
    def from_car(car: Car):
        return TrajectoryPoint(
            car.position.xz,
            car.rotation_y,
            car.speed,
            tuple(int(car.keys[k]) for k in "wsad")
        )
        
    @staticmethod
    def save_list_to_json(l:list[TrajectoryPoint],trackname:str,seg_i:int):
        data = []
        for d in l:
            data.append(d.to_json())
        
        fid = 0
        filename = f"best_trajectory_{trackname}_{fid}.json"
        while os.path.exists(filename):
            fid += 1
            filename = f"best_trajectory_{trackname}_{fid}_segc{seg_i}.json"
        with open(filename,"w") as f:
            json.dump(data,f,indent=4)
        print("saved to file:",filename)
        return os.path.abspath(filename)
        
        
