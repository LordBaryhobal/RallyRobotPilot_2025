import lzma
import os
import pickle

from ursina import Entity, held_keys

from rallyrobopilot.car import Car
from rallyrobopilot.sensing_message import SensingSnapshot


class Recorder(Entity):
    def __init__(self, car: Car) -> None:
        super().__init__()
        self.car: Car = car
        self.snapshots: list[SensingSnapshot] = []
        self.recording: bool = False

    def take_snapshot(self) -> SensingSnapshot:
        snapshot = SensingSnapshot()
        snapshot.current_controls = (
            held_keys["w"] or held_keys["up arrow"],
            held_keys["s"] or held_keys["down arrow"],
            held_keys["a"] or held_keys["left arrow"],
            held_keys["d"] or held_keys["right arrow"],
        )
        snapshot.car_position = self.car.world_position
        snapshot.car_speed = self.car.speed
        snapshot.car_angle = self.car.rotation_y
        snapshot.raycast_distances = self.car.multiray_sensor.collect_sensor_values()
        self.snapshots.append(snapshot)
        return snapshot

    def save(self):
        record_name = "record_%d.npz"
        fid = 0
        while os.path.exists(record_name % fid):
            fid += 1
        path: str = record_name % fid
        with lzma.open(path, "wb") as f:
            pickle.dump(self.snapshots, f)
        print(f"Saved recording in {path}")

    def update(self):
        if self.recording:
            self.take_snapshot()

    def input(self, key):
        if key == ",":
            self.save()
        elif key == ".":
            self.recording = not self.recording
            if self.recording:
                print("Started recording")
            else:
                print("Stopped recording")
