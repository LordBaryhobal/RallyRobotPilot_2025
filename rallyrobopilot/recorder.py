import lzma
import os
import pickle
import numpy as np
from ursina import Entity, held_keys
from ursina import *

from rallyrobopilot.car import Car
from rallyrobopilot.sensing_message import SensingSnapshot


class Recorder(Entity):
    def __init__(self, car: Car,callback_sensor) -> None:
        super().__init__()
        self.car: Car = car
        self.snapshots: list[SensingSnapshot] = []
        self.recording: bool = False
        self.callback_sensor = callback_sensor

    def take_snapshot(self) -> SensingSnapshot:
        snapshot = SensingSnapshot()
        snapshot.current_controls = (
            self.car.keys["w"] or held_keys["w"] or held_keys["up arrow"],
            self.car.keys["s"] or held_keys["s"] or held_keys["down arrow"],
            self.car.keys["a"] or held_keys["a"] or held_keys["left arrow"],
            self.car.keys["d"] or held_keys["d"] or held_keys["right arrow"],
        )
        snapshot.car_position = self.car.world_position
        snapshot.car_speed = self.car.speed
        snapshot.car_angle = self.car.rotation_y
        snapshot.raycast_distances = self.callback_sensor()
        tex = base.win.getDisplayRegion(0).getScreenshot()
        arr = tex.getRamImageAs("RGB")
        data = np.frombuffer(arr, np.uint8)
        image = data.reshape(tex.getYSize(), tex.getXSize(), 3)
        image = image[::-1, :, :]#   Image arrives with inverted Y axis

        snapshot.image = image

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
