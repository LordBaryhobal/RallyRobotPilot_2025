import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
from typing import Optional
import zipfile
from PyQt6 import QtWidgets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from rallyrobopilot.game_launcher import prepare_game_app
from rallyrobopilot.recorder import Recorder
from rallyrobopilot.sensing_message import SensingSnapshot
import tqdm
import lzma
from ursina import Entity

from .data_collector import DataCollectionUI


class SuperRacer(Entity):
    CONTROLS = ["forward", "back", "left", "right"]
    N_RAYCASTS = 15
    
    def __init__(self):
        super().__init__()
        self.augment_flipped: bool = True
        self.dist_deriv: Optional[int] = 3
        self.add_noise: bool = False
        self.app = None
        self.car = None
        self.n_inputs: int = self.N_RAYCASTS + 2 + len(self.CONTROLS)
        if self.dist_deriv is not None:
            self.n_inputs += self.N_RAYCASTS

        self.nn = nn.Sequential(
            nn.Linear(self.n_inputs, 64),
            nn.Sigmoid(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.Sigmoid(),
            nn.Linear(16, len(self.CONTROLS))
        )
        self.loss_fit = None  # nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.nn.parameters(), lr=0.001, weight_decay=1e-5)

        self.distances = np.zeros((self.dist_deriv, self.N_RAYCASTS)) if self.dist_deriv is not None else None
        self.speeds = np.zeros((10,))
        self.auto_pilot: bool = False
        

    def nn_infer(self, message: SensingSnapshot):
        if self.dist_deriv is not None:
            message.raycast_speeds = np.diff(self.distances, axis=0).mean(axis=0)
            self.distances = np.roll(self.distances, -1, axis=0)
            self.distances[-1] = message.raycast_distances
        
        message.acc = np.diff(self.speeds).mean()
        self.speeds = np.roll(self.speeds, -1)
        self.speeds[-1] = message.car_speed

        X, _ = self.snapshot_to_tensor(message)
        y = self.nn(X)
        y = torch.sigmoid(y) >= 0.5
        return y

    def process_message(self, message: SensingSnapshot, data_collector: DataCollectionUI):
        commands = self.nn_infer(message)

        for command, start in commands:
            data_collector.onCarControlled(command, start)

    def extend_snapshots(self, snapshots: list[SensingSnapshot]):
        if self.dist_deriv:
            distances = np.zeros((self.dist_deriv, self.N_RAYCASTS))
            for snap in snapshots:
                snap.raycast_speeds = np.diff(distances, axis=0).mean(axis=0)
                distances = np.roll(distances, -1, axis=0)
                distances[-1] = snap.raycast_distances
        
        speeds = np.zeros(10)
        for snap in snapshots:
            snap.acc = np.diff(speeds).mean()
            speeds = np.roll(speeds, -1)
            speeds[-1] = snap.car_speed

    def snapshot_to_tensor(self, snapshot: SensingSnapshot) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = np.zeros((self.n_inputs,))
        i0 = 0
        i1 = i0 + self.N_RAYCASTS
        i2 = i1 + 2
        i3 = i2 + len(self.CONTROLS)
        inputs[i0:i1] = snapshot.raycast_distances
        inputs[i1] = snapshot.car_speed
        inputs[i1+1] = snapshot.acc
        inputs[i2:i3] = list(map(int, snapshot.current_controls))
        if self.dist_deriv is not None:
            inputs[i3:] = snapshot.raycast_speeds
        y = torch.tensor(snapshot.current_controls, dtype=torch.float32)#.reshape(-1, 1)
        X = torch.tensor(inputs, dtype=torch.float32)
        
        return X, y
    
    def record_to_tensor(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        with lzma.open(path, "rb") as file:
            snapshots = pickle.load(file)
            Xs = []
            ys = []
            self.extend_snapshots(snapshots)
            for s in snapshots:
                X, y = self.snapshot_to_tensor(s)
                Xs.append(X)
                ys.append(y)
            batch_X = torch.vstack(Xs)  # shape (N, 16)
            batch_y = torch.vstack(ys)  # shape (N, 4)
        return batch_X, batch_y
    
    @staticmethod
    def rolling_mean_2d(tensor, window_size, dim=0):
        unfolded = tensor.unfold(dimension=dim, size=window_size, step=1)
        return unfolded.mean(dim=-1)

    def train(self):
        Xs = []
        ys = []
        for filename in tqdm.tqdm(filter(lambda fn: os.path.splitext(fn)[1] == ".npz", os.listdir("data")), desc="Loading recording files"):
            Xi, yi = self.record_to_tensor("data/" + filename)
            Xs.append(Xi)
            ys.append(yi)
        
        X = torch.vstack(Xs)
        y = torch.vstack(ys)
        if self.augment_flipped:
            inputs2 = X.numpy()
            i0 = 0
            i1 = i0 + self.N_RAYCASTS
            i2 = i1 + 2
            inputs2[:, i0:i1] = inputs2[:, i0:i1][:, ::-1]
            inputs2[:, i2+2], inputs2[:, i2+3] = inputs2[:, i2+3], inputs2[:, i2+2]
            X2 = torch.tensor(inputs2, dtype=torch.float32)
            ctrl2 = y.numpy()
            ctrl2[:, 2:] = ctrl2[:, 2:][:, ::-1]
            y2 = torch.tensor(ctrl2, dtype=torch.float32)
            X, y = torch.vstack((X, X2)), torch.vstack((y, y2))
        
        if self.add_noise:
            X[:, :self.N_RAYCASTS] += np.random.random((X.shape[0], self.N_RAYCASTS)) * 2 - 1
            X[:, :self.N_RAYCASTS] = X[:, :self.N_RAYCASTS].clip(0, 100)
        
        #y = self.rolling_mean_2d(y, window_size=2)
        X = X[:-1]
        y = y[1:]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

        pos = y_train.mean(dim=0)
        neg = 1 - pos
        pos_weight = (neg / (pos + 1e-8)).clamp(1.0, 100.0)
        self.loss_fit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        n_epochs = 80
        batch_size = 16

        train_loss = []
        test_loss = []
        
        print("Training")
        for epoch in tqdm.tqdm(range(n_epochs)):
            batch_loss = []
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_pred = self.nn(X_batch)
                y_batch = y_train[i:i+batch_size]
                loss = self.loss_fit(y_pred, y_batch)
                batch_loss.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            mean_batch_loss = np.array(batch_loss).mean()
            train_loss.append(mean_batch_loss)
            with torch.no_grad():
                y_pred = self.nn(X_test)
                test_loss.append(self.loss_fit(y_pred, y_test).item())
            print(f"Finished epoch {epoch}, latest loss {mean_batch_loss}")
        
        epochs = list(range(n_epochs))
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, test_loss, label="Test")
        plt.savefig("learning.png")

        # compute accuracy (no_grad is optional)
        with torch.no_grad():
            y_pred = self.nn(X_test)
        accuracy = (y_pred.round() == y_test).float().mean()
        print(f"Accuracy {accuracy}")

        self.save_model("models/super_racer8.pt")

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.nn.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }, path)

    def load_model(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.nn.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            
    def update(self):
        if self.auto_pilot:
            message: SensingSnapshot = SensingSnapshot().from_car(self.car)
            output =  racer.nn_infer(message)
            self.car.keys["w"] = output[0]
            self.car.keys["s"] = output[1]
            self.car.keys["a"] = output[2]
            self.car.keys["d"] = output[3]
    
    def input(self, key):
        if key == "u":
            self.auto_pilot = not self.auto_pilot
            if self.auto_pilot:
                print("Auto-pilot enabled")
            else:
                print("Auto-pilot disabled")

if  __name__ == "__main__":
    import sys
    train = len(sys.argv) > 1 and sys.argv[1] == "train"
    racer = SuperRacer()
    if train:
        racer.train()
    else:
        racer.load_model("models/super_racer6.pt")
        def except_hook(cls, exception, traceback):
            sys.__excepthook__(cls, exception, traceback)
        sys.excepthook = except_hook

        app, car, track = prepare_game_app("SimpleTrack/track_metadata.json", True)
        racer.car = car
        recorder: Recorder = Recorder(car)
        recorder.enable()
        activated = False
        app.run()
            