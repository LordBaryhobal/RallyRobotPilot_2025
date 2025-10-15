import os
import pickle
import zipfile
from PyQt6 import QtWidgets
from sklearn.model_selection import train_test_split
import numpy as np
from rallyrobopilot.sensing_message import SensingSnapshot
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from .data_collector import DataCollectionUI


class SuperRacer:
    CONTROLS = ["forward", "back", "left", "right"]
    
    def __init__(self):
        self.always_forward = True
        self.model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        self.loss_fit = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)

    def nn_infer(self, message: SensingSnapshot):
        X, _ = self.snapshot_to_tensor(message)
        y = self.model(X)
        y = torch.sigmoid(y) >= 0.5
        cmds = []
        for ctrl, state in zip(self.CONTROLS, y):
            cmds.append((ctrl, state))
        return cmds

    def process_message(self, message: SensingSnapshot, data_collector: DataCollectionUI):
        commands = self.nn_infer(message)

        for command, start in commands:
            data_collector.onCarControlled(command, start)

    def snapshot_to_tensor(self, snapshot: SensingSnapshot) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = np.zeros((16,))
        inputs[:15] = snapshot.raycast_distances
        inputs[15] = snapshot.car_speed
        y = torch.tensor(snapshot.current_controls, dtype=torch.float32)#.reshape(-1, 1)
        X = torch.tensor(inputs, dtype=torch.float32)
        return X, y
    
    def record_to_tensor(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        with zipfile.ZipFile(path) as f:
            snapshots = pickle.loads(f.read("data.pkl"))
            Xs = []
            ys = []
            for s in snapshots:
                X, y = self.snapshot_to_tensor(s)
                Xs.append(X)
                ys.append(y)
            batch_X = torch.vstack(Xs)  # shape (N, 16)
            batch_y = torch.vstack(ys)  # shape (N, 4)
        return batch_X, batch_y
    
    def train(self):
        Xs = []
        ys = []
        for filename in tqdm.tqdm(os.listdir("data"), desc="Loading recording files"):
            Xi, yi = self.record_to_tensor("data/" + filename)
            print(Xi.shape, yi.shape)
            Xs.append(Xi)
            ys.append(yi)
        
        X = torch.vstack(Xs)
        y = torch.vstack(ys)
        print(X.shape, y.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
        
        n_epochs = 100
        batch_size = 10
        
        print("Training")
        for epoch in tqdm.tqdm(range(n_epochs)):
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_pred = self.model(X_batch)
                y_batch = y_train[i:i+batch_size]
                loss = self.loss_fit(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Finished epoch {epoch}, latest loss {loss}")
        
        # compute accuracy (no_grad is optional)
        with torch.no_grad():
            y_pred = self.model(X_test)
        accuracy = (y_pred.round() == y_test).float().mean()
        print(f"Accuracy {accuracy}")

        self.save_model("models/super_racer.pt")

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }, path)

    def load_model(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

if  __name__ == "__main__":
    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = SuperRacer()
    nn_brain.load_model("models/super_racer.pt")
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()

    #racer = SuperRacer()
    #racer.train()
