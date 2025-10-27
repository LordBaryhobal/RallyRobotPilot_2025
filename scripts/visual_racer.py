import os
import pickle
from typing import Optional
import zipfile
from PyQt6 import QtWidgets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from rallyrobopilot.sensing_message import SensingSnapshot
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import pickle
import lzma

from .data_collector import DataCollectionUI

class VisualRacer:
    def __init__(self):
        self.always_forward = True
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 48, 3, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.25)
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 650,650)
            conv_out = self.conv_layers(dummy)
            conv_out_flat = conv_out.view(1, -1)
            conv_output_size = conv_out_flat.size(1)
            
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=conv_output_size, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=4),
        )
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(list(self.conv_layers.parameters()) + list(self.linear_layers.parameters()), lr=0.001)
        


    def forward(self, input):
        input = input.reshape(input.size(0), 3, 650, 650) 
        output = self.conv_layers(input)
        output = output.reshape(output.size(0), -1)
        output = self.linear_layers(output)
        return output
    
    
    def train(self):
        folder = "data"
        X = []
        y = []
        for filename in os.listdir(folder):
            if not filename.endswith(".npz"):
                continue
            with lzma.open(os.path.join(folder, filename), "rb") as file:
                data = pickle.load(file)
                for frame in data:
                    X.append(frame.image)
                    y.append(frame.current_controls)
                    
        X = torch.tensor(np.array(X)).float() / 255.0
        X = X.permute(0, 3, 1, 2)
        y = torch.tensor(np.array(y)).float()
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
        
        n_epochs = 80
        batch_size = 10

        train_loss = []
        test_loss = []
        
        print("Training")
        for epoch in tqdm.tqdm(range(n_epochs)):
            batch_loss = []
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_pred = self.forward(X_batch)
                y_batch = y_train[i:i+batch_size]
                loss = self.loss(y_pred, y_batch)
                batch_loss.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            mean_batch_loss = np.array(batch_loss).mean()
            train_loss.append(mean_batch_loss)
            with torch.no_grad():
                y_pred = self.model(X_test)
                test_loss.append(self.loss(y_pred, y_test).item())
            print(f"Finished epoch {epoch}, latest loss {mean_batch_loss}")
        
        epochs = list(range(n_epochs))
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, test_loss, label="Test")
        plt.savefig("learning.png")

        # compute accuracy (no_grad is optional)
        with torch.no_grad():
            y_pred = self.model(X_test)
        accuracy = (y_pred.round() == y_test).float().mean()
        print(f"Accuracy {accuracy}")

        self.save_model("models/caca.pt")

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }, path)
    
    def nn_infer(self, message):
        #   Do smart NN inference here
        image = message.image
        output = self.forward(image)
        command_list = ["forward", "back", "left", "right"]
        threshold = 0.5
        commands = [(command_list[i], output[0][i].item() > threshold) for i in range(4)]
        return commands


    def process_message(self, message, data_collector):

        commands = self.nn_infer(message)

        for command, start in commands:
            data_collector.onCarControlled(command, start)

if  __name__ == "__main__":
    import sys
    train = len(sys.argv) > 1 and sys.argv[1] == "train"

    if train:
        racer = VisualRacer()
        racer.train()
    else:
        def except_hook(cls, exception, traceback):
            sys.__excepthook__(cls, exception, traceback)
        sys.excepthook = except_hook

        app = QtWidgets.QApplication(sys.argv)

        nn_brain = VisualRacer()
        nn_brain.load_model("models/caca.pt")
        data_window = DataCollectionUI(nn_brain.process_message)
        data_window.show()

        app.exec()