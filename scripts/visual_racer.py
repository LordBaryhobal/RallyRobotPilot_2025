import torch
import torch.nn as nn
import torch.optim as optim
import lzma
import os
import pickle
from threading import Thread
from data_images.image_scaler import rescale
from ursina import Entity

import matplotlib.pyplot as plt
import numpy as np

import tqdm
from PyQt6 import QtWidgets
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split,DataLoader

from rallyrobopilot.game_launcher import prepare_game_app
from rallyrobopilot.recorder import Recorder
from rallyrobopilot.sensing_message import SensingSnapshot

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("No GPU available. Using CPU.")

NUMBER_LAST_IMAGES = 3
INPUT_WIDTH = 128
INPUT_HEIGHT = 128

class DriverDataset(torch.utils.data.Dataset):
    def __init__(self,folder,rdf=True):
        self.random_data_flip = rdf
        X = []
        self.Y = []
        for filename in os.listdir(folder):
            if not filename.endswith(".npz"):
                continue
            filepath = os.path.join(folder, filename)

            with lzma.open(filepath, "rb") as file:
                data = pickle.load(file)
                for i, frame in enumerate(data):
                    last_images = []
                    for i1 in range(i - NUMBER_LAST_IMAGES, i):
                        if i1 >= 0:
                            last_images.append(data[i1].image)
                        else:
                            last_images.append(data[i].image)
                    all_images_stacked = np.stack(last_images + [frame.image], axis=0)
                    frame_tensor = torch.from_numpy(all_images_stacked).float() / 255.0
                    X.append(frame_tensor)
                    self.Y.append(frame.current_controls)
                    # Data augmentation (add flipped last 5 images )
                    if self.random_data_flip and i % 10 == 0:
                        for i in range(len(last_images)):
                            last_images[i] = np.flipud(last_images[i])
                        all_images_stacked = np.stack(last_images + [np.flipud(frame.image)], axis=0)
                        frame_tensor_flipped = torch.from_numpy(all_images_stacked).float() / 255.0
                        X.append(frame_tensor)
                        self.Y.append(frame.current_controls)
                print("computed file",filename)
        
        self.X = torch.stack(X).to(device)
        self.Y = torch.Tensor(np.array(self.Y)).float().to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx],self.Y[idx]





class VisualRacer(Entity):
    def __init__(self):
        super().__init__()
        self.always_forward = True
        self.auto_pilot = False
        self.car = None
        self.conv_layers = nn.Sequential(
            nn.Conv2d((NUMBER_LAST_IMAGES+1)*3, 24, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 48, 3, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.25)
        ).to(device)
        self.last_images = []
        
        with torch.no_grad():
            dummy = torch.zeros(1,(NUMBER_LAST_IMAGES+1)*3, INPUT_WIDTH, INPUT_HEIGHT).to(device)
            conv_out = self.conv_layers(dummy)
            conv_out_flat = conv_out.view(1, -1)
            conv_output_size = conv_out_flat.size(1)
            
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=conv_output_size, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=4),
        ).to(device)
        self.loss = nn.BCEWithLogitsLoss().to(device)
        self.optimizer = optim.AdamW(list(self.conv_layers.parameters()) + list(self.linear_layers.parameters()), lr=0.001)
        


    def forward(self, input: torch.cuda.FloatTensor):
        input = input.view(input.size(0), (NUMBER_LAST_IMAGES+1)*3, INPUT_WIDTH, INPUT_HEIGHT).to(device)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output
    
    def train(self):
        print("starting train")
        folder = "data_images_reduced"
        full_dataset = DriverDataset(folder,True)
    
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        n_epochs = 80
        batch_size = 8

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_loss = []
        test_loss = []
        
        print("Training")
        for epoch in tqdm.tqdm(range(n_epochs)):
            batch_loss = []
            self.linear_layers.train()
            self.conv_layers.train()
            for X_batch, y_batch in train_dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = self.forward(X_batch)
                loss = self.loss(y_pred, y_batch)
                batch_loss.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            mean_batch_loss = np.array(batch_loss).mean()
            train_loss.append(mean_batch_loss)
            self.conv_layers.eval()
            self.linear_layers.eval()
            test_batch_loss = []
            with torch.no_grad():
                for X_test_batch, y_test_batch in test_dataloader:
                    X_test_batch = X_test_batch.to(device)
                    y_test_batch = y_test_batch.to(device)
                    
                    y_pred_test = self.forward(X_test_batch)
                    test_batch_loss.append(self.loss(y_pred_test, y_test_batch).item())

                mean_test_loss = np.array(test_batch_loss).mean()
                test_loss.append(mean_test_loss)
            print(f"Finished epoch {epoch}, latest loss {mean_batch_loss}")
        
        epochs = list(range(n_epochs))
        #plt.gca().clear()
        #plt.plot(epochs, train_loss, label="Train")
        #plt.plot(epochs, test_loss, label="Test")
        #plt.savefig("learning.png")

        
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for X_test_batch, y_test_batch in test_dataloader:
                X_test_batch = X_test_batch.to(device)
                y_test_batch = y_test_batch.to(device)
                
                y_pred_test = self.forward(X_test_batch)
                correct = (y_pred_test.round() == y_test_batch).float().sum()
                total_correct += correct
                total_samples += y_test_batch.numel()

        accuracy = total_correct / total_samples
        print(f"Accuracy {accuracy:.4f}")

        self.save_model("models/visual_model8.pt")


    def update(self):
        if self.auto_pilot:
            message: SensingSnapshot = SensingSnapshot().from_car(self.car)
            output =  self.nn_infer(message)
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

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "conv_state": self.conv_layers.state_dict(),
            "linear_state": self.linear_layers.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }, path)

    def load_model(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.conv_layers.load_state_dict(checkpoint["conv_state"])
        self.linear_layers.load_state_dict(checkpoint["linear_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    def nn_infer(self, message):
        # pass the last x images with the current image
        #   Do smart NN inference here
        if len(self.last_images)==0 :
            while len(self.last_images) != NUMBER_LAST_IMAGES:
                self.last_images.append(rescale(message.image))
        
        all_images_stacked = np.stack(self.last_images + [rescale(message.image)], axis=0)
        frame_tensor = torch.from_numpy(all_images_stacked).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0) 

        print(".:.",frame_tensor.shape)
        output = self.forward(frame_tensor)
        threshold = 0.5
        commands = [(output[0][i].item() > threshold) for i in range(4)]
        self.last_images.append(rescale(message.image))
        self.last_images.pop(0)
        return commands


if  __name__ == "__main__":
    import sys
    train = len(sys.argv) > 1 and sys.argv[1] == "train"
    racer = VisualRacer()
    if train:
        racer.train()
    else:
        
        def except_hook(cls, exception, traceback):
            sys.__excepthook__(cls, exception, traceback)
        sys.excepthook = except_hook

        app = QtWidgets.QApplication(sys.argv)

        racer.load_model("models/visual_model8.pt")

        app, car, track = prepare_game_app("SimpleTrack/track_metadata.json", True)
        racer.car = car
        app.run()