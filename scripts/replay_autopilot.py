import datetime
import time
from PyQt6 import QtWidgets
from .data_collector import DataCollectionUI
import pickle
import lzma

FILENAME = "record_0.npz"
REPLAY_FPS = 30

class ReplayTrack:
    def __init__(self, filename):
        self.always_forward = True
        self.data = self.load_replay(filename)
        self.index = 0
        self.replay_start_time = None  # Temps de démarrage du replay
        self.replay_period = 1.0 / REPLAY_FPS
        self.previous_commands = [("forward", False), ("back", False), 
                                 ("left", False), ("right", False)]
        
    def load_replay(self, filename):
        print("Loading replay from", filename)
        with lzma.open(filename, "rb") as file:
            data = pickle.load(file)
        print(f"Loaded {len(data)} snapshots")
        return data

    def nn_infer(self, message):
        """
        Avance dans le replay en se basant sur le temps écoulé depuis le début
        """

        
        # Initialisation au premier appel
    
        # Récupérer le snapshot actuel
        snapshot = self.data[self.index]


        # Extraire les commandes du snapshot
        command = snapshot.current_controls
        commands = ["forward", "back", "left", "right"]
        
        # Générer la liste de changements de commandes
        new_commands = [(commands[i], bool(command[i])) for i in range(len(commands))]
        
        self.index += 1
        return new_commands
    
    def process_message(self, message, data_collector):
        """
        Traite les messages de sensing et applique les commandes du replay
        """
        commands = self.nn_infer(message)

        for command, start in commands:
            data_collector.onCarControlled(command, start)


if __name__ == "__main__":
    import sys
    
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = ReplayTrack(FILENAME)
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()