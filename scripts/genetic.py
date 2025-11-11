from threading import Thread

from flask import Flask

from rallyrobopilot.checkpoint_manager import CheckpointManager
from rallyrobopilot.game_launcher import prepare_game_app
from rallyrobopilot.genetic_manager import GeneticManager


def main():

    flask_app = Flask(__name__)
    flask_thread = Thread(
        target=flask_app.run, kwargs={"host": "0.0.0.0", "port": 5000}
    )
    print("Flask server running on port 5000")
    flask_thread.start()

    app, _, track = prepare_game_app("SimpleTrack/track_metadata.json")

    cm: CheckpointManager = CheckpointManager()
    gm: GeneticManager = GeneticManager(app, track, cm.checkpoints[0])
    gm.execute()
    quit()


if __name__ == "__main__":
    main()
