from rallyrobopilot import prepare_game_app, RemoteController
from flask import Flask
from threading import Thread

from rallyrobopilot.recorder import Recorder


# Setup Flask

flask_app = Flask(__name__)
flask_thread = Thread(target=flask_app.run, kwargs={'host': "0.0.0.0", 'port': 5000})
print("Flask server running on port 5000")
flask_thread.start()

app, car, track = prepare_game_app("VisualTrack/track_metadata.json", True)
recorder: Recorder = Recorder(car)
recorder.enable()
app.run()