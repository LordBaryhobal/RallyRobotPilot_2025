from rallyrobopilot import prepare_game_app
from rallyrobopilot.recorder import Recorder

app, car, track = prepare_game_app("SlightlyHarder/track_metadata.json", True)
recorder: Recorder = Recorder(car)
recorder.enable()
app.run()
