import glob
import os
from pathlib import Path
from ursina.color import rgba

from rallyrobopilot.game_launcher import prepare_game_app
from rallyrobopilot.trajectory import Trajectory

TRACK = "SimpleTrack"
#TRACK = "SlightlyHarder"
#TRACK = "NotSoSimpleTrack"

app, car, track = prepare_game_app(f"{TRACK}/track_metadata.json", True)
if car is None:
    raise ValueError()

DIR = Path("generated")

for path in glob.glob(str(DIR / f"{TRACK}*")):
    bot_traj: Trajectory = Trajectory.from_recording(path)
    bot_traj.set_color(rgba(0, 255, 0, 200))
app.run()