import random
from threading import Thread

from flask import Flask

from rallyrobopilot.checkpoint_manager import CheckpointManager
from rallyrobopilot.game_launcher import prepare_game_app
from rallyrobopilot.genetic_manager import GeneticManager
from rallyrobopilot.genetic_player import GeneticPlayer
from rallyrobopilot.trajectory_optimizer import TrajectoryOptimizer
from rallyrobopilot.trajectory_point import TrajectoryPoint
from rallyrobopilot.trajectory_segment import TrajectorySegment


def main():

    flask_app = Flask(__name__)
    flask_thread = Thread(
        target=flask_app.run, kwargs={"host": "0.0.0.0", "port": 5000}
    )
    print("Flask server running on port 5000")
    flask_thread.start()

    app, _, track = prepare_game_app("SimpleTrack/track_metadata.json")

    cm: CheckpointManager = CheckpointManager()
    to: TrajectoryOptimizer = TrajectoryOptimizer("record_2.npz")
    #random.seed(11)
    gm: GeneticManager = GeneticManager(app, track)
    segment: TrajectorySegment = to.random_segment()
    traj_real: list[TrajectoryPoint] = [TrajectoryPoint.from_snapshot(s) for s in segment.snapshots][:-1]
    best: GeneticPlayer = gm.optimize(segment)
    traj_sim: list[TrajectoryPoint] = best.trajectory
    
    x = range(segment.length - 1)
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2)
    err00 = plt.twinx(axes[0][0])
    err01 = plt.twinx(axes[0][1])
    err10 = plt.twinx(axes[1][0])
    err11 = plt.twinx(axes[1][1])
    axes[0][0].plot(x, [tp.pos.x for tp in traj_real], label="Real pos (X)")
    axes[0][0].plot(x, [tp.pos.x for tp in traj_sim], label="Sim pos (X)")
    axes[0][0].set_title("Position (X)")
    err00.plot(x, [tpr.pos.x - tps.pos.x for tpr, tps in zip(traj_real, traj_sim)], label="Difference", linestyle="--", color="r")
    axes[0][1].plot(x, [tp.pos.y for tp in traj_real], label="Real pos (Y)")
    axes[0][1].plot(x, [tp.pos.y for tp in traj_sim], label="Sim pos (Y)")
    axes[0][1].set_title("Position (Y)")
    err01.plot(x, [tpr.pos.y - tps.pos.y for tpr, tps in zip(traj_real, traj_sim)], label="Difference", linestyle="--", color="r")
    axes[1][0].plot(x, [tp.speed for tp in traj_real], label="Real speed")
    axes[1][0].plot(x, [tp.speed for tp in traj_sim], label="Sim speed")
    axes[1][0].set_title("Speed")
    err10.plot(x, [tpr.speed - tps.speed for tpr, tps in zip(traj_real, traj_sim)], label="Difference", linestyle="--", color="r")
    axes[1][1].plot(x, [tp.angle for tp in traj_real], label="Real angle")
    axes[1][1].plot(x, [tp.angle for tp in traj_sim], label="Sim angle")
    axes[1][1].set_title("Angle")
    err11.plot(x, [tpr.angle - tps.angle for tpr, tps in zip(traj_real, traj_sim)], label="Difference", linestyle="--", color="r")
    
    axes[2][0].set_title("Controls")
    axes[2][1].set_title("Controls")
    for i, ctrl in enumerate(["Forward", "Backward", "Left", "Right"]):
        axes[2][0].plot(x, [s.current_controls[i] + 1.5 * i for s in segment.snapshots][1:], label=f"{ctrl} real")
        axes[2][0].plot(x, [f[i] + 1.5 * i + 0.1 for f in best.dna][1:], label=f"{ctrl} sim")
        axes[2][1].plot(x, [s.current_controls[i] + 1.5 * i for s in segment.snapshots][1:], label=f"{ctrl} real")
        axes[2][1].plot(x, [f[i] + 1.5 * i + 0.1 for f in best.dna][1:], label=f"{ctrl} sim")
    
    plt.legend()
    plt.show()
    
    #gm.optimize(to.random_segment())
    quit()


if __name__ == "__main__":
    main()
