import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from ursina import Ursina
from ursina.color import rgb
from ursina.vec2 import Vec2

from rallyrobopilot.car import Car
from rallyrobopilot.checkpoint_manager import CheckpointManager
from rallyrobopilot.generation_stats import GenerationStats
from rallyrobopilot.genetic_player import FrameInput, GeneticPlayer
from rallyrobopilot.genetic_settings import GeneticSettings
from rallyrobopilot.sun import SunLight
from rallyrobopilot.track import Track
from rallyrobopilot.trajectory import Trajectory
from rallyrobopilot.trajectory_segment import TrajectorySegment


class GeneticManager:
    def __init__(
        self, app: Ursina, track: Track, out_path: Path, settings: GeneticSettings
    ):
        self.app: Ursina = app
        self.track: Track = track
        self.segment: TrajectorySegment = TrajectorySegment()
        self.settings: GeneticSettings = settings
        self.cars: list[Car] = [self.new_car() for _ in range(self.settings.pop_size)]
        for car in self.cars:
            car.ignore_collisions = self.cars
        self.cars[0].camera_follow = True
        self.cars[0].change_camera = True
        self.sun = SunLight(
            direction=(-0.7, -0.9, 0.5), resolution=3072, car=self.cars[0]
        )
        self.population: list[GeneticPlayer] = self.generate_population()
        self.checkpoint_manager: CheckpointManager = CheckpointManager()
        self.ref_trajectory: Trajectory = Trajectory(color=rgb(210, 50, 50, 200))
        self.best_trajectory: Trajectory = Trajectory(color=rgb(50, 210, 50, 200))
        self.best_player: GeneticPlayer = GeneticPlayer(0, [])
        self.stats: list[GenerationStats] = []
        self.out_path: Path = out_path
        self.out_path.mkdir(parents=True, exist_ok=True)

    def new_car(self) -> Car:
        car = Car()
        car.sports_car()
        car.set_track(self.track)
        car.visible = True
        car.enable()
        return car

    def execute(self, plots: bool = False) -> GeneticPlayer:
        mean_evals: list[float] = []
        best_evals: list[float] = []
        best_steps: list[int] = []

        with tqdm.trange(self.settings.generations, desc="Generation", unit="gen") as p:
            for gen in p:
                self.evaluate_all()
                best: GeneticPlayer = sorted(
                    self.population, key=lambda p: p.evaluation
                )[0]
                best_steps.append(best.steps_till_gate)
                if gen == 0 or best.evaluation < self.best_player.evaluation:
                    self.best_player = best
                    self.best_trajectory.set_pts(best.trajectory)
                evals: np.ndarray = np.array([c.evaluation for c in self.population])
                steps: np.ndarray = np.array(
                    [c.steps_till_gate for c in self.population]
                )
                mean_eval: float = float(evals.mean())
                min_eval: float = float(evals.min())
                mean_evals.append(mean_eval)
                best_evals.append(min_eval)
                p.set_postfix_str(f"avg={mean_eval:.2f} min={min_eval:.2f}")
                median_eval: float = float(np.median(evals))
                max_eval: float = float(evals.max())
                min_steps: int = int(steps.min())
                mean_steps: float = float(steps.mean())
                median_steps: int = int(np.median(steps))
                self.stats.append(
                    GenerationStats(
                        gen,
                        self.settings.pop_size,
                        self.settings.dna_length,
                        min_eval,
                        mean_eval,
                        median_eval,
                        max_eval,
                        min_steps,
                        mean_steps,
                        median_steps,
                    )
                )

                parents: list[GeneticPlayer] = self.select()
                children: list[GeneticPlayer] = self.crossover(parents)
                self.mutate(children)
                self.population = children

        self.evaluate_all()

        best: GeneticPlayer = sorted(self.population, key=lambda p: p.evaluation)[0]

        with open("best.json", "w") as f:
            json.dump(best.dna, f)

        if plots:
            fig, axes = plt.subplots(1, 2)
            axes[0].plot(range(self.settings.generations), mean_evals, label="Mean")
            axes[0].plot(range(self.settings.generations), best_evals, label="Best")
            axes[0].set_xlabel("Generation")
            axes[0].set_ylabel("Evaluation")
            axes[0].set_yscale("symlog")
            axes[0].set_title("Evaluation evolution")
            axes[1].plot(
                range(self.settings.generations), best_steps, label="Steps until gate"
            )
            axes[1].set_xlabel("Generation")
            axes[1].set_ylabel("Steps")
            plt.legend()
            plt.savefig("evolution.png")

        return best

    def generate_population(self) -> list[GeneticPlayer]:
        return [
            GeneticPlayer.random(i, self.settings.dna_length)
            for i in range(self.settings.pop_size)
        ]

    def select(self) -> list[GeneticPlayer]:
        parents: list[GeneticPlayer] = []

        for _ in range(self.settings.pop_size):
            best: GeneticPlayer = random.choice(self.population)
            for _ in range(self.settings.rounds):
                other: GeneticPlayer = random.choice(self.population)
                if other.evaluation < best.evaluation:
                    best = other
            parents.append(best)

        return parents

    def crossover(self, parents: list[GeneticPlayer]) -> list[GeneticPlayer]:
        children: list[GeneticPlayer] = []

        for i in range(0, self.settings.pop_size, 2):
            p1: GeneticPlayer = parents[i]
            p2: GeneticPlayer = parents[i + 1]
            if random.random() < self.settings.passthrough_rate:
                c1: GeneticPlayer = GeneticPlayer(i, p1.dna)
                c2: GeneticPlayer = GeneticPlayer(i + 1, p2.dna)
                c1.set_evaluation(p1.evaluation)
                c2.set_evaluation(p2.evaluation)
                children.extend([c1, c2])
                continue

            j: int = random.randint(1, self.settings.dna_length - 2)

            dna1: list = p1.dna[:j] + p2.dna[j:]
            dna2: list = p2.dna[:j] + p1.dna[j:]
            children.append(GeneticPlayer(i, dna1))
            children.append(GeneticPlayer(i + 1, dna2))
        return children

    def mutate(self, children: list[GeneticPlayer]):
        for child in children:
            if random.random() >= self.settings.mutation_rate:
                continue
            self.settings.mutation_strategy.mutate(child.dna, self.settings.mutation_prob, self.settings.flip_prob)

    def reset_cars(self):
        for car in self.cars:
            self.segment.checkpoint.reset_car(car)

    def evaluate_all(self):
        if self.segment.checkpoint.end is None:
            return
        for player, car in zip(self.population, self.cars):
            self.segment.checkpoint.reset_car(car)
            player.prev_pos = car.position
        start_pos: Vec2 = self.cars[0].position.xz

        for _ in range(1, self.settings.dna_length):
            for player, car in zip(self.population, self.cars):
                player.infer(car, self.segment.checkpoint)
            self.app.step()  # type: ignore

        for player, car in zip(self.population, self.cars):
            end_pos: Vec2 = car.position.xz
            start_dist: float = (end_pos - start_pos).length()
            end_dist: float = self.segment.checkpoint.distance(end_pos)
            max_dist: float = self.segment.checkpoint.distance(start_pos)

            if player.reached_end:
                end_dist = 0
            else:
                player.steps_till_gate += 1

            d: float = end_dist / max_dist
            t: float = player.steps_till_gate / self.segment.length

            if player.wall_hits > 0:
                fitness = 1000 + player.wall_hits * 100
            elif player.reached_end:
                fitness = t
            else:
                fitness = 500 * d
            player.set_evaluation(fitness)
            print(
                f"  Player {player.id:3d}: {player.evaluation:9.3f} (wall_hits={player.wall_hits}, {start_dist=:6.2f}, {end_dist=:6.2f}, steps_till_gate={player.steps_till_gate:3d})"
            )

    def optimize(self, segment: TrajectorySegment) -> GeneticPlayer:
        self.segment = segment
        self.checkpoint_manager.add_entity(self.segment.checkpoint)
        self.ref_trajectory.set_pts(self.segment.trajectory)
        self.init_pop_from_segment(self.segment)
        best: GeneticPlayer = self.execute()
        self.save_stats()
        return best

    def init_pop_from_segment(self, segment: TrajectorySegment):
        self.population = []
        self.settings.dna_length = segment.length
        dna: list[FrameInput] = []
        for i in range(self.settings.dna_length):
            dna.append(segment.frame_at(i))

        for i in range(self.settings.pop_size):
            self.population.append(GeneticPlayer(i, dna.copy()))

    def save_stats(self):
        df: pd.DataFrame = pd.DataFrame([asdict(gen) for gen in self.stats])
        df.to_csv(self.out_path / "stats.csv", index=False)

        with open(self.out_path / "settings.json", "w") as f:
            json.dump(asdict(self.settings), f, indent=4)
