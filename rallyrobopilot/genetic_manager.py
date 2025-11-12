import json
import random

from matplotlib import pyplot as plt
from rallyrobopilot.sun import SunLight
from rallyrobopilot.trajectory import Trajectory
from ursina import Ursina
from ursina.vec2 import Vec2
from ursina.color import rgb

from rallyrobopilot.car import Car
from rallyrobopilot.checkpoint_manager import CheckpointManager
from rallyrobopilot.genetic_player import FrameInput, GeneticPlayer
from rallyrobopilot.track import Track
from rallyrobopilot.trajectory_segment import TrajectorySegment


class GeneticManager:
    def __init__(
        self,
        app: Ursina,
        track: Track,
        pop_size: int = 20,
        dna_length: int = 100,
        generations: int = 30,
        rounds: int = 3,
        passthrough_rate: float = 0.2,
        mutation_rate: float = 0.6,
        mutation_prob: float = 0.2,
    ):
        self.app: Ursina = app
        self.track: Track = track
        self.segment: TrajectorySegment = TrajectorySegment()
        self.pop_size: int = pop_size
        self.dna_length: int = dna_length
        self.generations: int = generations
        self.rounds: int = rounds
        self.passthrough_rate: float = passthrough_rate
        self.mutation_rate: float = mutation_rate
        self.mutation_prob: float = mutation_prob
        self.cars: list[Car] = [self.new_car() for _ in range(self.pop_size)]
        for car in self.cars:
            car.ignore_collisions = self.cars
        self.cars[0].camera_follow = True
        self.cars[0].change_camera = True
        self.sun = SunLight(direction = (-0.7, -0.9, 0.5), resolution = 3072, car = self.cars[0])
        self.population: list[GeneticPlayer] = self.generate_population()
        self.checkpoint_manager: CheckpointManager = CheckpointManager()
        self.ref_trajectory: Trajectory = Trajectory(color=rgb(210, 50, 50, 200))
        self.best_trajectory: Trajectory = Trajectory(color=rgb(50, 210, 50, 200))
        self.best_player: GeneticPlayer = GeneticPlayer(0, [])

    def new_car(self) -> Car:
        car = Car()
        car.sports_car()
        car.set_track(self.track)
        car.visible = True
        car.enable()
        return car

    def execute(self) -> GeneticPlayer:
        mean_evals: list[float] = []
        best_evals: list[float] = []
        
        for gen in range(self.generations):
            print(f"Generation {gen + 1}/{self.generations}")
            self.evaluate_all()
            best: GeneticPlayer = sorted(self.population, key=lambda p: p.evaluation)[0]
            if gen == 0 or best.evaluation < self.best_player.evaluation:
                self.best_player = best
                self.best_trajectory.set_pts(best.trajectory)
            evals: list[float] = [c.evaluation for c in self.population]
            tot_eval: float = sum(evals)
            mean_eval: float = tot_eval / self.pop_size
            min_eval: float = min(evals)
            mean_evals.append(mean_eval)
            best_evals.append(min_eval)
            print(f"Gen {gen} - avg={mean_eval:.2f} min={min_eval:.2f}")

            parents: list[GeneticPlayer] = self.select()
            children: list[GeneticPlayer] = self.crossover(parents)
            self.mutate(children)
            self.population = children

        self.evaluate_all()

        best: GeneticPlayer = sorted(self.population, key=lambda p: p.evaluation)[0]

        with open("best.json", "w") as f:
            json.dump(best.dna, f)

        plt.plot(range(self.generations), mean_evals, label="Mean")
        plt.plot(range(self.generations), best_evals, label="Best")
        plt.xlabel("Generation")
        plt.ylabel("Evaluation")
        plt.yscale("symlog")
        plt.title("Evaluation evolution")
        plt.legend()
        plt.savefig("evolution.png")
        
        return best

    def generate_population(self) -> list[GeneticPlayer]:
        return [GeneticPlayer.random(i, self.dna_length) for i in range(self.pop_size)]

    def select(self) -> list[GeneticPlayer]:
        parents: list[GeneticPlayer] = []

        for _ in range(self.pop_size):
            best: GeneticPlayer = random.choice(self.population)
            for _ in range(self.rounds):
                other: GeneticPlayer = random.choice(self.population)
                if other.evaluation < best.evaluation:
                    best = other
            parents.append(best)

        return parents

    def crossover(self, parents: list[GeneticPlayer]) -> list[GeneticPlayer]:
        children: list[GeneticPlayer] = []

        for i in range(0, self.pop_size, 2):
            p1: GeneticPlayer = parents[i]
            p2: GeneticPlayer = parents[i + 1]
            if random.random() < self.passthrough_rate:
                c1: GeneticPlayer = GeneticPlayer(i, p1.dna)
                c2: GeneticPlayer = GeneticPlayer(i + 1, p2.dna)
                c1.set_evaluation(p1.evaluation)
                c2.set_evaluation(p2.evaluation)
                children.extend([c1, c2])
                continue

            j: int = random.randint(1, self.dna_length - 2)

            dna1: list = p1.dna[:j] + p2.dna[j:]
            dna2: list = p2.dna[:j] + p1.dna[j:]
            children.append(GeneticPlayer(i, dna1))
            children.append(GeneticPlayer(i + 1, dna2))
        return children

    def mutate(self, children: list[GeneticPlayer]):
        for child in children:
            if random.random() >= self.mutation_rate:
                continue
            
            width: int = random.randint(1, 5)
            i: int = random.randint(0, self.dna_length - width)
            frame: FrameInput = child.random_frame()

            for j in range(width):
                child.dna[i + j] = frame

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
        
        for _ in range(1, self.dna_length):
            for player, car in zip(self.population, self.cars):
                player.infer(car, self.segment.checkpoint)
            self.app.step() # type: ignore

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
        return self.execute()

    def init_pop_from_segment(self, segment: TrajectorySegment):
        self.population = []
        self.dna_length = segment.length
        dna: list[FrameInput] = []
        for i in range(self.dna_length):
            dna.append(segment.frame_at(i))

        for i in range(self.pop_size):
            self.population.append(GeneticPlayer(i, dna.copy()))

        # self.mutate(self.population)
