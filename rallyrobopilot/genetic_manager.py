import json
import random

from rallyrobopilot.car import Car
from rallyrobopilot.checkpoint import Checkpoint
from rallyrobopilot.genetic_player import FrameInput, GeneticPlayer

from ursina import Ursina, held_keys, time, Vec3

from rallyrobopilot.track import Track


class GeneticManager:
    def __init__(
        self,
        app: Ursina,
        track: Track,
        checkpoint: Checkpoint,
        pop_size: int = 20,
        dna_length: int = 100,
        generations: int = 30,
        rounds: int = 5,
        passthrough_rate: float = 0.1,
        mutation_rate: float = 0.6,
        mutation_prob: float = 0.3,
    ):
        self.app: Ursina = app
        self.track: Track = track
        self.checkpoint: Checkpoint = checkpoint
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
        self.population: list[GeneticPlayer] = self.generate_population()

    def new_car(self) -> Car:
        car = Car()
        car.sports_car()
        car.set_track(self.track)
        car.visible = True
        car.enable()
        return car

    def execute(self):
        for gen in range(self.generations):
            print(f"Generation {gen + 1}/{self.generations}")
            self.evaluate_all()
            tot_eval: float = sum(c.evaluation for c in self.population)
            print(f"Gen {gen} - average evaluation: {tot_eval / self.pop_size:.2f}")

            parents: list[GeneticPlayer] = self.select()
            children: list[GeneticPlayer] = self.crossover(parents)
            self.mutate(children)
            self.population = children

        self.evaluate_all()

        best: GeneticPlayer = sorted(self.population, key=lambda p: p.evaluation)[0]

        with open("best.json", "w") as f:
            json.dump(best.dna, f)

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

            for i in range(self.dna_length):
                if random.random() < self.mutation_prob:
                    child.dna[i] = child.random_frame()

    def reset_cars(self):
        for car in self.cars:
            self.checkpoint.reset_car(car)

    def evaluate_all(self):
        if self.checkpoint.end is None:
            return
        for player, car in zip(self.population, self.cars):
            self.checkpoint.reset_car(car)
            player.prev_pos = car.position
        self.start_pos = self.cars[0].position
        
        for _ in range(self.dna_length):
            for player, car in zip(self.population, self.cars):
                player.infer(car, self.checkpoint)
            self.app.step()
        
        for player, car in zip(self.population, self.cars):
            end_pos: Vec3 = car.position
            start_dist: float = (end_pos - self.start_pos).length()
            end_dist1: float = (self.checkpoint.end[0] - self.start_pos.xz).length()
            end_dist2: float = (self.checkpoint.end[1] - self.start_pos.xz).length()
            end_dist: float = (end_dist1 + end_dist2) / 2
            player.set_evaluation(player.step_till_gate / start_dist * end_dist * (player.wall_hits + 1))
            print(f"  Player {player.id}: {player.evaluation}")
