import json
import random

from rallyrobopilot.car import Car
from rallyrobopilot.checkpoint import Checkpoint
from rallyrobopilot.genetic_player import FrameInput, GeneticPlayer

from ursina import Ursina, held_keys, time


class GeneticManager:
    def __init__(
        self,
        app: Ursina,
        car: Car,
        checkpoint: Checkpoint,
        pop_size: int = 100,
        dna_length: int = 100,
        generations: int = 30,
        rounds: int = 5,
        passthrough_rate: float = 0.1,
        mutation_rate: float = 0.5,
        mutation_prob: float = 0.1,
    ):
        self.app: Ursina = app
        self.car: Car = car
        self.checkpoint: Checkpoint = checkpoint
        self.pop_size: int = pop_size
        self.dna_length: int = dna_length
        self.generations: int = generations
        self.rounds: int = rounds
        self.passthrough_rate: float = passthrough_rate
        self.mutation_rate: float = mutation_rate
        self.mutation_prob: float = mutation_prob
        self.population: list[GeneticPlayer] = self.generate_population()

    def execute(self):
        for gen in range(self.generations):
            print(f"Generation {gen + 1}/{self.generations}")
            for player in self.population:
                self.evaluate(player)
            tot_eval: float = sum(c.evaluation for c in self.population)
            print(f"Gen {gen} - average evaluation: {tot_eval / self.pop_size:.2f}")

            parents: list[GeneticPlayer] = self.select()
            children: list[GeneticPlayer] = self.crossover(parents)
            self.mutate(children)
            self.population = children

        for player in self.population:
            self.evaluate(player)

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

    def evaluate(self, player: GeneticPlayer):
        if player.evaluated:
            return

        # Simulate player and evaluate
        self.checkpoint.reset_car(self.car)
        self.start_pos = self.car.position
        step_till_gate: int = 0
        prev_pos = self.car.position
        reached_end: bool = False
        fps = 0
        for _ in range(self.dna_length):
            controls: FrameInput = player.infer()
            held_keys["w"] = bool(controls[0])
            held_keys["s"] = bool(controls[1])
            held_keys["a"] = bool(controls[2])
            held_keys["d"] = bool(controls[3])
            for _ in range(3):
                if self.checkpoint.intersects(prev_pos, self.car.position):
                    reached_end = True
                self.app.step()
                fps += 1 / time.dt
                if not reached_end:
                    step_till_gate += 1
                prev_pos = self.car.position
        self.end_pos = self.car.position
        dist: float = (self.end_pos - self.start_pos).length()
        player.set_evaluation(dist * step_till_gate)
        print(f"  Player {player.id}: {player.evaluation}")
        print(f"  Average FPS: {fps / self.dna_length / 3}")
