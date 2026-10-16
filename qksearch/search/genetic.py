from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from qksearch.gates import GateOp, State
from qksearch.genome import (
    Genome,
    GenomeSpec,
    decode,
    l1_distance,
    random_genome,
    repair,
)

from .base import Heuristic, SearchStrategy


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    best: float
    mean: float
    diversity: float
    n_evaluated: int


class GeneticSearch(SearchStrategy[State, GateOp]):
    path_following = False

    def __init__(self, spec: GenomeSpec, *, population: Sequence[Genome],
                 n_generations: int = 50, tournament: int = 3,
                 p_crossover: float = 0.8, p_mutate: Optional[float] = None,
                 p_step: float = 0.5, elitism: int = 2,
                 max_support: int = 6, max_weight: Optional[int] = None,
                 patience: Optional[int] = None,
                 seed: Optional[int] = None) -> None:
        super().__init__(record_complete_only=False)
        if not population:
            raise ValueError("the initial population is empty")
        if n_generations < 1:
            raise ValueError(f"n_generations must be >= 1, got {n_generations}")
        if not 2 <= tournament <= len(population):
            raise ValueError(
                f"tournament must lie in [2, {len(population)}], got {tournament}")
        if not 0.0 <= p_crossover <= 1.0:
            raise ValueError(f"p_crossover must lie in [0, 1], got {p_crossover}")
        if not 0.0 <= p_step <= 1.0:
            raise ValueError(f"p_step must lie in [0, 1], got {p_step}")
        if not 0 <= elitism < len(population):
            raise ValueError(
                f"elitism must lie in [0, {len(population) - 1}], got {elitism}")
        if any(len(g) != len(spec.genes) for g in population):
            raise ValueError("a genome of the population does not match the spec")
        self.spec = spec
        self.population = tuple(population)
        self.n_generations = n_generations
        self.tournament = tournament
        self.p_crossover = p_crossover
        self.p_mutate = 1.0 / len(spec.genes) if p_mutate is None else p_mutate
        self.p_step = p_step
        self.elitism = elitism
        self.max_support = max_support
        self.max_weight = max_weight
        self.patience = patience
        self.seed = seed
        self._history: list[GenerationStats] = []
        self._best_genome: Optional[Genome] = None
        self._fitness: dict[Genome, float] = {}

    @property
    def history(self) -> list[GenerationStats]:
        return self._history

    @property
    def best_genome(self) -> Optional[Genome]:
        return self._best_genome

    def _reset(self) -> None:
        super()._reset()
        self._history = []
        self._best_genome = None
        self._fitness: dict[Genome, float] = {}
        self._rng = np.random.default_rng(self.seed)


    def _select(self, fitness: np.ndarray) -> int:
        picks = self._rng.integers(0, len(fitness), size=self.tournament)
        return int(picks[int(np.argmax(fitness[picks]))])

    def _crossover(self, a: Genome, b: Genome) -> Genome:
        take = self._rng.random(len(a)) < 0.5
        return tuple(int(x if t else y) for x, y, t in zip(a, b, take))

    def _mutate(self, g: Genome) -> Genome:
        out = list(g)
        for i, gene in enumerate(self.spec.genes):
            if self._rng.random() >= self.p_mutate:
                continue
            if self._rng.random() < self.p_step:
                step = 1 if self._rng.random() < 0.5 else -1
                out[i] = int(np.clip(out[i] + step, 0, gene.cap))
            else:
                out[i] = int(self._rng.integers(0, gene.cap + 1))
        return tuple(out)

    def _breed(self, pop: Sequence[Genome], fitness: np.ndarray) -> Genome:
        parent = pop[self._select(fitness)]
        if self._rng.random() < self.p_crossover:
            child = self._crossover(parent, pop[self._select(fitness)])
        else:
            child = parent
        child = self._mutate(child)
        return repair(child, self.spec, self._rng,
                      max_support=self.max_support, max_weight=self.max_weight)

    def _replace(self, pop: list[Genome], fitness: np.ndarray,
                 children: Sequence[Genome], child_fitness: Sequence[float]) -> None:
        elite = set(np.argsort(fitness)[len(fitness) - self.elitism:].tolist()) \
            if self.elitism else set()
        for child, score in zip(children, child_fitness):
            order = np.argsort([l1_distance(child, p) for p in pop])
            for j in order:
                j = int(j)
                if j in elite:
                    continue
                if score > fitness[j]:
                    pop[j], fitness[j] = child, score
                break


    def _evaluate(self, heuristic: Heuristic[State, GateOp],
                  genomes: Sequence[Genome]) -> np.ndarray:
        unknown = [g for g in dict.fromkeys(genomes) if g not in self._fitness]
        if unknown:
            states = [decode(g, self.spec) for g in unknown]
            scores = self._score_many(heuristic, states)
            self._fitness.update(zip(unknown, (float(s) for s in scores)))
        return np.array([self._fitness[g] for g in genomes], dtype=float)

    def _diversity(self, pop: Sequence[Genome]) -> float:
        if len(pop) < 2:
            return 0.0
        total = sum(l1_distance(a, b)
                    for i, a in enumerate(pop) for b in pop[i + 1:])
        return 2.0 * total / (len(pop) * (len(pop) - 1))

    def search(self, initial, heuristic, *, budget=None):
        self._reset()
        pop = list(self.population)
        fitness = self._evaluate(heuristic, pop)
        for g, score in zip(pop, fitness):
            self._keep(heuristic, g, float(score))

        stall = 0
        for generation in range(1, self.n_generations + 1):
            if not self._budget_left(budget):
                break
            self._n_expanded += 1
            children = [self._breed(pop, fitness)
                        for _ in range(len(pop) - self.elitism)]
            child_fitness = self._evaluate(heuristic, children)
            for g, score in zip(children, child_fitness):
                self._keep(heuristic, g, float(score))

            before = self._best_score
            self._replace(pop, fitness, children, child_fitness)
            self._history.append(GenerationStats(
                generation=generation,
                best=float(fitness.max()),
                mean=float(fitness.mean()),
                diversity=self._diversity(pop),
                n_evaluated=self._n_evaluated,
            ))

            if self.patience is not None:
                stall = 0 if self._best_score > before else stall + 1
                if stall >= self.patience:
                    break
        return self._result()

    def _keep(self, heuristic: Heuristic[State, GateOp],
              genome: Genome, score: float) -> None:
        if score > self._best_score:
            self._best_genome = genome
        self._record(heuristic, decode(genome, self.spec), score)


def seed_population(spec: GenomeSpec, size: int, rng: np.random.Generator, *,
                    seeds: Sequence[Genome] = (), max_support: int = 6,
                    max_weight: Optional[int] = None) -> list[Genome]:
    pop = [tuple(g) for g in seeds][:size]
    while len(pop) < size:
        pop.append(random_genome(spec, rng, max_support=max_support,
                                 max_weight=max_weight))
    return pop
