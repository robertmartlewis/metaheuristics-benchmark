import math
import numpy as np

from benchmark.problem import Problem
from benchmark.tsp import TSP
from benchmark.continuous_problem import ContinuousProblem

class DifferentialEvolution:
    def __init__(self, population_size, differential_weight, crossover_rate):
        if not isinstance(population_size, int) or isinstance(population_size, bool):
            raise TypeError("population_size must be an integer")
        if population_size < 4:
            raise ValueError("population_size must be >= 4")
        self.population_size = population_size
        if not isinstance(differential_weight, (int, float)) or isinstance(differential_weight, bool):
            raise TypeError("differential_weight must be a real number")
        if not math.isfinite(differential_weight):
            raise ValueError("differential_weight must be finite")
        if not 0 < differential_weight <= 2:
            raise ValueError("differential_weight must be between 0 and 2")
        self.differential_weight = differential_weight
        if not isinstance(crossover_rate, (int, float)) or isinstance(crossover_rate, bool):
            raise TypeError("crossover_rate must be a real number")
        if not math.isfinite(crossover_rate):
            raise ValueError("crossover_rate must be finite")
        if not 0 <= crossover_rate <= 1:
            raise ValueError("crossover_rate must be between 0 and 1")
        self.crossover_rate = crossover_rate

    def _initialize_population(self, problem, evaluator, rng):
        if not isinstance(problem, ContinuousProblem):
            raise TypeError("problem must be ContinuousProblem")

        population = []
        fitness_values = []

        for _ in range(self.population_size):
            candidate = problem.random_solution(rng)
            fitness = evaluator.evaluate(candidate)
            population.append(candidate)
            fitness_values.append(fitness)
        return population, fitness_values

    def _select_donor_indices(self, target_index, rng):
        available_indices = [
            i for i in range(self.population_size)
            if i!= target_index
        ]
        return rng.choice(available_indices, size=3, replace=False)

    def _differential_mutation(self, population, donor_indices):
        r1, r2, r3 = donor_indices

        x1 = population[r1]
        x2 = population[r2]
        x3 = population[r3]

        return x1 + self.differential_weight * (x2 - x3)

    def _binomial_crossover(self, target, mutant, rng):
        dimension = len(target)
        mask = rng.random(dimension) < self.crossover_rate
        j_rand = rng.integers(dimension)
        mask[j_rand] = True
        return np.where(mask, mutant, target)

    def _repair_bounds(self, trial, problem):
        return np.clip(trial, problem.lower_bound, problem.upper_bound)

    def _select_survivor(self, target, target_fitness, trial, trial_fitness):
        if trial_fitness < target_fitness:
            return trial, trial_fitness

        return target, target_fitness

    def run(self, problem, evaluator, rng):
        if not isinstance(problem, ContinuousProblem):
            raise TypeError("problem must be ContinuousProblem")
        if evaluator.max_evaluations < self.population_size:
            raise ValueError("Evaluation budget is smaller than population_size")
        population, fitness_values = self._initialize_population(problem, evaluator, rng)

        while evaluator.evaluations < evaluator.max_evaluations:
            population, fitness_values = self._evolve_generation(population, fitness_values, problem, evaluator, rng)

    def _evolve_generation(self, population, fitness_values, problem, evaluator, rng):
        next_population = []
        next_fitness_values = []

        for i in range(self.population_size):
            target = population[i]
            target_fitness = fitness_values[i]
            if evaluator.evaluations >= evaluator.max_evaluations:
                next_population.append(target)
                next_fitness_values.append(target_fitness)
                continue
            donor_indices = self._select_donor_indices(i, rng)
            mutant = self._differential_mutation(population, donor_indices)
            trial = self._binomial_crossover(target, mutant, rng)
            repaired_trial = self._repair_bounds(trial, problem)
            trial_fitness = evaluator.evaluate(repaired_trial)
            survivor, survivor_fitness = self._select_survivor(target, target_fitness, repaired_trial, trial_fitness)
            next_population.append(survivor)
            next_fitness_values.append(survivor_fitness)

        return next_population, next_fitness_values