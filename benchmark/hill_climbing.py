import numpy as np

from benchmark.continuous_problem import ContinuousProblem
from benchmark.tsp import TSP

class HillClimbing:
    def __init__(self):
        pass

    def run(self, problem, evaluator, rng):
        if not isinstance(problem, (ContinuousProblem, TSP)):
            raise TypeError("problem must be Continuous or TSP")

        current = problem.random_solution(rng)
        current_fitness = evaluator.evaluate(current)

        while evaluator.evaluations < evaluator.max_evaluations:
            neighbor = self._generate_neighbor(current, problem, rng)
            neighbor_fitness = evaluator.evaluate(neighbor)

            if neighbor_fitness < current_fitness:
                current = neighbor
                current_fitness = neighbor_fitness

    def _gaussian_neighbor(self, current, problem, rng):
        sigma = 0.1*(problem.upper_bound - problem.lower_bound)

        neighbor = current + rng.normal(loc=0.0, scale=sigma, size=len(current))
        return np.clip(neighbor, problem.lower_bound, problem.upper_bound)

    def _generate_neighbor(self, current, problem, rng):
        if not isinstance(problem, ContinuousProblem) and not isinstance(problem, TSP):
            raise TypeError("problem must be Continuous or TSP")

        if isinstance(problem, ContinuousProblem):
            return self._gaussian_neighbor(current, problem, rng)
        if isinstance(problem, TSP):
            return self._swap_neighbor(current, rng)

    def _swap_neighbor(self, current, rng):
        neighbor = current.copy()
        a, b = rng.choice(len(neighbor), size=2, replace=False)
        neighbor[a], neighbor[b] = neighbor[b], neighbor[a]
        return neighbor
