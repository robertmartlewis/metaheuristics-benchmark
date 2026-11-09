import math


class Evaluator:
    def __init__(self, problem, max_evaluations):
        if type(max_evaluations) is not int:
            raise TypeError("max_evaluations must be an integer")

        if max_evaluations <= 0:
            raise ValueError("Evaluation budget must be positive")

        self.problem = problem
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.best_fitness = float("inf")
        self.best_history = []

    def evaluate(self, candidate):
        if self.evaluations >= self.max_evaluations:
            raise RuntimeError("Evaluation budget exhausted")

        result = self.problem.evaluate(candidate)
        self.evaluations += 1

        if not math.isfinite(result):
            raise ValueError("Fitness must be finite")

        if result < self.best_fitness:
            self.best_fitness = result

        self.best_history.append(self.best_fitness)

        return result