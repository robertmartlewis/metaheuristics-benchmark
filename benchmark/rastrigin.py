import math
from benchmark.continuous_problem import ContinuousProblem


class Rastrigin(ContinuousProblem):

    def evaluate(self, candidate):
        result = 10 * self.dimension

        for x in candidate:
            result += x ** 2 - 10 * math.cos(2 * math.pi * x)

        return result

    def random_solution(self, rng):
        return rng.uniform(
            self.lower_bound,
            self.upper_bound,
            size=self.dimension
        )