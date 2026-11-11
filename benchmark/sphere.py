from benchmark.continuous_problem import ContinuousProblem


class Sphere(ContinuousProblem):

    def evaluate(self, candidate):
        result = 0.0

        for x in candidate:
            result += x ** 2

        return result

    def random_solution(self, rng):
        return rng.uniform(
            self.lower_bound,
            self.upper_bound,
            size=self.dimension
        )