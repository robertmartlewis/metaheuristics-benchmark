import numpy as np

from benchmark.problem import Problem


class TSP(Problem):

    def __init__(self, cities):
        self.cities = cities

    def validate(self, candidate):
        return (
            len(candidate) == len(self.cities)
            and all(
                isinstance(x, (int, np.integer))
                and not isinstance(x, (bool, np.bool_))
                for x in candidate
            )
            and set(candidate) == set(range(len(self.cities)))
        )

    def evaluate(self, candidate):
        if not self.validate(candidate):
            raise ValueError("Invalid TSP tour")
        distance = 0.0
        for i in range(len(candidate)):
            current_index = candidate[i]
            next_index = candidate[(i + 1) % len(candidate)]
            distance += np.linalg.norm(self.cities[current_index] - self.cities[next_index])
        return distance


    def random_solution(self, rng):
        return rng.permutation(len(self.cities))

def create_tsp_instance():
    rng = np.random.default_rng(314159)
    cities = rng.uniform(0, 100, size=(20, 2))
    return TSP(cities)