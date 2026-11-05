class Problem:

    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def validate(self, candidate):
        if len(candidate) != self.dimension: return False

        for x in candidate:
            if not (self.lower_bound <= x <= self.upper_bound):
                return False

        return True

    def evaluate(self, candidate):
        pass

    def random_solution(self, rng):
        pass