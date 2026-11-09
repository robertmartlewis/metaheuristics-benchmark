class Evaluator:
    def __init__(self, problem, max_evaluations):
        if not type(max_evaluations) is int:
            raise TypeError("max_evaluations must be an integer")
        if max_evaluations <= 0:
            raise ValueError("Evaluation budget must be positive")
        self.problem = problem
        self.max_evaluations = max_evaluations
        self.evaluations = 0

    def evaluate(self, candidate):
        if self.evaluations < self.max_evaluations:
            result = self.problem.evaluate(candidate)
            self.evaluations += 1
            return result
        raise RuntimeError("Evaluation budget exhausted")