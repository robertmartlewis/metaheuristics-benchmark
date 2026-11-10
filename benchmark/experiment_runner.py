import numpy as np

from benchmark.evaluator import Evaluator
from benchmark.run_result import RunResult

class ExperimentRunner:
    def __init__(self, max_evaluations):
        self.max_evaluations = max_evaluations

    def run(self, algorithm, problem, seed):
        rng = np.random.default_rng(seed)
        evaluator = Evaluator(problem, self.max_evaluations)

        algorithm.run(problem, evaluator, rng)

        result = RunResult(
            algorithm=type(algorithm).__name__,
            problem=type(problem).__name__,
            seed=seed,
            best_fitness=evaluator.best_fitness,
            evaluations=evaluator.evaluations,
            best_history=tuple(evaluator.best_history)
        )

        return result