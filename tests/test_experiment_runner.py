import numpy as np
import pytest
from benchmark.problem import Problem
from benchmark.experiment_runner import ExperimentRunner

class DummyProblem(Problem):
    def validate(self, candidate):
        return True

    def evaluate(self, candidate):
        return candidate[0] ** 2

    def random_solution(self, rng):
        return [rng.random()]

class DummyRandomSearch:
    def run(self, problem, evaluator, rng):
        for _ in range(evaluator.max_evaluations):
            candidate = problem.random_solution(rng)
            evaluator.evaluate(candidate)

class DummyOverBudgetSearch:
    def run(self, problem, evaluator, rng):
        for _ in range(evaluator.max_evaluations + 1):
            candidate = problem.random_solution(rng)
            evaluator.evaluate(candidate)

def test_same_seed_produces_same_random_sequence():
    rng_a = np.random.default_rng(202700)
    rng_b = np.random.default_rng(202700)

    sequence_a = rng_a.random(5)
    sequence_b = rng_b.random(5)

    assert np.array_equal(sequence_a, sequence_b)

def test_experiment_runner_is_reproducible():
    runner = ExperimentRunner(max_evaluations=3)

    result_a = runner.run(DummyRandomSearch(), DummyProblem(), seed=202700)
    result_b = runner.run(DummyRandomSearch(), DummyProblem(), seed=202700)

    assert result_a == result_b
    assert result_a.evaluations == result_b.evaluations == 3
    assert result_a.best_history == result_b.best_history
    assert len(result_a.best_history) == 3
    assert isinstance(result_a.best_history, tuple)

def test_experiment_runner_enforces_budget():
    runner = ExperimentRunner(max_evaluations=3)

    with pytest.raises(RuntimeError):
        runner.run(DummyOverBudgetSearch(), DummyProblem(), seed=202700)