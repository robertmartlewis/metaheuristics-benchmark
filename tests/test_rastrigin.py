import pytest
import numpy as np
from benchmark.rastrigin import Rastrigin
from benchmark.experiment_runner import ExperimentRunner


class DummyRandomSearch:
    def run(self, problem, evaluator, rng):
        for _ in range(evaluator.max_evaluations):
            candidate = problem.random_solution(rng)
            evaluator.evaluate(candidate)

def test_rastrigin_global_optimum():
    problem = Rastrigin(10, -5.12, 5.12)

    assert problem.evaluate([0.0] * 10) == pytest.approx(0.0)

def test_rastrigin_integer_candidate():
    problem = Rastrigin(10, -5.12, 5.12)
    assert problem.evaluate([1.0] + [0.0] * 9) == pytest.approx(1.0)

def test_rastrigin_fractional_candidate():
    problem = Rastrigin(10, -5.12, 5.12)
    assert problem.evaluate([0.5] + [0.0] * 9) == pytest.approx(20.25)

def test_rastrigin_random_solution_is_valid():
    problem = Rastrigin(10, -5.12, 5.12)
    rng = np.random.default_rng(202700)

    candidate = problem.random_solution(rng)

    assert len(candidate) == 10
    assert problem.validate(candidate)

def test_rastrigin_with_experiment_runner():
    problem = Rastrigin(10, -5.12, 5.12)
    runner = ExperimentRunner(max_evaluations=5)

    result_a = runner.run(DummyRandomSearch(), problem, seed=202700)
    result_b = runner.run(DummyRandomSearch(), problem, seed=202700)

    assert result_a == result_b
    assert result_a.evaluations == 5
    assert len(result_a.best_history) == 5
    assert result_a.best_fitness >= 0.0