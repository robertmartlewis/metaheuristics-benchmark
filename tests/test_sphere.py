import numpy as np
from benchmark.sphere import Sphere
from benchmark.experiment_runner import ExperimentRunner

class DummyRandomSearch:
    def run(self, problem, evaluator, rng):
        for _ in range(evaluator.max_evaluations):
            candidate = problem.random_solution(rng)
            evaluator.evaluate(candidate)

def test_sphere_global_optimum():
    sphere = Sphere(10, -5.12, 5.12)

    candidate = [0.0] * 10

    assert sphere.evaluate(candidate) == 0.0

def test_sphere_evaluates_sum_of_squares():
    sphere = Sphere(10, -5.12, 5.12)

    candidate = [1.0, -2.0] + [0.0] * 8

    assert sphere.evaluate(candidate) == 5.0

def test_sphere_random_solution_is_valid():
    sphere = Sphere(10, -5.12, 5.12)

    rng = np.random.default_rng(202700)
    candidate = sphere.random_solution(rng)

    assert np.all(candidate >= -5.12)
    assert np.all(candidate <= 5.12)
    assert len(candidate) == 10
    assert sphere.validate(candidate)

def test_sphere_with_experiment_runner():
    sphere = Sphere(10, -5.12, 5.12)
    runner = ExperimentRunner(max_evaluations=5)

    result_a = runner.run(DummyRandomSearch(), sphere, seed=202700)
    result_b = runner.run(DummyRandomSearch(), sphere, seed=202700)

    assert result_a == result_b
    assert result_a.evaluations == result_b.evaluations == 5
    assert len(result_a.best_history) == 5
    assert isinstance(result_a.best_history, tuple)
    assert result_a.best_fitness >= 0.0