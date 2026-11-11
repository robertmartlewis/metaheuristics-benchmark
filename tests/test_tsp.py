import numpy as np
import pytest

from benchmark.tsp import TSP
from benchmark.tsp import create_tsp_instance
from benchmark.experiment_runner import ExperimentRunner


class DummyRandomSearch:
    def run(self, problem, evaluator, rng):
        for _ in range(evaluator.max_evaluations):
            candidate = problem.random_solution(rng)
            evaluator.evaluate(candidate)

def test_tsp_calculates_closed_tour_distance():
    cities = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 4.0],
        [0.0, 4.0]
    ])

    problem = TSP(cities)

    assert problem.evaluate([0, 1, 2, 3]) == 14.0

def test_tsp_validates_permutations():
    cities = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 4.0],
        [0.0, 4.0]
    ])

    problem = TSP(cities)

    assert problem.validate([0, 1, 2, 3])
    assert not problem.validate([0, 1, 1, 3])
    assert not problem.validate([0, 1, 2])
    assert not problem.validate([0, 1, 2, 4])

def test_tsp_distance_depends_on_route_order():
    cities = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 4.0],
        [0.0, 4.0]
    ])

    problem = TSP(cities)

    assert problem.evaluate([0, 2, 1, 3]) == 18.0

def test_tsp_random_solution_is_valid():
    cities = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 4.0],
        [0.0, 4.0]
    ])

    problem = TSP(cities)
    rng_a = np.random.default_rng(202700)
    rng_b = np.random.default_rng(202700)
    candidate_a = problem.random_solution(rng_a)
    candidate_b = problem.random_solution(rng_b)

    assert len(candidate_a) == 4
    assert problem.validate(candidate_a)
    assert np.array_equal(candidate_a, candidate_b)

def test_tsp_rejects_non_integer_indices():
    cities = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 4.0],
        [0.0, 4.0]
    ])

    problem = TSP(cities)

    assert not problem.validate([0.0, 1, 2, 3])
    assert not problem.validate([False, 1, 2, 3])

def test_tsp_fixed_instance_is_reproducible():
    problem_a = create_tsp_instance()
    problem_b = create_tsp_instance()

    assert problem_a.cities.shape == (20, 2)
    assert np.array_equal(problem_a.cities, problem_b.cities)
    assert np.all(problem_a.cities >= 0)
    assert np.all(problem_a.cities < 100)

def test_tsp_rejects_invalid_tour_evaluation():
    cities = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 4.0],
        [0.0, 4.0]
    ])

    problem = TSP(cities)

    with pytest.raises(ValueError):
        problem.evaluate([0, 1, 1, 3])

def test_tsp_with_experiment_runner():
    problem = create_tsp_instance()
    runner = ExperimentRunner(max_evaluations=5)

    result_a = runner.run(DummyRandomSearch(), problem, seed=202700)
    result_b = runner.run(DummyRandomSearch(), problem, seed=202700)

    assert result_a == result_b
    assert result_a.evaluations == result_b.evaluations == 5
    assert len(result_a.best_history) == 5
    assert isinstance(result_a.best_history, tuple)
    assert result_a.best_fitness >= 0.0