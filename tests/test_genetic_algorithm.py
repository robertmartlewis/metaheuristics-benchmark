import math
import numpy as np
import pytest

from benchmark.genetic_algorithm import GeneticAlgorithm
from benchmark.experiment_runner import ExperimentRunner
from benchmark.sphere import Sphere


def test_ga_runs_on_sphere():
    problem = Sphere(10, -5.12, 5.12)
    runner = ExperimentRunner(max_evaluations=60)

    algorithm = GeneticAlgorithm(population_size=20)

    result = runner.run(
        algorithm,
        problem,
        seed=202700
    )

    assert result.evaluations == 60
    assert len(result.best_history) == 60
    assert math.isfinite(result.best_fitness)
    assert result.best_fitness >= 0.0

def test_ga_tournament_selects_lowest_fitness():
    algorithm = GeneticAlgorithm(population_size=3)
    rng = np.random.default_rng(202700)

    fitness_values = [14.0, 6.0, 11.0]

    winner = algorithm._tournament_select(fitness_values, rng)

    assert winner == 1

def test_ga_arithmetic_crossover():
    algorithm = GeneticAlgorithm(population_size=20)
    rng = np.random.default_rng(202700)

    parent_a = np.array([8.0, 2.0])
    parent_b = np.array([4.0, 6.0])

    child = algorithm._arithmetic_crossover(parent_a, parent_b, rng)

    assert len(child) == 2
    assert np.all(child >= np.minimum(parent_a, parent_b))
    assert np.all(child <= np.maximum(parent_a, parent_b))
    assert np.array_equal(parent_a, np.array([8.0, 2.0]))
    assert np.array_equal(parent_b, np.array([4.0, 6.0]))

def test_ga_gaussian_mutation():
    algorithm = GeneticAlgorithm(population_size=20)
    problem = Sphere(3, -5.12, 5.12)
    rng = np.random.default_rng(202700)

    candidate = np.array([5.12, 0.0, -5.12])

    mutated = algorithm._gaussian_mutation(candidate, problem, rng)

    assert len(mutated) == 3
    assert np.all(mutated >= problem.lower_bound)
    assert np.all(mutated <= problem.upper_bound)
    assert problem.validate(mutated)
    assert np.array_equal(candidate, np.array([5.12, 0.0, -5.12]))
    assert isinstance(mutated, np.ndarray)
    assert not np.array_equal(mutated, candidate)

def test_ga_rejects_budget_smaller_than_population():
    algorithm = GeneticAlgorithm(population_size=20)
    problem = Sphere(10, -5.12, 5.12)
    runner = ExperimentRunner(max_evaluations=10)

    with pytest.raises(ValueError):
        runner.run(algorithm, problem, seed=202700)