import math

import numpy as np
import pytest

from benchmark.experiment_runner import ExperimentRunner
from benchmark.sphere import Sphere
from benchmark.rastrigin import Rastrigin
from benchmark.tsp import TSP, create_tsp_instance
from benchmark.evolution_strategy import EvolutionStrategy

def test_es_runs_on_sphere():
    problem = Sphere(10, -5.12, 5.12)
    algorithm = EvolutionStrategy(mu=20, offspring_size=20)
    runner = ExperimentRunner(max_evaluations=100)

    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == 100
    assert len(result.best_history) == 100
    assert math.isfinite(result.best_fitness)
    assert result.best_fitness >= 0.0

def test_es_runs_on_rastrigin():
    problem = Rastrigin(10, -5.12, 5.12)
    algorithm = EvolutionStrategy(mu=20, offspring_size=20)
    runner = ExperimentRunner(max_evaluations=100)

    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == 100
    assert len(result.best_history) == 100
    assert math.isfinite(result.best_fitness)
    assert result.best_fitness >= 0.0

def test_es_selects_best_mu_survivors():
    algorithm = EvolutionStrategy(mu=3, offspring_size=3)

    candidates = ["A", "B", "C", "D", "E", "F"]
    fitness_values = [10.0, 5.0, 8.0, 3.0, 12.0, 6.0]

    survivors, survivor_fitness = algorithm._select_survivors(
        candidates, fitness_values
    )

    assert survivors == ["D", "B", "F"]
    assert survivor_fitness == [3.0, 5.0, 6.0]

@pytest.mark.parametrize(
    "mu, offspring_size, exception",
    [
        (0, 20, ValueError),
        (-1, 20, ValueError),
        (20, 0, ValueError),
        (20, -1, ValueError),
        (2.5, 20, TypeError),
        (20, 2.5, TypeError),
        ("20", 20, TypeError),
        (20, True, TypeError),
        (True, 20, TypeError),
    ]
)
def test_es_rejects_invalid_parameters(mu, offspring_size, exception):
    with pytest.raises(exception):
        EvolutionStrategy(mu=mu, offspring_size=offspring_size)

def test_es_rejects_budget_smaller_than_mu():
    problem = Sphere(10, -5.12, 5.12)
    algorithm = EvolutionStrategy(mu=20, offspring_size=20)
    runner = ExperimentRunner(max_evaluations=10)

    with pytest.raises(ValueError):
        runner.run(algorithm, problem, seed=202700)

@pytest.mark.parametrize("budget", [20, 21, 45])
def test_es_handles_partial_generations(budget):
    problem = Sphere(10, -5.12, 5.12)
    algorithm = EvolutionStrategy(mu=20, offspring_size=20)
    runner = ExperimentRunner(max_evaluations=budget)

    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == budget
    assert len(result.best_history) == budget
    assert math.isfinite(result.best_fitness)

def test_es_is_reproducible():
    problem = Sphere(10, -5.12, 5.12)
    algorithm = EvolutionStrategy(mu=20, offspring_size=20)
    runner = ExperimentRunner(max_evaluations=1000)
    result1 = runner.run(algorithm, problem, seed=202700)
    result2 = runner.run(algorithm, problem, seed=202700)

    assert result1.evaluations == result2.evaluations
    assert result1.best_fitness == result2.best_fitness
    assert result1.best_history == result2.best_history

def test_es_runs_on_tsp():
    problem = create_tsp_instance()
    algorithm = EvolutionStrategy(mu=20, offspring_size=20)
    runner = ExperimentRunner(max_evaluations=100)

    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == 100
    assert len(result.best_history) == 100
    assert math.isfinite(result.best_fitness)
    assert result.best_fitness > 0.0

def test_es_swap_mutation_preserves_permutation():
    problem = create_tsp_instance()
    algorithm = EvolutionStrategy(mu=20, offspring_size=20)

    parent = np.arange(20)
    original = parent.copy()

    rng = np.random.default_rng(seed=202700)

    mutated = algorithm._swap_mutation(parent, rng)

    assert problem.validate(mutated)
    assert np.count_nonzero(mutated != original) == 2
    assert np.array_equal(parent, original)
    assert len(mutated) == len(original)

def test_es_tsp_is_reproducible():
    problem = create_tsp_instance()
    algorithm = EvolutionStrategy(mu=20, offspring_size=20)
    runner = ExperimentRunner(max_evaluations=1000)

    result1 = runner.run(algorithm, problem, seed=202700)
    result2 = runner.run(algorithm, problem, seed=202700)

    assert result1.evaluations == result2.evaluations
    assert result1.best_fitness == result2.best_fitness
    assert result1.best_history == result2.best_history

def test_es_rejects_unsupported_problem():
    algorithm = EvolutionStrategy(mu=20, offspring_size=20)
    runner = ExperimentRunner(max_evaluations=100)
    with pytest.raises(TypeError):
        runner.run(algorithm, object(), seed=202700)

@pytest.mark.parametrize("budget", [20, 21, 45])
def test_es_tsp_handles_partial_generations(budget):
    problem = create_tsp_instance()
    algorithm = EvolutionStrategy(mu=20, offspring_size=20)
    runner = ExperimentRunner(max_evaluations=budget)

    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == budget
    assert len(result.best_history) == budget
    assert np.isfinite(result.best_fitness)