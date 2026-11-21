import math
import pytest
import numpy as np

from benchmark.hill_climbing import HillClimbing
from benchmark.sphere import Sphere
from benchmark.experiment_runner import ExperimentRunner
from benchmark.rastrigin import Rastrigin
from benchmark.tsp import create_tsp_instance

def test_hc_runs_on_sphere():
    problem = Sphere(dimension=10, lower_bound=-5.12, upper_bound=5.12)
    algorithm = HillClimbing()
    runner = ExperimentRunner(max_evaluations=100)

    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == 100
    assert len(result.best_history) == 100
    assert math.isfinite(result.best_fitness)
    assert result.best_fitness >= 0.0

def test_hc_runs_on_rastrigin():
    problem = Rastrigin(
        dimension=10,
        lower_bound=-5.12,
        upper_bound=5.12
    )
    algorithm = HillClimbing()
    runner = ExperimentRunner(max_evaluations=100)

    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == 100
    assert len(result.best_history) == 100
    assert math.isfinite(result.best_fitness)
    assert result.best_fitness >= 0.0

def test_hc_is_reproducible():
    problem = Sphere(
        dimension=10,
        lower_bound=-5.12,
        upper_bound=5.12
    )
    algorithm = HillClimbing()
    runner = ExperimentRunner(max_evaluations=1000)

    result1 = runner.run(algorithm, problem, seed=202700)
    result2 = runner.run(algorithm, problem, seed=202700)

    assert result1.evaluations == result2.evaluations
    assert result1.best_fitness == result2.best_fitness
    assert result1.best_history == result2.best_history

def test_hc_runs_on_tsp():
    problem = create_tsp_instance()
    algorithm = HillClimbing()
    runner = ExperimentRunner(max_evaluations=100)

    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == 100
    assert len(result.best_history) == 100
    assert math.isfinite(result.best_fitness)
    assert result.best_fitness > 0.0

def test_hc_rejects_unsupported_problem():
    algorithm = HillClimbing()
    runner = ExperimentRunner(max_evaluations=1)

    with pytest.raises(TypeError):
        runner.run(algorithm, object(), seed=202700)

def test_hc_swap_neighbor_preserves_permutation():
    problem = create_tsp_instance()
    algorithm = HillClimbing()
    rng = np.random.default_rng(202700)

    current = np.arange(20)
    original = current.copy()

    neighbor = algorithm._swap_neighbor(current, rng)

    assert problem.validate(neighbor)
    assert np.count_nonzero(neighbor != original) == 2
    assert np.array_equal(current, original)
    assert len(neighbor) == len(original)

def test_hc_tsp_is_reproducible():
    problem = create_tsp_instance()
    algorithm = HillClimbing()
    runner = ExperimentRunner(max_evaluations=1000)

    result1 = runner.run(algorithm, problem, seed=202700)
    result2 = runner.run(algorithm, problem, seed=202700)

    assert result1.evaluations == result2.evaluations
    assert result1.best_fitness == result2.best_fitness
    assert result1.best_history == result2.best_history

@pytest.mark.parametrize("budget", [1, 21, 45])
def test_hc_respects_evaluation_budget(budget):
    problem = create_tsp_instance()
    algorithm = HillClimbing()
    runner = ExperimentRunner(max_evaluations=budget)

    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == budget
    assert len(result.best_history) == budget
    assert np.isfinite(result.best_fitness)