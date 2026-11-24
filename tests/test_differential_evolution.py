import math
import pytest
import numpy as np

from benchmark.differential_evolution import DifferentialEvolution
from benchmark.experiment_runner import ExperimentRunner
from benchmark.sphere import Sphere
from benchmark.rastrigin import Rastrigin
from benchmark.evaluator import Evaluator


def test_de_runs_on_sphere():
    problem = Sphere(
        dimension=10,
        lower_bound=-5.12,
        upper_bound=5.12
    )

    algorithm = DifferentialEvolution(
        population_size=20,
        differential_weight=0.8,
        crossover_rate=0.9
    )

    runner = ExperimentRunner(max_evaluations=100)

    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == 100
    assert len(result.best_history) == 100
    assert math.isfinite(result.best_fitness)
    assert result.best_fitness >= 0.0


@pytest.mark.parametrize("population_size", [0, 1, 2, 3, -1])
def test_de_rejects_invalid_population_size(population_size):
    with pytest.raises(ValueError):
        DifferentialEvolution(
            population_size=population_size,
            differential_weight=0.8,
            crossover_rate=0.9
        )

@pytest.mark.parametrize("population_size", [4.5, "20", True, None])
def test_de_rejects_invalid_population_size_types(population_size):
    with pytest.raises(TypeError):
        DifferentialEvolution(
            population_size=population_size,
            differential_weight=0.8,
            crossover_rate=0.9
        )

@pytest.mark.parametrize("weight", [-0.5, 0.0, 2.1, float("nan"), float("inf"), -float("inf")])
def test_de_rejects_invalid_differential_weight(weight):
    with pytest.raises(ValueError):
        DifferentialEvolution(
            population_size=20,
            differential_weight=weight,
            crossover_rate=0.9
        )


@pytest.mark.parametrize("weight", ["0.8", None, True, [0.8]])
def test_de_rejects_invalid_differential_weight_types(weight):
    with pytest.raises(TypeError):
        DifferentialEvolution(
            population_size=20,
            differential_weight=weight,
            crossover_rate=0.9
        )

@pytest.mark.parametrize("rate", [
    -0.1, 1.1,
    float("nan"),
    float("inf"),
    -float("inf")
])
def test_de_rejects_invalid_crossover_rate(rate):
    with pytest.raises(ValueError):
        DifferentialEvolution(
            population_size=20,
            differential_weight=0.8,
            crossover_rate=rate
        )

@pytest.mark.parametrize("rate", [
    "0.9", None, True, [0.9]
])
def test_de_rejects_invalid_crossover_rate_types(rate):
    with pytest.raises(TypeError):
        DifferentialEvolution(
            population_size=20,
            differential_weight=0.8,
            crossover_rate=rate
        )

@pytest.mark.parametrize("rate", [0.0, 0.5, 1.0])
def test_de_accepts_valid_crossover_rate(rate):
    algorithm = DifferentialEvolution(
        population_size=20,
        differential_weight=0.8,
        crossover_rate=rate
    )

    assert algorithm.crossover_rate == rate

def test_de_initializes_population():
    problem = Sphere(
        dimension=10,
        lower_bound=-5.12,
        upper_bound=5.12
    )

    algorithm = DifferentialEvolution(
        population_size=4,
        differential_weight=0.8,
        crossover_rate=0.9
    )

    evaluator = Evaluator(problem, max_evaluations=4)
    rng = np.random.default_rng(202700)

    population, fitness_values = algorithm._initialize_population(
        problem, evaluator, rng
    )

    assert len(population) == 4
    assert len(fitness_values) == 4
    assert evaluator.evaluations == 4
    assert len(evaluator.best_history) == 4

    assert all(problem.validate(candidate) for candidate in population)
    assert all(math.isfinite(fitness) for fitness in fitness_values)

def test_de_selects_distinct_donor_indices():
    algorithm = DifferentialEvolution(
        population_size=20,
        differential_weight=0.8,
        crossover_rate=0.9
    )

    rng = np.random.default_rng(202700)

    for target_index in range(20):
        r1, r2, r3 = algorithm._select_donor_indices(
            target_index, rng
        )

        indices = [r1, r2, r3]

        assert len(set(indices)) == 3
        assert target_index not in indices
        assert all(0 <= index < 20 for index in indices)

def test_de_differential_mutation():
    algorithm = DifferentialEvolution(
        population_size=4,
        differential_weight=0.5,
        crossover_rate=0.9
    )

    population = [
        np.array([1.0, 2.0]),
        np.array([4.0, 6.0]),
        np.array([2.0, 1.0]),
        np.array([5.0, 3.0])
    ]

    donor_indices = (0, 1, 2)

    mutant = algorithm._differential_mutation(
        population, donor_indices
    )

    np.testing.assert_allclose(mutant, [2.0, 4.5])

def test_de_binomial_crossover_forces_one_mutant_coordinate():
    algorithm = DifferentialEvolution(
        population_size=4,
        differential_weight=0.8,
        crossover_rate=0.0
    )

    target = np.array([1.0, 2.0, 3.0, 4.0])
    mutant = np.array([10.0, 20.0, 30.0, 40.0])
    rng = np.random.default_rng(202700)

    trial = algorithm._binomial_crossover(target, mutant, rng)

    assert np.count_nonzero(trial != target) == 1
    assert np.count_nonzero(trial == mutant) == 1
    assert np.array_equal(target, [1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(mutant, [10.0, 20.0, 30.0, 40.0])

def test_de_binomial_crossover_with_rate_one():
    algorithm = DifferentialEvolution(
        population_size=4,
        differential_weight=0.8,
        crossover_rate=1.0
    )

    target = np.array([1.0, 2.0, 3.0, 4.0])
    mutant = np.array([10.0, 20.0, 30.0, 40.0])
    rng = np.random.default_rng(202700)

    trial = algorithm._binomial_crossover(target, mutant, rng)

    assert np.array_equal(trial, mutant)
    assert not np.shares_memory(trial, mutant)

def test_de_binomial_crossover_with_intermediate_rate():
    algorithm = DifferentialEvolution(
        population_size=4,
        differential_weight=0.8,
        crossover_rate=0.5
    )

    target = np.array([1.0, 2.0, 3.0, 4.0])
    mutant = np.array([10.0, 20.0, 30.0, 40.0])
    rng = np.random.default_rng(202700)

    trial = algorithm._binomial_crossover(target, mutant, rng)

    assert np.all((trial == target) | (trial == mutant))
    assert np.count_nonzero(trial == mutant) >= 1
    assert np.array_equal(target, [1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(mutant, [10.0, 20.0, 30.0, 40.0])

def test_de_repairs_trial_bounds():
    problem = Sphere(
        dimension=3,
        lower_bound=-5.12,
        upper_bound=5.12
    )

    algorithm = DifferentialEvolution(
        population_size=4,
        differential_weight=0.8,
        crossover_rate=0.9
    )

    trial = np.array([-6.0, 0.3, 9.0])
    original = trial.copy()

    repaired = algorithm._repair_bounds(trial, problem)

    np.testing.assert_allclose(repaired, [-5.12, 0.3, 5.12])
    assert problem.validate(repaired)
    assert np.array_equal(trial, original)

def test_de_selects_better_trial():
    algorithm = DifferentialEvolution(
        population_size=4,
        differential_weight=0.8,
        crossover_rate=0.9
    )

    target = np.array([3.0, 1.0])
    trial = np.array([1.0, 2.0])

    survivor, fitness = algorithm._select_survivor(
        target, 10.0,
        trial, 5.0
    )

    assert np.array_equal(survivor, trial)
    assert fitness == 5.0

def test_de_rejects_worse_trial():
    algorithm = DifferentialEvolution(
        population_size=4,
        differential_weight=0.8,
        crossover_rate=0.9
    )

    target = np.array([1.0, 2.0])
    trial = np.array([3.0, 1.0])

    survivor, fitness = algorithm._select_survivor(
        target, 5.0,
        trial, 10.0
    )

    assert np.array_equal(survivor, target)
    assert fitness == 5.0

def test_de_keeps_target_on_equal_fitness():
    algorithm = DifferentialEvolution(
        population_size=4,
        differential_weight=0.8,
        crossover_rate=0.9
    )

    target = np.array([1.0, 2.0])
    trial = np.array([3.0, 4.0])

    survivor, fitness = algorithm._select_survivor(
        target, 5.0,
        trial, 5.0
    )

    assert np.array_equal(survivor, target)
    assert fitness == 5.0

def test_de_stops_after_initial_population():
    problem = Sphere(
        dimension=10,
        lower_bound=-5.12,
        upper_bound=5.12
    )

    algorithm = DifferentialEvolution(
        population_size=4,
        differential_weight=0.8,
        crossover_rate=0.9
    )

    runner = ExperimentRunner(max_evaluations=4)
    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == 4
    assert len(result.best_history) == 4
    assert math.isfinite(result.best_fitness)

def test_de_rejects_budget_smaller_than_population():
    problem = Sphere(10, -5.12, 5.12)
    algorithm = DifferentialEvolution(4, 0.8, 0.9)
    runner = ExperimentRunner(max_evaluations=3)

    with pytest.raises(ValueError):
        runner.run(algorithm, problem, seed=202700)

def test_de_runs_one_generation():
    problem = Sphere(
        dimension=10,
        lower_bound=-5.12,
        upper_bound=5.12
    )

    algorithm = DifferentialEvolution(
        population_size=4,
        differential_weight=0.8,
        crossover_rate=0.9
    )

    runner = ExperimentRunner(max_evaluations=8)
    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == 8
    assert len(result.best_history) == 8
    assert math.isfinite(result.best_fitness)

@pytest.mark.parametrize("budget", [9, 10, 11, 12])
def test_de_handles_partial_generations(budget):
    problem = Sphere(10, -5.12, 5.12)
    algorithm = DifferentialEvolution(4, 0.8, 0.9)
    runner = ExperimentRunner(max_evaluations=budget)

    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == budget
    assert len(result.best_history) == budget
    assert math.isfinite(result.best_fitness)

def test_de_partial_generation_preserves_population():
    problem = Sphere(10, -5.12, 5.12)
    algorithm = DifferentialEvolution(4, 0.8, 0.9)

    evaluator = Evaluator(problem, max_evaluations=5)
    rng = np.random.default_rng(202700)

    population, fitness_values = algorithm._initialize_population(
        problem, evaluator, rng
    )

    next_population, next_fitness_values = algorithm._evolve_generation(
        population, fitness_values, problem, evaluator, rng
    )

    assert evaluator.evaluations == 5
    assert len(next_population) == 4
    assert len(next_fitness_values) == 4

    for i in range(1, 4):
        assert np.array_equal(next_population[i], population[i])
        assert next_fitness_values[i] == fitness_values[i]

def test_de_runs_on_rastrigin():
    problem = Rastrigin(10, -5.12, 5.12)
    algorithm = DifferentialEvolution(20, 0.8, 0.9)
    runner = ExperimentRunner(max_evaluations=1000)

    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == 1000
    assert len(result.best_history) == 1000
    assert math.isfinite(result.best_fitness)
    assert result.best_fitness >= 0.0

def test_de_is_reproducible():
    problem = Sphere(10, -5.12, 5.12)
    algorithm = DifferentialEvolution(20, 0.8, 0.9)
    runner = ExperimentRunner(max_evaluations=1003)

    result1 = runner.run(algorithm, problem, seed=202700)
    result2 = runner.run(algorithm, problem, seed=202700)

    assert result1.evaluations == result2.evaluations == 1003
    assert result1.best_fitness == result2.best_fitness
    assert result1.best_history == result2.best_history

def test_de_uses_previous_generation_for_all_donors(monkeypatch):
    problem = Sphere(1, -50.0, 50.0)
    algorithm = DifferentialEvolution(4, 0.5, 1.0)

    population = [
        np.array([10.0]),
        np.array([1.0]),
        np.array([2.0]),
        np.array([3.0])
    ]
    original = [candidate.copy() for candidate in population]
    fitness_values = [100.0, 1.0, 4.0, 9.0]

    donor_indices = {
        0: (1, 2, 3),
        1: (0, 2, 3),
        2: (0, 1, 3),
        3: (0, 1, 2)
    }

    monkeypatch.setattr(
        algorithm,
        "_select_donor_indices",
        lambda i, rng: donor_indices[i]
    )

    evaluator = Evaluator(problem, max_evaluations=4)
    rng = np.random.default_rng(202700)

    next_population, next_fitness_values = algorithm._evolve_generation(
        population, fitness_values, problem, evaluator, rng
    )

    assert next_population[0][0] == pytest.approx(0.5)
    assert next_population[1][0] == pytest.approx(1.0)
    assert next_fitness_values[1] == pytest.approx(1.0)
    assert evaluator.evaluations == 4

    for before, after in zip(original, population):
        assert np.array_equal(before, after)