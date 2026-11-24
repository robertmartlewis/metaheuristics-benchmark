import math
import pytest
import numpy as np

from benchmark.differential_evolution import DifferentialEvolution
from benchmark.experiment_runner import ExperimentRunner
from benchmark.sphere import Sphere
from benchmark.rastrigin import Rastrigin
from benchmark.tsp import TSP
from benchmark.tsp import create_tsp_instance
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

def test_de_decodes_random_keys():
    algorithm = DifferentialEvolution(4, 0.8, 0.9)

    keys = np.array([0.7, 0.2, 0.9, 0.4])
    original = keys.copy()

    route = algorithm._decode_random_keys(keys)

    assert np.array_equal(route, [1, 3, 0, 2])
    assert np.array_equal(keys, original)

def test_de_decodes_equal_keys_deterministically():
    algorithm = DifferentialEvolution(4, 0.8, 0.9)

    keys = np.array([0.2, 0.2, 0.1, 0.2])

    route = algorithm._decode_random_keys(keys)

    assert np.array_equal(route, [2, 0, 1, 3])

def test_de_different_keys_can_encode_same_route():
    algorithm = DifferentialEvolution(4, 0.8, 0.9)

    keys_a = np.array([0.7, 0.2, 0.9, 0.4])
    keys_b = np.array([7.0, 2.0, 9.0, 4.0])

    route_a = algorithm._decode_random_keys(keys_a)
    route_b = algorithm._decode_random_keys(keys_b)

    assert not np.array_equal(keys_a, keys_b)
    assert np.array_equal(route_a, route_b)

def test_de_evaluates_random_keys_as_tsp_route():
    cities = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    problem = TSP(cities)
    algorithm = DifferentialEvolution(4, 0.8, 0.9)
    evaluator = Evaluator(problem, max_evaluations=2)

    keys_a = np.array([0.1, 0.2, 0.3, 0.4])
    keys_b = np.array([1.0, 2.0, 3.0, 4.0])

    fitness_a = algorithm._evaluate_candidate(
        keys_a, problem, evaluator
    )
    fitness_b = algorithm._evaluate_candidate(
        keys_b, problem, evaluator
    )

    assert fitness_a == pytest.approx(4.0)
    assert fitness_b == fitness_a
    assert evaluator.evaluations == 2
    assert len(evaluator.best_history) == 2

def test_de_evaluates_continuous_candidate():
    problem = Sphere(3, -5.12, 5.12)
    algorithm = DifferentialEvolution(4, 0.8, 0.9)
    evaluator = Evaluator(problem, max_evaluations=1)

    candidate = np.array([1.0, 2.0, 3.0])
    original = candidate.copy()

    fitness = algorithm._evaluate_candidate(
        candidate, problem, evaluator
    )

    assert fitness == pytest.approx(14.0)
    assert evaluator.evaluations == 1
    assert np.array_equal(candidate, original)

def test_de_initializes_random_keys_population():
    problem = create_tsp_instance()
    algorithm = DifferentialEvolution(4, 0.8, 0.9)
    evaluator = Evaluator(problem, max_evaluations=4)
    rng = np.random.default_rng(202700)
    population, fitness_values = algorithm._initialize_population(
        problem, evaluator, rng
    )

    assert len(population) == 4
    assert len(fitness_values) == 4
    assert evaluator.evaluations == 4

    for keys, fitness in zip(population, fitness_values):
        assert len(keys) == len(problem.cities)
        assert np.all(keys >= 0.0)
        assert np.all(keys < 1.0)
        route = algorithm._decode_random_keys(keys)
        assert problem.validate(route)
        assert fitness == pytest.approx(problem.evaluate(route))

def test_de_runs_on_tsp_with_initial_population_only():
    problem = create_tsp_instance()
    algorithm = DifferentialEvolution(4, 0.8, 0.9)
    runner = ExperimentRunner(max_evaluations=4)

    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == 4
    assert len(result.best_history) == 4
    assert math.isfinite(result.best_fitness)
    assert result.best_fitness > 0.0

def test_de_runs_one_tsp_generation():
    problem = create_tsp_instance()
    algorithm = DifferentialEvolution(4, 0.8, 0.9)

    runner = ExperimentRunner(max_evaluations=8)
    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == 8
    assert len(result.best_history) == 8
    assert math.isfinite(result.best_fitness)
    assert result.best_fitness > 0.0

def test_de_tsp_fitness_ties_preserve_original_keys():
    cities = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    problem = TSP(cities)
    algorithm = DifferentialEvolution(4, 0.5, 1.0)

    population = [
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([10.0, 11.0, 12.0, 13.0]),
        np.array([-3.0, -2.0, -1.0, 0.0]),
        np.array([0.5, 1.5, 2.5, 3.5])
    ]

    original = [keys.copy() for keys in population]
    fitness_values = [4.0] * 4

    evaluator = Evaluator(problem, max_evaluations=4)
    rng = np.random.default_rng(202700)

    next_population, next_fitness_values = (
        algorithm._evolve_generation(
            population,
            fitness_values,
            problem,
            evaluator,
            rng
        )
    )

    assert evaluator.evaluations == 4

    for before, after in zip(original, next_population):
        np.testing.assert_array_equal(after, before)

    assert next_fitness_values == fitness_values

    for before, after in zip(original, population):
        np.testing.assert_array_equal(after, before)

def test_de_tsp_accepts_better_trial_as_random_keys(monkeypatch):
    cities = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    problem = TSP(cities)
    algorithm = DifferentialEvolution(4, 0.5, 1.0)

    population = [
        np.array([0.0, 2.0, 1.0, 3.0]),
        np.array([2.0, 3.0, 4.0, 5.0]),
        np.array([-1.0, 0.0, 1.0, 2.0]),
        np.array([3.0, 4.0, 5.0, 6.0])
    ]

    original = [keys.copy() for keys in population]

    fitness_values = [
        problem.evaluate(algorithm._decode_random_keys(keys))
        for keys in population
    ]

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

    next_population, next_fitness_values = (
        algorithm._evolve_generation(
            population,
            fitness_values,
            problem,
            evaluator,
            rng
        )
    )

    np.testing.assert_allclose(
        next_population[0],
        [0.0, 1.0, 2.0, 3.0]
    )

    assert next_fitness_values[0] == pytest.approx(4.0)
    assert next_fitness_values[0] < fitness_values[0]
    assert evaluator.evaluations == 4
    for before, after in zip(original, population):
        np.testing.assert_array_equal(before, after)

@pytest.mark.parametrize("budget", [5, 6, 7, 9, 11])
def test_de_tsp_handles_partial_generations(budget):
    problem = create_tsp_instance()
    algorithm = DifferentialEvolution(4, 0.8, 0.9)
    runner = ExperimentRunner(max_evaluations=budget)

    result = runner.run(algorithm, problem, seed=202700)

    assert result.evaluations == budget
    assert len(result.best_history) == budget
    assert math.isfinite(result.best_fitness)

    assert result.best_fitness == min(result.best_history)

def test_de_tsp_is_reproducible():
    problem = create_tsp_instance()
    algorithm = DifferentialEvolution(20, 0.8, 0.9)
    runner = ExperimentRunner(max_evaluations=1003)

    result1 = runner.run(algorithm, problem, seed=202700)
    result2 = runner.run(algorithm, problem, seed=202700)

    assert result1.evaluations == result2.evaluations == 1003
    assert result1.best_fitness == result2.best_fitness
    assert result1.best_history == result2.best_history

    assert all(
        later <= earlier
        for earlier, later in zip(
            result1.best_history,
            result1.best_history[1:]
        )
    )