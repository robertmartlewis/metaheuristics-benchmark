import pytest
import numpy as np

from benchmark.diversity import tsp_population_diversity
from benchmark.diversity import tsp_edge_distance
from benchmark.differential_evolution import DifferentialEvolution
from benchmark.diversity import continuous_solution_diversity

def test_tsp_edge_distance_equivalent_tours():
    route_a = [0, 1, 2, 3]
    route_rotated = [2, 3, 0, 1]
    route_reversed = [0, 3, 2, 1]

    assert tsp_edge_distance(route_a, route_rotated) == 0.0
    assert tsp_edge_distance(route_a, route_reversed) == 0.0


def test_tsp_edge_distance_different_tours():
    route_a = [0, 1, 2, 3]
    route_b = [0, 2, 1, 3]

    distance = tsp_edge_distance(route_a, route_b)

    assert distance == pytest.approx(0.5)

from benchmark.diversity import tsp_population_diversity


def test_tsp_population_diversity():
    population = [
        [0, 1, 2, 3],
        [0, 3, 2, 1],
        [0, 2, 1, 3]
    ]

    diversity = tsp_population_diversity(population)

    assert diversity == pytest.approx(1 / 3)


def test_identical_tours_have_zero_population_diversity():
    population = [
        [0, 1, 2, 3],
        [0, 3, 2, 1],
        [1, 2, 3, 0]
    ]

    assert tsp_population_diversity(population) == 0.0

def test_de_tsp_diversity_uses_decoded_routes():
    algorithm = DifferentialEvolution(4, 0.8, 0.9)

    population = [
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0.1, 0.4, 0.3, 0.2]),
        np.array([0.1, 0.3, 0.2, 0.4])
    ]

    routes = [
        algorithm._decode_random_keys(keys)
        for keys in population
    ]

    diversity = tsp_population_diversity(routes)

    assert diversity == pytest.approx(1 / 3)

def test_continuous_solution_diversity():
    solutions = [
        np.array([0.0, 0.0]),
        np.array([10.0, 10.0]),
        np.array([5.0, 5.0])
    ]

    diversity = continuous_solution_diversity(
        solutions,
        lower_bound=0.0,
        upper_bound=10.0
    )

    assert diversity == pytest.approx(2 / 3)