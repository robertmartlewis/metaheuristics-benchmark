import statistics

from benchmark.differential_evolution import DifferentialEvolution
from benchmark.experiment_runner import ExperimentRunner
from benchmark.sphere import Sphere
from benchmark.rastrigin import Rastrigin
from benchmark.tsp import create_tsp_instance


def diagnose(name, problem):
    algorithm = DifferentialEvolution(
        population_size=20,
        differential_weight=0.8,
        crossover_rate=0.9
    )

    runner = ExperimentRunner(max_evaluations=10000)

    result = runner.run(
        algorithm=algorithm,
        problem=problem,
        seed=303000
    )

    print(f"\n{name}")
    print(f"Final best fitness: {result.best_fitness:.8f}")

    for evaluations in [20, 100, 1000, 5000, 10000]:
        fitness = result.best_history[evaluations - 1]
        print(f"{evaluations:5d} evaluations: {fitness:.8f}")

    if result.diversity_history:
        diversity_by_evaluations = dict(result.diversity_history)

        print("\nPopulation diversity:")

        for evaluations in [20, 100, 1000, 5000, 10000]:
            diversity = diversity_by_evaluations[evaluations]
            print(f"{evaluations:5d} evaluations: {diversity:.8f}")


def diagnose_tsp_multiple_seeds():
    problem = create_tsp_instance()
    seeds = [303000, 303001, 303002, 303003, 303004]

    final_fitness_values = []

    for seed in seeds:
        algorithm = DifferentialEvolution(
            population_size=20,
            differential_weight=0.8,
            crossover_rate=0.9
        )

        runner = ExperimentRunner(max_evaluations=10000)
        result = runner.run(algorithm=algorithm, problem=problem, seed=seed)

        first_zero_diversity = None

        for evaluation_count, diversity in result.diversity_history:
            if diversity == 0.0:
                first_zero_diversity = evaluation_count
                break

        diversity_by_evaluations = dict(result.diversity_history)
        fitness_5000 = result.best_history[5000-1]
        fitness_10000 = result.best_history[10000-1]
        diversity_5000 = diversity_by_evaluations[5000]
        diversity_10000 = diversity_by_evaluations[10000]
        improved = fitness_5000 > fitness_10000
        final_fitness_values.append(fitness_10000)

        print(
            f"First zero diversity: {first_zero_diversity} "
            f"Seed {seed}: "
            f"5000={fitness_5000:.4f}, "
            f"10000={fitness_10000:.4f}, "
            f"Improved={improved}, "
            f"diversity 5000={diversity_5000:.4f}, "
            f"diversity 10000={diversity_10000:.4f}"
        )

    print("mean: " + str(statistics.mean(final_fitness_values)))
    print("std: " + str(statistics.stdev(final_fitness_values)))

if __name__ == "__main__":
    diagnose(
        "Sphere",
        Sphere(
            dimension=10,
            lower_bound=-5.12,
            upper_bound=5.12
        )
    )

    diagnose(
        "Rastrigin",
        Rastrigin(
            dimension=10,
            lower_bound=-5.12,
            upper_bound=5.12
        )
    )

    diagnose(
        "TSP",
        create_tsp_instance()
    )

    diagnose_tsp_multiple_seeds()