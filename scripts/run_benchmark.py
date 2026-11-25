import math
import csv
from pathlib import Path
from benchmark.experiment_runner import ExperimentRunner
from benchmark.genetic_algorithm import GeneticAlgorithm
from benchmark.evolution_strategy import EvolutionStrategy
from benchmark.hill_climbing import HillClimbing
from benchmark.differential_evolution import DifferentialEvolution

from benchmark.sphere import Sphere
from benchmark.rastrigin import Rastrigin
from benchmark.tsp import create_tsp_instance
from benchmark.metrics import evaluations_to_convergence

SEEDS = range(202800, 202820)
MAX_EVALUATIONS = 10000


def main():
    algorithms = {
        "GA": GeneticAlgorithm(population_size=20),
        "EE": EvolutionStrategy(mu=20, offspring_size=20),
        "HC": HillClimbing(),
        "DE": DifferentialEvolution(
            population_size=20,
            differential_weight=0.8,
            crossover_rate=0.9,
        ),
    }

    problems = {
        "Sphere": Sphere(10, -5.12, 5.12),
        "Rastrigin": Rastrigin(10, -5.12, 5.12),
        "TSP": create_tsp_instance(),
    }

    runner = ExperimentRunner(max_evaluations=100)

    total_runs = 0
    records = []

    for problem_name, problem in problems.items():
        for algorithm_name, algorithm in algorithms.items():
            for seed in SEEDS[:1]:
                print(f"{problem_name} | {algorithm_name} | seed={seed}")

                result = runner.run(
                    algorithm=algorithm,
                    problem=problem,
                    seed=seed
                )

                assert result.evaluations == 100
                assert len(result.best_history) == 100
                assert math.isfinite(result.best_fitness)
                assert result.best_candidate is not None

                assert math.isclose(
                    problem.evaluate(result.best_candidate),
                    result.best_fitness,
                    rel_tol=1e-9,
                    abs_tol=1e-9
                )

                convergence_evaluations = evaluations_to_convergence(
                    result.best_history,
                    patience=1000
                )

                records.append({
                    "problem": problem_name,
                    "algorithm": algorithm_name,
                    "seed": seed,
                    "best_fitness": result.best_fitness,
                    "evaluations": result.evaluations,
                    "convergence_evaluations": convergence_evaluations
                })

                print(
                    f"{problem_name} | {algorithm_name} | "
                    f"Fitness: {result.best_fitness:.6f}"
                )

                total_runs += 1

    assert len(records) == total_runs
    print(records[0])

    print(f"\nTotal runs: {total_runs}")

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "benchmark_runs.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "problem",
                "algorithm",
                "seed",
                "best_fitness",
                "evaluations",
                "convergence_evaluations"
            ]
        )
        writer.writeheader()
        writer.writerows(records)

if __name__ == "__main__":
    main()