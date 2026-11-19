import csv
from pathlib import Path

import numpy as np

from benchmark.genetic_algorithm import GeneticAlgorithm
from benchmark.sphere import Sphere
from benchmark.rastrigin import Rastrigin
from benchmark.experiment_runner import ExperimentRunner


class GeneticAlgorithmAllCoordinates(GeneticAlgorithm):
    def _gaussian_mutation(self, candidate, problem, rng):
        sigma = 0.1 * (problem.upper_bound - problem.lower_bound)
        noise = rng.normal(loc=0.0, scale=sigma, size=len(candidate))
        mutated_candidate = candidate + noise

        return np.clip(
            mutated_candidate,
            problem.lower_bound,
            problem.upper_bound
        )


def main():
    problems = [
        Sphere(10, -5.12, 5.12),
        Rastrigin(10, -5.12, 5.12)
    ]

    runner = ExperimentRunner(max_evaluations=10000)

    algorithms = {
        "original": GeneticAlgorithmAllCoordinates(population_size=20),
        "modified": GeneticAlgorithm(population_size=20)
    }

    records = []

    for problem in problems:
        print(f"\n=== {type(problem).__name__} 10D ===")

        for variant, algorithm in algorithms.items():
            fitness_values = []

            for seed in range(303000, 303020):
                result = runner.run(algorithm, problem, seed=seed)

                fitness_values.append(result.best_fitness)

                records.append({
                    "problem": type(problem).__name__,
                    "variant": variant,
                    "seed": seed,
                    "best_fitness": result.best_fitness
                })

            print(f"{variant} mean: {np.mean(fitness_values):.8f}")
            print(f"{variant} std:  {np.std(fitness_values, ddof=1):.8f}")
            print(f"{variant} median: {np.median(fitness_values):.8f}")

    # Guardar resultados individuales
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "ga_mutation_comparison.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["problem", "variant", "seed", "best_fitness"]
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved {len(records)} records to {output_file}")


if __name__ == "__main__":
    main()