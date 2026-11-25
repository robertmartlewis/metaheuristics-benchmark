import pandas as pd
import json
import numpy as np

from benchmark.diversity import (continuous_solution_diversity, tsp_population_diversity)
from pathlib import Path
from scipy.stats import friedmanchisquare


RESULTS_DIR = Path("results")


def main():
    runs = pd.read_csv(
        RESULTS_DIR / "benchmark_runs.csv"
    )

    print(runs.head())
    print(f"Total runs: {len(runs)}")

    grouped = runs.groupby(["problem", "algorithm"])
    summary = grouped["best_fitness"].agg([
        "min",
        "mean",
        "std",
        "count"
    ])

    convergence = grouped["convergence_evaluations"].agg([
        "mean",
        "count"
    ])

    convergence["percentage"] = (
        convergence["count"] / summary["count"] * 100
    )

    print("\nFitness statistics:")
    print(summary)

    print("\nConvergence statistics:")
    print(convergence)

    for problem_name, problem_runs in runs.groupby("problem"):
        paired = problem_runs.pivot(
            index="seed",
            columns="algorithm",
            values="best_fitness"
        )

        paired = paired[["GA", "EE", "HC", "DE"]]

        assert paired.shape == (20, 4)
        assert not paired.isna().any().any()

        statistic, p_value = friedmanchisquare(
            paired["GA"],
            paired["EE"],
            paired["HC"],
            paired["DE"]
        )

        print(f"\nProblem: {problem_name}")
        print(f"Friedman statistic: {statistic:.6f}")
        print(f"p-value: {p_value:.8g}")

    print("\nFinal solution diversity:")

    for problem_name, problem_runs in runs.groupby("problem"):
        for algorithm_name, algorithm_runs in problem_runs.groupby("algorithm"):

            solutions = []

            for candidate_json in algorithm_runs["best_candidate"]:
                solution = np.asarray(json.loads(candidate_json))
                solutions.append(solution)

            if(problem_name == "TSP"):
                diversity = tsp_population_diversity(solutions)

            else:
                diversity = continuous_solution_diversity(solutions, lower_bound=-5.12, upper_bound=5.12)

            print(f"{problem_name} | {algorithm_name} | {diversity:.6f}")


if __name__ == "__main__":
    main()