import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")


def main():
    history = pd.read_csv(
        RESULTS_DIR / "benchmark_history.csv"
    )

    grouped = history.groupby(
        ["problem", "algorithm", "evaluation"]
    )["best_fitness"]

    aggregated = grouped.agg(
        mean="mean",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75)
    ).reset_index()

    print(aggregated.head())
    print(f"Total aggregated points: {len(aggregated)}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for problem_name, problem_data in aggregated.groupby("problem"):

        fig, ax = plt.subplots(figsize=(9, 5))

        for algorithm_name, algorithm_data in problem_data.groupby("algorithm"):
            evaluation = algorithm_data["evaluation"]
            mean = algorithm_data["mean"]
            q25 = algorithm_data["q25"]
            q75 = algorithm_data["q75"]

            ax.plot(evaluation, mean, label=algorithm_name)
            ax.fill_between(evaluation, q25, q75, alpha=0.2)

        ax.set_title(f"Convergence on {problem_name}")
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Best fitness")
        ax.legend()

        if problem_name == "Sphere":
            ax.set_yscale("log")

        fig.savefig(FIGURES_DIR / f"convergence_{problem_name.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()