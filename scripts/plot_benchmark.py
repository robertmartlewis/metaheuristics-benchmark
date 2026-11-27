import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

from benchmark.tsp import create_tsp_instance
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

def plot_fitness_boxplots():
    runs = pd.read_csv(
        RESULTS_DIR / "benchmark_runs.csv"
    )

    for problem_name, problem_data in runs.groupby("problem"):
        fig, ax = plt.subplots(figsize=(8, 5))

        fitness_ga = problem_data.loc[
            problem_data["algorithm"] == "GA",
            "best_fitness"
        ]
        fitness_ee = problem_data.loc[
            problem_data["algorithm"] == "EE",
            "best_fitness"
        ]
        fitness_hc = problem_data.loc[
            problem_data["algorithm"] == "HC",
            "best_fitness"
        ]
        fitness_de = problem_data.loc[
            problem_data["algorithm"] == "DE",
            "best_fitness"
        ]

        ax.boxplot(
            [fitness_ga, fitness_ee, fitness_hc, fitness_de],
            tick_labels=["GA", "EE", "HC", "DE"]
        )

        ax.set_title(f"Final fitness distribution on {problem_name}")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Final best fitness")

        if problem_name == "Sphere":
            ax.set_yscale("log")

        fig.savefig(
            FIGURES_DIR / f"boxplot_{problem_name.lower()}.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

def plot_best_tsp_route():
    runs = pd.read_csv(
        RESULTS_DIR / "benchmark_runs.csv"
    )
    tsp_runs = runs[runs["problem"] == "TSP"]
    best_index = tsp_runs["best_fitness"].idxmin()
    best_run = tsp_runs.loc[best_index]
    route = np.asarray(
        json.loads(best_run["best_candidate"]),
        dtype=int
    )
    problem = create_tsp_instance()
    cities = problem.cities
    assert problem.validate(route)
    assert np.isclose(
        problem.evaluate(route),
        best_run["best_fitness"]
    )
    route_coordinates = cities[route]

    closed_route = np.vstack([
        route_coordinates,
        route_coordinates[0]
    ])

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        closed_route[:, 0],
        closed_route[:, 1],
        linewidth=1.5,
        alpha=0.8
    )

    ax.scatter(
        cities[:, 0],
        cities[:, 1],
        s=45,
        zorder=3
    )

    for i, (x, y) in enumerate(cities):
        ax.annotate(
            str(i),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

    ax.set_title(
        f"Best TSP route — {best_run['algorithm']}\n"
        f"Distance: {best_run['best_fitness']:.2f} | "
        f"Seed: {best_run['seed']}"
    )

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    fig.savefig(
        FIGURES_DIR / "best_tsp_route.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

if __name__ == "__main__":
    main()
    plot_fitness_boxplots()
    plot_best_tsp_route()