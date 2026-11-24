from benchmark.differential_evolution import DifferentialEvolution
from benchmark.experiment_runner import ExperimentRunner
from benchmark.sphere import Sphere
from benchmark.rastrigin import Rastrigin


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