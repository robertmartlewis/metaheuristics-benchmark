from benchmark.genetic_algorithm import GeneticAlgorithm
from benchmark.experiment_runner import ExperimentRunner
from benchmark.sphere import Sphere

problem = Sphere(10, -5.12, 5.12)
algorithm = GeneticAlgorithm(population_size=20)
runner = ExperimentRunner(max_evaluations=10000)

result = runner.run(algorithm, problem, seed=202700)

print("Initial best:", result.best_history[19])
print("Final best:", result.best_fitness)
for evaluations in [20, 100, 1000, 5000, 10000]:
    fitness = result.best_history[evaluations - 1]
    print(f"{evaluations} evaluations: {fitness:.6f}")