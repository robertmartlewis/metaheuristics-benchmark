from benchmark.sphere import Sphere
from benchmark.evolution_strategy import EvolutionStrategy
from benchmark.experiment_runner import ExperimentRunner

problem = Sphere(10, -5.12, 5.12)
algorithm = EvolutionStrategy(mu=20, offspring_size=20)
runner = ExperimentRunner(max_evaluations=10000)

result = runner.run(algorithm, problem, seed=303000)

for evaluations in [20, 100, 1000, 5000, 10000]:
    fitness = result.best_history[evaluations-1]
    print(f"{evaluations} evaluations: {fitness:.6f}")
