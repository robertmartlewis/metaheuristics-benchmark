import numpy as np

class EvolutionStrategy:
    def __init__(self, mu, offspring_size):
        if not isinstance(mu, int) or isinstance(mu, bool):
            raise TypeError("mu must be integer")
        if mu < 1:
            raise ValueError("mu must be >= 1")

        if not isinstance(offspring_size, int) or isinstance(offspring_size, bool):
            raise TypeError("offspring_size must be an integer")
        if offspring_size < 1:
            raise ValueError("offspring_size must be >= 1")

        self.mu = mu
        self.offspring_size = offspring_size


    def run(self, problem, evaluator, rng):
        if evaluator.max_evaluations < self.mu:
            raise ValueError("Evaluation budget is smaller than mu")

        population = []
        fitness_values = []

        for _ in range(self.mu):
            candidate = problem.random_solution(rng)
            fitness = evaluator.evaluate(candidate)

            population.append(candidate)
            fitness_values.append(fitness)

        while evaluator.evaluations < evaluator.max_evaluations:
            offspring = []
            offspring_fitness = []

            for _ in range(self.offspring_size):
                if evaluator.evaluations >= evaluator.max_evaluations:
                    break

                parent_index = rng.integers(len(population))
                parent = population[parent_index]

                child = self._gaussian_mutation(parent, problem, rng)
                fitness = evaluator.evaluate(child)

                offspring.append(child)
                offspring_fitness.append(fitness)

            combined_population = population + offspring
            combined_fitness = fitness_values + offspring_fitness

            population, fitness_values = self._select_survivors(combined_population, combined_fitness)

    def _gaussian_mutation(self, parent, problem, rng):
        sigma = 0.1*(problem.upper_bound - problem.lower_bound)
        noise = rng.normal(loc=0.0, scale=sigma, size=len(parent))

        child = parent + noise

        return np.clip(child, problem.lower_bound, problem.upper_bound)

    def _select_survivors(self, combined_population, combined_fitness):
        sorted_indices = sorted(range(len(combined_population)), key=lambda i: combined_fitness[i])

        best_indices = sorted_indices[:self.mu]

        survivors = [combined_population[i] for i in best_indices]
        survivor_fitness = [combined_fitness[i] for i in best_indices]

        return survivors, survivor_fitness
