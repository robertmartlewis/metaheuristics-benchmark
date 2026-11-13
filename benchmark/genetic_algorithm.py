import numpy as np


class GeneticAlgorithm:
    def __init__(self, population_size):
        self.population_size = population_size

    def run(self, problem, evaluator, rng):
        if evaluator.max_evaluations < self.population_size:
            raise ValueError("Evaluation budget is smaller than population size")

        population = []
        fitness_values = []

        for _ in range(0, self.population_size):
            candidate = problem.random_solution(rng)
            fitness = evaluator.evaluate(candidate)
            population.append(candidate)
            fitness_values.append(fitness)

        while evaluator.evaluations < evaluator.max_evaluations:
            best_index = min(range(len(population)), key=lambda i: fitness_values[i])
            new_population = [population[best_index].copy()]
            new_fitness_values = [fitness_values[best_index]]

            while len(new_population) < self.population_size and evaluator.evaluations < evaluator.max_evaluations:
                index_1 = self._tournament_select(fitness_values, rng)
                index_2 = self._tournament_select(fitness_values, rng)

                parent_a = population[index_1]
                parent_b = population[index_2]

                if rng.random() < 0.9:
                    child = self._arithmetic_crossover(parent_a, parent_b, rng)
                else:
                    child = parent_a.copy()

                mutated = self._gaussian_mutation(child, problem, rng)
                fitness = evaluator.evaluate(mutated)

                new_population.append(mutated)
                new_fitness_values.append(fitness)

            population = new_population
            fitness_values = new_fitness_values


    def _tournament_select(self, fitness_values, rng):
        indices = rng.choice(len(fitness_values), size=3, replace=False)
        return min(indices, key=lambda x: fitness_values[x])

    def _arithmetic_crossover(self, parent_a, parent_b, rng):
        alpha = rng.random()
        return parent_a*alpha + parent_b*(1-alpha)

    def _gaussian_mutation(self, candidate, problem, rng):
        sigma = 0.1 * (problem.upper_bound - problem.lower_bound)

        noise = rng.normal(
            loc=0.0,
            scale=sigma,
            size=len(candidate)
        )

        mutated_candidate = candidate + noise
        return np.clip(mutated_candidate, problem.lower_bound, problem.upper_bound)

