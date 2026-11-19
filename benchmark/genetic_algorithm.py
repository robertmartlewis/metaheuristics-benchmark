import numpy as np

from benchmark.tsp import TSP
from benchmark.continuous_problem import ContinuousProblem

class GeneticAlgorithm:
    def __init__(self, population_size):
        if not isinstance(population_size, int) or isinstance(population_size, bool):
            raise TypeError("population_size must be an integer")
        if population_size < 3:
            raise ValueError("population_size must be >= 3")
        self.population_size = population_size

    def run(self, problem, evaluator, rng):
        if evaluator.max_evaluations < self.population_size:
            raise ValueError("Evaluation budget is smaller than population size")
        if not isinstance(problem, (TSP, ContinuousProblem)):
            raise TypeError("problem must be an instance of TSP or ContinuousProblem")

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
                    if isinstance(problem, TSP):
                        child = self._order_crossover(parent_a, parent_b, rng)
                    else:
                        child = self._arithmetic_crossover(parent_a, parent_b, rng)
                else:
                    child = parent_a.copy()

                if isinstance(problem, TSP):
                    mutated = self._swap_mutation(child, rng)
                else:
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
        noise = rng.normal(loc=0.0, scale=sigma, size=len(candidate))
        mutation_probability = 1.0 / len(candidate)
        mask = rng.random(size=len(candidate)) < mutation_probability
        masked_noise = noise * mask
        mutated_candidate = candidate + masked_noise
        return np.clip(mutated_candidate, problem.lower_bound, problem.upper_bound)

    def _order_crossover(self, parent_a, parent_b, rng):
        cuts = rng.choice(len(parent_a), size=2, replace=False)
        left, right = sorted(cuts)

        child = np.full(len(parent_a), -1, dtype=int)
        child[left:right + 1] = parent_a[left:right + 1]

        write_index = (right + 1) % len(parent_a)

        for i in range(len(parent_a)):
            index = (right + 1 + i) % len(parent_a)
            city = parent_b[index]

            if city not in child:
                child[write_index] = city
                write_index = (write_index + 1) % len(parent_a)

        return child

    def _swap_mutation(self, candidate, rng):
        swapped = candidate.copy()
        a, b = rng.choice(len(candidate), size=2, replace=False)
        swapped[a], swapped[b] = swapped[b], swapped[a]
        return swapped