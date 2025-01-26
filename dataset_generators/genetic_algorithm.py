import random
from functools import lru_cache
import time
from utility import *
import numpy as np


class GeneticAlgorithm:

    population_size = 100
    selection_k = 8
    mutation_probability = 0.8
    initial_maximum_mutation_ratio = 0.4
    initial_minimum_mutation_ratio = 0.2
    final_maximum_mutation_ratio = 0.05
    final_minimum_mutation_ratio = 0.0
    minimum_mutation_threshold = 1
    mutation_decay = 50
    crossover_probability = 0.8
    initial_maximum_crossover_ratio = 0.4
    initial_minimum_crossover_ratio = 0.2
    final_maximum_crossover_ratio = 0.05
    final_minimum_crossover_ratio = 0.0
    minimum_crossover_threshold = 1
    crossover_decay = 50
    maximum_iterations = 1000
    moving_difference_window_size = 200
    convergence_tolerance = 0.02
    degradation_tolerance = -1.0
    current_iterations = 1000

    def __init__(self,
                 fitness_parameters,
                 compound_attribute_weight_list,
                 compound_attribute_class_reference, compound_attribute_class_count_table, compound_attribute_class_sparsity_list,
                 m_max, m_ideal, a_max, a_ideal,
                 compound_attribute_count, compound_attribute_class_count,
                 chromosome_dimensions, priority_chromosome_i, semi_priority_chromosome_i,
                 gene_attribute_expression_pool,
                 log_path=None, save_path=None,
                 create_save=False, load_save=False):

        self.fitness_parameters = fitness_parameters
        self.worst_fitness = 0.0

        self.chromosome_dimensions = chromosome_dimensions
        self.priority_chromosome_i = priority_chromosome_i
        self.semi_priority_chromosome_i = semi_priority_chromosome_i
        self.individual_size = np.sum(self.chromosome_dimensions)
        self.chromosome_offsets = np.append(np.array([0]), np.cumsum(self.chromosome_dimensions)[:-1])
        self.chromosome_count = self.chromosome_dimensions.shape[0]

        self.log_path = log_path
        self.save_path = save_path
        self.create_save = create_save
        self.load_save = load_save

        self.maximum_mutation_ratio = None
        self.minimum_mutation_ratio = None
        self.maximum_crossover_ratio = None
        self.minimum_crossover_ratio = None

        self.gene_attribute_expression_pool = gene_attribute_expression_pool
        self.gene_pool_size = len(self.gene_attribute_expression_pool)
        self.gene_pool = np.arange(self.gene_pool_size)

        self.compound_attribute_weight_list = compound_attribute_weight_list
        self.compound_attribute_class_reference = compound_attribute_class_reference
        self.compound_attribute_class_count_table = compound_attribute_class_count_table
        self.compound_attribute_class_sparsity_list = compound_attribute_class_sparsity_list
        self.compound_attribute_count = compound_attribute_count
        self.compound_attribute_class_count = compound_attribute_class_count
        self.total_compound_attribute_weight = np.sum(self.compound_attribute_weight_list)
        if self.compound_attribute_class_count > 0:
            self.compound_attribute_class_sum = np.bincount(self.compound_attribute_class_reference, minlength=self.compound_attribute_count)
        self.m_max = m_max
        self.m_ideal = m_ideal
        self.a_max = a_max
        self.a_ideal = a_ideal

        self.individual = None

    # ************************************************ INITIALISER ***************************************************

    def run(self):

        print("Running Genetic Algorithm...")

        if not (self.individual_size == self.gene_pool_size and len(self.chromosome_dimensions) == 1) and (self.compound_attribute_class_count != 0):

            has_converged = False

            iteration = 0
            best_fitness_moving_difference = 0
            average_fitness_list = np.array([], dtype=np.float64)
            best_fitness_list = np.array([], dtype=np.float64)

            best_individual = None
            absolute_best_individual = None

            historical_best_fitness_list, historical_absolute_best_fitness, historical_absolute_best_i,  historical_iterations = self.load_best_fitness_history()
            population = self.load_population()

            while iteration < (self.current_iterations - 1):

                self.update_change_ratios(historical_iterations)

                best_fitness_moving_difference = self.update_moving_difference(best_fitness_moving_difference, historical_best_fitness_list)

                start = time.time()
                population_fitness = self.determine_population_fitness(population)
                end = time.time()
                # print(f"fitness time: {end - start}")

                average_fitness = np.mean(population_fitness)
                best_i = np.argmax(population_fitness)
                best_fitness = population_fitness[best_i]

                if iteration % 50 == 0:
                    print(f"iteration -> {iteration + 1}")
                    print(f"average fitness: {average_fitness}")
                    print(f"best fitness: {best_fitness}")

                has_converged = self.determine_convergence(best_fitness, best_fitness_moving_difference, historical_iterations)

                average_fitness_list = np.append(average_fitness_list, average_fitness)
                best_fitness_list = np.append(best_fitness_list, best_fitness)
                historical_best_fitness_list = np.append(historical_best_fitness_list, best_fitness)
                best_individual = population[best_i]

                if has_converged:
                    break

                start = time.time()
                population = self.select_population(population, population_fitness)
                population = self.crossover_population(population)
                population = self.mutate_population(population)
                end = time.time()
                # print(f"mutation time: {end - start}")

                iteration += 1
                historical_iterations += 1

                if best_fitness > historical_absolute_best_fitness:
                    historical_absolute_best_fitness = best_fitness
                    absolute_best_individual = best_individual
                    historical_absolute_best_i = historical_iterations

            if not has_converged:

                self.update_change_ratios(historical_iterations)

                best_fitness_moving_difference = self.update_moving_difference(best_fitness_moving_difference, historical_best_fitness_list)

                start = time.time()
                population_fitness = self.determine_population_fitness(population)
                end = time.time()
                # print(f"fitness time: {end - start}")

                average_fitness = np.mean(population_fitness)
                best_i = np.argmax(population_fitness)
                best_fitness = population_fitness[best_i]

                print(f"iteration -> {iteration + 1}")
                print(f"average fitness: {average_fitness}")
                print(f"best fitness: {best_fitness}")

                has_converged = self.determine_convergence(best_fitness, best_fitness_moving_difference, historical_iterations)

                average_fitness_list = np.append(average_fitness_list, average_fitness)
                best_fitness_list = np.append(best_fitness_list, best_fitness)
                best_individual = population[best_i]
                historical_iterations += 1
                if best_fitness > historical_absolute_best_fitness:
                    historical_absolute_best_fitness = best_fitness
                    absolute_best_individual = best_individual
                    historical_absolute_best_i = historical_iterations

            # print(f"absolute best fitness: {historical_absolute_best_fitness}")

        else:

            has_converged = True

            average_fitness_list = np.array([0], dtype=np.float64)
            best_fitness_list = np.array([0], dtype=np.float64)
            historical_absolute_best_fitness = 1
            historical_absolute_best_i = 1
            historical_iterations = 1

            best_individual = self.gene_pool
            absolute_best_individual = self.gene_pool

        self.update_log(average_fitness_list, best_fitness_list, historical_absolute_best_fitness, historical_absolute_best_i, has_converged, historical_iterations)

        if not has_converged:
            self.update_save(population)
        else:
            self.delete_save()

        print()

        return best_individual, absolute_best_individual

    def load_population(self):
        if self.load_save and os.path.exists(self.save_path):
            population = np.load(self.save_path, allow_pickle=True)
        else:
            population = self.create_population()
        return population

    def load_best_fitness_history(self):
        if self.load_save and os.path.exists(self.save_path) and os.path.exists(self.log_path):
            log_dictionary = read_data_from_file_(self.log_path)
            historical_best_fitness_list = np.array(log_dictionary["best_fitness"])
            historical_absolute_best_fitness = log_dictionary["absolute_best_fitness"]
            historical_absolute_best_i = log_dictionary["absolute_best_i"]
            historical_iterations = log_dictionary["iterations"]
        else:
            historical_best_fitness_list = np.array([], dtype=np.float64)
            historical_absolute_best_fitness = self.worst_fitness
            historical_absolute_best_i = 0
            historical_iterations = 0
        return historical_best_fitness_list, historical_absolute_best_fitness, historical_absolute_best_i, historical_iterations

    def update_log(self, average_fitness_list, best_fitness_list, absolute_best_fitness, absolute_best_i, has_converged, iterations):
        if self.load_save and os.path.exists(self.log_path):
            log_dictionary = read_data_from_file_(self.log_path)
            log_dictionary["average_fitness"].extend(average_fitness_list.tolist())
            log_dictionary["best_fitness"].extend(best_fitness_list.tolist())
            log_dictionary["absolute_best_fitness"] = float(absolute_best_fitness)
            log_dictionary["absolute_best_i"] = absolute_best_i
            log_dictionary["has_converged"] = has_converged
            log_dictionary["iterations"] = iterations
        else:
            log_dictionary = {"average_fitness": average_fitness_list.tolist(),
                              "best_fitness": best_fitness_list.tolist(),
                              "absolute_best_fitness": float(absolute_best_fitness),
                              "absolute_best_i": absolute_best_i,
                              "has_converged": has_converged,
                              "iterations": iterations}
        save_data_to_file_(self.log_path, log_dictionary)

    def update_save(self, population):
        if self.create_save:
            np.save(self.save_path, population, allow_pickle=True)

    def delete_save(self):
        if os.path.exists(self.save_path) and self.load_save:
            os.remove(self.save_path)

    def update_moving_difference(self, moving_difference, historical_best_fitness_list):
        if historical_best_fitness_list.shape[0] > self.moving_difference_window_size:
            moving_difference =  historical_best_fitness_list[-1] - historical_best_fitness_list[-(self.moving_difference_window_size + 1)]
        else:
            moving_difference = 1.0
        return moving_difference

    def update_change_ratios(self, historical_iterations):

        def update_change_ratio(initial_ratio, final_ratio, decay, iteration):
            ratio = max(initial_ratio / float(2 ** (int(iteration / decay))), final_ratio)
            return ratio

        self.maximum_mutation_ratio = update_change_ratio(self.initial_maximum_mutation_ratio, self.final_maximum_mutation_ratio, self.mutation_decay, historical_iterations)
        self.minimum_mutation_ratio = update_change_ratio(self.initial_minimum_mutation_ratio, self.final_minimum_mutation_ratio, self.mutation_decay, historical_iterations)
        self.maximum_crossover_ratio = update_change_ratio(self.initial_maximum_crossover_ratio, self.final_maximum_crossover_ratio, self.crossover_decay, historical_iterations)
        self.minimum_crossover_ratio = update_change_ratio(self.initial_minimum_crossover_ratio, self.final_minimum_crossover_ratio, self.crossover_decay, historical_iterations)

    def determine_convergence(self, best_fitness, best_fitness_moving_difference, iteration):
        has_converged = False
        if (best_fitness_moving_difference < self.convergence_tolerance) or (iteration >= self.maximum_iterations):
            has_converged = True
        return has_converged

    def evaluate(self, individual):
        self.individual = individual
        attribute_fitness = self.determine_attribute_fitness()
        compound_attribute_frequency_table = self.determine_compound_attribute_class_frequency_table(individual)
        return compound_attribute_frequency_table, attribute_fitness

    # ************************************************ GENETIC ALGORITHM ***************************************************

    def create_population(self):
        population = np.empty((self.population_size, self.individual_size), dtype=np.int32)
        for i in range(self.population_size):
            population[i] = self.create_individual()
        return population

    def create_individual(self):
        individual = np.copy(self.gene_pool)
        np.random.shuffle(individual)
        if self.individual_size < self.gene_pool_size:
            individual = individual[:self.individual_size]
        return individual

    def determine_population_fitness(self, population):
        population_fitness = np.zeros(self.population_size, dtype=np.float64)
        for i in range(self.population_size):
            self.individual = population[i]
            individual_fitness = self.determine_attribute_fitness()
            population_fitness[i] = individual_fitness
        return population_fitness

    def determine_attribute_fitness(self):
        compound_attribute_class_frequency_table = self.determine_compound_attribute_class_frequency_table(self.individual)
        base_compound_attribute_class_fitness_list = self.determine_base_compound_attribute_class_fitness_table(compound_attribute_class_frequency_table)
        compound_attribute_fitness_list = np.bincount(self.compound_attribute_class_reference, weights=base_compound_attribute_class_fitness_list, minlength=self.compound_attribute_count)
        compound_attribute_fitness_list = np.divide(compound_attribute_fitness_list, self.compound_attribute_class_sum, out=np.zeros_like(compound_attribute_fitness_list), where=self.compound_attribute_class_sum != 0)
        compound_attribute_fitness_list *= self.compound_attribute_weight_list
        attribute_fitness = np.sum(compound_attribute_fitness_list)
        attribute_fitness /= self.total_compound_attribute_weight
        return attribute_fitness

    def determine_compound_attribute_class_frequency_table(self, individual):
        compound_attribute_class_frequency_table = np.zeros((self.compound_attribute_class_count, self.chromosome_count), dtype=np.int32)
        for i in range(self.chromosome_count):
            chromosome_start = self.chromosome_offsets[i]
            chromosome_end = chromosome_start + self.chromosome_dimensions[i]
            for j in range(chromosome_start, chromosome_end):
                compound_attribute_class_frequency_table[:, i] += np.bincount(self.gene_attribute_expression_pool[individual[j]], minlength=self.compound_attribute_class_count)
        return compound_attribute_class_frequency_table

    def determine_base_compound_attribute_class_fitness_table(self, compound_attribute_class_frequency_table):

        a = compound_attribute_class_frequency_table

        a_is_a_ideal = np.isclose(a, self.a_ideal)

        a_is_else = ~(np.isclose(a, 0)) & ~(np.isclose(a, self.a_max)) & ~a_is_a_ideal

        a_else = a[a_is_else]
        a_ideal_else = self.a_ideal[a_is_else]
        a_max_else = self.a_max[a_is_else]

        d_max = np.maximum(a_ideal_else, a_max_else - a_ideal_else)
        d = np.absolute(a_else - a_ideal_else)
        d_norm = d / d_max
        fi_else = 1.0 / (1 + np.exp(8.0 * (d_norm - 0.5)))

        fi = np.zeros((self.compound_attribute_class_count, self.chromosome_count), dtype=np.float64)
        fi[a_is_a_ideal] = 1.0
        fi[a_is_else] = fi_else

        f = np.mean(fi, axis=1)

        return f

    def select_population(self, population, population_fitness):
        new_population = np.empty((self.population_size, self.individual_size), dtype=np.int32)
        for i in range(self.population_size):
            random_population_j = np.random.choice(self.population_size, self.selection_k, replace=False)
            random_fitness_j = population_fitness[random_population_j]
            best_m = np.argmax(random_fitness_j)
            best_j = random_population_j[best_m]
            new_population[i] = population[best_j]
        return new_population

    def mutate_population(self, population):
        for i in range(self.population_size):
            population[i] = self.mutate(population[i])
        return population

    def mutate(self, individual):

        if random.uniform(0, 1) <= self.mutation_probability:

            if self.individual_size < self.gene_pool_size:
                individual = np.append(individual, np.delete(self.gene_pool, individual))
                chromosome_count = self.chromosome_count + 1
                chromosome_dimensions = np.append(self.chromosome_dimensions, [self.gene_pool_size - self.individual_size])
                chromosome_offsets = np.append(np.array([0]), np.cumsum(chromosome_dimensions)[:-1])
            else:
                chromosome_count = self.chromosome_count
                chromosome_dimensions = self.chromosome_dimensions
                chromosome_offsets = self.chromosome_offsets

            c = random.sample(range(chromosome_count), 2)
            c1 = c[0]
            c2 = c[1]

            c1_start = chromosome_offsets[c1]
            c1_end = chromosome_offsets[c1] + chromosome_dimensions[c1]
            c2_start = chromosome_offsets[c2]
            c2_end = chromosome_offsets[c2] + chromosome_dimensions[c2]

            maximum_mutation_count = min(chromosome_dimensions[c1], chromosome_dimensions[c2])
            mutation_count = self.determine_change_count(maximum_mutation_count, self.minimum_mutation_ratio, self.maximum_mutation_ratio, self.minimum_mutation_threshold)

            c1_genes_for_mutation = np.random.choice(np.arange(c1_start, c1_end), mutation_count, replace=False)
            c2_genes_for_mutation = np.random.choice(np.arange(c2_start, c2_end), mutation_count, replace=False)

            temp = np.copy(individual[c1_genes_for_mutation])
            individual[c1_genes_for_mutation] = individual[c2_genes_for_mutation]
            individual[c2_genes_for_mutation] = temp

            individual = individual[:self.individual_size]

        return individual

    def crossover_population(self, population):
        new_population = np.empty((self.population_size, self.individual_size), dtype=np.int32)
        for i in range(self.population_size):
            mother_i = np.random.randint(self.population_size)
            father_i = np.random.randint(self.population_size)
            mother = population[mother_i]
            father = population[father_i]
            child = self.crossover(mother, father)
            new_population[i] = child
        return new_population

    def crossover(self, mother, father):

        child = np.copy(mother)

        if random.uniform(0, 1) <= self.crossover_probability:

            if self.chromosome_count != 1:

                c = random.sample(range(self.chromosome_count), 2)
                c1 = c[0]
                c2 = c[1]

                c1_start = self.chromosome_offsets[c1]
                c1_end = self.chromosome_offsets[c1] + self.chromosome_dimensions[c1]
                c2_start = self.chromosome_offsets[c2]
                c2_end = self.chromosome_offsets[c2] + self.chromosome_dimensions[c2]

                c1_to_c2_comparison = c1_start + np.nonzero(np.isin(mother[c1_start: c1_end], father[c2_start: c2_end]))[0]
                c2_to_c1_comparison = c2_start + np.nonzero(np.isin(mother[c2_start: c2_end], father[c1_start: c1_end]))[0]

                maximum_crossover_count = min(c1_to_c2_comparison.shape[0], c2_to_c1_comparison.shape[0])
                crossover_count = self.determine_change_count(maximum_crossover_count, self.minimum_crossover_ratio, self.maximum_crossover_ratio, self.minimum_crossover_threshold)

                c1_genes_for_crossover = np.random.choice(c1_to_c2_comparison, crossover_count, replace=False)
                c2_genes_for_crossover = np.random.choice(c2_to_c1_comparison, crossover_count, replace=False)

                child[c1_genes_for_crossover] = mother[c2_genes_for_crossover]
                child[c2_genes_for_crossover] = mother[c1_genes_for_crossover]

            else:

                mother_to_father_comparison = np.nonzero(np.isin(mother, father, invert=True))[0]
                father_to_mother_comparison = np.nonzero(np.isin(father, mother, invert=True))[0]

                maximum_crossover_count = min(mother_to_father_comparison.shape[0], father_to_mother_comparison.shape[0])
                crossover_count = self.determine_change_count(maximum_crossover_count, self.minimum_crossover_ratio, self.maximum_crossover_ratio, self.minimum_crossover_threshold)

                mother_genes_for_crossover = np.random.choice(mother_to_father_comparison, crossover_count, replace=False)
                father_genes_for_crossover = np.random.choice(father_to_mother_comparison, crossover_count, replace=False)

                child[mother_genes_for_crossover] = father[father_genes_for_crossover]

        return child

    # ************************************************ HELPERS ***************************************************

    @staticmethod
    def determine_change_count(maximum_change_count, minimum_change_ratio, maximum_change_ratio, minimum_change_threshold):
        change_ratio = random.uniform(minimum_change_ratio, maximum_change_ratio)
        change_count = int(round(maximum_change_count * change_ratio))
        if change_count < minimum_change_threshold and maximum_change_count != 0:
            change_count = min(minimum_change_threshold, maximum_change_count)
        return change_count

