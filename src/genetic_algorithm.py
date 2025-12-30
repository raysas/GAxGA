'''
---------------------------- 
GA module
----------------------------

contains implementation of the metaheuristic on graph, particulalry tailored for solving Hamiltonian Path problem
it has:

- GA parameters: population size, mutation rate, crossover rate, number of generations
- GA operators: selection, crossover, mutation
'''

import random
import networkx as nx # -- using graph children classes from nx

class GeneticAlgorithm:
    def __init__(self, graph:nx.Graph, population_size:int=100, mutation_rate:float=0.01, crossover_rate:float=0.7, generations:int=500):
        self.graph = graph
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        nodes = list(self.graph.nodes)
        for _ in range(self.population_size):
            individual = nodes[:] #important to copy the list cause of mutability
            random.shuffle(individual)
            population.append(individual)
        return population

    def fitness(self, individual):
        score = 0
        for i in range(len(individual) - 1):
            if self.graph.has_edge(individual[i], individual[i + 1]):
                score += 1
        return score

    def selection(self):
        weighted_population = [(self.fitness(ind), ind) for ind in self.population]
        total_fitness = sum(fit for fit, _ in weighted_population)
        probabilities = [fit / total_fitness for fit, _ in weighted_population]
        selected = random.choices(self.population, weights=probabilities, k=self.population_size)
        return selected

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1[:]
        
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        child[start:end] = parent1[start:end]
        
        pointer = 0
        for i in range(size):
            if child[i] is None:
                while parent2[pointer] in child:
                    pointer += 1
                child[i] = parent2[pointer]
        
        return child

    def mutate(self, individual):
        '''swap mutation operator
        swaps 2 genes with a probability defined by mutation rate
        >>> ind = [1,2,3,4,5]
        >>> ga.mutate(ind)
        [1,3,2,4,5]  #an example output
        '''
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(individual) - 1)
                individual[i], individual[j] = individual[j], individual[i]

    def run(self):
        for generation in range(self.generations):
            selected = self.selection()
            next_generation = []
            
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[i + 1]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                self.mutate(child1)
                self.mutate(child2)
                next_generation.extend([child1, child2])
            self.population = next_generation
        best_individual = max(self.population, key=self.fitness)
        return best_individual, self.fitness(best_individual)

if __name__ == "__main__":
    # Example usage
    G = nx.Graph()
    edges = [(0,1), (1,2), (2,3), (3,4), (4,0), (1,3)]
    G.add_edges_from(edges)

    ga = GeneticAlgorithm(G, population_size=50, mutation_rate=0.05, crossover_rate=0.8, generations=100)
    best_path, best_fitness = ga.run()
    print("Best Path:", best_path)
    print("Best Fitness:", best_fitness)