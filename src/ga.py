import numpy as np
import random
import networkx as nx
from sklearn.base import BaseEstimator
import fitness as f
from generate_graphs import GraphFactory

class HamiltonianGA(BaseEstimator):
    """
    Genetic Algorithm for approximating Hamiltonian paths in graphs.

    This class optimizes permutations of graph nodes using genetic
    operators such as ordered crossover and swap mutation. Fitness
    evaluation is propagated from a fitness module

    Parameters
    ----------
    pop_size : int, default=100
        Number of individuals in the population

    n_gen : int, default=200
        Number of generations to run.

    mutation_rate : float, default=0.02
        Probability of mutating an offspring.

    fitness_mode : {'soft', 'hard'}, default='soft'
        fitness evaluation strategy

    penalty_weight : float, default=10
        penalty weight used in the soft fitness function
        
    verbose : bool, default=False
        if True, prints progress for each generation
    """

    def __init__(self, pop_size=100, n_gen=200, mutation_rate=0.02, ox_rate=1,
                 fitness_mode='soft', penalty_weight=10, verbose=False, memoize=False):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.mutation_rate = mutation_rate
        self.ox_rate = ox_rate
        self.fitness_mode = fitness_mode
        self.penalty_weight = penalty_weight
        self.verbose = verbose
        self.memoize = memoize
        
        self.history_ = []      # -- stats and best path per gen
        self.all_genomes_ = np.array([])  # -- full population snapshots (if memoize=True)
        self.best_path_ = None
        self.best_score_ = -float('inf')
        self.best_path_idx_ = None

        # -- printing hyperparameters
        if self.verbose:
            print(f'[GA INIT] pop_size={self.pop_size}, n_gen={self.n_gen}, mutation_rate={self.mutation_rate}, ox_rate={self.ox_rate}, fitness_mode={self.fitness_mode}, penalty_weight={self.penalty_weight} (for soft mode)')

    def _ordered_crossover(self, p1, p2):
        """Perform ordered crossover (OX) between two permutations."""
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b] = p1[a:b]
        p2_filtered = [item for item in p2 if item not in child]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_filtered[idx]
                idx += 1
        return child

    def _mutate(self, genome):
        """apply swap mutation to a permutation."""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(genome)), 2)
            genome[i], genome[j] = genome[j], genome[i]
        return genome

    def is_viable(self, G, path=None):
        '''
        checking if the best path (trained) is viable, reasons for invalid HP:  
            - eges in best path not in graph
            - not all nodes visited exactly once

        '''
        if path is None:
            path=self.best_path_
        if path is None:
            print(f" [!not viable] | no best path found, make sure to run fit() first")
            return False

        # -- a: check if every step in the path is a valid edge in the graph
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if not G.has_edge(u, v):
                print(f" [!not viable] | path invalid: no edge between {u} and {v}")
                return False

        # -- b: check if it visits every node exactly once
        # --- unique visits
        if len(set(path)) != len(G.nodes):
            print(f" [!not viable] | path invalid: nodes not visited exactly once")
            return False
        # --- all nodes visited
        if len(path) != len(G.nodes):
            print(f"    [!not viable] | path invalid: not all nodes visited")
            return False

        print(f" [✓ viable] | valid Hamiltonian Path!")
        return True


    def fit(self, G, y=None):
        '''fit the genetic algorithm to graph G'''
        adj_matrix = nx.to_numpy_array(G)
        n_nodes = len(G.nodes)
        node_list = list(G.nodes)
        idx_to_node = {i: node for i, node in enumerate(node_list)}

        if self.fitness_mode == 'soft':
            fitness_func = lambda ind: f.soft_penalty(ind, adj_matrix, self.penalty_weight)
        else:
            fitness_func = lambda ind: f.hard_viability(ind, adj_matrix)

        pop = [random.sample(range(n_nodes), n_nodes) for _ in range(self.pop_size)]
        
        self.history_ = []
        self.all_genomes_ = np.array([]) if self.memoize else []
        self.best_score_ = -float('inf')
        self.best_path_idx_ = None

        if self.verbose:
            print(f"Starting GA: {n_nodes} nodes, {self.pop_size} population, {self.n_gen} generations.")
            print("-" * 50)

        for gen in range(self.n_gen):
            scores = [fitness_func(ind) for ind in pop]

            current_max = max(scores)
            current_mean = np.mean(scores)
            best_idx = np.argmax(scores)
            
            if current_max > self.best_score_:
                self.best_score_ = current_max
                self.best_path_idx_ = pop[best_idx].copy()

            gen_best_path = [idx_to_node[i] for i in pop[best_idx]]

            if self.memoize:
                gen_snapshot = [[idx_to_node[i] for i in genome] for genome in pop]
                self.all_genomes_ = np.append(self.all_genomes_, [gen_snapshot], axis=0) if self.all_genomes_.size else np.array([gen_snapshot])

            self.history_.append({
                'max': current_max,
                'mean': current_mean,
                'best_path': gen_best_path
            })

            if self.verbose and (gen % 10 == 0 or gen == self.n_gen - 1):
                print(f"Gen {gen:4d} | Max Fitness: {current_max:7.2f} | Mean Fitness: {current_mean:7.2f} ", end='|')
                self.is_viable(G, path=gen_best_path)

            sorted_indices = np.argsort(scores)[::-1]
            sorted_pop = [pop[i] for i in sorted_indices]

            next_pop = []
            while len(next_pop) < self.pop_size:
                parents = random.sample(sorted_pop[:int(self.pop_size / 2)], 2)

                if random.random() < self.ox_rate:
                    child = self._ordered_crossover(parents[0], parents[1])
                else:
                    child = parents[0].copy()
            
                child = self._mutate(child) # -- embedded mut rate in function
                next_pop.append(child)

            pop = next_pop

        self.best_path_ = [idx_to_node[i] for i in self.best_path_idx_]
        
        if self.verbose:
            print("-" * 50)
            print(f"Optimization Finished.")
            print(f"Global Best Fitness found: {self.best_score_}")

        return self

    def score(self, G, y=None):
        """return the best fitness score obtained during optimization"""
        return self.best_score_

if __name__ == "__main__":
    from generate_graphs import GraphFactory

    n=50
    m=8
    mode='soft'

    G=GraphFactory('scale_free',n=n,m=m)()

    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = random.randint(1, 10)
    ga = HamiltonianGA(pop_size=100, n_gen=100, mutation_rate=0.05,
                       fitness_mode=mode, penalty_weight=10, verbose=True, memoize=True)
    ga.fit(G)
    ga.is_viable(G)
    print("\nResulting Path:", ga.best_path_)

    print(ga.all_genomes_.shape)  # -- (n_gen, pop_size, n_nodes)

    from plot import plot_fitness_evolution, plot_best_path, plot_population_distribution
    # fitness_evol=plot_fitness_evolution(ga.history_)
    # # fitness_evol.savefig(f"assets/fitness_evolution_{n}_{m}_{mode}.png")
    # fitness_evol.show()
    # best_path_graph=plot_best_path(G, ga.best_path_, ga.best_score_)
    # # best_path_graph.savefig(f"assets/best_path_{n}_{m}_{mode}.png")
    # best_path_graph.show()

    # population_distribution=plot_population_distribution(ga.all_genomes_, lambda ind: f.soft_penalty(ind, nx.to_numpy_array(G), penalty_weight=10))
    # # population_distribution.savefig(f"assets/population_distribution_{n}_{m}_{mode}.png")
    # population_distribution.show()

    # from viz import animate_ga_evolution
    # ani = animate_ga_evolution(G, ga.history_, interval=50)
    # ani.save(f"assets/ga_evolution_{n}_{m}_{mode}.gif", writer='pillow')

    muts=[0.01,0.05,0.1]
    co=[0.7,0.8,0.9]

    # ---- hyperparameter tuning example (gridsearch under experimentation still)
    for mut in muts:
        for ox in co:
            print(f"\n--- Running GA with mutation_rate={mut}, ox_rate={ox} ---")
            ga = HamiltonianGA(pop_size=100, n_gen=100, mutation_rate=mut,
                               ox_rate=ox, fitness_mode=mode, penalty_weight=10, verbose=True, memoize=True)
            ga.fit(G)
            ga.is_viable(G)
            print("Best Fitness:", ga.best_score_)
            print("Best Path:", ga.best_path_)

            fitness_evol=plot_fitness_evolution(ga.history_)
            fitness_evol.savefig(f"assets/fitness_evolution_mut{mut}_ox{ox}.png")
            # fitness_evol.show()

            best_path=plot_best_path(G, ga.best_path_, ga.best_score_)
            best_path.savefig(f"assets/best_path_mut{mut}_ox{ox}.png")
            # best_path.show()

            population_distribution=plot_population_distribution(ga.all_genomes_, lambda ind: f.soft_penalty(ind, nx.to_numpy_array(G), penalty_weight=10))
            population_distribution.savefig(f"assets/population_distribution_mut{mut}_ox{ox}.png")
            # population_distribution.show()