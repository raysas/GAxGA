import argparse
import random
import networkx as nx
import numpy as np
from generate_graphs import GraphFactory
import fitness as f

# Assuming your HamiltonianGA class is in a file named ga_model.py
from ga import HamiltonianGA

## --- RUNNER SCRIPT ---

def run_experiment(args):
    """
    Executes the GA based on command line arguments.
    """
    print(f"--- Running Experiment: {args.graph_type} graph with {args.nodes} nodes ---")
    
    # 1. Generate the Graph
    factory = GraphFactory(args.graph_type, n=args.nodes, m=args.m_edges)
    G = factory()

    # 2. Initialize the GA
    ga = HamiltonianGA(
        pop_size=args.pop_size,
        n_gen=args.generations,
        mutation_rate=args.mutation_rate,
        fitness_mode=args.fitness_mode,
        penalty_weight=args.penalty_weight,
        verbose=args.verbose,
        memoize=args.memoize
    )

    # 3. Fit the model
    ga.fit(G)

    # 4. Final Validation
    is_valid = ga.is_viable(G)
    
    print(f"\nFinal Results for {args.graph_type}:")
    print(f"Best Score: {ga.best_score_:.2f}")
    print(f"Viable Path Found: {is_valid}")
    
    if args.plot and is_valid:
        from plot import plot_fitness_evolution, plot_best_path
        plot_fitness_evolution(ga.history_).show()
        plot_best_path(G, ga.best_path_, ga.best_score_).show()

    return is_valid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA for Hamiltonian Path Discovery")

    # Graph Params
    parser.add_argument("--graph_type", type=str, default="scale_free", help="Type of graph (scale_free, erdos_renyi, etc.)")
    parser.add_argument("--nodes", type=int, default=20, help="Number of nodes in the graph")
    parser.add_argument("--m_edges", type=int, default=3, help="Edges to attach from a new node to existing nodes")

    # GA Hyperparameters
    parser.add_argument("--pop_size", type=int, default=100, help="Population size")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations")
    parser.add_argument("--mutation_rate", type=float, default=0.05, help="Mutation probability (0.0 to 1.0)")
    parser.add_argument("--fitness_mode", type=str, choices=['soft', 'hard'], default='soft')
    parser.add_argument("--penalty_weight", type=float, default=10.0)

    # Execution Options
    parser.add_argument("--verbose", action="store_true", help="Print progress per generation")
    parser.add_argument("--memoize", action="store_true", help="Store all genomes (memory intensive)")
    parser.add_argument("--plot", action="store_true", help="Show plots after completion")

    args = parser.parse_args()
    run_experiment(args)