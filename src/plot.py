import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_fitness_evolution(history):
    """
    Plots the convergence of the GA.
    Input: ga.history_
    """
    max_scores = [h['max'] for h in history]
    mean_scores = [h['mean'] for h in history]
    generations = range(len(history))

    plt.figure(figsize=(10, 5))
    plt.plot(generations, max_scores, label='Max Fitness', color='#1b9e77', linewidth=2)
    plt.plot(generations, mean_scores, label='Mean Fitness', color='#d95f02', linestyle='--')
    
    plt.title('Evolutionary Progress: Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()

    return plt



def plot_best_path(G, best_path, score):
    """
    Visualizes the best found path on the graph topology.
    """
    pos = nx.spring_layout(G, seed=42) # Consistent layout
    
    plt.figure(figsize=(8, 8))
    
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color='#f0f0f0', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')

    path_edges = list(zip(best_path, best_path[1:]))
    valid_edges = [e for e in path_edges if G.has_edge(*e)]
    invalid_edges = [e for e in path_edges if not G.has_edge(*e)]

    nx.draw_networkx_edges(G, pos, edgelist=valid_edges, width=4, edge_color='#2ca02c', label='Valid Edge')
    nx.draw_networkx_edges(G, pos, edgelist=invalid_edges, width=4, edge_color='#d62728', style='dotted', label='Missing Edge')

    plt.title(f"Final Solution Visualization\nBest Score: {score}")
    plt.legend(loc='upper left')
    plt.axis('off')
    # plt.show()

    return plt



def plot_population_distribution(all_genomes, fitness_func):
    """
    Plots a histogram of fitness distribution for Gen 0 vs the Final Gen.
    Requires memoize=True.
    """
    if len(all_genomes) == 0:
        print("Error: No memoized data found. Set memoize=True in GA.")
        return

    initial_scores = [fitness_func(ind) for ind in all_genomes[0]]
    final_scores = [fitness_func(ind) for ind in all_genomes[-1]]

    plt.figure(figsize=(10, 5))
    plt.hist(initial_scores, bins=20, alpha=0.5, label='Generation 0', color='gray')
    plt.hist(final_scores, bins=20, alpha=0.7, label=f'Generation {len(all_genomes)-1}', color='blue')
    
    plt.title('Shift in Population Fitness Distribution')
    plt.xlabel('Fitness Value')
    plt.ylabel('Number of Individuals')
    plt.legend()
    # plt.show()

    return plt