import numpy as np

def soft_penalty(genome, adj_matrix, penalty_weight=10):
    """
    fitness = (number of unique nodes) - (number of missing edges * weight)
    """
    n = len(genome)
    unique_nodes = len(set(genome))
    
    invalid_edges = 0
    for i in range(n - 1):
        if adj_matrix[genome[i], genome[i+1]] == 0:
            invalid_edges += 1
            
    return unique_nodes - (invalid_edges * penalty_weight)


def hard_viability(genome, adj_matrix, **kwargs):
    """
    Strict strategy: If a single edge is missing, the fitness is 0.
    useful for comparing how 'hard' constraints affect evolution
    """
    n = len(genome)
    unique_nodes = len(set(genome))
    
    for i in range(n - 1):
        if adj_matrix[genome[i], genome[i+1]] == 0:
            return 0 
            
    return unique_nodes