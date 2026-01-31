'''
generate_graphs module: utilities to create graphs for testing the HamiltonianGA  
> [!NOTE] all graphs are genrated as NetworkX graphs (and with networkx functions)

To wrap it in a more suitable interface, 
we will be using a generator class GraphGenerator that follow an oop paradigm known as Factory Method Pattern

Factory Method Pattern: define an interface for creating an object, but let subclasses decide which class to instantiate
As we will be considerign different type sof graphs (complete, random, real world BUT ALL CYCLIC GRAPHS), we will create a base class GraphGenerator with a method generate_graph() (or even a __call__() method) that will be implemented by subclasses for each type of graph

CLASSES:
- GraphGenerator (abstract base class)
- CompleteGraphGenerator (subclass of GraphGenerator)
- RandomGraphGenerator (subclass of GraphGenerator)
- ScaleFreeGraphGenerator (subclass of GraphGenerator)
- GraphFactory (factory class to create graph generators based on type)

'''

import networkx as nx
from abc import ABC,abstractmethod
import random

class GraphGenerator(ABC):
    """base class for graph generators"""   
    @abstractmethod
    def generate_graph(self):
        raise NotImplementedError("[!] subclasses must implement this method")
    def __call__(self):
        return self.generate_graph()
    
class CompleteGraphGenerator(GraphGenerator):
    """Generates a complete graph with n nodes"""
    def __init__(self, n):
        self.n = n
    
    def generate_graph(self):
        G = nx.complete_graph(self.n)
        # -- random weights to edges
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = random.randint(1, 10)
        return G
    
class RandomGraphGenerator(GraphGenerator):
    """Generates a random graph with n nodes and edge probability p"""
    def __init__(self, n, p=0.3):
        self.n = n
        self.p = p
    
    def generate_graph(self):
        # 1. start with a guaranteed cycle: 0-1-2-...-(n-1)-0: ensures at least one Hamiltonian Path/Cycle exists.
        G = nx.cycle_graph(self.n)
        
        # 2. extra random edges to make it a challenge (erdos renyi style)
        for u in range(self.n):
            for v in range(u + 1, self.n):
                if random.random() < self.p:
                    G.add_edge(u, v)

        # 3. -- random weights to all edges
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = random.randint(1, 10)
            
        return G
    
class ScaleFreeGraphGenerator(GraphGenerator):
    """Generates a scale-free graph with n nodes using Barabasi Albert model"""
    def __init__(self, n, m=2):
        self.n = n
        self.m = m
    
    def generate_graph(self):
        # 1 the Scale-Free structure
        G = nx.barabasi_albert_graph(self.n, self.m)
        
        # 2. a random permutation of nodes
        nodes = list(range(self.n))
        random.shuffle(nodes)
        
        # 3. the cycle based on the shuffled order
        for i in range(self.n):
            u = nodes[i]
            v = nodes[(i + 1) % self.n]
            if not G.has_edge(u, v):
                G.add_edge(u, v)
        
        # 4. weights
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = random.randint(1, 10)
            
        return G
    

class GraphFactory:
    """Factory class to create graph generators based on type
    """    
    _registry = {
        "complete": CompleteGraphGenerator,
        "random": RandomGraphGenerator,
        "scale_free": ScaleFreeGraphGenerator,
    }

    @staticmethod
    def create(graph_type: str, **kwargs) -> GraphGenerator:
        ''' Create a graph generator based on the specified type and parameters.
        Args:
            graph_type (str): Type of graph to generate ('complete', 'random', 'scale_free').
            **kwargs: Additional parameters required for the specific graph generator (p: probability for random graph, m: edges to attach for scale-free graph)
        Returns:
            GraphGenerator: An instance of the requested graph generator
        Raises:
            ValueError: If an unknown graph type is provided
        '''
        try:
            generator_cls = GraphFactory._registry[graph_type]
        except KeyError:
            raise ValueError(f"[!] unknown graph type: {graph_type}")

        return generator_cls(**kwargs)
    
    # -- callable interface
    def __init__(self, graph_type: str, **kwargs):
        self.generator = self.create(graph_type, **kwargs)

    def __call__(self):
        return self.generator()