# GAxGA
_GA²: Genetic Algorithms for Genome Assembly via Hamiltonian Path Optimization_

## Problem Definition

The Hamiltonian Path is defined as a path in a graph \( G = (V, E) \) that visits each vertex \( v \in V \) exactly once. This problem is NP-complete (can be verified through a polynomial-time reduction from the Traveling Salesman Problem). I chose it because of its relevance in bioinformatics, particularly in DNA fragment assembly, which I will be describing an application as a supplemntary to the project (graph is made of k-mer nodes and edges represent overlaps).

<p align="center">
  <img src="assets/HAG.png" alt="Hamiltonian Path Example" width="400"/>
</p>

For this project we will be targetting undirected graphs, to find a Hamiltonian Path in a given using Genetic Algorithms.  

> **Plan**: 
> - [x] Graph generation: random graphs with a guaranteed "hidden" path to verify success   
> - [x] HamiltonianGA: class that plugs into GridSearchCV for auto-tuning on different parameters
> - [x] permutation encoding with Ordered Crossover (OX) and Swap Mutation to visit every vertex exactly once
> - [ ] Viability logic: Compare "Hard Constraints" (viability) vs. "Soft" (fitness penalty for invalid edges), explore different fitness functions
> - [x] Performance benchmarking: plot max/mean fitness and diversity to analyze convergence
> - [ ] visulization of GA operations and best paths accross generations (animated dynamic graph viz)