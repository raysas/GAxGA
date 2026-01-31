import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

def animate_ga_evolution(G, history, interval=200):
    """
    Creates an animation of the best path found in each generation.
    
    Parameters:
    - G: The networkx graph object.
    - history: ga.history_ (list of dicts containing 'best_path').
    - interval: Delay between frames in milliseconds.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # Keep layout consistent across frames

    def update(frame):
        ax.clear()
        data = history[frame]
        best_path = data['best_path']
        score = data['max']
        
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='grey', ax=ax)

        path_edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path)-1)]
        
        existing_edges = [e for e in path_edges if G.has_edge(*e)]
        missing_edges = [e for e in path_edges if not G.has_edge(*e)]

        nx.draw_networkx_edges(G, pos, edgelist=existing_edges, 
                               edge_color='forestgreen', width=3, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=missing_edges, 
                               edge_color='crimson', width=2, style='dashed', ax=ax)

        ax.set_title(f"Generation: {frame}\nFitness Score: {score:.2f}", fontsize=14)
        ax.set_axis_off()

    ani = FuncAnimation(fig, update, frames=len(history), interval=interval, repeat=False)
    return ani