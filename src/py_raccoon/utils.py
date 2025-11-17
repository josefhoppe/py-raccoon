import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .spanning_trees import get_induced_cycle as get_induced_cycle_pyx

def estimate_er_params(G: nx.Graph):
    """
    Estimates n and p for a s.t. G ~ G(n,p)

    Returns: Tuple (n,p)
    """
    n = len(G.nodes)
    return n, len(G.edges) * 2 / (n*n - n)

def get_induced_cycle(edge: tuple[int,int], parent: NDArray[np.int32], depth: NDArray[np.int32]) -> tuple:
    """
    Gets the cycle induced by adding edge to the spanning tree modeled by node_level and parent_node
    """
    cycle = get_induced_cycle_pyx(edge, parent, depth)
    return tuple(int(node) for node in cycle)