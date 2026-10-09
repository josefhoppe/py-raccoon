from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import graph_tool.all as gt

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .spanning_trees import get_induced_cycle as get_induced_cycle_pyx

def estimate_er_params(G: "nx.Graph | gt.Graph"):
    """
    Estimates n and p s.t. G ~ G(n,p)

    G: NetworkX or graph_tool Graph

    Returns: Tuple (n,p)
    """
    if isinstance(G, nx.Graph):
        n = len(G.nodes)
        m = len(G.edges)
    else:
        # gt.Graph. Cannot directly check because that would require graph_tool dependency
        n = G.num_vertices()
        m = G.num_edges()
    return n, m * 2 / (n*n - n)

def get_induced_cycle(edge: tuple[int,int], parent: NDArray[np.int32], depth: NDArray[np.int32]) -> tuple:
    """
    Gets the cycle induced by adding edge to the spanning tree modeled by parent and depth
    """
    cycle = get_induced_cycle_pyx(edge, parent, depth)
    return tuple(int(node) for node in cycle)