from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import graph_tool.all as gt

import numpy as np
from numpy.typing import NDArray
import networkx as nx

from .spanning_trees import NP_EDGE

def uniform_cc_fast(G: "nx.Graph | gt.Graph", approx_prob: bool, p: float, np_P: NDArray[np.float64], samples: int = 100, seed:int|np.random.Generator|None=None) -> tuple[Union[nx.Graph, "gt.Graph"], set[tuple], int, int]:
    """
    Samples 2-cells according to parameters.

    **Parameters**  
    G:              Graph (1-skeleton) to generate 2-cells for
    approx_prob:    if True, uses the fast approximation for occurrence probability. May lead to bias in the distribution.
    p:              edge probability of G. Only used if approx_prob is True, but must be between 0 and 1.
    np_P:           log_2 of the probability with which to sample cells of each length. 
    samples:        number of spanning trees to sample. More samples take (linearly) more time, but reduce undersampling and overcorrelation of cells.
    seed:           random seed or generator, will use np.random.default_rng() if None.

    Returns: Graph (same as G), sampled cells, undersampled count, overcorrelated count
    """


def get_edgelist_gt(G: "gt.Graph") -> NDArray:
    """
    Gets edge list of graph tool graph
    """

def estimate_cycle_count_fast(G: nx.Graph | gt.Graph, edges: np.ndarray[NP_EDGE], p: float, approx_prob: bool, samples: int, seed: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.int32]]:
    """
    Estimates the number of cycles in `G` for each length.

    For the estimation, it simulates a sampling of cells with probability $P = 1$.
    The number of cells for each length is then the sum of the resulting sampling probability $p'_c$ (which may be greater than 1).

    **Parameters** 
    G:          the graph
    edges:      numpy array of edges, for efficiency
    p:          edge probability on G
    approx_prob: whether to use the fast approximation or the exact occurrence probability
    samples:    number of spanning trees to sample
    seed:       random generator to use

    Returns: log_cycle_counts, is_zero, length_occurred (see `interface.estimate_cycle_count`), indexed by length. Length of the arrays is number of nodes in `G` plus one.
    """

def spanning_trees_with_occ_prob(G: nx.Graph | "gt.Graph", approx_prob: bool, p: float, seed: int | np.random.Generator | None, trees: int = 1) -> list[tuple[NDArray[np.int32], int, NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]]:
    """
    Estimates the occurrence probabilities of cycles in the spanning tree of the given graph.

    **Parameters**
    G:      the graph
    approx_prob: whether to use the fast approximation or the exact occurrence probability
    p:      edge probability on G. Only used if approx_prob is True, but must be between 0 and 1.
    seed:   random seed or generator to use
    trees:  number of spanning trees to sample

    Returns: list of tuples (parent, root, depth, us, vs, lcas, p_cs) for each tree:
    - parent:    parent relationship on the spanning tree T (root has parent -1)
    - root:      the root of the spanning tree
    - depth:     distance of node from root in T
    - us, vs:    array of nodes u, v, representing edges (u,v) in G \ T
    - lcas:      array of lca(u,v)
    - p_cs:      array of log_2(p_c) of the cycle induced by (u,v) with T
    """