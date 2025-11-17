from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import graph_tool.all as gt

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .sampling import uniform_cc_fast, estimate_len_count_fast, get_edgelist_gt, spanning_trees_with_occ_prob
from .spanning_trees import NP_EDGE
from .utils import estimate_er_params

def sample_cycle_space(G: "nx.Graph | gt.Graph", p: float | None = None, seed: int | np.random.Generator | None = None, trees: int = 1) -> list[tuple[NDArray[np.int32], int, NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]] | tuple[NDArray[np.int32], int, NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
    """
    Estimates the occurrence probabilities of cycles in the spanning tree of the given graph.

    **Parameters**
    G:      the graph
    p:      edge probability on G
    seed:   random seed or generator to use

    Returns: list of tuples (parent, root, occ_probs) for each tree, or single tuple for trees=1:
    - parent:    parent relationship on the spanning tree T (root has parent -1)
    - root:      the root of the spanning tree
    - depth:     distance of node from root in T
    - occ_probs: np.ndarray of NP_OCCURENCE_PROB datatype (see below), essentially tuples (u, v, lca(u,v), log_2(p_c)) for each induced cycle

    ```
    NP_OCCURENCE_PROB = np.dtype([
        ('u', np.int32),
        ('v', np.int32),
        ('lca', np.int32),
        ('p_c', np.float64),
    ])
    ```
    """
    if seed is None:
        seed = np.random.default_rng()
    elif isinstance(seed, int):
        seed = np.random.default_rng(seed)

    if p is None:
        _, p = estimate_er_params(G)

    res = spanning_trees_with_occ_prob(G, p, seed, trees)
    if len(res) == 1:
        return res[0]
    return res

def uniform_cc(n: int, p: float, N: float | NDArray[np.float64] | None = None, P: NDArray | None = None, samples: int = 100, G: Union[nx.Graph, "gt.Graph", None] = None, seed:int|np.random.Generator|None=None) -> tuple[Union[nx.Graph, "gt.Graph"], set[tuple], int, int]:
    """
    Generates a uniform cell complex adhering to the given parameters

    Parameters:
    - n: Number of nodes.
    - p: Edge probability for G(n,p) graph.
    - N: Number of 2-cells (in expectation). Behavior depends on whether N is a single number or an array:
        - For a single number, the sampled CC contains – in expectation – N 2-cells sampled from all available lengths
        - For an array, the sampled CC contains – in expectation – N[l] 2-cells of length l if such cells are found. **The array must have length n + 1**
    - P: Logarithmic (base 2) sampling probability, based on length; must have length `n + 1`. $P_l$ in the paper is equivalent to `exp2(P[l])`.
        - Samples from the model according to its theoretical definition, leading to possibly large variations in the number of cells, even with the same configuration. To avoid this, use `N` instead. If `N` is used, `P` must be `None` and vice versa.
        - For $P_l = 0$, set `P[l] = -np.inf`; otherwise `P[l] = log2($P_l$)`.
    - samples: Random spanning trees to sample. Larger is more accurate. Should be greater than `N` (or `np.sum(N)`).
    - seed: Random seed or generator to use. Will generate a new `numpy.random.default_rng` if number or no seed is specified.
    - G: Underlying graph to generate 2-cells for. If `None`, a G(n,p) random graph will be sampled instead. If specified, `p` will still be used for the sampling process. Nodes must be integers `0, ..., n-1`.

    Returns: G, cells, undersampled, overcorrelated

    - G: parameter G, if provided, or 1-skeleton of the CC as nx.Graph
    - cells: set[tuple[int]], representing 2-cells as normalized tuples
    - undersampled: int, number of encountered cycles where $q_c > 1$
    - overcorrelated: int, number of encountered cycles where $q_c$ is large enough that, in expectation, multiple cells would be sampled from the same ST.
    """
    if isinstance(G, nx.Graph):
        if len(G.nodes) != n:
            raise ValueError(f"Argument inconsistency: G has {len(G.nodes)} nodes, but n={n}.")
        nodes = set(G.nodes)
        for i in range(n):
            if i not in nodes:
                raise ValueError("G must have the nodes `0, ..., n-1` (as integers). Hint: Use `nx.convert_node_labels_to_integers()`.")
    elif G is not None:
        # gt.Graph. Cannot directly check because that would require graph_tool dependency
        if G.is_directed():
            raise ValueError(f"Illegal Argument: G is directed.")
        if len(G.vertex_index) != n:
            raise ValueError(f"Argument inconsistency: G has {len(G.vertex_index)} nodes, but n={n}.")
    if (N is None) == (P is None):
        raise ValueError(f"P xor N must be None, but N={N} and P={P}.")
    return uniform_cc_fast(n, p, N, P, samples, seed, G)

def estimate_cycle_count(G: Union[nx.Graph, "gt.Graph"], samples: int = 1000, p: float | None = None, seed:int|np.random.Generator|None=None) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.int32]]:
    """
    Estimates the number of simple cycles in G using spanning-tree-based sampling.

    Originally designed for Erdös-Rényi Graphs, but is still relatively accurate on many other graphs.

    Parameters:
    - G: Graph to estimate the number of simple cycles for
    - samples: Number of spanning trees to sample for the estimation. More samples lead to a more accurate estimation.
    - p: Edge Probability used for generating G (assuming Erdös-Rényi). Will be inferred if not specified.
    - seed: Random seed or generator to use. Will generate a new `numpy.random.default_rng` if number or no seed is specified.

    Returns: log_cycle_counts, is_zero, length_occurred

    - log_cycle_counts: np.ndarray of length n + 1. Position l contains the log of the estimated number of cycles of length l.
    - is_zero: np.ndarray of length n + 1. Position l is True if the estimated number of cycles of length l is 0.
    - length_occurred: np.ndarray of length n + 1. Position l contains the number of times a cycles of length l was encountered during the sampling process.
    """
    if seed is None:
        seed = np.random.default_rng()
    elif isinstance(seed, int):
        seed = np.random.default_rng(seed)
    
    if p is None:
        _, p = estimate_er_params(G)

    edges = np.array([(u,v) for (u,v) in G.edges], dtype=NP_EDGE) if isinstance(G, nx.Graph) else get_edgelist_gt(G)
    
    return estimate_len_count_fast(G, edges, p, samples, seed)