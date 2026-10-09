from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import graph_tool.all as gt

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from collections import defaultdict

from .sampling import uniform_cc_fast, estimate_cycle_count_fast, get_edgelist_gt, spanning_trees_with_occ_prob
from .spanning_trees import NP_EDGE
from .utils import estimate_er_params

def sample_cycle_space(G: "nx.Graph | gt.Graph", p: float | None = None, seed: int | np.random.Generator | None = None, trees: int = 1, approx_prob: bool = True) -> list[tuple[NDArray[np.int32], int, NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]] | tuple[NDArray[np.int32], int, NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
    """
    Estimates the occurrence probabilities of cycles in the spanning tree of the given graph.

    **Parameters**
    G:      the graph
    p:      edge probability on G
    seed:   random seed or generator to use
    trees:  number of spanning trees to sample
    approx_prob: whether to use the fast approximation or the exact occurrence probability

    Returns: list of tuples (parent, root, depth, us, vs, lcas, p_cs) for each tree, or single tuple for trees=1:
    - parent:    parent relationship on the spanning tree T (root has parent -1)
    - root:      the root of the spanning tree
    - depth:     distance of node from root in T
    - us, vs:    array of nodes u, v, representing edges (u,v) of G that are not in T
    - lcas:      array of lca(u,v)
    - p_cs:      array of log_2(p_c) of the cycle induced by (u,v) with T
    """
    if seed is None:
        seed = np.random.default_rng()
    elif isinstance(seed, int):
        seed = np.random.default_rng(seed)

    if p is None:
        _, p = estimate_er_params(G)

    res = spanning_trees_with_occ_prob(G, approx_prob, p, seed, trees)
    if len(res) == 1:
        return res[0]
    return res

def uniform_cc(G: "int | nx.Graph | gt.Graph", p: float | None = None, N: float | NDArray[np.float64] | NDArray[np.int64] | None = None, exact_N: bool = False, P: NDArray | None = None, approx_prob: bool = True, samples: int = 100, seed:int|np.random.Generator|None=None) -> tuple[Union[nx.Graph, "gt.Graph"], set[tuple], int, int]:
    """
    Generates a uniform cell complex adhering to the given parameters.

    This function has different modes that can be combined arbitrarily:

    - Using a provided graph G as a 1-skeleton or generating a new G(n,p) graph for the CC
    - Sampling with a certain probability P (per length), an (expected|exact) number N (in total, distributed over lengths), or an (expected|exact) number N (per length)
    - using the approximated or exact occurrence probability

    The active mode depends on the parameters.

    Parameters:
    - G: NetworkX / graph_tool Graph to generate 2-cells for; or number of nodes to generate ER graph.
      - If graph: nodes must be integers `0, ..., n-1`.
    - p: Edge probability for G(n,p) graph. Used for generation if `type(G) == int`, otherwise only used if `approx_prob` is True. Inferred from G if not provided.
    - N: Number of 2-cells (in expectation or exactly, see `exact_N`). Behavior depends on whether N is a single number or an array:
        - For a single number, the sampled CC contains – in expectation – N 2-cells sampled from all available lengths
        - For an array, the sampled CC contains – in expectation or exactly – N[l] 2-cells of length l if such cells are found. **The array must have length n + 1**
    - exact_N: Whether the result will contain N cells in expectation (False) or exactly (True). Only relevant if N is an integer array.
    - P: Logarithmic (base 2) sampling probability, based on length; must have length `n + 1`. $P_l$ in the paper is equivalent to `exp2(P[l])`.
        - Samples from the model according to its theoretical definition, leading to possibly large variations in the number of cells, even with the same configuration. To avoid this, use `N` instead. If `N` is used, `P` must be `None` and vice versa.
        - For $P_l = 0$, set `P[l] = -np.inf`; otherwise `P[l] = log2($P_l$)`.
    - samples: Random spanning trees to sample. Larger is more accurate. Should be greater than `N` (or `np.sum(N)`).
    - approx_prob: whether to use the fast probability approximation (O(m^{1+eps})) or the exact probability (O(m*n^3))
    - seed: Random seed or generator to use. Will generate a new `numpy.random.default_rng` if number or no seed is specified.

    Returns: G, cells, undersampled, overcorrelated

    - G: parameter G, if it is a graph, or the generated G(n,p) graph as nx.Graph
    - cells: set[tuple[int]], representing 2-cells as normalized tuples
    - undersampled: int, number of encountered cycles where $q_c > 1$
    - overcorrelated: int, number of encountered cycles where $q_c$ is large enough that, in expectation, multiple cells would be sampled from the same ST.
    """
    if seed is None:
        seed = np.random.default_rng()
    elif isinstance(seed, int):
        seed = np.random.default_rng(seed)

    # Check / generate graph
    if isinstance(G, int):
        n = G
        if p is None:
            raise ValueError(f'G is a node count n={G}, but p is None. Please supply a Graph G or a probability p.')
        G = nx.gnp_random_graph(n, p, seed)
        while not nx.is_connected(G):
            G = nx.gnp_random_graph(n, p, seed)
    elif isinstance(G, nx.Graph):
        if isinstance(G, nx.DiGraph) or isinstance(G, nx.MultiGraph):
            raise ValueError("Unsupported NetworkX type: G is a DiGraph or MultiGraph")
        n = len(G.nodes)
        nodes = set(G.nodes)
        for i in range(n):
            if i not in nodes:
                raise ValueError("G must have the nodes `0, ..., n-1` (as integers). Hint: Use `nx.convert_node_labels_to_integers()`.")
        if not nx.is_connected(G):
            raise ValueError("Illegal Argument: G is not connected.")
    else:
        # gt.Graph. Cannot directly check because that would require graph_tool dependency
        if G.is_directed():
            raise ValueError(f"Illegal Argument: G is directed.")
        n = len(G.vertex_index)
        # graph_tool must be installed if G is a gt.Graph
        from graph_tool.topology import label_components
        _, hist = label_components(G)
        if len(hist) > 1:
            raise ValueError("Illegal Argument: G is not connected.")

    if p is None:
        _, p = estimate_er_params(G)

    # Check N and P / generate
    if (N is None) == (P is None):
        raise ValueError(f"P xor N must be None, but N={N} and P={P}.")

    if P is None:
        # estimate number of cycles and calculate P
        count_samples = samples // 10 if samples > 100 else samples
        len_counts, len_count_is_zero, sample_counts = estimate_cycle_count(G, approx_prob, count_samples, p, seed)
        if np.isscalar(N):
            # distribute among existing lengths
            existing_lengths = (sample_counts > count_samples)
            scalar_N = float(N)
            N = np.zeros(shape = n + 1, dtype=np.float64)
            N[existing_lengths] = scalar_N / np.count_nonzero(existing_lengths)
            if exact_N:
                print('[WARN] exact N count sampling not supported for scalar N, setting exact_N=False')
                exact_N = False
        elif N.dtype.kind == 'f':
            # also no exact sampling for floating point, only integer
            if exact_N:
                print('[WARN] exact N count sampling not supported for non-integer Ns, setting exact_N=False')
                exact_N = False

        if np.any(sample_counts < N):
            print('[WARN] attempting to sample very infrequent lengths. Increase samples.')
            if exact_N:
                print('[WARN] exact N is active but likely to not terminate.')

        # calculate sampling probability s.t. N cells are sampled in expectation
        with np.errstate(divide='ignore', invalid='ignore'):
            # log2(0) = -np.inf -> correct behavior in sampler
            # => ignore numpy warning
            log_N = np.log2(N)
            # 0 / count = 0
            # -np.inf - log(count) = -np.inf => ignore numpy warning
            P = log_N - len_counts
            # if count is 0, N / count = 0/0 = -np.inf - -np.inf = np.nan
            # Could lead to errors -> replace with -np.inf
            P[np.isnan(P)] = -np.inf
    else:
        if exact_N:
            print('[WARN] exact N count sampling not supported for probabilities P, setting exact_N=False')
            exact_N = False # no exact sampling with P provided

    if exact_N:
        # if exact_N then N is an int array
        # sample twice as many 2-cells as required in expectation to later subsample
        P += np.log2(2)

        # for small numbers a factor of 2 may not suffice, increase
        P[N < 10] += np.log2(10)

    rnd_state = seed.bit_generator.state
    seed.bit_generator.state = rnd_state

    G, cells, undersample, overcorrelate = uniform_cc_fast(G, approx_prob, p, P, samples, seed)

    if exact_N:
        # check if enough cells found, if so subsample, else re-run with higher probabilities
        sampled_cell_counts = np.zeros_like(N)
        for c in cells:
            sampled_cell_counts[len(c)] += 1

        # rerun until enough are found
        iters = 0
        while np.any(sampled_cell_counts < N):
            iters += 1
            print(f"[WARN] resampling cells for exact N, iteration {iters}")
            P[sampled_cell_counts < N] += np.log2(2)
            G, cells, undersample, overcorrelate = uniform_cc_fast(G, approx_prob, p, P, samples, seed)
            sampled_cell_counts = np.zeros_like(N)
            for c in cells:
                sampled_cell_counts[len(c)] += 1

        # subsample
        cell_candidates = defaultdict(lambda: [])
        for c in cells:
            cell_candidates[len(c)].append(c)

        final_cells = set()
        for l, cs in cell_candidates.items():
            for sel in seed.choice(len(cs), int(N[l]), replace=False):
                final_cells.add(cs[sel])

        return G, final_cells, undersample, overcorrelate
    
    return G, cells, undersample, overcorrelate

def estimate_cycle_count(G: Union[nx.Graph, "gt.Graph"], approx_prob: bool = True, samples: int = 1000, p: float | None = None, seed:int|np.random.Generator|None=None) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.int32]]:
    """
    Estimates the number of simple cycles in G using spanning-tree-based sampling.

    The approximated probability introduces a strong bias for some graph classes, especially (nearly) planar graphs and graphs with a high diameter.

    Parameters:
    - G: Graph to estimate the number of simple cycles for
    - approx_prob: Whether to use the (faster) approximation for the cycle sampling probability. 
    - samples: Number of spanning trees to sample for the estimation. More samples lead to a more accurate estimation.
    - p: Edge Probability used for generating G (assuming Erdős-Rényi). Will be inferred if not specified.
    - seed: Random seed or generator to use. Will generate a new `numpy.random.default_rng` if number or no seed is specified.

    Returns: log_cycle_counts, is_zero, length_occurred

    - log_cycle_counts: np.ndarray of length n + 1. Position l contains the log of the estimated number of cycles of length l.
    - is_zero: np.ndarray of length n + 1. Position l is True if the estimated number of cycles of length l is 0.
    - length_occurred: np.ndarray of length n + 1. Position l contains the number of times a cycle of length l was encountered during the sampling process.
    """
    if seed is None:
        seed = np.random.default_rng()
    elif isinstance(seed, int):
        seed = np.random.default_rng(seed)
    
    if p is None:
        _, p = estimate_er_params(G)

    edges = np.array([(u,v) for (u,v) in G.edges], dtype=NP_EDGE) if isinstance(G, nx.Graph) else get_edgelist_gt(G)
    
    return estimate_cycle_count_fast(G, edges, p, approx_prob, samples, seed)