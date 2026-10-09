import networkx as nx
import scipy
import numpy as np
from math import log2, log
from .spanning_trees import get_induced_cycle
from typing import Generator
try:
    from sksparse.cholmod import cho_factor as cholesky
except ImportError:
    # optional dependency, only needed for exact occurrence probabilities
    cholesky = None

def tree_incl_lapl(A: scipy.sparse.sparray, cycle: tuple[int]) -> scipy.sparse.csc_array:
    """
    Creates the Laplacian whose determinant characterizes spanning trees that include `cycle`.

    For this, all edges in the cycle are contracted (i.e., all nodes merged) into a multigraph.
    Returns the Laplacian of the new multigraph (with fewer nodes than in `A`).

    Parameters:
    - A: Sparse adjacency matrix. Will be converted to coo format initially; if it already is in coo format, it is modified in place.
    - cycle: Tuple of nodes in the cycle.

    Returns: Laplacian of the contracted multigraph as `scipy.sparse.csc_array`
    """
    A = A.tocoo() # coo format is much faster for merging nodes
    data = A.data
    i = A.row
    j = A.col
    new_cols = list(range(A.shape[0])) # which cols will exist afterwards
    collapsed_node = cycle[0] # node to collapse into
    for node in cycle[1:]:
        # simply change coordinates of all entries to the collapsed node
        # duplicate entries will be summed up later
        i[i == node] = collapsed_node
        j[j == node] = collapsed_node
        new_cols.remove(node)
    # set diagonals to zero (only the entry for the collapsed node can be non-zero, but it will exist multiple times in general)
    data[i == j] = 0
    A = scipy.sparse.coo_array((data, (i,j)), shape=tuple(A.shape))
    degs = A.sum(axis=1)

    # construct laplacian coo matrix
    data = np.concatenate((-data, degs))
    i = np.concatenate((i, np.arange(len(degs))))
    j = np.concatenate((j, np.arange(len(degs))))
    L = scipy.sparse.coo_array((data, (i,j)), shape=tuple(A.shape))

    L = L.tocsc()
    L = L[new_cols, :][:, new_cols]
    return L

LN_2 = log(2)

def spanning_tree_count_sparse_log(L: scipy.sparse.sparray) -> float:
    """
    Given the Laplacian L, calculates the number of spanning trees on the (multi-) graph represented by L via the determinant.

    Parameters:
    - L: Sparse Laplacian of a (multi-) graph

    Returns: log_2 of the number of spanning trees
    """
    if cholesky is None:
        raise ImportError("Exact occurrence probabilities require scikit-sparse >= 0.5.0, see README for installation. Alternatively, use approx_prob=True.")
    factor = cholesky(L[1:,1:])
    return factor.logdet() / LN_2

def occurrence_probability_exact(G: nx.Graph | scipy.sparse.sparray, cycles: list[tuple] | Generator[tuple, None, None] | None = None, tree: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None) -> list[float]:
    """
    Calculates the exact occurrence probability p_c of all `cycles` (as log_2(p_c)).

    Either `cycles` or `tree` must be specified. If both are given, `tree` is ignored.

    Parameters:
    - G: NetworkX Graph or sparse adjacency matrix
    - cycles: List or generator of cycles, each as tuple of nodes
    - tree: Tuple (parent, depth, us, vs), where parent and depth define the spanning tree and us, vs define the non-tree edges (u,v) whose induced cycles will be used

    Returns: list of log_2(p_c), in the same order as `cycles` (or `us`, `vs`)
    """
    if type(G) is nx.Graph:
        A = nx.adjacency_matrix(G, weight=None).tocoo()
        L = nx.laplacian_matrix(G, weight=None).T
    else:
        A = G.tocoo()
        L = (scipy.sparse.diags(np.squeeze(np.asarray(G.sum(axis=1)))) - A).tocsc()
    total_number_trees = spanning_tree_count_sparse_log(L)
    if cycles is None:
        # yield cycles from tree
        if tree is None:
            raise AssertionError("need either cycles or tree as argument")
        parent, depth, us, vs = tree
        cycles = (get_induced_cycle((u, v), parent, depth) for u,v in zip(us,vs))
    res = []
    for cycle in cycles:
        L = tree_incl_lapl(A.copy(), cycle)
        try:
            log_tree_count = spanning_tree_count_sparse_log(L)
        except AssertionError as ex:
            raise AssertionError(f"Error encountered: '{str(ex)}' for cycle {cycle}")
        log_prob = log_tree_count + log2(len(cycle)) - total_number_trees
        res.append(log_prob)
    return res