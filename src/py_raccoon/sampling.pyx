# distutils: language=c++

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import graph_tool.all as gt

from libc.stdlib cimport malloc, free
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from math import log2
from libc.math cimport log2 as clog2

# Backport since exp2 was introduced in python 3.11
# Note: The improved accuracy of exp2 will not be noticeable since the 
#       accuracy of 2**x is greater than that of our approximation.
cdef inline double cexp2(double x) nogil:
    return 2**x

import cython

from .spanning_trees cimport lowest_common_ancestor, Edge, LcaResult, calc_property_fast, graph_to_neighbors, graph_to_neighbors_gt, free_graph_neighbors, uniform_spanning_tree_c, Edge
from .spanning_trees import NP_EDGE, calc_depth, uniform_spanning_tree, get_induced_cycle

cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937() # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed) # not worrying about matching the exact int type for seed
        unsigned int operator()()
    
    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen) # ignore the possibility of using other classes for "gen"

cdef packed struct OccurenceProb:
    int u
    int v
    int lca
    double p_c

NP_OCCURENCE_PROB = np.dtype([
    ('u', np.int32),
    ('v', np.int32),
    ('lca', np.int32),
    ('p_c', np.float64),
])

@cython.wraparound(False)
cdef double deg_prod_update(int node, int p, double parent_val, int[:] degree, double mean_degree):
    """
    # root r, node u, parent v
    # Degree product: $\pi(r,u) = \pi(r,v) * (d(v) - 1)$
    deg_prod_update = lambda _, p, parent_val: parent_val + (log2(degree_np[p] - 1) if degree_np[p] > 1 else 0) # if the degree is 1 we can ignore the node -- should only happen if the root has degree 1
    """
    if degree[node] <= 1:
        return parent_val # only relevant if the root has degree 1, where this leads to numerical problems
    return parent_val + clog2(tau_i_inacc(degree[node], len(degree)))

@cython.wraparound(False)
cdef double deg_sum_update(int u, int v, double parent_val, int[:] degree, double mean_degree):
    """
    # Degree sum: $\sigma(r,u) = \sigma(r,v) + (d(v) - 1)(d(u)-1)$
    deg_sum_update = lambda u, v, parent_val: parent_val + (degree_np[u]-1) * (degree_np[v]-1)
    """
    if degree[u] <= 1 or degree[v] <= 1:
        return parent_val # only relevant if the root has degree 1, where this leads to numerical problems
    return parent_val + (tau_1(degree[u], len(degree), mean_degree) + tau_1(degree[v], len(degree), mean_degree)) / 2 / tau_i_inacc(degree[u], len(degree)) / tau_i_inacc(degree[v], len(degree))

cdef inline double tau_l_1(double d, int n, int l, double d_exp):
    #\tilde\tau_c^{l-1}
    return 1 / (1 + (d-2)*(n-l)*(d_exp-1)*(n-1)/(n-3)/l/d_exp/(n-2))

cdef inline double tau_l_2(double d, int n, int l, d_exp):
    #\tilde\tau_c^{l-2}
    cdef double f_prime_prime = (1 + (d_exp-2)*(n-l)/(n-3)/(l-1))/d_exp
    cdef double f_prime = (d_exp - 1) / d_exp * (n-1) / (n-2) / (l - 1)
    return f_prime_prime / (f_prime_prime + (d-2)/(n-3)*((n-l)*f_prime + 1))

cdef inline double tau_i_acc(double d, int n, int i, d_exp):
    #\tilde\tau_c^{i}
    cdef double f = (d_exp-1)*(n-1)/d_exp/(n-2)/(i+1)
    return 1 / (1 + (d-2)*(n-i-2 + 1/f)/(n-3))

cdef inline double tau_i_inacc(double d, int n):
    #\tilde\tilde\tau_c^{i}
    return 1 / (1 + (d-2)*(n-1)/(n-3))

cdef inline double tau_1(double d, int n, double d_exp):
    return 1 / (2*d_exp/(d_exp-1)*(n-2)/(n-1) + d - 1)

cdef inline double correction_factor_log(double d_exp, int n, int l):
    """
    The correction factor for the given length
    """
    cdef double corr = clog2(tau_l_1(d_exp,n,l,d_exp) * tau_l_2(d_exp, n,l,d_exp)  / tau_i_inacc(d_exp,n) / tau_i_inacc(d_exp,n))
    cdef int i
    for i in range(2, l-2):
        corr += clog2(tau_i_acc(d_exp, n, i, d_exp) / tau_i_inacc(d_exp, n))
    return corr

def spanning_trees_with_occ_prob(G: nx.Graph | "gt.Graph", p: float, seed: int | np.random.Generator | None, trees: int = 1) -> list[tuple[NDArray[np.int32], int, NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]]:
    """
    Estimates the occurrence probabilities of cycles in the spanning tree of the given graph.

    **Parameters**
    G:      the graph
    p:      edge probability on G
    seed:   random seed or generator to use

    Returns: list of tuples (parent, root, us, vs, lcas, p_cs) for each tree:
    - parent:    parent relationship on the spanning tree T (root has parent -1)
    - root:      the root of the spanning tree
    - depth:     distance of node from root in T
    - us, vs:    array of nodes u, v, representing edges (u,v) in G \ T
    - lcas:      array of lca(u,v)
    - p_cs:      array of log_2(p_c) of the cycle induced by (u,v) with T
    """
    assert p > 0
    if seed is None:
        seed = np.random.default_rng()
    elif isinstance(seed, int):
        seed = np.random.default_rng(seed)

    cdef int n = len(G.nodes) if isinstance(G, nx.Graph) else len(G.vertex_index)
    cdef int samples = trees
    if isinstance(G, nx.Graph):
        node_degree = np.array(G.degree, dtype=np.int32)
        degree_np = node_degree[:,1][node_degree[:,0].argsort()] # nx may not consider the nodes to be in ascending order by their id
    else:
        degree_np = np.array(G.get_total_degrees(G.vertex_index), dtype=np.int32)
    cdef int[:] degree = degree_np

    cdef int** neighbors = graph_to_neighbors(n, degree, G) if isinstance(G, nx.Graph) else graph_to_neighbors_gt(n, degree, G)
    edges = np.array([(u,v) for (u,v) in G.edges], dtype=NP_EDGE) if isinstance(G, nx.Graph) else get_edgelist_gt(G)

    cdef int l
    cdef int i, j
    cdef int[:] parent, depth, us, vs, lcas
    cdef double[:] p_cs
    cdef OccurenceProb[:] occ_probs
    cdef OccurenceProb op
    cdef int tree_root

    result = []

    for i in range(samples):
        np_parent = np.ndarray(n, np.int32)
        parent = np_parent
        tree_root = uniform_spanning_tree_c(n, degree, neighbors, parent, seed)
        np_depth = calc_depth(parent)
        depth = np_depth
        
        occ_probs = occurence_probability_fast(edges, p, np_parent, tree_root, np_depth, degree_np)


        np_us = np.ndarray(len(occ_probs), np.int32)
        np_vs = np.ndarray(len(occ_probs), np.int32)
        np_lcas = np.ndarray(len(occ_probs), np.int32)
        np_p_cs = np.ndarray(len(occ_probs), np.float64)

        for j, op in enumerate(occ_probs):
            np_us[j] = op.u
            np_vs[j] = op.v
            np_lcas[j] = op.lca
            np_p_cs[j] = op.p_c

        result.append((np_parent, tree_root, np_depth, np_us, np_vs, np_lcas, np_p_cs))

    free_graph_neighbors(n, neighbors)

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef OccurenceProb[:] occurence_probability_fast(Edge[:] edges, double p, int[:] parent, int tree_root, int[:] depth, int[:] degree):
    """
    Estimates the probability with which each cycle induced by the given spanning tree appears in a uniformly sampled spanning tree.

    Due to limitations of floating point numbers, it returns the logarithm of the result.

    **Parameters** 
    G:      underlying ER-Graph
    p:      edge probability of G
    parent: parent relationship on the spanning tree T (root has parent -1)
    depth:  distance of node from the root in T

    Returns: List of tuples (u, v, lca(u,v), log_2(p_c)) for each $(u, v) \in G \setminus T$
    """
    cdef int n = len(parent)
    cdef int l, i
    cdef double c_prob = p
    cdef double d_exp = (n - 1) * p

    # calculate correction factor by length
    corr_np = np.zeros(n + 1, np.float64)
    cdef double[:] corr = corr_np
    corr_calc_np = np.zeros(n + 1, np.int8)
    cdef char[:] corr_calc = corr_calc_np
    # Correction factor is calculated on demand:
    # - calculating all correction factors takes Omega(n^2), which is longer than the runtime of sampling
    # - we don't know (a priori) which lengths will occur

    # calculate cumulative properties $\pi(r,u)$ and $\sigma(r,u)$
    cum_prod_np = np.ndarray(len(parent), np.float64)
    root_prod = deg_prod_update(tree_root, -1, 0, degree, d_exp)
    calc_property_fast(parent, cum_prod_np, root_prod, degree, d_exp, deg_prod_update)
    cdef double[:] cum_prod = cum_prod_np
    cum_sum_np = np.ndarray(len(parent), np.float64)
    calc_property_fast(parent, cum_sum_np, 0, degree, d_exp, deg_sum_update)
    cdef double[:] cum_sum = cum_sum_np

    cdef int j
    cdef int num_candidates = len(edges) - n + 1
    np_candidate_edges = np.empty(num_candidates, dtype=NP_EDGE)
    cdef Edge[:] candidate_edges = np_candidate_edges
    cdef Edge e
    i = 0

    parent_count = np.zeros(n, dtype=np.int32)
    cdef int[:] p_count = parent_count

    for e in edges:
        if e.a > n or e.a < 0 or e.b > n or e.b < 0:
            break
        if parent[e.a] != e.b and parent[e.b] != e.a:
            candidate_edges[i] = e
            i += 1

    cdef int u, v, lca
    cdef double deg_sum, deg_prod, p_c
    cdef LcaResult* lca_result = lowest_common_ancestor(parent, candidate_edges)
    cdef int res_count = candidate_edges.shape[0]
    cdef OccurenceProb[:] result
    try:
        np_result = np.ndarray(res_count, dtype=NP_OCCURENCE_PROB)
        result = np_result
        for i in range(res_count):
            lca_res = lca_result[i]
            u = lca_res.a
            v = lca_res.b
            lca = lca_res.lca

            l = depth[u] + depth[v] - 2*depth[lca] + 1

            if not corr_calc[l]:
                corr[l] = correction_factor_log(d_exp, n, l)
                corr_calc[l] = 1

            # $\sigma(u,v) = \sigma(r,u) + \sigma(r,v) - 2 \sigma(r,lca(u,v)) + (d(v) - 1)(d(u) - 1)
            deg_sum = cum_sum[u] + cum_sum[v] - 2*cum_sum[lca]
            deg_sum = deg_sum + (tau_1(degree[u], n, d_exp) + tau_1(degree[v], n, d_exp)) / 2 / tau_i_inacc(degree[u], n) / tau_i_inacc(degree[v], n)

            # $\pi(u,v) = (d(lca(u,v)) - 1) * \pi(r,u) * \pi(r,v) / \pi(r,lca(u,v))^2
            deg_prod = cum_prod[u] + cum_prod[v] + clog2(tau_i_inacc(degree[lca], n)) - (2*cum_prod[lca])

            # add correction factor
            p_c = clog2(deg_sum) + deg_prod + corr[l]

            if p_c > 0: #logarithmic -> actual p_c > 1
                p_c = 0

            result[i] = OccurenceProb(u, v, lca, p_c)
    finally:
        free(lca_result)

    return result

@cython.wraparound(False)
def get_edgelist_gt(G: "gt.Graph") -> NDArray:
    edges_2dim = G.get_edges().astype(np.int32)
    np_edges = np.zeros(shape=edges_2dim.shape[0], dtype=NP_EDGE)
    cdef Edge[:] edges = np_edges
    cdef int i
    cdef size = G.num_vertices()
    for i in range(edges_2dim.shape[0]):
        edges[i].a = edges_2dim[i,0]
        edges[i].b = edges_2dim[i,1]
    return np_edges

@cython.boundscheck(False)
@cython.wraparound(False)
def uniform_cc_fast(n: int, p: float, N: float | NDArray[np.float64] | None, np_P: NDArray[np.float64] | None = None, samples: int = 100, seed:int|np.random.Generator|None=None, G:nx.Graph | "gt.Graph" | None = None) -> tuple[Union[nx.Graph, "gt.Graph"], set[tuple], int, int]:
    """
    Samples 2-cells according to parameters.

    **Parameters**  
    n:          number of nodes for the generated graph
    p:          edge probability for the generated graph
    N:          approximate number of cells to sample (not guaranteed); or, if array, approx. number for each length.
    np_P:       log_2 of the probability with which to sample cells of each length. 
    samples:    number of spanning trees to sample. More samples take (linearly) more time, but reduce undersampling and overcorrelation of cells.
    seed:       random seed or generator, will use np.random.default_rng() if None.
    G:          Graph to use as 1-skeleton; if None, a new ER graph is generated.

    Returns: Graph (same as G if provided), sampled cells, undersampled count, overcorrelated count
    """
    if seed is None:
        seed = np.random.default_rng()
    elif isinstance(seed, int):
        seed = np.random.default_rng(seed)
    cdef mt19937 c_rnd = mt19937(seed.integers(0, 1 << 32))
    cdef uniform_real_distribution[double] sampling_dist = uniform_real_distribution[double](0,1)
    if G is None:
        G = nx.gnp_random_graph(n, p, seed)
        while not nx.is_connected(G):
            G = nx.gnp_random_graph(n, p, seed)
    cdef int m = len(G.edges) if isinstance(G, nx.Graph) else len(G.edge_index)
    if m - n + 1 == 0:
        print("WARN: Graph is a tree, returning empty result")
        return G, set(), 0, 0
    edges = np.array([(u,v) for (u,v) in G.edges], dtype=NP_EDGE) if isinstance(G, nx.Graph) else get_edgelist_gt(G)
    if isinstance(G, nx.Graph):
        node_degree = np.array(G.degree, dtype=np.int32)
        degree_np = node_degree[:,1][node_degree[:,0].argsort()] # nx may not consider the nodes to be in ascending order by their id
    else:
        degree_np = np.array(G.get_total_degrees(G.vertex_index), dtype=np.int32)
    cdef int[:] degree = degree_np
    cdef int** neighbors = graph_to_neighbors(n, degree, G) if isinstance(G, nx.Graph) else graph_to_neighbors_gt(n, degree, G)

    cdef int c_samples = samples
    cdef char[:] len_count_is_zero
    cdef int est_samples, occuring_lengths
    cdef double log_occuring_lengths
    cdef double[:] len_counts
    if np_P is None and N is not None:
        # Use significantly fewer samples for estimation to save time
        est_samples = c_samples // 10 if c_samples > 100 else c_samples
        np_len_counts, np_len_count_is_zero, sample_counts = estimate_len_count_fast(G, edges, p, est_samples, seed)
        np_len_count_is_zero[np_len_counts < 1] = True #small values make calculation weird; log_2(2) = 1
        np_len_count_is_zero[sample_counts < min(est_samples, 10)] = True #not actually occuring cells are also weird
        occuring_lengths = np.size(np_len_count_is_zero) - np.count_nonzero(np_len_count_is_zero)
        log_occuring_lengths = clog2(occuring_lengths)
        len_counts = np_len_counts
        with np.errstate(divide='ignore'):
            log_N = np.log2(N) # works for both scalars and arrays
        if np.isscalar(N):
            # approx sample count is distributed evenly among all lengths that occur
            np_P = log_N - np_len_counts - log_occuring_lengths # numpy auto broadcasts to arrays where necessary.
        else:
            np_P = log_N - np_len_counts
        len_count_is_zero = np_len_count_is_zero
    else:
        # treat all lengths as existing, the np_P parameter takes precedence (and may be `log2(0)`).
        len_count_is_zero = np.zeros(n, dtype=np.int8)
    cdef double[:] P = np_P

    cells = set()
    undersample = 0
    overcorrelate = 0
    cdef int l
    cdef double p_c, p_c_prime
    cdef OccurenceProb[:] occ_probs
    cdef OccurenceProb op
    cdef int i
    cdef int[:] depth
    # overcorrelation <=> if all cells had this probability, we would sample > 1 from this spanning tree in expectation.
    cdef double overcorrelate_thresh = 1.0 / (m - n + 1)
    for i in range(c_samples):
        parent = np.ndarray(n, dtype=np.int32)
        tree_root = uniform_spanning_tree_c(n, degree, neighbors, parent, seed)
        np_depth = calc_depth(parent)
        depth = np_depth
        
        occ_probs = occurence_probability_fast(edges, p, parent, tree_root, np_depth, degree_np)
        for op in occ_probs:
            l = depth[op.u] + depth[op.v] - 2*depth[op.lca] + 1
            if not len_count_is_zero[l]:
                p_c_prime = calc_sampling_probability(P[l], c_samples, op.p_c, log_input=True)
                if p_c_prime > 1:
                    undersample += 1
                elif p_c_prime > overcorrelate_thresh:
                    overcorrelate += 1
                if sampling_dist(c_rnd) < p_c_prime:
                    cell = get_induced_cycle((op.u, op.v), parent, depth)
                    cells.add(tuple(int(node) for node in cell)) # nodes are still np integers

    free_graph_neighbors(n, neighbors)
    return G, cells, undersample, overcorrelate

@cython.boundscheck(False)
@cython.wraparound(False)
def estimate_len_count_fast(G: nx.Graph | "gt.Graph", edges: np.ndarray[NP_EDGE], p: float, samples: int, seed: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[bool], NDArray[np.int32]]:
    """
    Estimates the number of cycles in `G` for each length.

    For the estimation, it simulates a sampling of cells with probability $P = 1$.
    The number of cells for each length is then the sum of the resulting sampling probability $p'_c$ (which may be greater than 1).

    **Parameters** 
    G:          the graph
    p:          edge probability on G
    samples:    number of spanning trees to sample

    Returns: array of count, indexed by length. Length of the array is number of nodes in `G` plus one.
    """
    assert p > 0
    cdef int n = len(G.nodes) if isinstance(G, nx.Graph) else len(G.vertex_index)
    if isinstance(G, nx.Graph):
        node_degree = np.array(G.degree, dtype=np.int32)
        degree_np = node_degree[:,1][node_degree[:,0].argsort()] # nx may not consider the nodes to be in ascending order by their id
    else:
        degree_np = np.array(G.get_total_degrees(G.vertex_index), dtype=np.int32)
    cdef int[:] degree = degree_np
    cdef int** neighbors = graph_to_neighbors(n, degree, G) if isinstance(G, nx.Graph) else graph_to_neighbors_gt(n, degree, G)
    np_expected_counts = np.zeros(n + 1, np.float64)
    cdef double[:] expected_counts = np_expected_counts
    np_occured = np.zeros(n + 1, np.int32)
    cdef int[:] occured = np_occured
    np_P = np.ndarray(n + 1, np.float64)
    cdef double[:] P = np_P
    
    np_P[[0,1,2]] = 0
    cdef int l
    for l in range(3,n + 1):
        # log to avoid floating point limitations
        P[l] = clog2(samples) + clog2(l) + clog2(n-2)*(l-2) - clog2(n)*(2*l - 4) - clog2(p)*(l-3)
    #P[:] = 0

    undersample = 0
    cdef int i, j
    cdef int[:] parent, depth
    cdef double p_c_prime
    cdef OccurenceProb[:] occ_probs
    cdef OccurenceProb op
    cdef int tree_root
    for i in range(samples):
        np_parent = np.ndarray(n, np.int32)
        parent = np_parent
        tree_root = uniform_spanning_tree_c(n, degree, neighbors, parent, seed)
        np_depth = calc_depth(parent)
        depth = np_depth
        
        occ_probs = occurence_probability_fast(edges, p, np_parent, tree_root, np_depth, degree_np)
        for j in range(occ_probs.shape[0]):
            op = occ_probs[j]
            l = depth[op.u] + depth[op.v] - 2*depth[op.lca] + 1
            
            p_c_prime = cexp2(P[l] - op.p_c) / samples

            if p_c_prime > 1:
                undersample += 1
            expected_counts[l] += p_c_prime
            occured[l] += 1
    
    free_graph_neighbors(n, neighbors)

    zeros = np_expected_counts == 0

    with np.errstate(divide='ignore'):
        np_est_counts = np.log2(np_expected_counts) - np_P
    return np_est_counts, zeros, np_occured

def last_step(d, n, l):
    return 1 / (1 + (d-2)*(n-l)/(n-3)/l)

@cython.cpow(True)
cdef inline double calc_sampling_probability(float P, int samples, float p_c, char log_input = False) nogil:
    """Calculates the probability with which we should sample the given cell obtained from a spanning tree.
    
    **Parameters**
    P:          Overall Probability with which we should choose the cell
    samples:    Number of sampled spanning trees
    p_c:        Probability that a uniformly sampled spanning tree induces the cell
    log_input:  Whether the input is the logarithm of P and p_c or not
    """
    if log_input:
        if p_c + clog2(samples) < clog2(.001):
            # very close to correct; exp2(p_c) may be 0 due to floating point limitations
            return cexp2(P - p_c - clog2(samples))
        else:
            # have to be more accurate; fortunately, exp2(p_c) will be > 0
            P = cexp2(P)
            p_c = cexp2(p_c)
    if p_c * samples < 0.001 or P >= 1:
        # floating point calculation errors; here, this is very close to correct
        return P / p_c / samples
    return (1-(1-P)**(1/samples)) / p_c