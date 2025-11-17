

cdef packed struct Edge:
    int a
    int b

cdef packed struct LcaResult:
    int a
    int b
    int lca

cdef LcaResult* lowest_common_ancestor(int[:] parent, Edge[:] node_pairs) nogil;

cdef void calc_property_fast(int[:] parent, double[:] result, double root_val, int[:] degree, double mean_degree, double (*update_fun)(int, int, double, int[:], double));

cdef int uniform_spanning_tree_c(int size, int[:] degree, int** neighbors, int[:] parent, rnd);

cdef int** graph_to_neighbors(int size, int[:] degree, G);

cdef int** graph_to_neighbors_gt(int size, int[:] degree, G);

cdef void free_graph_neighbors(int size, int** neighbors);