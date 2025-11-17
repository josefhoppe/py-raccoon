import py_raccoon as pr
import pandas as pd
import numpy as np
import networkx as nx

def test_uniform_ccs():
    G, cells, _, _ = pr.uniform_cc(20, 0.5, 20)
    assert len(cells) > 0, "No 2-cells sampled"

def test_estimate_cycle_count():
    # G is constructed by joining two cycles:
    # 1 - 2 - 3 -- 6 - 7
    #  \     /      \ /
    #   4 - 5        8
    G = nx.from_edgelist([(1,2), (2,3), (3,6), (6,7), (1,4), (3,5), (6,8), (7,8), (4,5)])
    log_counts, is_zero, sampled = pr.estimate_cycle_count(G, samples=10)

    assert not is_zero[3], "Should find cycles of length 3"
    assert is_zero[4], "Should not find cycles of length 4"
    assert not is_zero[5], "Should find cycles of length 5"
