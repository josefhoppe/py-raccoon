# 🦝 PyRaCCooN: Random Cell Complexes on Networks

<img align="right" width="200" style="margin-top:-5px" src="https://raw.githubusercontent.com/josefhoppe/py-raccoon/main/readme_src/LOGO_ERC-FLAG_FP.png">

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/josefhoppe/py-raccoon/blob/main/LICENSE)
[![arXiv:2406.01999](https://img.shields.io/badge/arXiv-2406.01999-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2406.01999)
![Python version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fjosefhoppe%2Fpy-raccoon%2Fmain%2Fpyproject.toml&logo=python&logoColor=ffd242)
[![Package version on PyPI](https://img.shields.io/pypi/v/py-raccoon?logo=pypi&logoColor=ffd242)](https://pypi.org/project/py-raccoon/)

PyRaCCooN (**Ra**ndom **C**ell **Co**mplexes **o**n **N**etworks) randomly generates cell complexes and and provides an approximation for the number of simple cycles (by length) on a graph.
PyRaCCooN also exposes the spanning-tree-based algorithm that samples the cycle space via an API, to enable easy adoption for further analyses.
To see how to use PyRaCCooN, check out the Jupyter [examples](https://github.com/josefhoppe/py-raccoon/tree/main/examples) or the short examples below.

For the sampling, PyRaCCooN

- generates random cell complexes by sampling an Erdös-Rényi Graph and random 2-cells, or
- samples random 2-cells on arbitrary graphs.

For all tasks, it uses the same sampling algorithm that is designed to work on ER-graphs, so the approximation may be less accurate on other graphs.
In general, it works well if the graph is globally well-connected and has a small diameter.
If the graph is planar, it tends to significantly underestimate the occurrence probability.

For more information on the theory, algorithmics, and accuracy, see our paper [*Random Abstract Cell Complexes*](https://arxiv.org/abs/2406.01999) on arXiv.
The [Evaluation Code](https://github.com/josefhoppe/random-abstract-cell-complexes) is also available on Github.

If you use PyRaCCooN, please cite the following paper:

```
@misc{hoppe2024random,
      title={{Random Abstract Cell Complexes}}, 
      author={Josef Hoppe and Michael T. Schaub},
      year={2024},
      eprint={2406.01999},
      archivePrefix={arXiv},
      primaryClass={cs.DS}
}
```

## Installation

```bash
pip install py-raccoon
```

## Generating Random Cell Complexes (by expected number of 2-cells)

Externally, PyRaCCooN supports `NetworkX` and `graph_tool` graph classes.
Since graph\_tool is only available via conda, PyRaCCooN uses NetworkX by default (graph\_tool is an optional dependency).
You may want to use graph\_tool for very large graphs to increase performance (in our experiments, the time networkx takes to load a graph from disk exceeded the runtime of our algorithm for very large graphs).

2-cells are represented as tuples of nodes in the boundary of the 2-cell.
Tuples are normalized to start with the smallest node (by integer label), continuing with its smallest neighbor.
For example, `(2,3,1,4)` would be normalized to `(1,3,2,4)`.

```py
import py_raccoon as pr

# Generates CC based on G(20,0.5) with (in expectation) 50 cells, sampled using 100 spanning trees.
G, cells, _, _ = pr.uniform_cc(20, 0.5, 50, samples=100)
```

You may also specify an expected number of 2-cells for each length. Note that longer cells may be impossible to sample. In that case, the supplied expected number is ignored.

```py
import py_raccoon as pr

n = 20
N = np.zeros(n + 1)
N[3] = 10
N[9] = 10
N[19] = 10 # Cells of length 19 will be impossible to sample, so the result will not contain any. See our paper for more information.

# Generates CC based on G(20,0.5) with (in expectation) N[l] cells of length l, sampled using 100 spanning trees.
G, cells, _, _ = pr.uniform_cc(n, 0.5, N, samples=100)
```

If you have a graph you'd like to add random 2-cells to, you can also supply the graph:

```py
import py_raccoon as pr

G = ... # nx.Graph or gt.Graph
n, p = pr.utils.estimate_er_params(G)

_, cells, _, _ = pr.uniform_cc(n, p, 50, samples=100, G=G)
```

Since `cells` is a set of tuples, the result can easily be imported into any library of your choosing.

## Generating Random Cell Complexes with given probability $P_l$

The previous examples all used the integrated functionality to sample an expected number of cells.
While this is more useful in many practical contexts, PyRaCCooN also supports sampling from the 'vanilla' model with a fixed probability $P_l$ for all cells of length $l$:

```py
import py_raccoon as pr
import numpy as np

n = 20
P = np.zeros(n + 1)
P[3] = .5
P[4] = .1
P[19] = 1.0 # Cells of length 19 will be impossible to sample, so the result will not contain any. See our paper for more information.

with np.errstate(divide='ignore'):
    log_P = np.log2(P) # Practical probabilities for greater $l$ are very small, thus represented logarithmically.

# Generates CC based on G(20,0.5). Each possible cell of length l is selected with probability P[l], sampled using 100 spanning trees.
G, cells, _, _ = pr.uniform_cc(n, 0.5, P=log_P, samples=100)
```

## Estimating the number of simple cycles

```py
import py_raccoon as pr
import numpy as np

G = ... # nx.Graph or gt.Graph
log_counts, is_zero, sampled = pr.estimate_cycle_count(G, samples=1000)

# Assuming all cycle counts are in the range of 64-bit floats
cycle_counts = np.exp2(log_counts)
cycle_counts[is_zero] = 0
for l in range(3, len(G.nodes) + 1):
    print(f'G has approx. {cycle_counts[l]} simple cycles of length {l} ({sampled[l]} samples).')
```

When using the estimation, you should also check the number of sampled cells for each length.
As a rule of thumb, if `sampled[l] < 100`, the estimation for length $l$ is inaccurate.
Also note that eventually, longer cycles won't occur at all, leading to an incorrect estimation of 0.

## Taming the cycle space

The core idea of sampling a uniform spanning tree and calculating the occurence probability for all induced cycles is exposed directly through the method `sample_cycle_space`:

```py
import py_raccoon as pr
import numpy as np

G = ... # nx.Graph or gt.Graph
n, p = pr.utils.estimate_er_params(G)

# we will use a single tree for simplicity
parent, root, depth, us, vs, lcas, p_cs = pr.interface.sample_cycle_space(G, p, seed=42)
```

The method returns the spanning tree and properties of the cycles, divided into seven variables:

- spanning tree
  - parent: array listing the parent of every node; parent[root] = -1
  - root: integer, root node
  - depth: array listing the depth of each node in the tree (distance to root)
- cycles (represented via edges (u,v) that induce the cycles)
  - us: list of first nodes of the edges
  - vs: list of second nodes of the edgs
  - lcas: list of lowest common ancestors of u and v
  - p_cs: occurence probability p_c (rho_c in the paper)

This enables us to calculate new properties, either directly or indirectly.
For example, it is simple to calculate the length of the induced cycles:

```py
# Lengths of induced cycles
lengths = depth[us] + depth[vs] - 2 * depth[lcas] + 1
```

More complicated calculations require us to traverse the tree ourselves, calculating an array similar to `depth` or the aggregates used for calculating the occurence probability.
For an example, see this [Jupyter Notebook](examples/CycleSpace.ipynb).

## Runtime behavior

PyRaCCooN is both algorithmically optimized and efficiently implemented using Cython.
The runtime figure below is from our paper; it shows the average runtime for one run of `pr.uniform_cc(n, p, N=10*n, samples=1000)`.

![Runtime Behavior](https://raw.githubusercontent.com/josefhoppe/py-raccoon/main/readme_src/runtime.svg)

## Acknowledgements

Funded by the European Union (ERC, HIGH-HOPeS, 101039827). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.
