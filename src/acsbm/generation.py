from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse, stats

from . import models


class NetworkTooSmallError(RuntimeError):
    pass


@dataclass
class GeneratedNetwork:
    A: Any # scipy.sparse
    theta: np.ndarray
    Z: np.ndarray
    Z_tilde: np.ndarray
    block_sizes: list[int]
    model: models.MultiCovariateModel

    @property
    def n_blocks(self) -> int:
        return len(self.block_sizes)

    @property
    def n_nodes(self) -> int:
        return self.A.shape[0]
    
    @property
    def n_edges(self) -> int:
        return self.A.count_nonzero()


def generate_sparse_block(size, prob, symmetric=False):
    """
    Generates a random block of binary entries where each entry is 1 w.p. prob
    If symmetric=True, returns a symmetric block with a zero on the diagonal.
    """
    density = stats.binom.rvs(size[0] * size[1], prob, size=1).item() / (size[0] * size[1])
    m = sparse.random(size[0], size[1], density)
    m.data[:] = 1
    
    if symmetric:
        if size[0] != size[1]:
            raise RuntimeError("symmetric matrix must be square")
        m = sparse.triu(m, k=1) + sparse.triu(m, k=1).transpose()
    
    return m

def generate_sparse_sbm(block_sizes, block_probs):
    """
    Generate a stochastic block model using fixed block sizes and connectivity matrix
    """
    k = len(block_sizes)
    blocks = [[None for i in range(k)] for j in range(k)]
    
    for i in range(k):
        for j in range(i, k):
            blocks[i][j] = generate_sparse_block(
                (block_sizes[i], block_sizes[j]),
                block_probs[i,j],
                symmetric=(i == j)
            )
            if i < j:
                blocks[j][i] = blocks[i][j].transpose()
    
    return sparse.bmat(blocks)

def generate_network(model: models.MultiCovariateModel, ndd: models.NodeDataDistribution, n: int):
    theta, Z = ndd.draw(n)
    Z_tilde = model.flatten_Z(Z)
    L_tilde = np.prod([c.n_levels for c in model.covariates])
    theta_tilde = L_tilde * theta + Z_tilde # same as applying tuple_id(...) over rows
    counts = Counter(theta_tilde)
    block_sizes = [counts[i] for i in range(model.n_communities * L_tilde)]

    # This could probably be handled more gracefully...
    if len(counts.keys()) < model.n_communities * L_tilde:
        raise NetworkTooSmallError("Generated network does not have a node of every type. Consider using a larger n.")

    return GeneratedNetwork(
        A=generate_sparse_sbm(block_sizes, model.B_tilde()),
        theta=theta[np.argsort(theta_tilde)],
        Z=Z[np.argsort(theta_tilde), :],
        Z_tilde=Z_tilde[np.argsort(theta_tilde)],
        block_sizes=block_sizes,
        model=model
    )
