from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg, optimize, sparse, spatial
from sklearn.mixture import GaussianMixture
import statsmodels.genmod.families as sm_families
from statsmodels.genmod.generalized_linear_model import GLM as sm_GLM

from . import generation
from .utils import from_tuple_id, tuple_id


@dataclass
class NetworkInitialClusterResult:
    k: int
    partitions: dict[int, list[int]]
    labels: dict[int, list[int]]
    theta_tilde: list[int]


@dataclass
class NetworkClusterResult:
    k: int
    theta: list[int]
    theta_tilde: list[int]
    B_tilde: np.ndarray


@dataclass
class NetworkEstimationResult:
    B_hat: np.ndarray
    coefficients: np.ndarray


def count_connections(A: Any, memberships: list[int], dtype: Any = int) -> np.ndarray:
    """
    Count the number of connections, broken down by block memberships

    This is primarily used to estimate B_tilde.

    Params:

    - `A`: sparse adjacency matrix
    - `memberships`: 0-indexed block memberships (e.g., theta_tilde)
    - `dtype` (optional): dtype to return (default: `int`)
    """
    n_blocks = max(memberships) + 1
    counts = np.zeros((n_blocks, n_blocks), dtype=dtype)
    
    for idx_1, idx_2 in zip(*A.nonzero()):
        if idx_1 > idx_2:
            continue # don't double-count edges!

        t1, t2 = memberships[idx_1], memberships[idx_2]
        counts[min(t1, t2), max(t1, t2)] += 1

    return np.maximum(counts, counts.T) # symmetrize, since lower triangle is zero

def count_dyads(memberships: list[int], dtype: Any = int) -> np.ndarray:
    """
    Calculate the dyad counts for each pair of blocks.

    Params:

    - `memberships`: 0-indexed block memberships (e.g., theta_tilde)
    - `dtype` (optional): dtype to return (default: `int`)
    """
    n_blocks = max(memberships) + 1
    est_block_sizes = Counter(memberships)
    
    counts = np.zeros((n_blocks, n_blocks), dtype=dtype)

    # 1 <= i <= j <= n_blocks
    for i in range(n_blocks):
        for j in range(i, n_blocks):
            if i == j:
                # on-diagonal block: (n choose 2) dyads
                counts[i, j] = est_block_sizes[i] * (est_block_sizes[i] - 1) // 2
            else:
                # off-digonal block: (m x n) dyads
                counts[i, j] = est_block_sizes[i] * est_block_sizes[j]

    return np.maximum(counts, counts.T) # symmetrize, since lower triangle is zero

def scaled_embeddings(M: Any, d: int = None) -> np.ndarray:
    """
    Return scaled embeddings (of dimension `d`) from matrix `M`

    `M` can be a sparse adjacency matrix or a dense estimate of SBM probabilities (e.g., `B_tilde`)

    If `d` is not specified, `M.shape[0]` is used.
    """
    if d is None:
        l, U = linalg.eigh(M)
    else:
        l, U = sparse.linalg.eigsh(M, d)
    return np.multiply(np.abs(l)**0.5, U)

def initial_cluster(net: generation.GeneratedNetwork, k: int, d: int) -> NetworkInitialClusterResult:
    """
    ACSBM Clustering Part 1:
    Recover theta_tilde (up to a permutation)

    Params:

    - `net`: generated network
    - `k`: number of latent communities
    - `d`: embedding dimension
    """
    # recode covariate:
    # (Within the context of this function, we pretend it's just a flattened covariate.
    Z = net.Z_tilde
    n_z = max(Z) + 1

    # get embeddings
    X = scaled_embeddings(net.A, d)

    # partition nodes by covariate
    S = defaultdict(list)
    for i in range(len(Z)):
        S[Z[i]].append(i)

    # cluster per covariate level
    # also: build initial theta_tilde (arbitrary order)
    labels = {}
    theta_tilde = [0 for _ in range(len(Z))]
    for z in range(n_z):
        clusters = GaussianMixture(n_components=k).fit(X[S[z],:])
        labels[z] = clusters.predict(X[S[z],:])

        for i_S, i_Z in enumerate(S[z]):
            theta_tilde[i_Z] = tuple_id((labels[z][i_S], z), (k, n_z))

    return NetworkInitialClusterResult(
        k=k,
        partitions=S,
        labels=labels,
        theta_tilde=theta_tilde
    )

def reconcile_clusters(net: generation.GeneratedNetwork, initial_cluster: NetworkInitialClusterResult) -> NetworkClusterResult:
    """
    ACSBM Clustering Parts 2 and 3:    
    Given an initial clustering (which separately clusters over the partitioned network),
    reconcile clusterings into a single set of clusters
    """
    Z = net.Z_tilde
    n_z = max(Z) + 1
    k, S, labels = initial_cluster.k, initial_cluster.partitions, initial_cluster.labels
    theta_tilde = initial_cluster.theta_tilde

    # estimate B_tilde, X_B
    B_tilde = count_connections(net.A, theta_tilde, dtype=float)
    B_tilde /= np.maximum(count_dyads(theta_tilde), 1)
    X_B = scaled_embeddings(B_tilde)
    
    # group estimated positions by covariate
    positions = []
    for z in range(n_z):
        rows = [tuple_id((j, z), (k, n_z)) for j in range(k)]
        positions.append(X_B[rows,:])

    # find optimal matching of clusters, invert to create lookups
    cluster_map = [invert_permutation(m) for m in optimal_matching(positions)]

    # reconcile permutations for final labels
    reorder = [0 for _ in range(k * len(S))] # old theta_tilde value => sorted theta_tilde value
    theta = [0 for _ in range(len(Z))] # reconciled labels
    for z in range(n_z):
        for j in range(k):
            theta_tilde_old = tuple_id((j, z), (k, n_z))
            theta_tilde_new = tuple_id((cluster_map[z][j], z), (k, n_z))
            reorder[theta_tilde_old] = theta_tilde_new

        for i_S, i_Z in enumerate(S[z]):
            theta[i_Z] = cluster_map[z][labels[z][i_S]]
    
    # reorder theta_tilde, B_tilde
    reorder_inv = invert_permutation(reorder)
    theta_tilde = [reorder[x] for x in theta_tilde]
    B_tilde = B_tilde[reorder_inv, :]
    B_tilde = B_tilde[:, reorder_inv]
    
    return NetworkClusterResult(
        k=k, theta=theta, theta_tilde=theta_tilde, B_tilde=B_tilde
    )


# O(lk^3 + lk^2d)?
# distance matrices: O(l k^2 d), where d is dimension
# assignment: O(l k^3)
def optimal_matching(positions: list[np.ndarray]) -> list[list[int]]:
    """
    Returns a 2D array [matching_1, ... matching_k]:
    Each entry gives indices mapping the cluster labels from the 0-th level
    to the i-th level

    Params:

    - `positions`: list of position arrays per covariate level
    """
    l = len(positions)
    k = positions[0].shape[0]

    best_matching = []
    best_matching.append(list(range(k))) # mapping level 0 => level 0 is just the identity

    for i in range(1, l):
        _, opt = optimize.linear_sum_assignment(
            spatial.distance_matrix(positions[0], positions[i]) ** 2
        )
        best_matching.append(opt.tolist())

    return best_matching

def invert_permutation(p):
    """
    Given a permutation p with entries 0..(n-1) in any order,
    return the permutation p^-1 with entries 0..(n-1) such that
    p^-1 [ p [i] ] = i for any i in 0..(n-1).
    """
    return [x[0] for x in sorted(enumerate(p), key=lambda x: x[1])]

def estimate(net: generation.GeneratedNetwork, cluster_result: NetworkClusterResult) -> NetworkEstimationResult:
    """
    Estimate B matrix and coefficients of ACSBM model (assuming simple covariates only) via GLM
    """
    theta_tilde = cluster_result.theta_tilde
    n_communities = cluster_result.k
    n_blocks = max(theta_tilde) + 1
    n_covariates = len(net.model.covariates)

    dyad_count = count_dyads(theta_tilde)
    connections = np.rint(cluster_result.B_tilde * dyad_count)

    # see from_tuple_id, which maps block index back to (theta, z_1, ... z_P)
    tuple_levels = [n_communities] + [c.n_levels for c in net.model.covariates]

    n_block_pairs = n_blocks * (n_blocks + 1) // 2
    covariate_indicators = np.zeros((n_block_pairs, n_covariates))
    response = np.zeros((n_block_pairs, 2))
    # base block indicators: complicated since we only want params for the upper triangle of B
    B_idx = np.triu_indices(n_communities)
    n_base_blocks = len(B_idx[0])
    B_idx_lookup = { (B_idx[0][i], B_idx[1][i]): i for i in range(n_base_blocks) }
    base_block_indicators = np.zeros((n_block_pairs, n_base_blocks))

    # Step 1: Build response and indicator matrices for regression
    # 1 <= i <= j <= n_blocks
    row = 0
    for i in range(n_blocks):
        for j in range(i, n_blocks):
            t_i = from_tuple_id(i, tuple_levels)
            t_j = from_tuple_id(j, tuple_levels)
            base_block = (min(t_i[0], t_j[0]), max(t_i[0], t_j[0])) # again, upper triangle commplications
            base_block_id = B_idx_lookup[base_block]
            base_block_indicators[row, base_block_id] = 1
            ind = np.where(np.array(t_i) == np.array(t_j), 1, 0)[1:] # covariate match indicators
            covariate_indicators[row,:] = ind
            response[row,:] = np.array([connections[i, j], dyad_count[i, j] - connections[i, j]])

            row += 1

    # Step 2: Fit GLM to estimate coefficients
    non_empty = np.sum(response[:,], axis=1) > 0
    model = sm_GLM(
        response[non_empty, :],
        np.hstack([base_block_indicators, covariate_indicators])[non_empty, :],
        family=sm_families.Binomial(link=net.model.link._statsmodels_link)
    )
    results = model.fit()
    coef = results.params[n_base_blocks:]

    # Step 3: Assemble B_hat, the estimation of B
    B_hat = np.zeros((n_communities, n_communities))
    B_hat[np.triu_indices_from(B_hat)] = results.params[0:n_base_blocks]
    B_hat[np.tril_indices_from(B_hat)] = results.params[0:n_base_blocks][::-1]

    return NetworkEstimationResult(
        B_hat=B_hat,
        coefficients=coef
    )
