from collections import Counter
from dataclasses import dataclass
import functools
import itertools
from typing import Any, Iterable

import numpy as np
from scipy import optimize, sparse, spatial, stats
from sklearn import metrics
from sklearn.mixture import GaussianMixture
import statsmodels.genmod.families as sm_families
from statsmodels.genmod.generalized_linear_model import GLM as sm_GLM


def kron_combine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    kron(A, 11^T) + kron(11^T, B)
    Note: This is not the same as a Kronecker sum, which uses identities in place of the ones matrices.
    """
    return np.kron(A, np.ones(B.shape)) + np.kron(np.ones(A.shape), B)

def tuple_id(t: Iterable[int], levels: Iterable[int]) -> int:
    """
    A bijection that maps a tuple of ints to a scalar integer
    Example with levels = (2, 3):
    (0, 0) -> 0
    (0, 1) -> 1
    (0, 2) -> 2
    (1, 0) -> 3
    (1, 1) -> 4
    (1, 2) -> 5
    """
    values = list(t)
    multiplier = 0
    for i in reversed(range(len(values))):
        if multiplier > 0:
            values[i] *= multiplier
        multiplier += levels[i]
    
    return sum(values)

def from_tuple_id(id: int, levels: Iterable[int]) -> tuple[int]:
    """
    Inverse of tuple_id
    """
    entries = []
    for i in reversed(range(len(levels))):
        remainder = id % levels[i]
        entries.append(remainder)
        id = id // levels[i]
    return tuple(reversed(entries))


@dataclass
class Covariate:
    beta_matrix: np.ndarray # L x L matrix of pairwise group effects, usually diagonal

    @classmethod
    def simple(cls, beta: float, L: int):
        return Covariate(beta_matrix=beta * np.eye(L))

    @property
    def n_levels(self) -> int:
        return self.beta_matrix.shape[0]
    
    def combine(self, other):
        return Covariate(beta_matrix=kron_combine(self.beta_matrix, other.beta_matrix))


@dataclass
class LinkFunction():
    """
    Wrapper around statsmodels link functions.
    Why? Not sure.
    """
    _statsmodels_link: sm_families.family.Family

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return self._statsmodels_link.inverse(x)
    
    @classmethod
    def identity(cls):
        return LinkFunction(sm_families.links.identity())
    
    @classmethod
    def log(cls):
        return LinkFunction(sm_families.links.log())
    
    @classmethod
    def logit(cls):
        return LinkFunction(sm_families.links.logit())
    
    @classmethod
    def probit(cls):
        return LinkFunction(sm_families.links.probit())


@dataclass
class MultiCovariateModel:
    link: LinkFunction
    B: np.ndarray # K x K matrix of base edge probabilities
    covariates: list[Covariate]

    @property
    def n_communities(self):
        return self.B.shape[0]

    def flatten_covariate(self):
        """
        Return a single covariate whose beta_matrix reflects the combined effects of all covariates
        """
        return functools.reduce(lambda x, y: x.combine(y), self.covariates)

    def flatten_Z(self, Z: np.ndarray) -> np.ndarray:
        """
        Map each covariate combination to a unique scalar
        Essentially tuple_id, but implemented in numpy instead.
        """
        levels = [c.n_levels for c in self.covariates]
        multiplier = 0
        Z2 = Z.copy()
        for i in reversed(range(Z2.shape[1])):
            if multiplier > 0:
                Z2[:,i] *= multiplier
            multiplier += levels[i]

        return Z2.sum(axis=1)
    
    def B_tilde(self):
        flattened = self.flatten_covariate()
        M = kron_combine(self.B, flattened.beta_matrix)
        return self.link.inverse(M)


class NodeDataDistribution:
    """
    A distribution on latent communities (theta) and covariates (Z)
    pmf is a K x L_1 x L_2 x ... x L_P array of probabilities, where
    pmf[k, l1, ... lP] = P(theta = k, Z_1 = l1, ... Z_P = lP)
    """
    
    def __init__(self, pmf: np.ndarray):
        self.pmf = pmf / np.sum(pmf) # normalize

    def draw(self, n: int, rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        all_indices = list(np.ndindex(*self.pmf.shape))
        data = rng.choice(all_indices, n, p = self.pmf.reshape(len(all_indices),))
        return data[:,0], data[:,1:] # theta, Z

    @classmethod
    def _model_shape(cls, model: MultiCovariateModel) -> tuple[int]:
        return tuple([model.n_communities] + [c.n_levels for c in model.covariates])

    @classmethod
    def uniform_for_model(cls, model: MultiCovariateModel):
        return NodeDataDistribution(np.ones(cls._model_shape(model)))

    @classmethod
    def random_for_model(cls, model: MultiCovariateModel):
        """
        An arbitrary random, non-uniform distribution
        """
        return NodeDataDistribution(np.random.default_rng().beta(2, 3, cls._model_shape(model)) + 0.05)


class NetworkTooSmallError(RuntimeError):
    pass


# pmf = np.array([[
#     [1, 3],
#     [0, 6]
# ],
# [
#     [0, 5],
#     [5, 0]
# ]])
# empirical = np.zeros_like(pmf)
# gen = multicovariates.NodeDataDistribution(pmf)
# n = 1000
# for x in gen.draw(n):
#     empirical[tuple(x)] += 1
# empirical / n

@dataclass
class GeneratedNetwork:
    A: Any # scipy.sparse
    theta: np.ndarray
    Z: np.ndarray
    Z_tilde: np.ndarray
    block_sizes: list[int]
    model: MultiCovariateModel


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

def generate_network(model: MultiCovariateModel, ndd: NodeDataDistribution, n: int):
    theta, Z = ndd.draw(n)
    Z_tilde = model.flatten_Z(Z)
    L_tilde = np.prod([c.n_levels for c in model.covariates])
    theta_tilde = L_tilde * theta + Z_tilde # same as applying tuple_id(...) over rows
    counts = Counter(theta_tilde)
    block_sizes = [counts[i] for i in range(model.n_communities * L_tilde)]

    # TODO: this doesn't belong here
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


@dataclass
class NetworkClusterResult:
    n_communities: int
    theta: list[int]
    theta_tilde: list[int]


@dataclass
class NetworkEstimationResult:
    B_star: np.ndarray
    B_hat: np.ndarray
    coefficients: np.ndarray


def cluster(net: GeneratedNetwork, k: int, d: int, ignore_covariates: bool = False) -> NetworkClusterResult:
    """
    Cluster ACSBM by separately clustering each covariate configuration, then matching clusterings

    Params:
    k: number of latent communities
    d: embedding dimension
    ignore_covariates: if True, estimates theta as one would without covariates (single-round, no matching)

    Note: `ignore_covariates` only affects estimation of `theta`. `theta_tilde` will
    still reflect covariate differences, but it will do so based on the naive 
    estimation of `theta`.
    """
    # recode covariate:
    # (Within the context of this function, we pretend it's just a flattened covariate.
    # If `ignore_covariates` is `True`, we further assume the flattened covariate is a constant.)
    if ignore_covariates:
        Z = np.zeros_like(net.Z_tilde)
    else:
        Z = net.Z_tilde
    zmax = max(Z)

    # get embeddings
    l, U = sparse.linalg.eigsh(net.A, d)
    X = np.multiply(np.abs(l)**0.5, U)

    # partition nodes by covariate value, cluster by partition
    S = []
    clusters = []
    labels = []
    for i in range(zmax + 1):
        S.append([j for j in range(len(Z)) if Z[j] == i])
        clusters.append(GaussianMixture(n_components=k).fit(X[S[i],:]))
        labels.append(clusters[i].predict(X[S[i],:]))

    # find optimal matching of clusters, invert to create lookups
    cluster_map = [invert_matching(m) for m in optimal_matching(clusters)]

    # reconcile labels for final labels
    theta = [0 for _ in range(len(Z))]
    for z in range(zmax + 1):
        for i_S, i_Z in enumerate(S[z]):
            theta[i_Z] = cluster_map[z][labels[z][i_S]]
    
    # theta tilde!
    theta_tilde = [tuple_id((theta[i], net.Z_tilde[i]), (k, max(net.Z_tilde) + 1)) for i in range(len(theta))]
    
    return NetworkClusterResult(n_communities=k, theta=theta, theta_tilde=theta_tilde)


# O(lk^3 + lk^2d)?
# distance matrices: O(l k^2 d), where d is dimension
# assignment: O(l k^3)
def optimal_matching(clusters):
    """
    Returns a 2D array [matching_1, ... matching_k]:
    Each entry gives indices mapping the cluster labels from the 0-th level
    to the i-th level
    """
    l = len(clusters)
    k = clusters[0].means_.shape[0]

    best_matching = []
    best_matching.append(list(range(k))) # mapping level 0 => level 0 is just the identity

    for i in range(1, l):
        _, opt = optimize.linear_sum_assignment(
            spatial.distance_matrix(clusters[0].means_, clusters[i].means_) ** 2
        )
        best_matching.append(opt.tolist())

    return best_matching

def invert_matching(matching):
    """
    A matching [i_1, ... i_k] for a given level l means:
    j-th community in level 0 is equivalent to community i_j in level l.
    
    We want to work our way back from this, so we invert the mapping.
    The inverted mapping [i'_1, ... i'_k] satisfies:
    j-th community in level i is equivalent to community i'_j in level 1.
    """
    return [x[0] for x in sorted(enumerate(matching), key=lambda x: x[1])]

def estimate(net: GeneratedNetwork, cluster_result: NetworkClusterResult) -> NetworkEstimationResult:
    theta_tilde = cluster_result.theta_tilde
    n_blocks = max(theta_tilde) + 1
    est_block_sizes = Counter(theta_tilde)
    n_communities = cluster_result.n_communities
    n_covariates = len(net.model.covariates)

    # for recovering B*
    dyad_count = np.zeros((n_blocks, n_blocks))
    connections = np.zeros((n_blocks, n_blocks))

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

    # Step 1: Count connections
    for idx_1, idx_2 in zip(*net.A.nonzero()):        
        if idx_1 > idx_2:
            continue # don't double-count edges!

        t1, t2 = theta_tilde[idx_1], theta_tilde[idx_2]
        connections[min(t1, t2), max(t1, t2)] += 1

    # Step 2: Build response and indicator matrices for regression
    # 1 <= i <= j <= n_blocks
    row = 0
    for i in range(n_blocks):
        for j in range(i, n_blocks):
            # count dyads
            if i == j:
                # on-diagonal block: (n choose 2) dyads
                dyad_count[i, i] = est_block_sizes[i] * (est_block_sizes[i] - 1) // 2
            else:
                # off-digonal block: (m x n) dyads
                dyad_count[i, j] = est_block_sizes[i] * est_block_sizes[j]
            
            t_i = from_tuple_id(i, tuple_levels)
            t_j = from_tuple_id(j, tuple_levels)
            base_block = (min(t_i[0], t_j[0]), max(t_i[0], t_j[0])) # again, upper triangle commplications
            base_block_id = B_idx_lookup[base_block]
            base_block_indicators[row, base_block_id] = 1
            ind = np.where(np.array(t_i) == np.array(t_j), 1, 0)[1:] # covariate match indicators
            covariate_indicators[row,:] = ind
            response[row,:] = np.array([connections[i, j], dyad_count[i, j] - connections[i, j]])

            row += 1

    # Step 3: Fit GLM to estimate coefficients
    model = sm_GLM(
        response, np.hstack([base_block_indicators, covariate_indicators]),
        family=sm_families.Binomial(link=net.model.link._statsmodels_link)
    )
    results = model.fit()
    coef = results.params[n_base_blocks:]

    # Step 4: Assemble B_star, the empirical equivalent of B_tilde
    B_star = connections / np.maximum(dyad_count, 1) # safely divide
    B_star = np.maximum(B_star, B_star.T) # symmetrize, since lower triangle is zero

    # Step 5: Assemble B_hat, the estimation of B
    B_hat = np.zeros((n_communities, n_communities))
    B_hat[np.triu_indices_from(B_hat)] = results.params[0:n_base_blocks]
    B_hat[np.tril_indices_from(B_hat)] = results.params[0:n_base_blocks][::-1]

    return NetworkEstimationResult(
        B_star=B_star,
        B_hat=B_hat,
        coefficients=coef
    )

def label_accuracy(labels, truth):
    """
    max(accuracy) over the set of all label permutations
    truth should be an list of 0-indexed integer labels of length n
    """
    accuracy = 0
    k = max(truth) + 1 # number of labels
    
    # This is not optimal, but we're using small k, so it's no biggie.
    for p in itertools.permutations(range(k)):
        compare = [p[t] for t in truth]
        accuracy = max(accuracy, metrics.accuracy_score(labels, compare))
    
    return accuracy
