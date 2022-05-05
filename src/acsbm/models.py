from dataclasses import dataclass
import functools

import numpy as np
import statsmodels.genmod.families as sm_families

from .utils import kron_combine


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

    def draw(self, n: int, random_state: np.random.Generator = None) -> np.ndarray:
        if random_state is None:
            random_state = np.random.default_rng()

        all_indices = list(np.ndindex(*self.pmf.shape))
        data = random_state.choice(all_indices, n, p = self.pmf.reshape(len(all_indices),))
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
