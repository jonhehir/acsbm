from timeit import default_timer as timer
import warnings

import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tools.sm_exceptions import DomainWarning

from acsbm import generation, estimation, utils


# ignore DomainWarning from statsmodels. We know...
warnings.simplefilter("ignore", DomainWarning)

def rng(seed: int = 123):
    return np.random.default_rng(seed)

def run_simulation(model, ndd, n, ignore_covariates=False, ignore_communities=False, seed_rng=True):
    random_state = rng() if seed_rng else None

    net = generation.generate_network(model, ndd, n, random_state=random_state)

    k = 1 if ignore_communities else model.n_communities
    d = len(net.block_sizes)

    # Show model, network
    if n <= 1000:
        plt.matshow(net.A.toarray())
        plt.title("Adjacency Matrix")
    plt.matshow(model.B_tilde())
    plt.title("B_tilde (true)")
    plt.colorbar()

    # Ignore covariates?
    if ignore_covariates:
        net.Z_tilde = np.zeros_like(net.Z_tilde)

    # Cluster!
    start = timer()
    ic_result = estimation.initial_cluster(net, k, d)
    mid = timer()
    c_result = estimation.reconcile_clusters(net, ic_result)
    end = timer()
    
    # Report clustering accuracy only if not ignoring communities (in which case accuracy is meaningless)
    if not ignore_communities:
        print(f"Accuracy: {utils.label_accuracy(c_result.theta, net.theta)}")
        print(f"Clustering Time: {mid-start} + {end-mid} = {end-start}")

    # Estimate coefficients (unless ignoring covariates)
    if not ignore_covariates:
        start = timer()
        e_result = estimation.estimate(net, c_result)
        end = timer()

        plt.matshow(c_result.B_tilde)
        plt.title("B_tilde (estimated)")
        plt.colorbar()
        plt.matshow(e_result.B_hat)
        plt.title("B (estimated)")
        plt.colorbar()

        print(f"Coefficients: {e_result.coefficients}")
        print(f"Estimation Time: {end-start}")
