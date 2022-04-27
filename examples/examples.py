from timeit import default_timer as timer
import warnings

import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tools.sm_exceptions import DomainWarning

from acsbm import generation, estimation, utils


# ignore DomainWarning from statsmodels. We know...
warnings.simplefilter("ignore", DomainWarning)

def run_simulation(model, ndd, n, ignore_covariates=False, ignore_communities=False):
    net = generation.generate_network(model, ndd, n)

    k = 1 if ignore_communities else model.n_communities
    d = len(net.block_sizes)

    # Show model, network
    if n <= 1000:
        plt.matshow(net.A.toarray())
    plt.matshow(model.B_tilde())
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
        plt.colorbar()
        plt.matshow(e_result.B_hat)
        plt.colorbar()

        print(f"Coefficients: {e_result.coefficients}")
        print(f"Estimation Time: {end-start}")
