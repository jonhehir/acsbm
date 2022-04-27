import argparse
import copy
from dataclasses import dataclass
import datetime
import hashlib
import itertools
import multiprocessing
from timeit import default_timer as timer
import warnings

import numpy as np
from statsmodels.tools.sm_exceptions import DomainWarning

from acsbm import models, generation, estimation
from acsbm.utils import label_accuracy


# ignore DomainWarnings
warnings.simplefilter("ignore", DomainWarning)

# We're outputting tab-delimited data with columns for each covariate.
# To ensure we always have the same number of columns, we'll pad with
# blanks up to MAX_COVARIATES when the actual number of covariates is smaller.
MAX_COVARIATES = 5

@dataclass
class SimulationSetting:
    base_model: models.MultiCovariateModel
    sparsity: float
    ndd: models.NodeDataDistribution = None

    def __post_init__(self):
        if self.ndd is None:
            self.ndd = models.NodeDataDistribution.uniform_for_model(self.base_model)
    
    def model_at(self, n: int) -> models.MultiCovariateModel:
        # add log-scale sparsity if called for (do not mutate base_model!)
        if self.sparsity > 0:
            B = np.copy(self.base_model.B)
            model = copy.copy(self.base_model)
            model.B = B - np.log(n ** self.sparsity)
            return model
        
        return self.base_model

    def hash(self) -> str:
        """
        Quick-and-dirty way to try to ensure that repeated runs that claim to be
        the same setting truly are the same.
        """
        h = hashlib.sha1()
        data = [
            self.base_model.B,
            self.base_model.flatten_covariate().beta_matrix,
            self.ndd.pmf,
            np.array([self.sparsity])
        ]
        for a in data:
            h.update(a.tobytes())
        
        return h.hexdigest()


settings = {
    "simpson": SimulationSetting(
        base_model=models.MultiCovariateModel(
            B=(3 * np.eye(2) - 5),
            covariates=[models.Covariate.simple(-0.5, 2)],
            link=models.LinkFunction.log()
        ),
        ndd = models.NodeDataDistribution([[4, 1], [1, 4]]),
        sparsity = 0
    ),
    "nonuniform": SimulationSetting(
        base_model = models.MultiCovariateModel(
            B=np.array([
                [2, 1],
                [1, 0.2]
            ]) + 1,
            covariates=[models.Covariate.simple(0.5, 2), models.Covariate.simple(-0.5, 2)],
            link=models.LinkFunction.logit()
        ),
        sparsity=0.9
    ),
    "uniform": SimulationSetting(
        base_model = models.MultiCovariateModel(
            B=-2 * np.eye(2) + 2,
            covariates=[models.Covariate.simple(0.5, 2), models.Covariate.simple(-0.5, 2)],
            link=models.LinkFunction.logit()
        ),
        sparsity=0.9
    ),
    "big-covariate": SimulationSetting(
        base_model = models.MultiCovariateModel(
            B=np.log(.6 * np.eye(2) + 1),
            covariates=[models.Covariate.simple(4, 2)],
            link=models.LinkFunction.logit()
        ),
        sparsity=0.9
    ),
    "probit-dense": SimulationSetting(
        base_model = models.MultiCovariateModel(
            B=-.5 * np.eye(3) - 1,
            covariates=[
                models.Covariate.simple(-0.7, 2),
                models.Covariate.simple(0.1, 2)
            ],
            link=models.LinkFunction.probit()
        ),
        sparsity = 0
    ),
    "logit-dense": SimulationSetting(
        base_model = models.MultiCovariateModel(
            B=-.5 * np.eye(3) - 1,
            covariates=[
                models.Covariate.simple(-0.7, 2),
                models.Covariate.simple(0.1, 2)
            ],
            link=models.LinkFunction.logit()
        ),
        sparsity = 0
    ),
    "identity-dense": SimulationSetting(
        base_model = models.MultiCovariateModel(
            B=-0.1 * np.eye(3) + 0.2,
            covariates=[
                models.Covariate.simple(0.05, 2),
                models.Covariate.simple(-0.05, 2)
            ],
            link=models.LinkFunction.identity()
        ),
        sparsity = 0
    )
}

def run_simulation(name: str, n: int):
    # generate random seed explicitly each time
    np.random.seed()
    
    setting = settings[name]
    model = setting.model_at(n)
    try:
        net = generation.generate_network(model, setting.ndd, n)
    except generation.NetworkTooSmallError:
        warnings.warn("Skipping simulation due to NetworkTooSmallError.")
        return None
    
    start = timer()
    ic_result = estimation.initial_cluster(net, model.n_communities, net.n_blocks)
    mid = timer()
    c_result = estimation.reconcile_clusters(net, ic_result)
    c1_time = mid - start
    c2_time = timer() - mid
    
    accuracy = label_accuracy(net.theta, c_result.theta)

    start = timer()
    e_result = estimation.estimate(net, c_result)
    e_time = timer() - start

    data = [name, datetime.datetime.now(), setting.hash(), setting.sparsity, n, net.n_edges, c1_time, c2_time, e_time, accuracy]
    n_covariates = len(model.covariates)
    for i in range(n_covariates):
        data.append(model.covariates[i].beta_matrix[0, 0])
        data.append(e_result.coefficients[i])
    blanks = [''] * (MAX_COVARIATES - n_covariates) * 2
    data.extend(blanks)
    
    return data

def print_result(result):
    if result is not None:
        print("\t".join([str(x) for x in result]))


parser = argparse.ArgumentParser(description="Run ACSBM simulations")
parser.add_argument("--name", type=str, help="Name of simulation setting")
parser.add_argument("--runs", type=int, help="Number of simulations to run", default=1)
parser.add_argument("--n", type=int, nargs="+", help="Number of nodes in each simulated network")

args = parser.parse_args()

with multiprocessing.Pool() as pool:
    jobs = []
    
    for (n, _) in itertools.product(args.n, range(args.runs)):
        a = (args.name, n)
        jobs.append(pool.apply_async(run_simulation, a, callback=print_result))

    for job in jobs:
        job.get() # for the sake of re-raising any exceptions in the child process

    pool.close()
    pool.join()
