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

import acsbm


# ignore DomainWarnings
warnings.simplefilter("ignore", DomainWarning)

# We're outputting tab-delimited data with columns for each covariate.
# To ensure we always have the same number of columns, we'll pad with
# blanks up to MAX_COVARIATES when the actual number of covariates is smaller.
MAX_COVARIATES = 5

@dataclass
class SimulationSetting:
    base_model: acsbm.MultiCovariateModel
    sparsity: float
    ndd: acsbm.NodeDataDistribution = None

    def __post_init__(self):
        if self.ndd is None:
            self.ndd = acsbm.NodeDataDistribution.uniform_for_model(self.base_model)
    
    def model_at(self, n: int) -> acsbm.MultiCovariateModel:
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
        base_model=acsbm.MultiCovariateModel(
            B=(3 * np.eye(2) - 5),
            covariates=[acsbm.Covariate.simple(-0.5, 2)],
            link=acsbm.LinkFunction.log()
        ),
        ndd = acsbm.NodeDataDistribution([[4, 1], [1, 4]]),
        sparsity = 0
    ),
    "nonuniform": SimulationSetting(
        base_model = acsbm.MultiCovariateModel(
            B=np.array([
                [2, 1],
                [1, 0.2]
            ]) + 1,
            covariates=[acsbm.Covariate.simple(0.5, 2), acsbm.Covariate.simple(-0.5, 2)],
            link=acsbm.LinkFunction.logit()
        ),
        sparsity=0.9
    ),
    "uniform": SimulationSetting(
        base_model = acsbm.MultiCovariateModel(
            B=-2 * np.eye(2) + 2,
            covariates=[acsbm.Covariate.simple(0.5, 2), acsbm.Covariate.simple(-0.5, 2)],
            link=acsbm.LinkFunction.logit()
        ),
        sparsity=0.9
    ),
    "big-covariate": SimulationSetting(
        base_model = acsbm.MultiCovariateModel(
            B=np.log(.6 * np.eye(2) + 1),
            covariates=[acsbm.Covariate.simple(4, 2)],
            link=acsbm.LinkFunction.logit()
        ),
        sparsity=0.9
    ),
    "probit-dense": SimulationSetting(
        base_model = acsbm.MultiCovariateModel(
            B=-.5 * np.eye(3) - 1,
            covariates=[
                acsbm.Covariate.simple(-0.7, 2),
                acsbm.Covariate.simple(0.1, 2)
            ],
            link=acsbm.LinkFunction.probit()
        ),
        sparsity = 0
    ),
    "logit-dense": SimulationSetting(
        base_model = acsbm.MultiCovariateModel(
            B=-.5 * np.eye(3) - 1,
            covariates=[
                acsbm.Covariate.simple(-0.7, 2),
                acsbm.Covariate.simple(0.1, 2)
            ],
            link=acsbm.LinkFunction.logit()
        ),
        sparsity = 0
    ),
    "identity-dense": SimulationSetting(
        base_model = acsbm.MultiCovariateModel(
            B=-0.1 * np.eye(3) + 0.2,
            covariates=[
                acsbm.Covariate.simple(0.05, 2),
                acsbm.Covariate.simple(-0.05, 2)
            ],
            link=acsbm.LinkFunction.identity()
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
        net = acsbm.generate_network(model, setting.ndd, n)
    except acsbm.NetworkTooSmallError:
        warnings.warn("Skipping simulation due to NetworkTooSmallError.")
        return None
    
    start = timer()
    c_result = acsbm.cluster(net, model.n_communities, net.n_blocks)
    c_time = timer() - start
    
    accuracy = acsbm.label_accuracy(net.theta, c_result.theta)

    start = timer()
    e_result = acsbm.estimate(net, c_result)
    e_time = timer() - start

    data = [name, datetime.datetime.now(), setting.hash(), setting.sparsity, n, net.n_edges, c_time, e_time, accuracy]
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
