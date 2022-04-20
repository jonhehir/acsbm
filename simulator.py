import argparse
import copy
from dataclasses import dataclass
import datetime
import hashlib
import itertools
import multiprocessing
from timeit import default_timer as timer

import numpy as np

import acsbm


@dataclass
class SimulationSetting:
    base_model: acsbm.MultiCovariateModel
    ndd: acsbm.NodeDataDistribution = None

    def __post_init__(self):
        if self.ndd is None:
            self.ndd = acsbm.uniform_for_model(self.model)
    
    def model_at(self, n: int, sparsity: float) -> acsbm.MultiCovariateModel:
        # add log-scale sparsity if called for
        if sparsity > 0:
            model = copy.copy(self.base_model)
            model.B -= np.log(n ** sparsity)
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
            self.ndd.pmf
        ]
        for a in data:
            h.update(a.tobytes())
        
        return h.hexdigest()


settings = {
    "simpson": SimulationSetting(
        base_model=acsbm.MultiCovariateModel(
            B=(3 * np.eye(2) - 4),
            covariates=[acsbm.Covariate.simple(-0.5, 2)],
            link=acsbm.LinkFunction.log()
        ),
        ndd = acsbm.NodeDataDistribution([[4, 1], [1, 4]])
    )
}

def run_simulation(name: str, n: int, sparsity: float):
    # generate random seed explicitly each time
    np.random.seed()
    
    setting = settings[name]
    model = setting.model_at(n, sparsity)
    # TODO: Catch too small exception
    net = acsbm.generate_network(model, setting.ndd, n)
    start = timer()
    # TODO: Suppress warnings!
    c_result = acsbm.cluster(net, model.n_communities, net.n_blocks)
    c_time = timer() - start
    
    accuracy = acsbm.label_accuracy(net.theta, c_result.theta)

    start = timer()
    e_result = acsbm.estimate(net, c_result)
    e_time = timer() - start

    data = [name, datetime.datetime.now(), setting.hash(), sparsity, n, net.n_edges, c_time, e_time, accuracy]
    for i in range(len(model.covariates)):
        data.append(model.covariates[i].beta_matrix[0, 0])
        data.append(e_result.coefficients[i])
    
    return data

def print_result(result):
    print("\t".join([str(x) for x in result]))


parser = argparse.ArgumentParser(description="Run ACSBM simulations")
parser.add_argument("--name", type=str, help="Name of simulation setting")
parser.add_argument("--runs", type=int, help="Number of simulations to run", default=1)
parser.add_argument("--n", type=int, nargs="+", help="Number of nodes in each simulated network")
parser.add_argument("--sparsity", type=float, help="If set, model will be adjusted by approx. n^-sparsity", default=0.0)

args = parser.parse_args()

with multiprocessing.Pool() as pool:
    jobs = []
    
    for (n, _) in itertools.product(args.n, range(args.runs)):
        a = (args.name, n, args.sparsity)
        jobs.append(pool.apply_async(run_simulation, a, callback=print_result))

    for job in jobs:
        job.get() # for the sake of re-raising any exceptions in the child process

    pool.close()
    pool.join()
