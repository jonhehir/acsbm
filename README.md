# Additive-Covariate Stochastic Block Model (`acsbm`)

This code accompanies a paper on a model called the additive-covariate stochastic block model (ACSBM), an extension of the popular stochastic block model. The purpose of this model and the proposed spectral fitting method is to separate observed homophily from latent network structure. For details on the spectral fitting method, please see the paper.

## Examples

Examples can be found in the [/examples](/examples) directory. In particular:

- [`simpsons-paradox`](/examples/simpsons-paradox.ipynb): This illustrates a Simpson's paradox example in the network setting. Each node has two binary covariates. The first covariate induces positive homophily, while the second induces negative homophily.
- [`misc-examples`](/examples/misc-examples.ipynb): A notebook filled with a smattering of miscellaneous examples of networks fit by the proposed algorithm.

## Reproducibility

Simulation code can be found in [`/src/simulator.py`](/src/simulator.py). The included `Dockerfile` can be used to generate a Docker image capable of running all code. This code has been tested on Python 3.10 with the modules listed in `requirements.txt`.
