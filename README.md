# PPSeq

The original PPSeq algorithm worked in continuous time and used a collapsed Gibbs sampler for nonparametric Bayesian inference. Unfortunately, the collapsed algorithm introduced serial dependencies that could make computation slow on some datasets. Here, we implement a simpler inference algorithm that works in discrete time takes advantage of parallelization across time bins using a GPU.