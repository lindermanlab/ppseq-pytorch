import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim

# Plotting stuff
import matplotlib.pyplot as plt

# Some helper utilities
from tqdm.auto import trange

def pred(w, a, b):
  """
  predict mean

  args:
  - w,a,b

  returns:
  - prediction of E[X]
  """

  K,N,D = w.shape
  lambdas = b.view(N,1) + F.conv1d(a, torch.flip(w.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]

  return lambdas

def log_probability(X, a, b, W):
  """
  calculate the log probability given data X 
  and estimated parameters a, b, W
  """
  N, T = X.shape
  K, N, D = W.shape
  lambda_nt = b.view(N,1) + F.conv1d(a, torch.flip(W.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
  lambda_nt = torch.clamp(lambda_nt, min=1e-5)
  return torch.sum(dist.Poisson(lambda_nt).log_prob(X))

def m_step(X, b, a, scale, mu, log_sigma, D, optimizer):
  """
  Args:
    - X: (N, T)
    - b: (N)
    - a: (K, T)
    - scale: (K, N) the log_scale that softmax to the true scale of the gaussian pdf
    - mu: (K, N) mean of the gaussian
    - log_sigma: (K, N) log of std of gaussian
      the weight W is determined by (scale, mu, log_sigma)
    - D: delay

  Returns:
  updated parameters in the m-step of an EM algorithm:
  b, a, scale, mu, log_sigma
  """

  N, T = X.shape
  K, N = scale.shape

  W = torch.exp(dist.Normal(mu, torch.exp(log_sigma)).log_prob(torch.arange(D)[:, None, None, ...])).permute(1,2,0)\
   * F.softmax(scale, dim=1).unsqueeze(-1).expand(-1, -1, D) #(K,N,D)

  # update b
  lambda_nt = b.view(N,1) + F.conv1d(a, torch.flip(W.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
  r_nt = X / (lambda_nt + 1e-4) # (N, T)
  b = torch.sum(r_nt, axis=1) * b / T

  # update a
  lambda_nt = b.view(N,1) + F.conv1d(a, torch.flip(W.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
  r_nt = X / (lambda_nt + 1e-4)
  beta_kt = torch.sum(W, dim=(1,2)).unsqueeze(1).repeat(1,T) # (K, T)
  a = a * F.conv1d(r_nt, W,  padding=D-1)[:,D-1:] / beta_kt

  # target W
  lambda_nt = b.view(N,1) + F.conv1d(a, torch.flip(W.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
  r_nt = X / (lambda_nt + 1e-4)
  beta_knd = torch.sum(a, dim=1)[:,None,None].repeat(1,N,D)
  conv = torch.flip(F.conv1d(a.unsqueeze(1), r_nt.unsqueeze(1),padding=D-1)[:,:,:-D+1], [2])
  W_hat = (W * conv / beta_knd).detach()

    
  for i in range(200):
    mse_loss = nn.MSELoss()
    loss = mse_loss(W_hat, W)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return b, a, scale, mu, log_sigma

def em(X,
       K,
       D,
       n_iter=50,
       ):
    """
    EM algorithm.

    Args:
    - X: (N, T)
    - K, D: scalar
    - n_iter: number of iterations of EM.

    Returns:
    - b, a, W
    - lps: the history of log probabilities
    - scale, mu, log_sigma
    """
    lps = []
    N, T = X.shape

    # Initialize  parameters
    b = torch.rand(N)
    a = torch.rand(K,T)
    scale = torch.randn(K, N, requires_grad=True)
    mu = torch.ones(K, N, requires_grad=True) * 5
    log_sigma = torch.ones(K, N, requires_grad=True)
    optimizer = optim.Adam([scale, mu, log_sigma], lr=0.01)
    # Run EM
    for _ in trange(n_iter):
        b, a, scale, mu, log_sigma = m_step(X, b, a, scale, mu, log_sigma, D, optimizer)
        W = torch.exp(dist.Normal(mu, torch.exp(log_sigma)).log_prob(torch.arange(D)[:, None, None, ...])).permute(1,2,0)\
         * F.softmax(scale, dim=1).unsqueeze(-1).expand(-1, -1, D).detach()
        lps.append(log_probability(X, a, b, W))

    return  b, a, W, lps, scale.detach(), mu.detach(), log_sigma.detach()