import numpy as np 
import pandas as pd 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim

from tqdm.auto import trange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('cpu')


def pred(w, a, b):
  """
  predict mean, i.e. the intensities of each spike
  args:
  - w: weights (K,N,D)
  - a: amplitutes (K,T)
  - b: base rate (N,)
  returns:
  - prediction of E[X], shape (N,T)
  """
  K,N,D = w.shape
  lambdas = b.view(N,1) + F.conv1d(a, torch.flip(w.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
  return lambdas

def log_probability(X, a, b, W):
  """
  calculate the log probability given data X
  and estimated parameters a, b, W
  args:
  - X: spike train data (N,T)
  - w: weights (K,N,D)
  - a: amplitutes (K,T)
  - b: base rate (N,)
  returns: 
  - a scalar of the log probability
  """
  N, T = X.shape
  K, N, D = W.shape
  lambda_nt = b.view(N,1) + F.conv1d(a, torch.flip(W.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
  lambda_nt = torch.clamp(lambda_nt, min=1e-7)
  return torch.sum(dist.Poisson(lambda_nt).log_prob(X))


class compute_weight(nn.Module):
    """
    This module compute the weight w given scale, mu, sigma
    the weight is modeled as scale * Normal(mu, sigma)
    args:
    - D: delay (scalar)
    - scale: the log scale, 
    we use a softmax to force the scale sum up to one  (K,N) 
    - mu: the mean delay (K,N)
    - sigma: log of std of gaussian (K,N)
    returns: 
    - w: weights (K,N,D)
    """
    def __init__(self, D):
        super(compute_weight, self).__init__()
        self.D = D

    def forward(self, scale, mu, log_sigma):
        delay = torch.arange(self.D, device=scale.device)
        return torch.exp(dist.Normal(mu, torch.exp(log_sigma)).log_prob(delay[:, None, None, ...])).permute(1,2,0)\
        * F.softmax(scale, dim=1).unsqueeze(-1).expand(-1, -1, self.D)

def em(X,
       K,
       D,
       n_iter=50,
       sgd_iter=1e5,
       alpha_a0=0.5, beta_a0=0, 
       alpha_b0=0, beta_b0=0
       ):
    """
    EM algorithm.

    Args:
    - X: (N, T)
    - K: type of actions, scalar
    - D: delay, scalar
    - n_iter: number of iterations of EM, scalar
    - sgd_iter: number of iterations of sgd to fit scale, mu, sigma, scalar
    - alpha_a0, beta_a0: gamma prior of base rate b, scalar
    - alpha_b0, beta_b0: gamma prior of amplitudes a, scalar

    Returns:
    - b: base rate (N)
    - a: amplitudes (K,T)
    - W: weights based on em_algorithm (K,N,D) 
    - lps: the history of log probabilities, a list
    - scale: the log scale, 
    we use a softmax to force the scale sum up to one  (K,N) 
    - mu: the mean delay (K,N)
    - log_sigma: log of std of gaussian (K,N)
    - loss_history: the loss of sgd iterations of scale mu log_sigma, a list
    - W_prediction: the predicted weights from scale, mu, log_sigma (K,N,D)
    """
    lps = []

    X = X.to(device)
    N, T = X.shape

    # Initialize parameters
    b = torch.rand(N, device=device)
    a = torch.rand(K, T, device=device) + 1e-4
    scale = torch.rand(K, N, requires_grad=True, device=device)
    mu = torch.tensor(torch.ones(K, N) * D/2, requires_grad=True, device=device)
    log_sigma = torch.ones(K, N, requires_grad=True, device=device)
    W = (torch.exp(dist.Normal(mu, torch.exp(log_sigma)).log_prob(torch.arange(D, device=device)[:, None, None, ...])).permute(1,2,0)\
   * F.softmax(scale, dim=1).unsqueeze(-1).expand(-1, -1, D)).detach() #(K,N,D)

    def m_step(X):
      """
      Args:
      - X: (N,T)
      - b: (N)
      - a: (K,T)
      - W: (K,N,D)
      - D: delay, scalar
      - alpha_a0, beta_a0, alpha_b0, beta_b0: prior parameters
      Returns:
      updated parameters in the m-step of an EM algorithm:
      b, a, W
      """
      nonlocal b, a, W, alpha_a0, beta_a0, alpha_b0, beta_b0

      N, T = X.shape
      K, N = scale.shape

      # update b
      lambda_nt = b.view(N,1) + F.conv1d(a, torch.flip(W.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
      r_nt = X / (lambda_nt +1e-7) # add 1e-7 for numerical stability (N, T)
      b = torch.clip((torch.sum(r_nt, dim=1) * b + alpha_b0 - 1) / (T + beta_b0), 0)

      # update a
      lambda_nt = b.view(N,1) + F.conv1d(a, torch.flip(W.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
      r_nt = X / (lambda_nt +1e-7)
      beta_kt = torch.sum(W, dim=(1,2)).unsqueeze(1).repeat(1,T) # (K, T)
      a = torch.clip((a * F.conv1d(r_nt, W,  padding=D-1)[:,D-1:] + alpha_a0 - 1) / (beta_kt + beta_a0), 0)

      # update W
      lambda_nt = b.view(N,1) + F.conv1d(a, torch.flip(W.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
      r_nt = X / (lambda_nt +1e-7)
      beta_knd = torch.sum(a, dim=1)[:,None,None].repeat(1,N,D)
      conv = torch.flip(F.conv1d(a.unsqueeze(1), r_nt.unsqueeze(1),padding=D-1)[:,:,:-D+1], [2])
      W = (W * conv) / beta_knd


    # Run EM
    for _ in trange(n_iter):
        m_step(X)
        lps.append(log_probability(X, a, b, W).detach().cpu())

    # initialize SGD 
    loss_hist = []
    model = compute_weight(D)
    model.to(device)
    optimizer = optim.Adam([scale, mu, log_sigma], lr=0.01)
    criterion = nn.MSELoss()
    # run SGD 
    for i in trange(int(sgd_iter)):
        optimizer.zero_grad()
        W_prediction = model(scale, mu, log_sigma)
        loss = criterion(W, W_prediction)
        loss_hist.append(loss)
        loss.backward()
        optimizer.step()

    return b.detach().cpu(), a.detach().cpu(), W.detach().cpu(), lps, scale.detach().cpu(), mu.detach().cpu(), log_sigma.detach().cpu(), loss_hist, W_prediction.detach().cpu()


"""
# deprecated functions

def quantile_mean(arr):
    low = np.quantile(arr, 0.0)
    high = np.quantile(arr, 0.85)
    filtered_data = arr[(arr >= low) & (arr <= high)]
    return np.mean(filtered_data)


def estimate_b(X):
    N, T = X.shape
    b = np.zeros(N)
    df = pd.DataFrame(X.cpu().T)
    averages = df.rolling(window=int(np.sqrt(T))).mean().dropna().to_numpy().T
    for i in range(N):
        #b[i] = quantile_mean(averages[i])
        b[i] = np.median(averages[i])
    return torch.tensor(b).float().to(device) 
"""