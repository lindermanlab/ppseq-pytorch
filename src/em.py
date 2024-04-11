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
  lambda_nt = torch.clamp(lambda_nt, min=1e-7)
  return torch.sum(dist.Poisson(lambda_nt).log_prob(X))

class compute_weight(nn.Module):
    def __init__(self, D):
        super(compute_weight, self).__init__()
        self.D = D

    def forward(self, scale, mu, log_sigma):
        delay = torch.arange(self.D, device=scale.device)
        return torch.exp(dist.Normal(mu, torch.exp(log_sigma)).log_prob(delay[:, None, None, ...])).permute(1,2,0)\
   * F.softmax(scale, dim=1).unsqueeze(-1).expand(-1, -1, self.D)

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

def em(X,
       K,
       D,
       n_iter=50,
       alpha_a0 = 0.5, beta_a0 = 0, 
       alpha_b0=0, beta_b0=0
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
    X = X.to(device)
    lps = []
    N, T = X.shape

    # Initialize  parameters
    b = torch.rand(N, device=device)
    a = torch.rand(K, T, device=device) + 1e-4
    scale = torch.rand(K, N, requires_grad=True, device=device)
    mu = torch.tensor(torch.ones(K, N) * D/2, requires_grad=True, device=device)
    log_sigma = torch.ones(K, N, requires_grad=True, device=device)

    model = compute_weight(D)
    model.to(device)

    W = (torch.exp(dist.Normal(mu, torch.exp(log_sigma)).log_prob(torch.arange(D, device=device)[:, None, None, ...])).permute(1,2,0)\
   * F.softmax(scale, dim=1).unsqueeze(-1).expand(-1, -1, D)).detach() #(K,N,D)

    b = estimate_b(X)

    def m_step(X):
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
      nonlocal b, a, W, alpha_a0, beta_a0, alpha_b0, beta_b0

      N, T = X.shape
      K, N = scale.shape

      # update b
      lambda_nt = b.view(N,1) + F.conv1d(a, torch.flip(W.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
      r_nt = X / (lambda_nt +1e-7) # (N, T)
      b = torch.clip((torch.sum(r_nt, dim=1) * b + alpha_b0 - 1) / (T + beta_b0), 0)

      # update a
      lambda_nt = b.view(N,1) + F.conv1d(a, torch.flip(W.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
      r_nt = X / (lambda_nt +1e-7 )
      beta_kt = torch.sum(W, dim=(1,2)).unsqueeze(1).repeat(1,T) # (K, T)
      a = torch.clip((a * F.conv1d(r_nt, W,  padding=D-1)[:,D-1:] + alpha_a0 - 1) / (beta_kt + beta_a0), 0)

      # update W, scale, mu, log_sigma
      lambda_nt = b.view(N,1) + F.conv1d(a, torch.flip(W.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
      r_nt = X / (lambda_nt +1e-7 )
      beta_knd = torch.sum(a, dim=1)[:,None,None].repeat(1,N,D)
      conv = torch.flip(F.conv1d(a.unsqueeze(1), r_nt.unsqueeze(1),padding=D-1)[:,:,:-D+1], [2])
      W = (W * conv - 0) / beta_knd.detach()


    # Run EM
    for _ in trange(n_iter):
        m_step(X)
        ll = log_probability(X, a, b, W).detach().cpu()
        lps.append(ll)

    loss_hist = []
    
    optimizer = optim.Adam([scale, mu, log_sigma], lr=0.01)
    #optimizer = optim.LBFGS([scale, mu, log_sigma], lr=0.01)
    criterion = nn.MSELoss()

    for i in trange(10000):
        optimizer.zero_grad()
        W_prediction = model(scale, mu, log_sigma)
        loss = criterion(W, W_prediction)
        loss_hist.append(loss)
        loss.backward()
        #max_norm = 1.0  # Adjust this value according to your needs
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

    """
    for i in trange(10):
      def closure():
        optimizer.zero_grad()
        W_prediction = model(scale, mu, log_sigma)
        loss = criterion(W, W_prediction)
        loss_hist.append(loss.item())
        loss.backward()
        return loss

      optimizer.step(closure)

    prev_loss = float('inf')
    loss_stable_count = 0
    stability_threshold = 1e-4  # Define your own threshold for stability

    for epoch in trange(10000):
      optimizer.zero_grad()
      W_prediction = model(scale, mu, log_sigma)
      loss = criterion(W, W_prediction)
      loss_hist.append(loss)
      loss.backward()
      optimizer.step()

      if np.abs(loss.item() - prev_loss) < stability_threshold:
        loss_stable_count += 1
        if loss_stable_count >= 10000:  # Adjust as needed
            print("Loss has stabilized. Early stopping...")
            break
      else:
        loss_stable_count = 0

      prev_loss = loss.item()

    """
    #W = (torch.exp(dist.Normal(mu, torch.exp(log_sigma)).log_prob(torch.arange(D, device=device)[:, None, None, ...])).permute(1,2,0)\
        # * F.softmax(scale, dim=1).unsqueeze(-1).expand(-1, -1, D)).detach()
    #ll = log_probability(X, a, b, W).detach().cpu()
    #lps.append(ll)

    return b.detach().cpu(), a.detach().cpu(), W.detach().cpu(), lps, scale.detach().cpu(), mu.detach().cpu(), log_sigma.detach().cpu(), loss_hist, W_prediction.detach().cpu()