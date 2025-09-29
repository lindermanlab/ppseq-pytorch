#PPSeq sets to mode, CAVI sets to mean


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from fastprogress import progress_bar
from torch import Tensor
from jaxtyping import Float


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from fastprogress import progress_bar
from torch import Tensor
from jaxtyping import Float


class GLMPPSeq:
    """PPSeq is a probabilistic model for detecting sequences of spikes
    embedded in multi-neuronal spike trains. It is based on a Poisson
    latent variable model, akin to a non-negative, convolutional matrix
    factorization.
    """
    base_rates : Float[Tensor, "num_neurons"]
    template_scales: Float[Tensor, "num_templates num_neurons"]
    template_offsets: Float[Tensor, "num_templates num_neurons"]
    template_widths: Float[Tensor, "num_templates num_neurons"]

    def __init__(self,
                 num_templates: int,
                 num_neurons: int,
                 template_duration: int,
                 alpha_a0: float=0.,
                 beta_a0: float=0.,
                 alpha_b0: float=0.,
                 beta_b0: float=0.,
                 alpha_t0: float=0.,
                 beta_t0: float=0.,
                 device=None,
                 use_bias=True,
                 empirical_glm=True,
                 n_covariates=6,
                 ):
        self.num_templates = num_templates
        self.num_neurons = num_neurons
        self.template_duration = template_duration

        # Set the device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not torch.cuda.is_available():
                print('Could not find a GPU. Defaulting to CPU instead.')
        self.device = device


        self.base_rates = torch.ones(num_neurons, device=device)
        self.template_scales = torch.ones(num_templates, num_neurons, device=device) / num_neurons
        self.template_offsets = template_duration * torch.rand(num_templates, num_neurons, device=device)
        self.template_widths = torch.ones(self.num_templates, self.num_neurons, device=device)
        self.use_bias = use_bias
        self.covariate_dim = n_covariates
        if self.use_bias:
          self.beta = torch.zeros(num_neurons, self.covariate_dim +1, device=device)
        else:
          self.beta = torch.zeros(num_neurons, self.covariate_dim, device=device)

        # Set prior hyperparameters
        self.alpha_a0 = alpha_a0
        self.beta_a0 = beta_a0
        self.alpha_b0 = alpha_b0
        self.beta_b0 = beta_b0
        self.alpha_t0 = alpha_t0
        self.beta_t0 = beta_t0
        self.empirical_glm = empirical_glm

    @property
    def templates(self) -> Float[Tensor, "num_templates num_neurons duration"]:
        """Compute the templates from the mean, std, and amplitude of the Gaussian kernel.
        """
        D = self.template_duration
        amp, mu, sigma = self.template_scales, self.template_offsets, self.template_widths
        ds = torch.arange(D, device=self.device)[:, None, None]
        p = dist.Normal(mu, sigma)
        W = p.log_prob(ds).exp().permute(1,2,0)
        return W / W.sum(dim=2, keepdim=True) * amp[:, :, None]

    def get_background_rates(self, X=None):
      """
      Returns background rates: either constant or time-varying via GLM
      X: (covariate_dim, num_timesteps) covariates
      """

      # Add intercept to covariates
      T = X.shape[1]

      # Compute GLM predictions: exp(beta^T * X)
      eta = torch.matmul(self.beta, X)  # (N, T)
      eta = torch.clamp(eta, min=-10, max=10)  # ADD THIS LINE
      return torch.exp(eta)  # (N, T)

    def reconstruct(self,
                    amplitudes: Float[Tensor, "num_templates num_timesteps"],
                    X: Float[Tensor, "covariate_dim num_timesteps"]) \
                    -> Float[Tensor, "num_neurons num_timesteps"]:
        """
        Reconstruct the firing rate given the model parameters and latent variables.

        Parameters
        ----------
        amplitudes: the amplitudes for each template as a function of time
        X: covariates for GLM
        """
        D = self.template_duration
        kernel = torch.flip(self.templates.permute(1,0,2), [2])
        background = self.get_background_rates(X)
        return background + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]

    def log_likelihood(self,
                       data: Float[Tensor, "num_neurons num_timesteps"],
                       amplitudes: Float[Tensor, "num_templates num_timesteps"],
                       X: Float[Tensor, "covariate_dim num_timesteps"],
                       rows = None,
                       cols = None) -> float:
        """
        Calculate the log probability given data X
        and estimated parameters a, b, W

        Parameters
        ----------
        data: spike count matrix
        amplitudes: amplitudes of each template over time
        X: covariates for GLM
        rows,cols: torch.tensor([list of row/col indices]) that represent a subset of rows * cols where the log likelihood is calculated

        Returns
        -------
        Scalar log probability
        """
        D = self.template_duration
        kernel = torch.flip(self.templates.permute(1,0,2),[2])
        background = self.get_background_rates(X)
        rates = background + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        rates = torch.clamp(rates, min=1e-7)
        if rows is None or cols is None:
            return torch.sum(dist.Poisson(rates).log_prob(data))

        data_selected = data[rows,:][:,cols]
        rates_selected = rates[rows,:][:,cols]
        poisson_dist = dist.Poisson(rates_selected)
        log_probs = poisson_dist.log_prob(data_selected)
        log_likelihood = torch.sum(log_probs)
        return log_likelihood


    def _update_amplitudes(self, data, amplitudes, X):
        D, T = self.template_duration, data.shape[1]
        W = self.templates
        kernel = torch.flip(W.permute(1,0,2), [2])
        background = self.get_background_rates(X)  # FIXED: Use GLM rates instead of self.base_rates
        rates = background + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]  # FIXED: Use GLM background
        ratio = data / (rates + 1e-7)

        alpha_post = amplitudes * F.conv1d(ratio, W, padding=D-1)[:,D-1:] + self.alpha_a0
        beta_post = torch.sum(W, dim=(1,2)).unsqueeze(1).repeat(1,T) + self.beta_a0
        return torch.clip((alpha_post - 1) / beta_post, 0)

    def create_smooth_covariates(self, data=None):
      """Create smooth temporal features using Gaussian filtering or RBF basis functions"""
      
      if data is not None:
          # Original implementation: smooth temporal features from data
          N, T = data.shape
          pop_rate = data.mean(dim=0)  # (T,)

          # Create multiple timescales of smoothing
          sigmas = [10, 25, 50, 100, 200]  # Different smoothing windows
          X = []

          for sigma in sigmas:
              # Create Gaussian kernel
              kernel_size = int(4 * sigma) + 1
              x = torch.arange(kernel_size, dtype=torch.float32, device=data.device) - kernel_size // 2
              kernel = torch.exp(-x**2 / (2 * sigma**2))
              kernel = kernel / kernel.sum()

              # Smooth the population rate
              smoothed = F.conv1d(
                  pop_rate.unsqueeze(0).unsqueeze(0),
                  kernel.unsqueeze(0).unsqueeze(0),
                  padding=kernel_size//2
              ).squeeze()

              X.append(smoothed)

          return torch.stack(X)  # (5, T)
      
      else:
          # Use slow RBF basis functions when data is not provided
          # Assumes self.data_shape was set during initialization or first call
          if not hasattr(self, 'T_for_rbf'):
              raise ValueError("Need to call create_smooth_covariates with data at least once, or set T_for_rbf")
          
          T = self.T_for_rbf
          device = self.device
          
          # Create 5 RBF basis functions with centers spread across time
          n_basis = self.covariate_dim 
          centers = torch.linspace(0, T-1, n_basis, device=device)
          width = T / (n_basis - 1) * 0.2  # Slow/wide RBFs
          
          # Create time indices
          t = torch.arange(T, dtype=torch.float32, device=device)
          
          # Compute RBF basis functions
          X = []
          for c in centers:
              rbf = torch.exp(-(t - c)**2 / (2 * width**2))
              X.append(rbf)
          
          return torch.stack(X)  # (5, T)

    def _update_base_rates(self, data, amplitudes, max_iter=10, tol=1e-4):
      """
      Vectorized Newton-Raphson update for GLM parameters
      Updates all neurons simultaneously for speed
      """
      D = self.template_duration
      T = data.shape[1]
      N = self.num_neurons
      
      # Store T for RBF basis functions
      self.T_for_rbf = T

      # Create smooth temporal covariates - consider using RBF basis functions
      if self.empirical_glm:
        X = self.create_smooth_covariates(data)
      else:
        X = self.create_smooth_covariates()
      if self.use_bias:
        X = torch.cat([torch.ones(1, T, device=X.device), X], dim=0)  # (P+1, T)

      # expected background
      W = self.templates
      kernel = torch.flip(W.permute(1,0,2), [2])
      seq_rates = F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]

      #  total rates
      # NOTE: probably shouldnt be using Beta to get background rates
      background = self.get_background_rates(X)
      total_rates = background + seq_rates

      # resid. ratio
      # X_n,t / lambda_nt - WHY DO WE NEED THIS?
      ratio = data / (total_rates + 1e-7)
      expected_background = ratio * background  # E[z_{n,t,0}], shape: (N, T)

      # newton updates
      for iter in range(max_iter):
          # Current predictions for all neurons
          eta = torch.matmul(self.beta, X)  # (N, T)
          mu = torch.exp(eta)  # (N, T)

          # Gradient for all neurons: ∇L = X * (y - μ)
          # should residuals be - X - E[z_{n,t,0}] or X - e^(B^Tx) (yes or no - determines whetehr residuals is X-EB or X-mu)
          # note that this is EM on coordinated ascent VI not vanilla GLM M step
          residuals = expected_background - mu  # (N, T)
          # should be ^ X - **something**
          gradient = torch.matmul(residuals, X.T)  # (N, P+1)

          # Diagonal Fisher Information approximation
          # For Poisson: Fisher = X * diag(μ) * X^T
          # We approximate with diagonal: Fisher_diag = Σ_t μ_{n,t} * x_t^2
          fisher_diag = torch.matmul(mu, (X.T)**2)  # (N, P+1)

          # Newton step with diagonal approximation
          # β_new = β_old + Fisher^(-1) * gradient
          delta = gradient / (fisher_diag + 1e-4)  # Add regularization for stability
          # Update all betas simultaneously
          # self.beta = self.beta + delta
          self.beta = self.beta + 0.1 * delta  # Add step size of 0.1


          """
          we couldve done SGD
          or first order updates
          self.beta += delta * gradient

          or second order as done above
          """

          # Check convergence (using Frobenius norm for matrix)
          if torch.norm(delta) < tol:
              break


    def _update_templates(self,
                          data,
                          amplitudes):

        D = self.template_duration
        b, W = self.base_rates, self.templates
        kernel = torch.flip(W.permute(1,0,2), [2])
        rates = b[:, None] + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        ratio = data / (rates + 1e-7)

        alpha_post = self.alpha_t0 + W * torch.flip(
            F.conv1d(amplitudes.unsqueeze(1),
                     ratio.unsqueeze(1),
                     padding=D-1)[:,:,:-D+1], [2])
        beta_post = self.beta_t0 + torch.sum(amplitudes, dim=1)[:,None,None]
        # targets = torch.clip((alpha_post - 1) / (beta_post), 1e-4)
        targets = torch.clip((alpha_post) / (beta_post), 1e-4)
        norm_targets = targets / torch.clip(targets.sum(dim=2, keepdim=True), 1e-4) # ensure no division by zero

        # Estimate the Gaussian template parameters by matching moments
        ds = torch.arange(self.template_duration, device=self.device)
        scales = targets.sum(dim=2) # (K,N)
        delays = torch.sum(ds * norm_targets, dim=2) # (K,N)
        widths = torch.sqrt(torch.sum((ds - delays[:, :, None])**2 * norm_targets, dim=2)) + 1e-4
        assert torch.all(torch.isfinite(scales))
        assert torch.all(torch.isfinite(delays))
        assert torch.all(torch.isfinite(widths))

        # Make the model identifiable by constraining the scales to sum to one across neurons
        scales /= scales.sum(axis=1, keepdim=True)
        self.template_scales = scales
        self.template_offsets = delays
        self.template_widths = widths

    def initialize_random(self,
                          data: Float[Tensor, "num_neurons num_timesteps"],
                          sequence_frac: float=0.5,
                          concentration: float=10.) \
                          -> None:
        """Initialize the model parameters randomly, while matching gross
        statistics of the data.

        Parameters
        ----------
        data: neurons x time array of spike counts
        sequence_frac: what fraction of spikes are due to sequences rather than background
        """
        K, N, D = self.num_templates, self.num_neurons, self.template_duration
        T = data.shape[1]
        avg_rate = data.mean(dim=1)
        self.base_rates = avg_rate * (1 - sequence_frac)
        self.template_scales = dist.Dirichlet(torch.clip(concentration * avg_rate, 1e-7)).sample(sample_shape=(K,))
        self.template_offsets = D * torch.rand(K, N, device=self.device)
        self.template_widths = torch.ones(K, N, device=self.device)

        # expected num spikes = .8 * total num spikes
        # unit amplitude produces 1 spike in expectation
        # need amplitudes.sum() = .2 * total num spikes
        amplitudes = dist.Dirichlet(0.1 * torch.ones(K, T, device=self.device)
        ).sample()
        amplitudes *= sequence_frac * data.sum() / K
        return amplitudes

    def initialize_default(self,
                          data: Float[Tensor, "num_neurons num_timesteps"],
                          sequence_frac: float=0.5,
                          concentration: float=10.) \
                          -> None:
        """Initialize the model parameters randomly, while matching gross
        statistics of the data.

        Parameters
        ----------
        data: neurons x time array of spike counts
        sequence_frac: what fraction of spikes are due to sequences rather than background
        """
        K, N, D = self.num_templates, self.num_neurons, self.template_duration
        T = data.shape[1]
        avg_rate = data.mean(dim=1)
        self.base_rates = avg_rate * (1 - sequence_frac)
        self.template_scales = dist.Dirichlet(torch.clip(concentration * avg_rate, 1e-7)).sample(sample_shape=(K,))
        self.template_offsets = D * torch.rand(K, N, device=self.device)
        self.template_widths = torch.ones(K, N, device=self.device)

        # expected num spikes = .8 * total num spikes
        # unit amplitude produces 1 spike in expectation
        # need amplitudes.sum() = .2 * total num spikes
        data = data.to(self.device)
        amplitudes = torch.clamp(data.sum(dim=0) + torch.normal(
            mean=sequence_frac * data.sum() / K,
            std=data.std(), size=(K, T)).to(self.device), min=1e-7)
        amplitudes /= amplitudes.sum()
        amplitudes *= sequence_frac * data.sum() / K
        return amplitudes

    def initialize_none(self, data):
        """
        Don't change the templates and base rates.
        Initialize the amplitudes to zeros.
        """
        K = self.num_templates
        T = data.shape[1]

        # amplitudes = torch.ones((K, T))
        amplitudes = dist.Uniform(0, 1).sample((K, T))
        return amplitudes

    def fit(self,
            data: Float[Tensor, "num_neurons num_timesteps"],
            num_iter: int=50,
            initialization='default',
            fit_templates=True,
            fit_base_rates=True,
            ):
        """
        Fit the model with expectation-maximization (EM).
        """
        K = self.num_templates
        T = data.shape[1]
        self.T_for_rbf = T

        init_method = dict(
            random=self.initialize_random,
            default=self.initialize_default,
            none=self.initialize_none,
            )[initialization.lower()]
        amplitudes = init_method(data)

        # data on right device and dtype
        amplitudes = amplitudes.to(self.device)
        data = data.to(self.device)

        # Create covariates
        if self.empirical_glm:
          phi = self.create_smooth_covariates(data)
        else:
          phi = self.create_smooth_covariates()
        if self.use_bias:
          phi = torch.cat([torch.ones(1, T, device=phi.device), phi], dim=0)  # (P+1, T)


        # Run EM
        lps = []
        for _ in progress_bar(range(num_iter)):
            amplitudes = self._update_amplitudes(data, amplitudes, phi)  # FIXED: Pass X
            if fit_base_rates: self._update_base_rates(data, amplitudes)
            if fit_templates: self._update_templates(data, amplitudes)
            lps.append(self.log_likelihood(data, amplitudes, phi))  # FIXED: Pass X

        lps = torch.stack(lps) if num_iter > 0 else torch.tensor([])
        return lps, amplitudes


class PPSeq:
    """PPSeq is a probabilistic model for detecting sequences of spikes
    embedded in multi-neuronal spike trains. It is based on a Poisson
    latent variable model, akin to a non-negative, convolutional matrix
    factorization. 
    """
    base_rates : Float[Tensor, "num_neurons"]
    template_scales: Float[Tensor, "num_templates num_neurons"]
    template_offsets: Float[Tensor, "num_templates num_neurons"]
    template_widths: Float[Tensor, "num_templates num_neurons"]

    def __init__(self,
                 num_templates: int,
                 num_neurons: int,
                 template_duration: int,
                 alpha_a0: float=0., 
                 beta_a0: float=0., 
                 alpha_b0: float=0., 
                 beta_b0: float=0.,
                 alpha_t0: float=0.,
                 beta_t0: float=0.,
                 device=None
                 ):
        self.num_templates = num_templates
        self.num_neurons = num_neurons
        self.template_duration = template_duration
        
        # Set the device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not torch.cuda.is_available():
                print('Could not find a GPU. Defaulting to CPU instead.')
        self.device = device
    
        # TODO[GLM]: The current implementation assumes a *constant* background rate per neuron:
        #            `self.base_rates: Float[Tensor, "num_neurons"]` (shape = N, broadcast across time).
        #            We want a *dynamic* background, modeled by a Poisson GLM with a log link:
        #              λ_bg[n,t] = exp( α[n] + X_t @ β[n] )   where X_t are covariates at time t.
        # Plan:
        #   • Add optional constructor args:
        #       - `bg_model: Optional[nn.Module] = None` or `covariate_dim: Optional[int] = None`
        #   • If `covariate_dim` is provided, instantiate a GLM and store:
        #       - `self.bg_model: Optional[PoissonGLM]`
        #       - `self.use_dynamic_background: bool = self.bg_model is not None`
        #   • Keep `self.base_rates` only as a fallback (constant background) AND to seed GLM intercepts.
        #   • Add a helper:
        #       def background_rate(self, covariates: Optional[Tensor], T: int) -> Tensor:
        #           if self.bg_model is None: return self.base_rates[:, None].expand(self.num_neurons, T)
        #           else: return self.bg_model(covariates)  # returns (N, T)
        # Training (during the M-step for background parameters):
        #   Given current total rate λ[n,t] and data x[n,t], compute expected background counts
        #       y_bg = x * λ_bg / clamp(λ, 1e-7)
        #   and maximize:  L = sum_{n,t} [ y_bg * η - exp(η) ]  - λ_reg * ||β||_2^2,  where η = α + Xβ
        #   with a few gradient steps (Adam/LBFGS). See TODOs in `_update_base_rates` and `fit` below.

        self.base_rates = torch.ones(num_neurons, device=device)
        self.template_scales = torch.ones(num_templates, num_neurons, device=device) / num_neurons
        self.template_offsets = template_duration * torch.rand(num_templates, num_neurons, device=device)
        self.template_widths = torch.ones(self.num_templates, self.num_neurons, device=device)

        # Set prior hyperparameters
        self.alpha_a0 = alpha_a0
        self.beta_a0 = beta_a0
        self.alpha_b0 = alpha_b0
        self.beta_b0 = beta_b0
        self.alpha_t0 = alpha_t0
        self.beta_t0 = beta_t0

    @property
    def templates(self) -> Float[Tensor, "num_templates num_neurons duration"]:
        """Compute the templates from the mean, std, and amplitude of the Gaussian kernel.
        """
        D = self.template_duration
        amp, mu, sigma = self.template_scales, self.template_offsets, self.template_widths
        ds = torch.arange(D, device=self.device)[:, None, None]
        p = dist.Normal(mu, sigma)
        W = p.log_prob(ds).exp().permute(1,2,0) 
        return W / W.sum(dim=2, keepdim=True) * amp[:, :, None]
        
    def reconstruct(self,
                    amplitudes: Float[Tensor, "num_templates num_timesteps"]) \
                    -> Float[Tensor, "num_neurons num_timesteps"]: 
        """
        Reconstruct the firing rate given the model parameters and latent variables.

        Parameters
        ----------
        amplitudes: the amplitudes for each template as a function of time 
        """
        # TODO[GLM]: Make background dynamic.
        # CURRENT: uses constant background `self.base_rates[:, None]` (N×1) broadcast across time.
        # CHANGE: extend signature to accept `covariates: Optional[Tensor]` and compute:
        #     bg  = self.background_rate(covariates, T)  # (N, T)
        #     seq = F.conv1d(amplitudes, kernel, padding=D-1)[:, :-D+1]
        #     return torch.clamp(bg + seq, min=1e-7)
        # Notes:
        #   • `covariates` can be (T, P) shared across neurons or (N, T, P) per-neuron.
        #   • Keep current path when `self.bg_model is None`.
        return self.base_rates[:, None] + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]

        D = self.template_duration
        kernel = torch.flip(self.templates.permute(1,0,2), [2])
        return self.base_rates[:, None] + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        
    def log_likelihood(self,
                       data: Float[Tensor, "num_neurons num_timesteps"], 
                       amplitudes: Float[Tensor, "num_templates num_timesteps"],
                       rows = None,
                       cols = None) -> float:
        """
        Calculate the log probability given data X
        and estimated parameters a, b, W

        Parameters
        ----------
        data: spike count matrix
        amplitudes: amplitudes of each template over time
        rows,cols: torch.tensor([list of row/col indices]) that represent a subset of rows * cols where the log likelihood is calculated
        
        Returns
        -------
        Scalar log probability
        """
        # TODO[GLM]: When background is dynamic, compute rates via `self.reconstruct(amplitudes, covariates)`
        # and thread `covariates` through `log_likelihood`:
        #   def log_likelihood(self, data, amplitudes, rows=None, cols=None, covariates=None)

        D = self.template_duration
        kernel = torch.flip(self.templates.permute(1,0,2),[2])
        rates = self.base_rates[:, None] + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        rates = torch.clamp(rates, min=1e-7)
        if rows is None or cols is None:
            return torch.sum(dist.Poisson(rates).log_prob(data))

        data_selected = data[rows,:][:,cols]
        rates_selected = rates[rows,:][:,cols]
        poisson_dist = dist.Poisson(rates_selected)
        log_probs = poisson_dist.log_prob(data_selected)
        log_likelihood = torch.sum(log_probs)
        return log_likelihood
        
    
    def _update_amplitudes(self, data, amplitudes):
        # TODO[GLM]: Replace `b[:, None]` with dynamic background:
        #   bg   = self.background_rate(covariates, T)     # (N, T)
        #   rates = torch.clamp(bg + F.conv1d(...), 1e-7)
        # Thread `covariates` into this method's signature to avoid global state.

        D, T = self.template_duration, data.shape[1]
        b, W = self.base_rates, self.templates
        kernel = torch.flip(W.permute(1,0,2), [2])
        rates = b[:, None] + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        ratio = data / (rates + 1e-7) 

        alpha_post = amplitudes * F.conv1d(ratio, W, padding=D-1)[:,D-1:] + self.alpha_a0
        beta_post = torch.sum(W, dim=(1,2)).unsqueeze(1).repeat(1,T) + self.beta_a0
        return torch.clip((alpha_post - 1) / beta_post, 0)
    
    def _update_base_rates(self, data, amplitudes):
        # TODO[GLM]: Replace this constant-background update with a Poisson GLM M-step.
        #   1) Compute current bg (constant or GLM) and total rate:
        #        bg   = self.background_rate(covariates, T)            # (N, T)
        #        seq  = F.conv1d(amplitudes, kernel, padding=D-1)[:, :-D+1]
        #        lam  = torch.clamp(bg + seq, 1e-7)
        #   2) Compute expected background counts:
        #        y_bg = data * (bg / lam)
        #   3) If using GLM:
        #        • take a few optimizer steps to maximize Σ y_bg * η - exp(η) w.r.t. (α, β),
        #          where η = α[n] + X_t @ β[n] and X are covariates provided by the caller.
        #        • support ℓ2 regularization on β and optional time-basis smoothing.
        #      Else (no GLM), keep existing constant-rate posterior update for `self.base_rates`.
        #   4) Accept `covariates` as an argument to this method and plumb it through from `.fit(...)`.
        D, T = self.template_duration, data.shape[1]
        b, W = self.base_rates, self.templates
        kernel = torch.flip(W.permute(1,0,2),[2])
        rates = b[:, None] + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        ratio = data / (rates + 1e-7)

        alpha_post = torch.sum(ratio, dim=1) * b + self.alpha_b0
        beta_post = T + self.beta_b0
        # self.base_rates = torch.clip((alpha_post - 1) / beta_post, 0)
        self.base_rates = torch.clip(alpha_post / beta_post, 1e-4)
        
    def _update_templates(self, 
                          data, 
                          amplitudes):
        # TODO[GLM]: Compute rates using dynamic background if enabled:
        #   bg  = self.background_rate(covariates, data.shape[1])
        #   rates = torch.clamp(bg + F.conv1d(...), 1e-7)
        D = self.template_duration
        b, W = self.base_rates, self.templates
        kernel = torch.flip(W.permute(1,0,2), [2])
        rates = b[:, None] + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        ratio = data / (rates + 1e-7) 

        alpha_post = self.alpha_t0 + W * torch.flip(
            F.conv1d(amplitudes.unsqueeze(1), 
                     ratio.unsqueeze(1), 
                     padding=D-1)[:,:,:-D+1], [2])
        beta_post = self.beta_t0 + torch.sum(amplitudes, dim=1)[:,None,None]
        # targets = torch.clip((alpha_post - 1) / (beta_post), 1e-4)
        targets = torch.clip((alpha_post) / (beta_post), 1e-4)
        norm_targets = targets / torch.clip(targets.sum(dim=2, keepdim=True), 1e-4) # ensure no division by zero

        # Estimate the Gaussian template parameters by matching moments
        ds = torch.arange(self.template_duration, device=self.device)
        scales = targets.sum(dim=2) # (K,N)
        delays = torch.sum(ds * norm_targets, dim=2) # (K,N)
        widths = torch.sqrt(torch.sum((ds - delays[:, :, None])**2 * norm_targets, dim=2)) + 1e-4
        assert torch.all(torch.isfinite(scales))
        assert torch.all(torch.isfinite(delays))
        assert torch.all(torch.isfinite(widths))

        # Make the model identifiable by constraining the scales to sum to one across neurons
        scales /= scales.sum(axis=1, keepdim=True)
        self.template_scales = scales
        self.template_offsets = delays
        self.template_widths = widths

    def initialize_random(self, 
                          data: Float[Tensor, "num_neurons num_timesteps"],
                          sequence_frac: float=0.5,
                          concentration: float=10.) \
                          -> None:
        """Initialize the model parameters randomly, while matching gross 
        statistics of the data.

        Parameters
        ----------
        data: neurons x time array of spike counts
        sequence_frac: what fraction of spikes are due to sequences rather than background
        """
        # TODO[GLM]: If using a background GLM, initialize its intercept with
        #            log(avg_rate * (1 - sequence_frac)) and its weights near zero.
        #            Keep `self.base_rates` only for the constant-background fallback.
        K, N, D = self.num_templates, self.num_neurons, self.template_duration
        T = data.shape[1]
        avg_rate = data.mean(dim=1)
        self.base_rates = avg_rate * (1 - sequence_frac)
        self.template_scales = dist.Dirichlet(torch.clip(concentration * avg_rate, 1e-7)).sample(sample_shape=(K,))
        self.template_offsets = D * torch.rand(K, N, device=self.device)
        self.template_widths = torch.ones(K, N, device=self.device)

        # expected num spikes = .8 * total num spikes
        # unit amplitude produces 1 spike in expectation
        # need amplitudes.sum() = .2 * total num spikes
        amplitudes = dist.Dirichlet(0.1 * torch.ones(K, T, device=self.device)
        ).sample()
        amplitudes *= sequence_frac * data.sum() / K
        return amplitudes

    def initialize_default(self, 
                          data: Float[Tensor, "num_neurons num_timesteps"],
                          sequence_frac: float=0.5,
                          concentration: float=10.) \
                          -> None:
        """Initialize the model parameters randomly, while matching gross 
        statistics of the data.

        Parameters
        ----------
        data: neurons x time array of spike counts
        sequence_frac: what fraction of spikes are due to sequences rather than background
        """
        # TODO[GLM]: As above, seed GLM intercepts and zero the weights instead of
        #            storing a constant background when a GLM is configured.
        K, N, D = self.num_templates, self.num_neurons, self.template_duration
        T = data.shape[1]
        avg_rate = data.mean(dim=1)
        self.base_rates = avg_rate * (1 - sequence_frac)
        self.template_scales = dist.Dirichlet(torch.clip(concentration * avg_rate, 1e-7)).sample(sample_shape=(K,))
        self.template_offsets = D * torch.rand(K, N, device=self.device)
        self.template_widths = torch.ones(K, N, device=self.device)

        # expected num spikes = .8 * total num spikes
        # unit amplitude produces 1 spike in expectation
        # need amplitudes.sum() = .2 * total num spikes
        data = data.to(self.device)
        amplitudes = torch.clamp(data.sum(dim=0) + torch.normal(
            mean=sequence_frac * data.sum() / K,
            std=data.std(), size=(K, T)).to(self.device), min=1e-7)
        amplitudes /= amplitudes.sum()
        amplitudes *= sequence_frac * data.sum() / K
        return amplitudes
    
    def initialize_none(self, data):
        """
        Don't change the templates and base rates.
        Initialize the amplitudes to zeros.
        """
        K = self.num_templates
        T = data.shape[1]

        # amplitudes = torch.ones((K, T))
        amplitudes = dist.Uniform(0, 1).sample((K, T))
        return amplitudes
    
    def fit(self,
            data: Float[Tensor, "num_neurons num_timesteps"],
            num_iter: int=50,
            initialization='default',
            fit_templates=True,
            fit_base_rates=True,
            # TODO[GLM]: new args to support dynamic background training:
            # covariates=None,           # (T, P) or (N, T, P) design matrix for the GLM
            # num_glm_steps=5,           # gradient steps for the GLM per EM iteration
            # glm_lr=1e-2,               # learning rate for the GLM optimizer
            # glm_l2=0.0,                # L2 regularization on β
            ):
        """
        Fit the model with expectation-maximization (EM).
        """
        K = self.num_templates
        T = data.shape[1]

        init_method = dict(
            random=self.initialize_random,
            default=self.initialize_default,
            none=self.initialize_none,
            )[initialization.lower()]
        amplitudes = init_method(data)
        # TODO[GLM]: If using a GLM, initialize an optimizer here (e.g., Adam) and store it on `self` or in a local closure; reuse it across EM iterations.


     
        # Run EM
        lps = []
        for _ in progress_bar(range(num_iter)):
            amplitudes = self._update_amplitudes(data, amplitudes)
            # TODO[GLM]: pass `covariates`
            # if fit_base_rates:
                # TODO[GLM]: call `_update_base_rates(data, amplitudes, covariates)` which, in GLM mode,
                #            performs a few optimizer steps on the GLM with expected background counts y_bg.
                # self._update_base_rates(data, amplitudes)
            if fit_base_rates: self._update_base_rates(data, amplitudes)
            if fit_templates: self._update_templates(data, amplitudes)
            lps.append(self.log_likelihood(data, amplitudes))

        lps = torch.stack(lps) if num_iter > 0 else torch.tensor([])
        return lps, amplitudes






import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from fastprogress import progress_bar
from torch import Tensor
from jaxtyping import Float


class CAVI:
    """PPSeq is a probabilistic model for detecting sequences of spikes
    embedded in multi-neuronal spike trains. It is based on a Poisson
    latent variable model, akin to a non-negative, convolutional matrix
    factorization.
    """
    base_rates : Float[Tensor, "num_neurons"]
    template_scales: Float[Tensor, "num_templates num_neurons"]
    template_offsets: Float[Tensor, "num_templates num_neurons"]
    template_widths: Float[Tensor, "num_templates num_neurons"]

    def __init__(self,
                 num_templates: int,
                 num_neurons: int,
                 template_duration: int,
                 alpha_a0: float=0.5,
                 beta_a0: float=0.,
                 alpha_b0: float=0.,
                 beta_b0: float=0.,
                 alpha_t0: float=0.,
                 beta_t0: float=0.,
                 device=None
                 ):
        self.num_templates = num_templates
        self.num_neurons = num_neurons
        self.template_duration = template_duration
        # TODO[GLM]: Mirror the PPSeq changes:
        #   • allow optional `bg_model` / `covariate_dim`
        #   • keep `self.base_rates` as fallback/initialization only
        #   • provide `self.background_rate(...)`

        # Set the device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not torch.cuda.is_available():
                print('Could not find a GPU. Defaulting to CPU instead.')
        self.device = device

        # TODO: Initialize parameters with values if not None
        self.base_rates = torch.ones(num_neurons, device=device)
        self.template_scales = torch.ones(num_templates, num_neurons, device=device) / num_neurons
        self.template_offsets = template_duration * torch.rand(num_templates, num_neurons, device=device)
        self.template_widths = torch.ones(self.num_templates, self.num_neurons, device=device)

        # Set prior hyperparameters
        self.alpha_a0 = alpha_a0
        self.beta_a0 = beta_a0
        self.alpha_b0 = alpha_b0
        self.beta_b0 = beta_b0
        self.alpha_t0 = alpha_t0
        self.beta_t0 = beta_t0

    @property
    def templates(self) -> Float[Tensor, "num_templates num_neurons duration"]:
        """Compute the templates from the mean, std, and amplitude of the Gaussian kernel.
        """
        D = self.template_duration
        amp, mu, sigma = self.template_scales, self.template_offsets, self.template_widths
        ds = torch.arange(D, device=self.device)[:, None, None]
        p = dist.Normal(mu, sigma)
        W = p.log_prob(ds).exp().permute(1,2,0)
        return W / W.sum(dim=2, keepdim=True) * amp[:, :, None]

    def reconstruct(self,
                    amplitudes: Float[Tensor, "num_templates num_timesteps"]) \
                    -> Float[Tensor, "num_neurons num_timesteps"]:
        """
        Reconstruct the firing rate given the model parameters and latent variables.

        Parameters
        ----------
        amplitudes: the amplitudes for each template as a function of time
        """
        D = self.template_duration
        kernel = torch.flip(self.templates.permute(1,0,2), [2])
        rates = self.base_rates[:, None] + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        # TODO[GLM]: As in PPSeq, compute background via GLM if configured, and add to sequence term.
        # Extend signature to accept `covariates` and clamp the sum.
        return torch.clamp(rates, min=1e-7)

    def log_likelihood(self,
                       data: Float[Tensor, "num_neurons num_timesteps"],
                       amplitudes: Float[Tensor, "num_templates num_timesteps"],
                       rows = None,
                       cols = None) -> float:
        """
        Calculate the log probability given data X
        and estimated parameters a, b, W

        Parameters
        ----------
        data: spike count matrix
        amplitudes: amplitudes of each template over time
        rows,cols: torch.tensor([list of row/col indices]) that represent a subset of rows * cols where the log likelihood is calculated

        Returns
        -------
        Scalar log probability
        """
        rates = self.reconstruct(amplitudes)
        # TODO[GLM]: Thread `covariates` through and call `self.reconstruct(amplitudes, covariates)`.
        if rows is None or cols is None:
            return torch.sum(dist.Poisson(rates).log_prob(data))

        data_selected = data[rows,:][:,cols]
        rates_selected = rates[rows,:][:,cols]
        poisson_dist = dist.Poisson(rates_selected)
        log_probs = poisson_dist.log_prob(data_selected)
        log_likelihood = torch.sum(log_probs)
        return log_likelihood


    def _update_amplitudes(self, data, amplitudes):
        D, T = self.template_duration, data.shape[1]
        W = self.templates
       
        rates = self.reconstruct(amplitudes)
        # TODO[GLM]: call `self.reconstruct(amplitudes, covariates)` (accept new arg).
        ratio = data / rates 

        alpha_post = amplitudes * F.conv1d(ratio, W, padding=D-1)[:,D-1:] + self.alpha_a0
        beta_post = torch.sum(W, dim=(1,2)).unsqueeze(1).repeat(1,T) + self.beta_a0
        return (alpha_post ) / (beta_post+1e-7)

    def _update_base_rates(self, data, amplitudes):
        T = data.shape[1]
        b = self.base_rates
        rates = self.reconstruct(amplitudes)
        # TODO[GLM]: Replace constant-background update with GLM M-step as in PPSeq:
        #   y_bg = data * bg / reconstruct(...), then optimize GLM params (α,β).
        # Accept `covariates` arg and use a few gradient steps per EM iteration
        ratio = data / rates 

        alpha_post = torch.sum(ratio, dim=1) * b + self.alpha_b0
        beta_post = T + self.beta_b0
        self.base_rates = (alpha_post ) / (beta_post+1e-7)

    def _update_templates(self,
                          data,
                          amplitudes):
        D = self.template_duration
        b, W = self.base_rates, self.templates
        rates = self.reconstruct(amplitudes)
        # TODO[GLM]: ensure this uses dynamic background; thread `covariates` through.
        ratio = data / rates 

        # TODO: Double check this line
        alpha_post = W * torch.flip(F.conv1d(amplitudes.unsqueeze(1),
                                             ratio.unsqueeze(1),
                                             padding=D-1)[:,:,:-D+1], [2]) + self.alpha_t0
        beta_post = torch.sum(amplitudes, dim=1)[:,None,None] + self.beta_t0

       
        targets = (alpha_post ) / (beta_post+1e-7)
        norm_targets = targets / torch.clip(targets.sum(dim=2, keepdim=True), 1e-4)

        # Estimate the Gaussian template parameters by matching moments
        ds = torch.arange(self.template_duration, device=self.device)
        scales = targets.sum(dim=2) # (K,N)
        delays = torch.sum(ds * norm_targets, dim=2) # (K,N)
        widths = torch.sqrt(torch.sum((ds - delays[:, :, None])**2 * norm_targets, dim=2)) + 1e-4
        assert torch.all(torch.isfinite(scales))
        assert torch.all(torch.isfinite(delays))
        assert torch.all(torch.isfinite(widths))

        # Make the model identifiable by constraining the scales to sum to one across neurons
        scales /= scales.sum(axis=1, keepdim=True)
        self.template_scales = scales
        self.template_offsets = delays
        self.template_widths = widths

    def initialize_random(self,
                          data: Float[Tensor, "num_neurons num_timesteps"],
                          sequence_frac: float=0.5,
                          concentration: float=10.) \
                          -> None:
        """Initialize the model parameters randomly, while matching gross
        statistics of the data.

        Parameters
        ----------
        data: neurons x time array of spike counts
        sequence_frac: what fraction of spikes are due to sequences rather than background
        """
        K, N, D = self.num_templates, self.num_neurons, self.template_duration
        T = data.shape[1]
        avg_rate = data.mean(dim=1)
        self.base_rates = avg_rate * (1 - sequence_frac)
        # TODO[GLM]: Initialize GLM intercepts with log(avg_rate*(1-sequence_frac)) and β near zero.
        self.template_scales = dist.Dirichlet(torch.clip(concentration * avg_rate, 1e-7)).sample(sample_shape=(K,))
        self.template_offsets = D * torch.rand(K, N, device=self.device)
        self.template_widths = torch.ones(K, N, device=self.device)

        # expected num spikes = .8 * total num spikes
        # unit amplitude produces 1 spike in expectation
        # need amplitudes.sum() = .2 * total num spikes
        amplitudes = dist.Dirichlet(0.1 * torch.ones(K, T, device=self.device)
        ).sample()
        amplitudes *= sequence_frac * data.sum() / K
        return amplitudes

    def initialize_default(self,
                          data: Float[Tensor, "num_neurons num_timesteps"],
                          sequence_frac: float=0.5,
                          concentration: float=10.) \
                          -> None:
        """Initialize the model parameters randomly, while matching gross
        statistics of the data.

        Parameters
        ----------
        data: neurons x time array of spike counts
        sequence_frac: what fraction of spikes are due to sequences rather than background
        """
        K, N, D = self.num_templates, self.num_neurons, self.template_duration
        T = data.shape[1]
        avg_rate = data.mean(dim=1)
        self.base_rates = avg_rate * (1 - sequence_frac)
        # TODO[GLM]: As above, seed GLM intercepts instead of a constant background when GLM is active.
        self.template_scales = dist.Dirichlet(concentration *
        avg_rate).sample(sample_shape=(K,))
        self.template_offsets = D * torch.rand(K, N, device=self.device)
        self.template_widths = torch.ones(K, N, device=self.device)

        # expected num spikes = .8 * total num spikes
        # unit amplitude produces 1 spike in expectation
        # need amplitudes.sum() = .2 * total num spikes
        data = data.to(self.device)
        amplitudes = torch.clamp(data.sum(dim=0) + torch.normal(
            mean=sequence_frac * data.sum() / K,
            std=data.std(), size=(K, T)).to(self.device), min=1e-7)
        amplitudes /= amplitudes.sum()
        amplitudes *= sequence_frac * data.sum() / K
        return amplitudes

    def fit(self,
            data: Float[Tensor, "num_neurons num_timesteps"],
            num_iter: int=50,
            initialization='default',
            # TODO[GLM]: add `covariates=None`, `num_glm_steps=5`, `glm_lr=1e-2`, `glm_l2=0.0`
            ):
        """
        Fit the model with expectation-maximization (EM).
        """
        K = self.num_templates
        T = data.shape[1]

        init_method = dict(
            random=self.initialize_random,
            default=self.initialize_default,
            )[initialization.lower()]
        amplitudes = init_method(data)


        # Run EM
        lps = []
        for _ in progress_bar(range(num_iter)):
            amplitudes = self._update_amplitudes(data, amplitudes) # TODO[GLM]: pass `covariates`
            self._update_base_rates(data, amplitudes) # TODO[GLM]: pass `covariates`
            self._update_templates(data, amplitudes) # TODO[GLM]: pass `covariates`
            lps.append(self.log_likelihood(data, amplitudes))

        lps = torch.stack(lps) if num_iter > 0 else torch.tensor([])
        return lps, amplitudes
