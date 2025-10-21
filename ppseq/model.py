#PPSeq sets to mode, CAVI sets to mean

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from fastprogress import progress_bar
from torch import Tensor
from jaxtyping import Float


class GLMPPSeq:
    """
    GLMPPSeq is a probabilistic model for detecting sequential patterns in multi-neuron spike train data.
    
    This class generalizes the Poisson latent variable model underlying PPSeq by incorporating a Generalized Linear Model (GLM)
    for the background firing rate, allowing for time-varying covariate effects in addition to the usual sequence-driven components.

    Key Features:
    -------------
    - Models observed spikes as arising from a time-varying background activity (via a GLM) and discrete sequence events.
    - Sequence events are defined by time-varying amplitudes and Gaussian-shaped templates for each sequence and neuron.
    - Gaussian templates are parameterized by per-template, per-neuron scale (amplitude), delay (offset), and width (spread).
    - EM algorithm (CAVI) for fitting, updating sequence amplitudes, background GLM weights, and template parameters.
    - Background covariates can be constructed either from empirical population activity or RBF basis functions.
    - Flexible regularization for l1/l2 sparsity and temporal smoothness.

    Parameters:
    -----------
    num_templates (int)    : Number of latent sequence templates.
    num_neurons (int)      : Number of neurons.
    template_duration (int): Duration (bins) for each template.
    alpha_a0, beta_a0      : Prior hyperparameters for amplitude updates.
    alpha_b0, beta_b0      : Prior hyperparameters for background/GLM.
    alpha_t0, beta_t0      : Prior hyperparameters for template weights.
    device                 : Torch device (defaults to CUDA if available).
    use_bias (bool)        : If True, include intercept in GLM.
    empirical_glm (bool)   : If True, use data-derived covariates; else use fixed RBFs.
    n_covariates (int)     : Number of covariates for GLM.
    l1, l1_amp (float)     : L1 regularization weights for sparsity.
    rbf_width (float)      : Width for RBF covariates (if used).

    Methods:
    --------
    - templates: returns current Gaussian templates tensor.
    - get_background_rates(X): computes GLM or baseline background rates.
    - reconstruct(amplitudes, X): computes overall firing rates.
    - log_likelihood(...): computes (subset) Poisson log-likelihood.
    - fit(...): fits the model to data using variational EM.
    - initialization/random/default/none: strategies for initial amplitudes and parameters.

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
                 l1=0.0,
                 l1_amp=0.0,
                 rbf_width=2.5
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
        self.l1 = l1
        self.l1_amp = l1_amp
        self.rbf_width = rbf_width

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
        background = self.get_background_rates(X)
        rates = background + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        ratio = data / (rates + 1e-7)

        alpha_post = amplitudes * F.conv1d(ratio, W, padding=D-1)[:,D-1:] + self.alpha_a0
        beta_post = torch.sum(W, dim=(1,2)).unsqueeze(1).repeat(1,T) + self.beta_a0

        # soft L1 penalty
        if self.l1_amp > 0:
            alpha_post -= self.l1_amp
        
        return torch.clip((alpha_post - 1) / beta_post, 0)

    def create_smooth_covariates(self):
        """Create smooth temporal features using Gaussian filtering or RBF basis functions"""
        # Use slow RBF basis functions when data is not provided
        if not hasattr(self, 'T_for_rbf'):
            raise ValueError("Need to call create_smooth_covariates with data at least once, or set T_for_rbf")
        
        T = self.T_for_rbf
        device = self.device
        
        # Create RBF basis functions with centers spread across time
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
        
        return torch.stack(X)  # (covariate_dim, T)

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
        X = self.create_smooth_covariates()
        if self.use_bias:
            X = torch.cat([torch.ones(1, T, device=X.device), X], dim=0)  # (P+1, T)

        # expected background
        W = self.templates
        kernel = torch.flip(W.permute(1,0,2), [2])
        seq_rates = F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]

        #  total rates
        background = self.get_background_rates(X)
        total_rates = background + seq_rates

        ratio = data / (total_rates + 1e-7)
        expected_background = ratio * background  # E[z_{n,t,0}], shape: (N, T)

        for iter in range(max_iter):
            eta = torch.matmul(self.beta, X)  # (N, T)
            eta = torch.clamp(eta, min=-10, max=10)  # Clamp eta before exp
            mu = torch.exp(eta)  # (N, T)

            residuals = expected_background - mu  # (N, T)
            gradient = torch.matmul(residuals, X.T)  # (N, P+1)

            fisher_diag = torch.matmul(mu, (X.T)**2)  # (N, P+1)

            # Add stronger regularization
            delta = gradient / (fisher_diag + 1e-2)  # Increased from 1e-4

            # Clip gradients to prevent explosion
            delta = torch.clamp(delta, min=-1.0, max=1.0)

            # Check for NaN/Inf before updating
            if torch.isnan(delta).any() or torch.isinf(delta).any():
                print("Warning: NaN or Inf detected in GLM update, stopping iteration")
                break

            self.beta = self.beta + 0.1 * delta

            # Check beta for NaN/Inf
            if torch.isnan(self.beta).any() or torch.isinf(self.beta).any():
                print("Warning: NaN or Inf in beta parameters")
                self.beta = torch.nan_to_num(self.beta, nan=0.0, posinf=10.0, neginf=-10.0)

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

        # ---- L1 prox on scales (encourages neuron-level sparsity per template)
        if self.l1 > 0:
            # soft-threshold (since scales ≥ 0 we can skip sign/abs handling)
            scales = torch.relu(scales - self.l1)

        # Keep identifiability: scales sum to 1 across neurons for each template
        eps = 1e-12
        row_sums = scales.sum(dim=1, keepdim=True)
        # If a whole row goes to zero, put uniform mass back to avoid NaNs
        zero_rows = (row_sums <= eps).squeeze(1)
        if zero_rows.any():
            scales[zero_rows] = 1.0 / scales.shape[1]
            row_sums = scales.sum(dim=1, keepdim=True)

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
        phi = self.create_smooth_covariates()
        if self.use_bias:
          phi = torch.cat([torch.ones(1, T, device=phi.device), phi], dim=0)  # (P+1, T)


        # Run EM
        lps = []
        for _ in progress_bar(range(num_iter)):
            amplitudes = self._update_amplitudes(data, amplitudes, phi)
            if fit_base_rates: self._update_base_rates(data, amplitudes)
            if fit_templates: self._update_templates(data, amplitudes)
            lps.append(self.log_likelihood(data, amplitudes, phi))

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

        D, T = self.template_duration, data.shape[1]
        b, W = self.base_rates, self.templates
        kernel = torch.flip(W.permute(1,0,2), [2])
        rates = b[:, None] + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        ratio = data / (rates + 1e-7) 

        alpha_post = amplitudes * F.conv1d(ratio, W, padding=D-1)[:,D-1:] + self.alpha_a0
        beta_post = torch.sum(W, dim=(1,2)).unsqueeze(1).repeat(1,T) + self.beta_a0
        return torch.clip((alpha_post - 1) / beta_post, 0)
    
    def _update_base_rates(self, data, amplitudes):
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

        init_method = dict(
            random=self.initialize_random,
            default=self.initialize_default,
            none=self.initialize_none,
            )[initialization.lower()]
        amplitudes = init_method(data)


     
        # Run EM
        lps = []
        for _ in progress_bar(range(num_iter)):
            amplitudes = self._update_amplitudes(data, amplitudes)
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
