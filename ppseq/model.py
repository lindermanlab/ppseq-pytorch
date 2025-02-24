#PPSeq sets to mode, CAVI sets to mean


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from fastprogress import progress_bar
from torch import Tensor
from jaxtyping import Float


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

    def to(self, map_location: str | torch.DeviceObjType | torch.dtype):

        self.base_rates = self.base_rates.to(map_location)
        self.template_scales = self.template_scales.to(map_location)
        self.template_offsets = self.template_offsets.to(map_location)
        self.template_widths = self.template_widths.to(map_location)

        if not isinstance(map_location, torch.dtype):
            self.device=map_location


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
        self.base_rates = torch.clip((alpha_post - 1) / beta_post, 0)
        
    def _update_templates(self, 
                          data, 
                          amplitudes):
        D = self.template_duration
        b, W = self.base_rates, self.templates
        kernel = torch.flip(W.permute(1,0,2), [2])
        rates = b[:, None] + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        ratio = data / (rates + 1e-7) 

        alpha_post = W * torch.flip(F.conv1d(amplitudes.unsqueeze(1), 
                                             ratio.unsqueeze(1),
                                             padding=D-1)[:,:,:-D+1], [2]) + self.alpha_t0
        beta_post = torch.sum(amplitudes, dim=1)[:,None,None] + self.beta_t0
        targets = torch.clip((alpha_post - 1) / (beta_post), 0)
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
    
    def fit(self,
            data: Float[Tensor, "num_neurons num_timesteps"],
            num_iter: int=50,
            initialization='default',
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
            amplitudes = self._update_amplitudes(data, amplitudes)
            self._update_base_rates(data, amplitudes)
            self._update_templates(data, amplitudes)
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
        ratio = data / rates 

        alpha_post = amplitudes * F.conv1d(ratio, W, padding=D-1)[:,D-1:] + self.alpha_a0
        beta_post = torch.sum(W, dim=(1,2)).unsqueeze(1).repeat(1,T) + self.beta_a0
        return (alpha_post ) / (beta_post+1e-7)

    def _update_base_rates(self, data, amplitudes):
        T = data.shape[1]
        b = self.base_rates
        rates = self.reconstruct(amplitudes)
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
            amplitudes = self._update_amplitudes(data, amplitudes)
            self._update_base_rates(data, amplitudes)
            self._update_templates(data, amplitudes)
            lps.append(self.log_likelihood(data, amplitudes))

        lps = torch.stack(lps) if num_iter > 0 else torch.tensor([])
        return lps, amplitudes
