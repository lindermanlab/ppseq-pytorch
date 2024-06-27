import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim

from jaxtyping import Array, Float


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if not torch.cuda.is_available():
#     print('cpu')


class PPSeq:
    """PPSeq is a probabilistic model for detecting sequences of spikes
    embedded in multi-neuronal spike trains. It is based on a Poisson
    latent variable model, akin to a non-negative, convolutional matrix
    factorization. 
    """
    # templates : Float[Array, "num_templates num_neurons seq_duration"]
    base_rates : Float[Array, "num_neurons"]
    _template_scales_unc: Float[Array, "num_templates num_neurons"]
    template_delays: Float[Array, "num_templates num_neurons"]
    _template_widths_unc: Float[Array, "num_templates num_neurons"]

    def __init__(self,
                 num_templates: int,
                 num_neurons: int,
                 template_duration: int,
                 initial_base_rates: Float[Array, "num_neurons"]=None,
                 initial_template_scales: Float[Array, "num_templates num_neurons"] = None,
                 initial_template_delays: Float[Array, "num_templates num_neurons"] = None,
                 initial_template_widths: Float[Array, "num_templates num_neurons"] = None,
                 alpha_a0: float=0.5, 
                 beta_a0: float=0., 
                 alpha_b0: float=0., 
                 beta_b0: float=0.
                 ):
        self.num_templates = num_templates
        self.num_neurons = num_neurons
        self.template_duration = template_duration

        # TODO: Initialize templates and base rates
        # scale = torch.rand(K, N, requires_grad=True, device=device)
        # mu = torch.tensor(torch.ones(K, N) * D/2, requires_grad=True, device=device)
        # log_sigma = torch.ones(K, N, requires_grad=True, device=device)

        # Set prior hyperparameters
        self.alpha_a0 = alpha_a0
        self.beta_a0 = beta_a0
        self.alpha_b0 = alpha_b0
        self.beta_b0 = beta_b0
    
    @property
    def template_scales(self) -> Float[Array, "num_templates num_neurons"]:
        return F.softmax(self._template_scales_unc, axis=1)
    
    @property
    def template_widths(self) -> Float[Array, "num_templates num_neurons"]:
        return F.softplus(self._template_widths_unc)

    @property
    def templates(self) -> Float[Array, "num_templates num_neurons duration"]:
        """Compute the templates from the mean, std, and amplitude of the Gaussian kernel.
        """
        ds = torch.arange(self.template_duration, device=self.device)[:, None, None]
        p = dist.Normal(self.template_delays, self.template_widths)
        W = p.log_prob(ds).exp().permute(1,2,0)
        W *= self.template_scales[:, :, None]
        return W

    def reconstruct(self,
                    amplitudes: Float[Array, "num_templates num_timesteps"]) \
                    -> Float[Array, "num_neurons num_timesteps"]: 
        """
        Reconstruct the firing rate given the model parameters and latent variables.

        Parameters
        ----------
        amplitudes: the amplitudes for each template as a function of time 
        """
        D = self.template_duration
        kernel = torch.flip(self.templates.permute(1,0,2), [2])
        return self.base_rates[:, None] + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        
    def log_probability(self,
                        data: Float[Array, "num_neurons num_timesteps"], 
                        amplitudes: Float[Array, "num_templates num_timesteps"]) -> float:
        """
        Calculate the log probability given data X
        and estimated parameters a, b, W

        Parameters
        ----------
        data: spike count matrix
        amplitudes: amplitudes of each template over time
        
        Returns
        -------
        Scalar log probability
        """
        D = self.template_duration
        kernel = torch.flip(self.templates.permute(1,0,2),[2])
        rates = self.base_rates[:, None] + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        rates = torch.clamp(rates, min=1e-7)
        return torch.sum(dist.Poisson(rates).log_prob(data))
    
    def _update_base_rates(self, data, amplitudes):
        D, T = self.template_duration, data.shape[1]
        b, W = self.base_rates, self.templates
        kernel = torch.flip(W.permute(1,0,2),[2])
        rates = b[:, None] + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        ratio = data / (rates + 1e-7)

        alpha_post = torch.sum(ratio, dim=1) * b + self.alpha_b0
        beta_post = T + self.beta_b0
        self.base_rates = torch.clip((alpha_post - 1) / beta_post, 0)
    
    def _update_amplitudes(self, data, amplitudes):
        D, T = self.template_duration, data.shape[1]
        b, W = self.base_rates, self.templates
        kernel = torch.flip(W.permute(1,0,2), [2])
        rates = b[:, None] + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        ratio = data / (rates + 1e-7) 

        alpha_post = amplitudes * F.conv1d(ratio, W, padding=D-1)[:,D-1:] + self.alpha_a0
        beta_post = torch.sum(W, dim=(1,2)).unsqueeze(1).repeat(1,T) + self.beta_a0
        return torch.clip((alpha_post - 1) / beta_post, 0)
        
    def _update_templates(self, 
                          data, 
                          amplitudes,
                          num_sgd_iter=100,
                          lr=0.01):
        D = self.template_duration
        b, W = self.base_rates, self.templates
        kernel = torch.flip(W.permute(1,0,2), [2])
        rates = b[:, None] + F.conv1d(amplitudes, kernel, padding=D-1)[:,:-D+1]
        ratio = data / (rates + 1e-7) 

        # TODO: Double check this line
        alpha_post = W * torch.flip(F.conv1d(amplitudes.unsqueeze(1), 
                                             ratio.unsqueeze(1),
                                             padding=D-1)[:,:,:-D+1], [2])
        beta_post = torch.sum(amplitudes, dim=1)[:,None,None]
        
        # Note: setting target to conditional mean rather than mode?
        targets = alpha_post / beta_post   

        # initialize SGD 
        loss_hist = []
        optimizer = optim.Adam([self._template_scales_unc, 
                                self.template_delays, 
                                self._template_widths_unc], 
                                lr=lr)
        
        # run SGD 
        for _ in range(int(num_sgd_iter)):
            optimizer.zero_grad()
            loss = nn.MSELoss(self.templates, targets)
            loss_hist.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()

    def fit(self,
            data: Float[Array, "num_neurons num_timesteps"],
            num_iter: int=50,
            num_inner_sgd_iter=100,
            sgd_learning_rate=0.01
            ):
        """
        Fit the model with expectation-maximization (EM).

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
        K = self.num_templates
        T = data.shape[1]
        lps = []

        # TODO: Initialize amplitudes more intelligently?
        amplitudes = torch.rand(K, T, device=self.device) + 1e-4
        
        # Run EM
        for _ in range(num_iter):
            amplitudes = self._update_amplitudes(data, amplitudes)
            self._update_base_rates(data, amplitudes)
            self._update_templates(data, amplitudes, num_sgd_iter=num_inner_sgd_iter, lr=sgd_learning_rate)
            lps.append(self.log_probability(data, amplitudes))

        return torch.stack(lps), amplitudes
