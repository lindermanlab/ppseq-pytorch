import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from fastprogress import progress_bar
from torch import Tensor
from jaxtyping import Float

from .model import PPSeq

class batchPPseq(PPSeq):
    def __init__(self,
                 num_templates: int,
                 num_neurons: int,
                 template_duration: int,
                 alpha_a0: float=0.5, 
                 beta_a0: float=0., 
                 alpha_b0: float=0., 
                 beta_b0: float=0.,
                 alpha_t0: float=0.,
                 beta_t0:float=0.,
                 device=None):
                 super().__init__(num_templates,
                 num_neurons,
                 template_duration,
                 alpha_a0, 
                 beta_a0, 
                 alpha_b0, 
                 beta_b0,
                 alpha_t0,
                 beta_t0,
                 device)
    
    def fit(self,
            data_batches,
            num_iter: int=50,
            initialization="random",
            ):
        """
        Fit the model with expectation-maximization (EM).
        """
        K = self.num_templates
        

        init_method = dict(random=self.initialize_random)[initialization.lower()]
        amplitude_batches =[init_method(data.squeeze()) for data in data_batches]

        # TODO: Initialize amplitudes more intelligently?
        # amplitudes = torch.rand(K, T, device=self.device) + 1e-4
        
        # Run EM
        lps = []
        for _ in progress_bar(range(num_iter)):
            ll = 0
            for i, data in enumerate(data_batches):
                data = data.squeeze() # prevents indexing error when data_shape = (1, N, T) (e.g in a torch dataloader)
                amplitude_batches[i] = self._update_amplitudes(data, 
                    amplitude_batches[i])
                self._update_base_rates(data, amplitude_batches[i])
                self._update_templates(data, amplitude_batches[i])
                ll += self.log_likelihood(data, amplitude_batches[i])
            lps.append(ll) #return the sum or avg log likelihood?

        lps = torch.stack(lps) if num_iter > 0 else torch.tensor([])
        return lps, amplitude_batches
        
