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
                 device=None):
                 super().__init__(num_templates,
                 num_neurons,
                 template_duration,
                 alpha_a0, 
                 beta_a0, 
                 alpha_b0, 
                 beta_b0,
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
        amplitude_batches =[init_method(data) for data in data_batches]

        # TODO: Initialize amplitudes more intelligently?
        # amplitudes = torch.rand(K, T, device=self.device) + 1e-4
        
        # Run EM
        lps = []
        for _ in progress_bar(range(num_iter)):
            ll = 0
            for i, data in enumerate(data_batches):
                amplitude_batches[i] = self._update_amplitudes(data, 
                    amplitude_batches[i])
                self._update_base_rates(data, amplitude_batches[i])
                self._update_templates(data, amplitude_batches[i])
                ll += self.log_likelihood(data, amplitude_batches[i])
            lps.append(ll)

        lps = torch.stack(lps) if num_iter > 0 else torch.tensor([])
        return lps, amplitude_batches
        
