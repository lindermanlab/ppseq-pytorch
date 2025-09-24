import torch
import torch.nn.functional as F
import torch.distributions as dist

from scipy.signal import find_peaks


def generate_data_v4(N=50, T=1000, K=2, D=5):
  """
  -X: (N, T)
  """
  mu = D/2 + (torch.rand(K,N) - 0.5) * D/3
  true_w = torch.exp(dist.Normal(mu, 0.5).log_prob(torch.arange(D).unsqueeze(1).unsqueeze(1))).permute(1,2,0).expand(K,N,D)
  #\ * 2 * torch.tensor(torch.rand(K,N, 1)).expand(K, N, D)

  #true_w = torch.linspace(D, D/2, D).repeat(K,N,1)
  true_w[0, N//2:,:] = 0
  true_w[1, :N//2, :] = 0

  true_a = torch.zeros((K,T))
  t = 10
  t1 = np.random.choice(T-7,  t, replace=False)
  t2 = np.random.choice(T-10, t, replace=False)
  true_a[0, t1] = 15
  true_a[1, t2] = 15
  # TODO[GLM]: Replace constant background with a dynamic GLM background, e.g.:
  #   X_cov = design_matrix(T, P)                 # (T, P) e.g., low-freq trends, task events, motion
  #   beta  = torch.randn(N, P) * 0.1            # (N, P)
  #   true_b = torch.exp(X_cov @ beta.T).T       # (N, T)
  # For legacy behavior keep constant for now:
  true_b = torch.ones(N) * 0.04
  lambdas = true_b.view(N,1) + F.conv1d(true_a, torch.flip(true_w.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
  X = torch.poisson(lambdas)
  return X, lambdas, true_b, true_a, true_w

def generate_data_v3(N=50, T=1000, K=2, D=10):
  """
  -X: (N, T)
  """
  mu = D/2 + (torch.rand(K,N) - 0.5) * D/3
  true_w = torch.exp(dist.Normal(mu, 1).log_prob(torch.arange(D).unsqueeze(1).unsqueeze(1))).permute(1,2,0).expand(K,N,D)
  #\ * 2 * torch.tensor(torch.rand(K,N, 1)).expand(K, N, D)

  #true_w = torch.linspace(D, D/2, D).repeat(K,N,1)
  true_w[0, N//2:,:] = 0
  true_w[1, :N//2, :] = 0

  true_a = torch.zeros((K,T))
  t = int(T / np.maximum(200, T**0.7))
  t1 = np.random.choice(T-7,  t, replace=False)
  t2 = np.random.choice(T-10, t, replace=False)
  true_a[0, t1] = 100
  true_a[1, t2] = 100
  # TODO[GLM]: As above, switch to λ_bg[n,t] = exp(X_t β_n) to create time-varying backgrounds.
  true_b = torch.ones(N) * 0.01
  lambdas = true_b.view(N,1) + F.conv1d(true_a, torch.flip(true_w.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
  X = torch.poisson(lambdas)
  return X, lambdas, true_b, true_a, true_w




def generate_templates(num_channels, len_waveform, num_neurons):
    # Make (semi) random templates
    templates = []
    for k in range(num_neurons):
        center = dist.Uniform(0.0, num_channels).sample()
        width = dist.Uniform(1.0, 1.0 + num_channels / 10.0).sample()
        spatial_factor = torch.exp(-0.5 * (torch.arange(num_channels) - center)**2 / width**2)

        dt = torch.arange(len_waveform)
        period = len_waveform / (dist.Uniform(1.0, 2.0).sample())
        z = (dt - 0.75 * period) / (.25 * period)
        warp = lambda x: -torch.exp(-x) + 1
        window = torch.exp(-0.5 * z**2)
        shape = torch.sin(2 * torch.pi * dt / period)
        temporal_factor = warp(window * shape)

        template = torch.outer(spatial_factor, temporal_factor)
        template /= torch.linalg.norm(template)
        templates.append(template)

    return torch.abs(torch.stack(templates))



def generate_data_v1(num_timesteps,
             num_channels,
             len_waveform,
             num_neurons,
             mean_amplitude=15,
             shape_amplitude=3.0,
             noise_std=1,
             sample_freq=1000):
    """Create a random set of model parameters and sample data.

    Parameters:
    num_timesteps: integer number of time samples in the data
    num_channels: integer number of channels
    len_waveform: integer duration (number of samples) of each template
    num_neurons: integer number of neurons
    """
    # Make semi-random templates
    templates = generate_templates(num_channels, len_waveform, num_neurons)

    # Make random amplitudes
    amplitudes = torch.zeros((num_neurons, num_timesteps))
    for k in range(num_neurons):
        num_spikes = dist.Poisson(num_timesteps / sample_freq * 10.0).sample()
        sample_shape = (1 + int(num_spikes),)
        times = dist.Categorical(torch.ones(num_timesteps) / num_timesteps).sample(sample_shape)
        amps = dist.Gamma(shape_amplitude, shape_amplitude / mean_amplitude).sample(sample_shape)
        amplitudes[k, times] = amps

        # Only keep spikes separated by at least D
        times, props = find_peaks(amplitudes[k], distance=len_waveform, height=1e-3)
        amplitudes[k] = 0
        amplitudes[k, times] = torch.tensor(props['peak_heights'], dtype=torch.float32)

    amplitudes = torch.abs(amplitudes)

    # Convolve the signal with each row of the multi-channel template
    #data = F.conv1d(amplitudes.unsqueeze(0),
    #               templates.permute(1, 0, 2).flip(dims=(2,)),
    #               padding=len_waveform-1)[0, :, :-(len_waveform-1)]

    #data += dist.Normal(0.0, noise_std).sample(data.shape)
    true_a = amplitudes
    true_w = templates
    # TODO[GLM]: Replace constant `true_b` with a dynamic background:
    #     X_cov = design_matrix(num_timesteps, P)     # (T, P)
    #     beta  = torch.randn(num_neurons, P) * 0.1   # (N, P)
    #     true_bg = torch.exp(X_cov @ beta.T).T       # (N, T)
    # Then use:
    #     lambdas = true_bg + F.conv1d(true_a, ...)
    true_b = torch.rand(N) + 0.2
    lambdas = true_b.view(N,1) + F.conv1d(true_a, torch.flip(true_w.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
    data = torch.poisson(lambdas)

    return templates, amplitudes, true_b, data

def generate_data_v0(N=8, T=2000, K=5, D=10):
  """
  -X: (N, T)
  """

  true_w = torch.linspace(10, 0, D).repeat(K,N,1)
  true_a = torch.rand((K,T)) * 10
  # TODO[GLM]: Make the background dynamic here too for GLM training/validation.
  true_b = torch.rand(N) *10
  lambdas = true_b.view(N,1) + F.conv1d(true_a, torch.flip(true_w.permute(1,0,2),[2]), padding=D-1)[:,:-D+1]
  X = torch.poisson(lambdas)
  return X, true_b, true_a, true_w

def generate_data_glm(N=50, T=1000, K=2, D=5, P=3):
    """
    # TODO[GLM]: New helper to synthesize data with a *dynamic* background rate.
    #   1) X_cov: (T, P) time covariates (e.g., constant 1, slow trend, event indicators).
    #   2) α: (N,), β: (N, P)  → λ_bg[n,t] = exp( α[n] + X_cov[t] @ β[n] ).
    #   3) Sequence parameters a, W as in existing generators.
    #   4) λ = λ_bg + (W ⊛ a) and X ~ Poisson(λ).
    # Return:
    #   X, λ, α, β, X_cov, true_a, true_w  (and optionally λ_bg)
    """
    raise NotImplementedError("TODO[GLM]: implement generate_data_glm as described above.")

  