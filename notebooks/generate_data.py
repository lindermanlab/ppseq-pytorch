import torch
import numpy as np
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

def generate_data_glm(N=50,
                      T=1000,
                      K=2,
                      D=5,
                      P=3,
                      seed=None,
                      num_segments=None,
                      dense_strength_range=(0.3, 0.9),
                      sparse_strength_range=(0.1, 0.6)):
    """
    Generate Poisson spike trains with a dynamic GLM background.

    Args:
      N: number of neurons
      T: number of timebins
      K: number of latent sequences/events
      D: filter length (history length) for sequence kernels
      P: number of time-varying covariates for GLM background
      seed: optional integer seed for reproducibility
      num_segments: optional integer number of background segments (alternating sparse/dense).
        If None, a heuristic based on T is used.
      dense_strength_range: tuple(low, high) for Uniform sampling of dense log-rate boosts
        per dense segment (positive offset magnitudes).
      sparse_strength_range: tuple(low, high) for Uniform sampling of sparse log-rate dips
        per sparse segment (negative offset magnitudes).

    Returns:
      X: (N, T) sampled spikes
      lambdas: (N, T) total firing rates
      alpha: (N,) neuron-specific intercepts (log-rate baseline)
      beta: (N, P) neuron-specific GLM covariate weights
      X_cov: (T, P) design matrix of time covariates
      true_a: (K, T) latent sequence amplitudes/events
      true_w: (K, N, D) sequence kernels per neuron
      lambda_bg: (N, T) background firing rates from GLM
    """
    # Optional seed for reproducibility of stochastic components
    if seed is not None:
      torch.manual_seed(int(seed))
      np.random.seed(int(seed))

    # Validate and normalize segment controls
    if num_segments is None:
      num_segments = max(2, min(12, int(T // 250) + 3))
    else:
      num_segments = int(num_segments)
      num_segments = max(2, min(num_segments, max(2, T)))

    def _check_range(name, r):
      if not (isinstance(r, (tuple, list)) and len(r) == 2):
        raise ValueError(f"{name} must be a (low, high) tuple")
      low, high = float(r[0]), float(r[1])
      if not (low >= 0 and high > low):
        raise ValueError(f"{name} must satisfy 0 <= low < high, got {r}")
      return low, high

    dense_low, dense_high = _check_range('dense_strength_range', dense_strength_range)
    sparse_low, sparse_high = _check_range('sparse_strength_range', sparse_strength_range)

    # 1) Time-varying covariates X_cov: include constant, slow trend, and sinusoids.
    t = torch.arange(T, dtype=torch.float32)
    features = []
    # Constant term
    features.append(torch.ones(T))
    if P >= 2:
      # Slow linear trend in [-1, 1]
      features.append((t - (T - 1) / 2.0) / ((T - 1) / 2.0 + 1e-8))
    if P >= 3:
      # Very low-frequency sinusoid
      features.append(torch.sin(2 * torch.pi * t / max(50.0, T / 5.0)))
    if P >= 4:
      features.append(torch.cos(2 * torch.pi * t / max(80.0, T / 4.0)))
    # Add additional random-phase low-frequency sinusoids if P > 4
    while len(features) < P:
      period = torch.clamp(torch.rand(1) * (T / 3.0 - T / 10.0) + T / 10.0, min=10.0).item()
      phase = torch.rand(1).item() * 2 * np.pi
      features.append(torch.sin(2 * torch.pi * t / period + phase))
    X_cov = torch.stack(features[:P], dim=1)  # (T, P)

    # 2) Neuron-specific GLM parameters: alpha (intercepts) and beta (weights)
    # Slightly denser background: center around exp(-2.5) ≈ 0.08 spikes/bin
    alpha = torch.randn(N) * 0.3 - 0.05               # (N,)
    beta = torch.randn(N, P) * 0.3                   # (N, P)

    # Background segments for sparse/denser periods (piecewise-constant offset in log-rate)
    # Use requested number of segments and alternate signs to ensure variety
    # Random change-points (ensure non-empty segments)
    if num_segments > 1 and T > num_segments:
      change_points = torch.sort(torch.randperm(T - 1)[: num_segments - 1] + 1).values.tolist()
    else:
      change_points = []
    boundaries = [0] + change_points + [T]
    # Alternate +/- with random magnitudes to ensure sparse and dense blocks
    start_sign = 1 if torch.rand(1).item() > 0.5 else -1
    num_blocks = len(boundaries) - 1
    dense_mags = torch.empty(num_blocks).uniform_(dense_low, dense_high)
    sparse_mags = torch.empty(num_blocks).uniform_(sparse_low, sparse_high)
    segment_effects = []
    for i in range(num_blocks):
      sign = start_sign if i % 2 == 0 else -start_sign
      if sign > 0:
        segment_effects.append(dense_mags[i].item())
      else:
        segment_effects.append(-sparse_mags[i].item())
    eta_segment = torch.zeros(T)
    for i in range(len(boundaries) - 1):
      s, e = boundaries[i], boundaries[i + 1]
      eta_segment[s:e] = segment_effects[i]

    # Background firing rate λ_bg[n, t] = exp(alpha[n] + X_cov[t] @ beta[n] + segment_offset[t])
    eta_bg = alpha.view(N, 1) + (X_cov @ beta.T).T + eta_segment.view(1, T)
    lambda_bg = torch.exp(eta_bg)  # (N, T)

    # 3) Latent sequence amplitudes true_a and kernels true_w
    # Generate neuron-specific filters with peaks around D/2 and modest width
    mu = D / 2 + (torch.rand(K, N) - 0.5) * (D / 3)
    true_w = torch.exp(
      dist.Normal(mu, 0.5).log_prob(torch.arange(D).unsqueeze(1).unsqueeze(1))
    ).permute(1, 2, 0).expand(K, N, D)               # (K, N, D)

    # Make two groups of neurons selective to different sequences (if K>=2)
    if K >= 2:
      true_w[0, N // 2:, :] = 0
      true_w[1, :N // 2, :] = 0

    # Sparse event trains per sequence
    true_a = torch.zeros((K, T))
    num_events = max(1, int(T / max(200, T ** 0.7)))
    rng1 = np.random.choice(T - 7, num_events, replace=False) if T > 7 else np.array([], dtype=int)
    rng2 = np.random.choice(T - 10, num_events, replace=False) if T > 10 else np.array([], dtype=int)
    if K >= 1:
      true_a[0, rng1] = 15.0
    if K >= 2:
      true_a[1, rng2] = 15.0
    # Optionally add tiny jitter noise
    true_a = true_a + 0.0 * torch.rand_like(true_a)  # keep deterministic magnitude

    # Convolution (W ⊛ a) using 1D conv with flipped kernels
    conv_term = F.conv1d(
      true_a,                                    # (K, T)
      torch.flip(true_w.permute(1, 0, 2), [2]),  # weight: (N, K, D)
      padding=D - 1
    )[:, : -D + 1]                                # (N, T)

    # 4) Total rate and spikes
    lambdas = lambda_bg + conv_term               # (N, T)
    X = torch.poisson(lambdas)

    return X, lambdas, alpha, beta, X_cov, true_a, true_w, lambda_bg


def generate_data_glm_full(N=50,
                            T=1000,
                            K=2,
                            D=5,
                            P=3,
                            num_sessions=1,
                            seed=None,
                            num_segments=None,
                            dense_strength_range=(0.3, 0.9),
                            sparse_strength_range=(0.1, 0.6)):
    """
    Generate multiple sessions of Poisson spike trains with dynamic GLM background
    and return the ground truth spike sequences (non-background activity).

    Args:
      N: number of neurons
      T: number of timebins per session
      K: number of latent sequences/events
      D: filter length (history length) for sequence kernels
      P: number of time-varying covariates for GLM background
      num_sessions: number of sessions to generate
      seed: optional integer seed for reproducibility
      num_segments: optional integer number of background segments (alternating sparse/dense)
      dense_strength_range: tuple(low, high) for dense log-rate boosts
      sparse_strength_range: tuple(low, high) for sparse log-rate dips

    Returns:
      X_sessions: list of (N, T) spike train tensors, one per session
      lambdas_sessions: list of (N, T) total firing rate tensors
      alpha: (N,) shared neuron-specific intercepts across sessions
      beta: (N, P) shared neuron-specific GLM weights across sessions
      X_cov_sessions: list of (T, P) design matrices
      true_a_sessions: list of (K, T) latent sequence amplitudes
      true_w: (K, N, D) shared sequence kernels across sessions
      lambda_bg_sessions: list of (N, T) background firing rates
      correct_sequences: list of (N, T) spike trains from sequence activity only (no background)
    """
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    # Validate segment parameters
    if num_segments is None:
        num_segments = max(2, min(12, int(T // 250) + 3))
    else:
        num_segments = int(num_segments)
        num_segments = max(2, min(num_segments, max(2, T)))

    def _check_range(name, r):
        if not (isinstance(r, (tuple, list)) and len(r) == 2):
            raise ValueError(f"{name} must be a (low, high) tuple")
        low, high = float(r[0]), float(r[1])
        if not (low >= 0 and high > low):
            raise ValueError(f"{name} must satisfy 0 <= low < high, got {r}")
        return low, high

    dense_low, dense_high = _check_range('dense_strength_range', dense_strength_range)
    sparse_low, sparse_high = _check_range('sparse_strength_range', sparse_strength_range)

    # Shared parameters across sessions
    # Alpha (intercepts) and beta (GLM weights) are consistent across sessions
    alpha = torch.randn(N) * 0.3 - 0.05  # (N,)
    beta = torch.randn(N, P) * 0.3        # (N, P)

    # Shared sequence kernels across sessions
    mu = D / 2 + (torch.rand(K, N) - 0.5) * (D / 3)
    true_w = torch.exp(
        dist.Normal(mu, 0.5).log_prob(torch.arange(D).unsqueeze(1).unsqueeze(1))
    ).permute(1, 2, 0).expand(K, N, D)  # (K, N, D)

    if K >= 2:
        true_w[0, N // 2:, :] = 0
        true_w[1, :N // 2, :] = 0

    # Session-specific outputs
    X_sessions = []
    lambdas_sessions = []
    X_cov_sessions = []
    true_a_sessions = []
    lambda_bg_sessions = []
    correct_sequences = []

    for session_idx in range(num_sessions):
        # 1) Time-varying covariates for this session
        t = torch.arange(T, dtype=torch.float32)
        features = []
        features.append(torch.ones(T))
        if P >= 2:
            features.append((t - (T - 1) / 2.0) / ((T - 1) / 2.0 + 1e-8))
        if P >= 3:
            features.append(torch.sin(2 * torch.pi * t / max(50.0, T / 5.0)))
        if P >= 4:
            features.append(torch.cos(2 * torch.pi * t / max(80.0, T / 4.0)))
        while len(features) < P:
            period = torch.clamp(torch.rand(1) * (T / 3.0 - T / 10.0) + T / 10.0, min=10.0).item()
            phase = torch.rand(1).item() * 2 * np.pi
            features.append(torch.sin(2 * torch.pi * t / period + phase))
        X_cov = torch.stack(features[:P], dim=1)  # (T, P)

        # 2) Session-specific background segments
        if num_segments > 1 and T > num_segments:
            change_points = torch.sort(torch.randperm(T - 1)[: num_segments - 1] + 1).values.tolist()
        else:
            change_points = []
        boundaries = [0] + change_points + [T]
        start_sign = 1 if torch.rand(1).item() > 0.5 else -1
        num_blocks = len(boundaries) - 1
        dense_mags = torch.empty(num_blocks).uniform_(dense_low, dense_high)
        sparse_mags = torch.empty(num_blocks).uniform_(sparse_low, sparse_high)
        segment_effects = []
        for i in range(num_blocks):
            sign = start_sign if i % 2 == 0 else -start_sign
            if sign > 0:
                segment_effects.append(dense_mags[i].item())
            else:
                segment_effects.append(-sparse_mags[i].item())
        eta_segment = torch.zeros(T)
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            eta_segment[s:e] = segment_effects[i]

        # Background firing rate
        eta_bg = alpha.view(N, 1) + (X_cov @ beta.T).T + eta_segment.view(1, T)
        lambda_bg = torch.exp(eta_bg)  # (N, T)

        # 3) Session-specific latent sequence amplitudes
        true_a = torch.zeros((K, T))
        num_events = max(1, int(T / max(200, T ** 0.7)))
        rng1 = np.random.choice(T - 7, num_events, replace=False) if T > 7 else np.array([], dtype=int)
        rng2 = np.random.choice(T - 10, num_events, replace=False) if T > 10 else np.array([], dtype=int)
        if K >= 1:
            true_a[0, rng1] = 15.0
        if K >= 2:
            true_a[1, rng2] = 15.0

        # 4) Convolution for sequence activity
        conv_term = F.conv1d(
            true_a,
            torch.flip(true_w.permute(1, 0, 2), [2]),
            padding=D - 1
        )[:, : -D + 1]  # (N, T)

        # 5) Total rate and spikes
        lambdas = lambda_bg + conv_term  # (N, T)
        X = torch.poisson(lambdas)

        # 6) Ground truth sequences - identify which spikes came from sequence activity
        # We mark regions where sequence activity is present (conv_term > threshold)
        # These are the "correct" sequence locations that should be detected
        X_seq = (conv_term > 0.1).float()  # (N, T) binary mask of sequence regions

        # Store session outputs
        X_sessions.append(X)
        lambdas_sessions.append(lambdas)
        X_cov_sessions.append(X_cov)
        true_a_sessions.append(true_a)
        lambda_bg_sessions.append(lambda_bg)
        correct_sequences.append(X_seq)

    return (X_sessions, lambdas_sessions, alpha, beta, X_cov_sessions,
            true_a_sessions, true_w, lambda_bg_sessions, correct_sequences)

