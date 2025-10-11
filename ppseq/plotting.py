import numpy as np

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


def sort_neurons(X, scale, mu):
    """
    Given Neural spike trains X of shape (N,T), return the order (N,) of the neurons
    """
    N, T, K = X.shape[0], X.shape[1], scale.shape[0]
    color_list = [[] for _ in range(K)]
    #A, B = [], []
    for i in range(N):
        color_list[int(np.argmax(scale[:,i]))].append(i)
        #if scale[0][i] > scale[1][i]:
        #    A.append(i)
        #else:
        #    B.append(i)
    #A.sort(key=lambda x: mu[0][x])
    #B.sort(key=lambda x: mu[1][x])
    color_list = [sorted(color_list[k], key=lambda x: mu[k][x]) 
    for k in range(len(color_list))]
    #return A + B
    return [x  for l in color_list for x in l]


def plot_sorted_neurons(data):
    """
    given a data matrix of shape (N, T) Neuron * Time plot the neural spike trains
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(torch.nonzero(data.T)[:, 0], torch.nonzero(data.T)[:, 1], s=10)
    plt.show()

named_colors = [
    'black',
    'red',
    'blue',
    'green',
    'cyan',
    'magenta',
    'yellow',
    'white',
    'orange',
    'purple',
    'brown',
    'pink',
    'gray',
    'olive',
    'navy',
    'gold',
    'silver',
    'maroon',
    'lime',
    'teal',
    'aqua',
    'fuchsia',
    'indigo',
    'violet',
    'coral',
    'turquoise',
    'tan',
    'chocolate',
    'salmon',
    'plum'
]

def color_plot(data, model, amplitudes, save_path=None):
    """
    Plot the neural spike trains and 
    color the spikes into red, blue and black according to their intensities
    supports at most 30 colors
    
    Args:
        save_path: Optional path to save the figure as PDF. If None, figure is not saved.
    """
    b, W, scale, mu = model.base_rates.cpu(),model.templates.cpu(),model.template_scales.cpu(),model.template_offsets.cpu()
    a = amplitudes
    order = sort_neurons(data, scale, mu)
    N, T, K = data.shape[0], data.shape[1], scale.shape[0]
    D = W.shape[2]
    black_nt = b.view(N, 1).expand(N, T)
    # TODO[GLM]: If background is dynamic (Poisson GLM), `model.base_rates` is not sufficient.
    # Replace with:
    #   black_nt = model.background_rate(covariates, T)   # shape (N, T)
    # and extend `color_plot(...)` to accept a `covariates` argument and pass it along.
    
    #red_nt = F.conv1d(a[[0], :], torch.flip(
    #    W[[0]].permute(1, 0, 2), [2]), padding=D-1)[:, :-D+1]
   # blue_nt = F.conv1d(a[[1], :], torch.flip(
     #   W[[1]].permute(1, 0, 2), [2]), padding=D-1)[:, :-D+1]
    #sum_nt = F.conv1d(a, torch.flip(
       # W.permute(1, 0, 2), [2]), padding=D-1)[:, :-D+1]
    #assert torch.allclose(red_nt + blue_nt, sum_nt)
    matrices = np.array([black_nt] + [F.conv1d(a[[i], :], torch.flip(
       W[[i]].permute(1, 0, 2), [2]), padding=D-1)[:, :-D+1] for i in range(K)])
    
    def f(i, j):
        if data[i, j] == 0:
            return -1
        large = max(black_nt[i, j], red_nt[i, j], blue_nt[i, j])
        if black_nt[i, j] >= large:
            return 0
        if red_nt[i, j] == max(red_nt[i, j], blue_nt[i, j]):
            return 1
        return 2

    def f(i,j):
        if data[i, j] == 0:
            return -1
        else:
            return np.argmax(matrices[:,i,j])
    colors = np.array([[f(i, j) for i in range(N)] for j in range(T)])
    colors = colors[:, order]
    #black_indices = np.argwhere(colors == 0)
    #red_indices = np.argwhere(colors == 1)
    #blue_indices = np.argwhere(colors == 2)
    color_indices = [np.argwhere(colors == i) for i in range(K+1)]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(K+1):
        plt.scatter(color_indices[i][:, 0],
            color_indices[i][:, 1], c=named_colors[i], s=4, alpha=0.7)
    #plt.scatter(black_indices[:, 0],
    #        black_indices[:, 1], c='black', s=4, alpha=0.7)
    #plt.scatter(red_indices[:, 0], red_indices[:, 1], c='red', s=4, alpha=0.7)
    #plt.scatter(blue_indices[:, 0], blue_indices[:, 1],
    #            c='blue', s=4, alpha=0.7)
    plt.title('')
    plt.xlabel('Time')
    plt.ylabel('channel')
    #plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


palette = sns.xkcd_palette(["windows blue",
                            "red",
                            "medium green",
                            "dusty purple",
                            "orange",
                            "amber",
                            "clay",
                            "pink",
                            "greyish"])
sns.set_context("notebook")


def plot_templates(templates,
                   indices,
                   scale=0.1,
                   n_cols=8,
                   panel_height=6,
                   panel_width=1.25,
                   colors=('k',),
                   label="neuron",
                   sample_freq=30000,
                   fig=None,
                   axs=None,
                   save_path=None):
    n_subplots = len(indices)
    n_cols = min(n_cols, n_subplots)
    n_rows = int(torch.ceil(torch.tensor(n_subplots / n_cols)))

    if fig is None and axs is None:
        fig, axs = plt.subplots(n_rows, n_cols,
                                figsize=(panel_width * n_cols,
                                         panel_height * n_rows),
                                sharex=True, sharey=True)

    n_units, n_channels, spike_width = templates.shape
    timestamps = torch.arange(-spike_width // 2, spike_width//2) / sample_freq
    for i, ind in enumerate(indices):
        row, col = i // n_cols, i % n_cols
        ax = axs[row, col] if n_rows > 1 else axs[col]
        color = colors[i % len(colors)]
        ax.plot(timestamps * 1000,
                templates[ind].T - scale * torch.arange(n_channels),
                '-', color=color, lw=1)

        ax.set_title("{} {:d}".format(label, ind + 1))
        ax.set_xlim(timestamps[0] * 1000, timestamps[-1] * 1000)
        ax.set_yticks(-scale * torch.arange(0, n_channels+1, step=4))
        ax.set_yticklabels(torch.arange(0, n_channels+1, step=4).numpy() + 1)
        ax.set_ylim(-scale * n_channels, scale)

        if i // n_cols == n_rows - 1:
            ax.set_xlabel("time [ms]")
        if i % n_cols == 0:
            ax.set_ylabel("channel")

        # plt.tight_layout(pad=0.1)

    # hide the remaining axes
    for i in range(n_subplots, len(axs)):
        row, col = i // n_cols, i % n_cols
        ax = axs[row, col] if n_rows > 1 else axs[col]
        ax.set_visible(False)

    if save_path is not None:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')

    return fig, axs


def plot_model(templates, amplitude, data, scores=None, lw=2, figsize=(12, 6), spc=0.1, save_path=None):
    """Plot the raw data as well as the underlying signal amplitudes and templates.

    amplitude: (K,T) array of underlying signal amplitude
    template: (K,N,D) array of template that is convolved with signal
    data: (N, T) array (channels x time)
    scores: optional (K,T) array of correlations between data and template
    save_path: Optional path to save the figure as PDF. If None, figure is not saved.
    """
    # prepend dimension if data and template are 1d
    data = torch.atleast_2d(data)
    N, T = data.shape
    amplitude = torch.atleast_2d(amplitude)
    K, _ = amplitude.shape
    templates = templates.reshape(K, N, -1)
    D = templates.shape[-1]
    dt = torch.arange(D)
    if scores is not None:
        scores = torch.atleast_2d(scores)

    # Set up figure with 2x2 grid of panels
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        2, K + 1, height_ratios=[1, 2], width_ratios=[1] * K + [2 * K])

    # plot the templates
    t_spc = spc * abs(templates).max()
    for n in range(K):
        ax = fig.add_subplot(gs[1, n])
        # ax.plot(dt, templates[n].T - t_spc * torch.arange(N),
        #         '-', color=palette[n % len(palette)], lw=lw)
        plt.imshow(templates[n], aspect='auto')
        ax.set_xlabel("delay $d$")
        # ax.set_xlim([0, D])
        # ax.set_yticks(-t_spc * torch.arange(N))
        # ax.set_yticklabels([])
        # ax.set_ylim(-N * t_spc, t_spc)
        if n == 0:
            ax.set_ylabel("neurons $n$")
        ax.set_title("template $W_{{ {} }}$".format(n+1))

    # plot the amplitudes for each neuron
    ax = fig.add_subplot(gs[0, -1])
    a_spc = 1.05 * abs(amplitude).max()
    if scores is not None:
        a_spc = max(a_spc, 1.05 * abs(scores).max())

    for n in range(K):
        ax.plot(amplitude[n] - a_spc * n, '-',
                color=palette[n % len(palette)], lw=lw)
        ax.axhline(-a_spc * n, color='gray', lw=0.5)

        if scores is not None:
            ax.plot(scores[n] - a_spc * n, ':', color=palette[n % len(palette)], lw=lw,
                    label="$X \star W$")

    ax.set_xlim([0, T])
    ax.set_xticklabels([])
    # ax.set_yticks(-a_spc * torch.arange(K).numpy())
    # ax.set_yticklabels([])
    ax.set_ylabel("num spikes")
    ax.set_title("amplitude $a$")
    if scores is not None:
        ax.legend()

    # plot the data
    ax = fig.add_subplot(gs[1, -1])
    d_spc = 1.05 * abs(data).max()
    masked_data = data.clone()
    masked_data[masked_data == 0] = torch.nan
    ax.plot(masked_data.T - d_spc * torch.arange(N), '|', color='gray', lw=lw, ms=2)
    ax.set_xlabel("time $t$")
    ax.set_xlim([0, T])
    ax.set_yticks(-d_spc * torch.arange(N).numpy())
    ax.set_yticklabels([])
    ax.set_ylim(-N * d_spc, d_spc)
    # ax.set_ylabel("channels $c$")
    ax.set_title(r"spikes $X$")
    
    if save_path is not None:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')

####

def plot_synthetic_model(true_a, true_w, X_full, figsize=(12, 6), save_path=None):
    """
    Plot the ground truth amplitudes and templates from synthetic data.
    
    Args:
        true_a: (K, T) tensor of ground truth latent sequence amplitudes
        true_w: (K, N, D) tensor of ground truth sequence kernels/templates
        X_full: (N, T) tensor of spike train data
        figsize: tuple specifying figure size
        save_path: Optional path to save the figure as PDF. If None, figure is not saved.
    """
    K, T = true_a.shape
    K, N, D = true_w.shape
    
    # Color palette for different sequences
    palette = plt.cm.Set2.colors
    
    # Set up figure with 2x2 grid of panels
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, K + 1, height_ratios=[1, 2], width_ratios=[1] * K + [2 * K])
    
    # Plot the templates
    for k in range(K):
        ax = fig.add_subplot(gs[1, k])
        im = ax.imshow(true_w[k].cpu().numpy(), aspect='auto', cmap='viridis')
        ax.set_xlabel("Delay $d$")
        ax.set_ylabel("Neuron $n$" if k == 0 else "")
        ax.set_title(f"Template $W_{{{k+1}}}$")
        plt.colorbar(im, ax=ax)
    
    # Plot the amplitudes
    ax = fig.add_subplot(gs[0, -1])
    a_spc = (1.05 * abs(true_a).max()).item()  # Convert to scalar
    
    for k in range(K):
        ax.plot(true_a[k].cpu().numpy() - a_spc * k, '-',
                color=palette[k % len(palette)], lw=2, label=f"Sequence {k+1}")
        ax.axhline(-a_spc * k, color='gray', lw=0.5, linestyle='--', alpha=0.5)
    
    ax.set_xlim([0, T])
    ax.set_xticklabels([])
    ax.set_ylabel("Amplitude")
    ax.set_title("Ground Truth Amplitudes $a$")
    ax.legend(loc='upper right')
    
    # Plot the spike raster
    ax = fig.add_subplot(gs[1, -1])
    d_spc = (1.05 * abs(X_full).max()).item()  # Convert to scalar
    masked_data = X_full.clone()
    masked_data[masked_data == 0] = torch.nan
    ax.plot(masked_data.T.cpu().numpy() - d_spc * torch.arange(N).numpy(), 
            '|', color='gray', lw=1, ms=2)
    ax.set_xlabel("Time $t$")
    ax.set_xlim([0, T])
    ax.set_yticks(-d_spc * torch.arange(N).numpy())
    ax.set_yticklabels([])
    ax.set_ylim(-N * d_spc, d_spc)
    ax.set_ylabel("Neuron" if K == 0 else "")
    ax.set_title("Spikes $X$")
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


def plot_background_activity(X_full, lambda_bg, X_cov, alpha, beta, figsize=(12, 8), save_path=None):
    """
    Plot the background activity from synthetic data.
    
    Args:
        X_full: (N, T) tensor of full spike train data
        lambda_bg: (N, T) tensor of background firing rates
        X_cov: (T, P) tensor of time-varying covariates
        alpha: (N,) tensor of neuron-specific intercepts
        beta: (N, P) tensor of neuron-specific GLM weights
        figsize: tuple specifying figure size
        save_path: Optional path to save the figure as PDF. If None, figure is not saved.
    """
    N, T = X_full.shape
    P = X_cov.shape[1]
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[1, 1, 2])
    
    # Plot 1: Time-varying covariates
    ax = axes[0]
    for p in range(P):
        ax.plot(X_cov[:, p].cpu().numpy(), label=f'Covariate {p+1}', alpha=0.7)
    ax.set_ylabel('Covariate value')
    ax.set_title('Time-varying covariates')
    ax.legend(loc='upper right', ncol=P)
    ax.set_xlim([0, T])
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Background firing rates (heatmap)
    ax = axes[1]
    im = ax.imshow(lambda_bg.cpu().numpy(), aspect='auto', cmap='YlOrRd', 
                   interpolation='nearest')
    ax.set_ylabel('Neuron')
    ax.set_title('Background firing rate $\\lambda_{bg}$')
    plt.colorbar(im, ax=ax, label='Rate (spikes/bin)')
    ax.set_xlim([0, T])
    
    # Plot 3: Background spikes (raster plot)
    ax = axes[2]
    
    # Get all spike coordinates
    spike_coords = torch.nonzero(X_full.T)
    spike_times = spike_coords[:, 0].cpu().numpy()
    spike_neurons = spike_coords[:, 1].cpu().numpy()
    
    # Compute sequence activity to identify background vs sequence spikes
    # (assuming we have access to true_a and true_w, otherwise plot all spikes)
    # For now, plot all spikes as they represent total activity including background
    
    ax.scatter(spike_times, spike_neurons, c='black', s=5, alpha=0.5, marker='|')
    ax.set_xlabel('Time (bins)')
    ax.set_ylabel('Neuron')
    ax.set_title('All spikes (including background)')
    ax.set_xlim([0, T])
    ax.set_ylim([-0.5, N - 0.5])
    
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    # Print statistics
    mean_bg_rate = lambda_bg.mean().item()
    print(f"Mean background firing rate: {mean_bg_rate:.3f} spikes/bin")
    print(f"Background rate range: [{lambda_bg.min().item():.3f}, {lambda_bg.max().item():.3f}]")

def plot_background_spikes_only(X_full, lambda_bg, X_cov, true_a, true_w, 
                                 alpha, beta, figsize=(12, 8), save_path=None):
    """
    Plot only the background activity, excluding sequence spikes.
    
    Args:
        X_full: (N, T) tensor of full spike train data
        lambda_bg: (N, T) tensor of background firing rates
        X_cov: (T, P) tensor of time-varying covariates
        true_a: (K, T) tensor of ground truth sequence amplitudes
        true_w: (K, N, D) tensor of ground truth sequence kernels
        alpha: (N,) tensor of neuron-specific intercepts
        beta: (N, P) tensor of neuron-specific GLM weights
        figsize: tuple specifying figure size
        save_path: Optional path to save the figure as PDF. If None, figure is not saved.
    """
    import torch.nn.functional as F
    
    N, T = X_full.shape
    P = X_cov.shape[1]
    K, _, D = true_w.shape
    
    # Compute sequence activity mask
    conv_term = F.conv1d(
        true_a,
        torch.flip(true_w.permute(1, 0, 2), [2]),
        padding=D - 1
    )[:, : -D + 1]
    seq_mask = conv_term > 0.1
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[1, 1, 2])
    
    # Plot 1: Time-varying covariates
    ax = axes[0]
    for p in range(P):
        ax.plot(X_cov[:, p].cpu().numpy(), label=f'Covariate {p+1}', alpha=0.7)
    ax.set_ylabel('Covariate value')
    ax.set_title('Time-varying covariates')
    ax.legend(loc='upper right', ncol=P)
    ax.set_xlim([0, T])
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Background firing rates (heatmap)
    ax = axes[1]
    im = ax.imshow(lambda_bg.cpu().numpy(), aspect='auto', cmap='YlOrRd', 
                   interpolation='nearest')
    ax.set_ylabel('Neuron')
    ax.set_title('Background firing rate $\\lambda_{bg}$')
    plt.colorbar(im, ax=ax, label='Rate (spikes/bin)')
    ax.set_xlim([0, T])
    
    # Plot 3: Background spikes only (raster plot)
    ax = axes[2]
    
    # Get all spike coordinates
    spike_coords = torch.nonzero(X_full.T)
    spike_times = spike_coords[:, 0].cpu().numpy()
    spike_neurons = spike_coords[:, 1].cpu().numpy()
    
    # Identify background spikes (not sequence spikes)
    is_sequence_spike = seq_mask.T[spike_coords[:, 0], spike_coords[:, 1]].cpu().numpy()
    bg_mask = ~is_sequence_spike
    
    # Plot only background spikes
    ax.scatter(spike_times[bg_mask], spike_neurons[bg_mask], 
               c='darkblue', s=5, alpha=0.6, marker='|')
    ax.set_xlabel('Time (bins)')
    ax.set_ylabel('Neuron')
    ax.set_title('Background spikes only')
    ax.set_xlim([0, T])
    ax.set_ylim([-0.5, N - 0.5])
    
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    # Print statistics
    num_bg_spikes = bg_mask.sum()
    num_total_spikes = len(spike_times)
    mean_bg_rate = lambda_bg.mean().item()
    print(f"Mean background firing rate: {mean_bg_rate:.3f} spikes/bin")
    print(f"Background rate range: [{lambda_bg.min().item():.3f}, {lambda_bg.max().item():.3f}]")
    print(f"Background spikes: {num_bg_spikes} ({100*num_bg_spikes/num_total_spikes:.1f}%)")
    print(f"Total spikes: {num_total_spikes}")

def plot_templates_comparison(true_w, model_w, figsize=(14, 6), save_path=None):
    """
    Plot ground truth templates and model templates side-by-side for comparison.
    
    Args:
        true_w: (K, N, D) tensor of ground truth sequence kernels/templates
        model_w: (K, N, D) tensor of estimated sequence kernels/templates
        figsize: tuple specifying figure size
        save_path: Optional path to save the figure as PDF. If None, figure is not saved.
    """
    K, N, D = true_w.shape
    
    fig, axes = plt.subplots(K, 2, figsize=figsize)
    
    # Handle case where K=1 (axes won't be 2D)
    if K == 1:
        axes = axes.reshape(1, -1)
    
    for k in range(K):
        # Plot ground truth template
        ax = axes[k, 0]
        im1 = ax.imshow(true_w[k].cpu().numpy(), aspect='auto', cmap='viridis')
        ax.set_ylabel(f'Neuron (Seq {k+1})')
        ax.set_xlabel('Delay $d$')
        if k == 0:
            ax.set_title('Ground Truth Templates $W_{true}$', fontweight='bold')
        plt.colorbar(im1, ax=ax)
        
        # Plot model template
        ax = axes[k, 1]
        im2 = ax.imshow(model_w[k].cpu().numpy(), aspect='auto', cmap='viridis')
        ax.set_ylabel('')
        ax.set_xlabel('Delay $d$')
        if k == 0:
            ax.set_title('Model Templates $W_{model}$', fontweight='bold')
        plt.colorbar(im2, ax=ax)
    
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    # Compute and print similarity metrics
    print("Template Comparison Metrics:")
    for k in range(K):
        # Correlation
        true_flat = true_w[k].flatten()
        model_flat = model_w[k].flatten()
        correlation = torch.corrcoef(torch.stack([true_flat, model_flat]))[0, 1].item()
        
        # Normalized MSE
        mse = torch.mean((true_w[k] - model_w[k])**2).item()
        norm_mse = mse / (torch.mean(true_w[k]**2).item() + 1e-10)
        
        print(f"  Sequence {k+1}:")
        print(f"    Correlation: {correlation:.4f}")
        print(f"    Normalized MSE: {norm_mse:.4f}")


def plot_amplitudes_comparison(true_a, model_a, figsize=(14, 8), save_path=None):
    """
    Plot ground truth amplitudes and model amplitudes side-by-side for comparison.
    
    Args:
        true_a: (K, T) tensor of ground truth latent sequence amplitudes
        model_a: (K, T) tensor of estimated latent sequence amplitudes
        figsize: tuple specifying figure size
        save_path: Optional path to save the figure as PDF. If None, figure is not saved.
    """
    K, T = true_a.shape
    
    # Color palette
    palette = plt.cm.Set2.colors
    
    fig, axes = plt.subplots(K, 1, figsize=figsize)
    
    # Handle case where K=1 (axes won't be an array)
    if K == 1:
        axes = [axes]
    
    for k in range(K):
        ax = axes[k]
        
        # Plot ground truth
        ax.plot(true_a[k].cpu().numpy(), '-', 
                color=palette[k % len(palette)], lw=2, 
                label='Ground Truth', alpha=0.8)
        
        # Plot model estimate
        ax.plot(model_a[k].cpu().numpy(), '--', 
                color=palette[(k + 1) % len(palette)], lw=2, 
                label='Model', alpha=0.8)
        
        ax.set_xlim([0, T])
        ax.set_ylabel(f'Amplitude\n(Seq {k+1})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if k == 0:
            ax.set_title('Amplitude Comparison: Ground Truth vs Model', fontweight='bold')
        if k == K - 1:
            ax.set_xlabel('Time $t$')
    
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    # Compute and print similarity metrics
    print("\nAmplitude Comparison Metrics:")
    for k in range(K):
        # Correlation
        correlation = torch.corrcoef(torch.stack([true_a[k], model_a[k]]))[0, 1].item()
        
        # Normalized MSE
        mse = torch.mean((true_a[k] - model_a[k])**2).item()
        norm_mse = mse / (torch.mean(true_a[k]**2).item() + 1e-10)
        
        # Count detection accuracy (for sparse amplitudes)
        true_events = (true_a[k] > 0.1).sum().item()
        model_events = (model_a[k] > 0.1).sum().item()
        
        print(f"  Sequence {k+1}:")
        print(f"    Correlation: {correlation:.4f}")
        print(f"    Normalized MSE: {norm_mse:.4f}")
        print(f"    True events: {true_events}, Model events: {model_events}")