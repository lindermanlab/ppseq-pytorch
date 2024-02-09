# Import PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim

# Plotting stuff
import matplotlib.pyplot as plt

# Some helper utilities
from tqdm.auto import trange
# Plotting stuff
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.gridspec import GridSpec
import seaborn as sns

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
                   axs=None):
    n_subplots = len(indices)
    n_cols = min(n_cols, n_subplots)
    n_rows = int(torch.ceil(torch.tensor(n_subplots / n_cols)))

    if fig is None and axs is None:
        fig, axs = plt.subplots(n_rows, n_cols,
                                figsize=(panel_width * n_cols, panel_height * n_rows),
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

    return fig, axs

def plot_model(templates, amplitude, data, scores=None, lw=2, figsize=(12, 6)):
    """Plot the raw data as well as the underlying signal amplitudes and templates.

    amplitude: (K,T) array of underlying signal amplitude
    template: (K,N,D) array of template that is convolved with signal
    data: (N, T) array (channels x time)
    scores: optional (K,T) array of correlations between data and template
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
    gs = GridSpec(2, K + 1, height_ratios=[1, 2], width_ratios=[1] * K + [2 * K])

    # plot the templates
    t_spc = 1.05 * abs(templates).max()
    for n in range(K):
        ax = fig.add_subplot(gs[1, n])
        ax.plot(dt, templates[n].T - t_spc * torch.arange(N),
                '-', color=palette[n % len(palette)], lw=lw)
        ax.set_xlabel("delay $d$")
        ax.set_xlim([0, D])
        ax.set_yticks(-t_spc * torch.arange(N))
        ax.set_yticklabels([])
        ax.set_ylim(-N * t_spc, t_spc)
        if n == 0:
            ax.set_ylabel("channels $n$")
        ax.set_title("$W_{{ {} }}$".format(n+1))

    # plot the amplitudes for each neuron
    ax = fig.add_subplot(gs[0, -1])
    a_spc = 1.05 * abs(amplitude).max()
    if scores is not None:
        a_spc = max(a_spc, 1.05 * abs(scores).max())

    for n in range(K):
        ax.plot(amplitude[n] - a_spc * n, '-', color=palette[n % len(palette)], lw=lw)

        if scores is not None:
            ax.plot(scores[n] - a_spc * n, ':', color=palette[n % len(palette)], lw=lw,
                label="$X \star W$")

    ax.set_xlim([0, T])
    ax.set_xticklabels([])
    ax.set_yticks(-a_spc * torch.arange(K).numpy())
    ax.set_yticklabels([])
    ax.set_ylabel("neurons $k$")
    ax.set_title("amplitude $a$")
    if scores is not None:
        ax.legend()

    # plot the data
    ax = fig.add_subplot(gs[1, -1])
    d_spc = 1.05 * abs(data).max()
    ax.plot(data.T - d_spc * torch.arange(N), '-', color='gray', lw=lw)
    ax.set_xlabel("time $t$")
    ax.set_xlim([0, T])
    ax.set_yticks(-d_spc * torch.arange(N).numpy())
    ax.set_yticklabels([])
    ax.set_ylim(-N * d_spc, d_spc)
    # ax.set_ylabel("channels $c$")
    ax.set_title("data $\mathbb{E}[X]$")