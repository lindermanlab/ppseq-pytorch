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
    A, B = [], []
    for i in range(N):
        if scale[0][i] > scale[1][i]:
            A.append(i)
        else:
            B.append(i)
    A.sort(key=lambda x: mu[0][x])
    B.sort(key=lambda x: mu[1][x])
    return A + B


def plot_sorted_neurons(data):
    """
    given a data matrix of shape (N, T) Neuron * Time plot the neural spike trains
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(torch.nonzero(data.T)[:, 0], torch.nonzero(data.T)[:, 1], s=10)
    plt.show()


def color_plot(data, b, a, W, scale, mu):
    """
    Plot the neural spike trains and 
    color the spikes into red, blue and black according to their intensities 
    """
    order = sort_neurons(data, scale, mu)
    N, T = data.shape
    D = W.shape[2]
    black_nt = b.view(N, 1).expand(N, T)
    red_nt = F.conv1d(a[[0], :], torch.flip(
        W[[0]].permute(1, 0, 2), [2]), padding=D-1)[:, :-D+1]
    blue_nt = F.conv1d(a[[1], :], torch.flip(
        W[[1]].permute(1, 0, 2), [2]), padding=D-1)[:, :-D+1]
    sum_nt = F.conv1d(a, torch.flip(
        W.permute(1, 0, 2), [2]), padding=D-1)[:, :-D+1]
    assert torch.allclose(red_nt + blue_nt, sum_nt)

    def f(i, j):
        if data[i, j] == 0:
            return -1
        large = max(black_nt[i, j], red_nt[i, j], blue_nt[i, j])
        if black_nt[i, j] >= large:
            return 0
        if red_nt[i, j] == max(red_nt[i, j], blue_nt[i, j]):
            return 1
        return 2

    colors = np.array([[f(i, j) for i in range(N)] for j in range(T)])
    colors = colors[:, order]
    black_indices = np.argwhere(colors == 0)
    red_indices = np.argwhere(colors == 1)
    blue_indices = np.argwhere(colors == 2)

    sizes = 10  # Adjust size based on matrix values

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(black_indices[:, 0],
                black_indices[:, 1], c='black', s=4, alpha=0.7)
    plt.scatter(red_indices[:, 0], red_indices[:, 1], c='red', s=4, alpha=0.7)
    plt.scatter(blue_indices[:, 0], blue_indices[:, 1],
                c='blue', s=4, alpha=0.7)
    plt.title('')
    plt.xlabel('Time')
    plt.ylabel('channel')
    plt.grid(True)
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
                   axs=None):
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

    return fig, axs


def plot_model(templates, amplitude, data, scores=None, lw=2, figsize=(12, 6), spc=0.1):
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
    gs = GridSpec(
        2, K + 1, height_ratios=[1, 2], width_ratios=[1] * K + [2 * K])

    # plot the templates
    t_spc = spc * abs(templates).max()
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
        ax.plot(amplitude[n] - a_spc * n, '-',
                color=palette[n % len(palette)], lw=lw)

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
