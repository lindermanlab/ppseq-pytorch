import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import torch.nn.functional as F

# Import ppseq modules
from ppseq.plotting import plot_model, plot_synthetic_model
from ppseq.model import GLMPPSeq
from notebooks.generate_data import generate_data_glm_full

# Set page config
st.set_page_config(page_title="PPSeq Hyperparameter Tuning", layout="wide")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    st.warning('Could not find a GPU. Defaulting to CPU instead.')

# Progress bar utility
try:
    from fastprogress.fastprogress import progress_bar
except Exception:
    try:
        from tqdm.auto import tqdm as progress_bar
    except Exception:
        def progress_bar(x):
            return x


def generate_scale_values():
    """Generate scale values from 1e-4 to 9e-1 with 0.5 increments within each order of magnitude."""
    values = []
    # For each power of 10: -4, -3, -2, -1
    for power in range(-4, 0):
        base = 10 ** power
        # Generate values: 1.0, 1.5, 2.0, 2.5, ..., 9.0 times the base
        for multiplier in np.arange(1.0, 9.5, 0.5):
            values.append(multiplier * base)
    return values


@st.cache_resource
def generate_synthetic_data():
    """Generate synthetic data once and cache it."""
    seed = 2
    N, T, K, D, P = 20, 1000, 2, 10, 2
    num_segments = 4
    dense_strength_range = (0.4, 1.0)
    sparse_strength_range = (0.2, 0.8)
    
    # Generate single session
    (X_sessions, lambdas_sessions, alpha, beta, X_cov_sessions,
     true_a_sessions, true_w, lambda_bg_sessions, correct_sequences) = generate_data_glm_full(
        N=N, T=T, K=K, D=D, P=P, 
        num_sessions=1,
        seed=seed,
        num_segments=num_segments,
        dense_strength_range=dense_strength_range,
        sparse_strength_range=sparse_strength_range,
    )
    
    # Extract single session data
    X_full = X_sessions[0]
    X_seq = correct_sequences[0]
    data = X_full.to(device)
    num_neurons, num_timesteps = data.shape
    
    return {
        'X_full': X_full,
        'X_seq': X_seq,
        'data': data,
        'num_neurons': num_neurons,
        'num_timesteps': num_timesteps,
        'true_a': true_a_sessions[0],
        'true_w': true_w,
        'K': K,
        'D': D,
    }


def train_glmppseq(data, num_neurons, K, D, l1, l1_amp):
    """Train GLMPPSeq model with specified hyperparameters."""
    torch.manual_seed(0)
    model = GLMPPSeq(
        num_templates=K,
        num_neurons=num_neurons,
        template_duration=D,
        alpha_a0=1.5,
        beta_a0=0.2,
        alpha_b0=1,
        beta_b0=0.1,
        alpha_t0=1.2,
        beta_t0=0.1,
        n_covariates=2,
        l1=l1,
        l1_amp=l1_amp,
        device=device,
    )
    
    # Train the model
    with st.spinner('Training GLMPPSeq model...'):
        lps, amplitudes = model.fit(data, num_iter=100)
    
    return model, amplitudes, lps


def main():
    st.title("PPSeq Hyperparameter Tuning")
    st.write("Tune `l1` and `l1_amp` hyperparameters for GLMPPSeq model training")
    
    # Generate scale values
    scale_values = generate_scale_values()
    
    # Create sidebar for hyperparameters
    st.sidebar.header("Hyperparameters")
    
    # Create sliders with the scale values
    l1_idx = st.sidebar.select_slider(
        "l1",
        options=range(len(scale_values)),
        value=0,
        format_func=lambda x: f"{scale_values[x]:.2e}"
    )
    l1 = scale_values[l1_idx]
    
    l1_amp_idx = st.sidebar.select_slider(
        "l1_amp",
        options=range(len(scale_values)),
        value=0,
        format_func=lambda x: f"{scale_values[x]:.2e}"
    )
    l1_amp = scale_values[l1_amp_idx]
    
    # Display selected values
    st.sidebar.write(f"**Selected l1:** {l1:.2e}")
    st.sidebar.write(f"**Selected l1_amp:** {l1_amp:.2e}")
    
    # Generate synthetic data (cached)
    synthetic_data = generate_synthetic_data()
    
    # Create two columns for plots
    col1, col2 = st.columns(2)
    
    # Plot synthetic model (ground truth) in the first column
    with col1:
        st.subheader("Ground Truth (Synthetic Model)")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        plot_synthetic_model(
            synthetic_data['true_a'],
            synthetic_data['true_w'],
            synthetic_data['X_full']
        )
        st.pyplot(plt.gcf())
        plt.close()
    
    # Train GLMPPSeq and plot results in the second column
    with col2:
        st.subheader("GLMPPSeq Results")
        
        # Train the model
        model, amplitudes, lps = train_glmppseq(
            synthetic_data['data'],
            synthetic_data['num_neurons'],
            synthetic_data['K'],
            synthetic_data['D'],
            l1,
            l1_amp
        )
        
        # Plot the model results
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        plot_model(
            model.templates.cpu(),
            amplitudes.cpu(),
            synthetic_data['X_full'].cpu(),
            spc=0.33
        )
        st.pyplot(plt.gcf())
        plt.close()
    
    # Plot log likelihood curve
    st.subheader("Training Log Likelihood")
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(lps.detach().cpu())
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Log Likelihood')
    ax3.set_title('GLMPPSeq Training Progress')
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    plt.close()


if __name__ == "__main__":
    main()

