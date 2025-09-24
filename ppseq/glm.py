import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float


class PoissonGLM(nn.Module):
    """
    # TODO[GLM]: Implement a per-neuron Poisson GLM for the *background* rate.
    #
    # Model:
    #   For neuron n at time t:
    #       η[n,t]   = α[n] +  X_t @ β[n]                  (shared covariates; X: (T, P))
    #       λ_bg[n,t] = exp(η[n,t])                        (log link; returns a *rate*)
    # Optional extensions:
    #   • Per-neuron covariates Z_n[t] with weights γ[n].
    #   • Offset term o[t] (log-exposure) added to η.
    #
    # Shapes:
    #   α: (N,), β: (N, P)
    #   X: (T, P)   (or broadcastable (N, T, P) for per-neuron covariates)
    #   forward(X) -> λ_bg: (N, T)
    #
    # Training target (used inside PPSeq/CAVI M-step):
    #   Given current total rate λ and observed counts x, compute expected background counts
    #       y_bg = x * λ_bg / clamp(λ, 1e-7)
    #   and maximize w.r.t. (α, β):
    #       L(α,β) = Σ_{n,t} [ y_bg[n,t] * η[n,t] - exp(η[n,t]) ]  - λ_l2 * ||β||_2^2
    # Use a handful of gradient steps (Adam/LBFGS) per EM iteration.
    #
    # API:
    #   • __init__(num_neurons: int, covariate_dim: int, l2: float = 0.0, device=None)
    #   • forward(X) -> λ_bg
    #   • neg_loglik(y_bg, X) -> scalar
    #   • step(y_bg, X, optimizer) -> None   (single optimization step)
    """

    def __init__(self, num_neurons: int, covariate_dim: int, l2: float = 0.0, device=None):
        super().__init__()
        # TODO[GLM]: initialize parameters and store regularization.
        # Example:
        #   self.alpha = nn.Parameter(torch.zeros(num_neurons, device=device))
        #   self.beta  = nn.Parameter(torch.zeros(num_neurons, covariate_dim, device=device))
        #   self.l2 = l2
        # Consider initializing `alpha` from log of constant base rate if available.
        raise NotImplementedError("TODO[GLM]: implement PoissonGLM parameters and initialization.")

    def forward(self, X: Float[Tensor, "time covariates"]) -> Float[Tensor, "neurons time"]:
        # TODO[GLM]: compute η = α[:,None] + β @ X^T, then λ_bg = η.exp()
        # Support X of shape (T, P) and broadcast β across neurons.
        raise NotImplementedError

    def neg_loglik(self,
                   y_bg: Float[Tensor, "neurons time"],
                   X: Float[Tensor, "time covariates"]) -> Tensor:
        # TODO[GLM]: return -Σ y_bg * η + Σ exp(η)  + l2 * ||β||^2.
        # Use torch.sum over all axes; ensure shapes align.
        raise NotImplementedError

    # Optional helper for an optimizer step:
    # def step(self, y_bg, X, optimizer):
    #     optimizer.zero_grad()
    #     loss = self.neg_loglik(y_bg, X)
    #     loss.backward()
    #     optimizer.step()
    #     return loss
