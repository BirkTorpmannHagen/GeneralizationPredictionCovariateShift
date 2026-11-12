"""Batch-level unsupervised-accuracy-estimation statistics on a softmax matrix.

Dependency-free (numpy/scipy only) so that both the GPU-side collector
(collect_softmax_stats.py) and offline analyses (experiments/label_shift.py) can use it.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def nuclear_norm(P):
    N, C = P.shape
    G = P.T @ P  # C x C, cheaper than an SVD of N x C
    eig = np.clip(np.linalg.eigvalsh(G), 0, None)
    return float(np.sqrt(eig).sum() / np.sqrt(min(N, C) * N))


def _labels_from_marginal(pi, n, rng):
    """n one-hot labels whose empirical distribution matches pi as closely as possible."""
    raw = pi * n
    counts = np.floor(raw).astype(int)
    rem = n - counts.sum()
    if rem > 0:
        frac = raw - counts
        # break ties randomly so no class is systematically favoured
        order = np.lexsort((rng.random(len(pi)), -frac))
        counts[order[:rem]] += 1
    return np.repeat(np.arange(len(pi)), counts)


def cot_error(P, pi, rng):
    """COT predicted error: W(P_pred, P_source_labels) with cost ||p - e_y||_inf."""
    N, C = P.shape
    labels = _labels_from_marginal(pi, N, rng)
    order = np.argsort(-P, axis=1)
    top1 = order[:, 0]
    m1 = P[np.arange(N), top1]
    m2 = P[np.arange(N), order[:, 1]] if C > 1 else np.zeros(N)
    # ||p - e_y||_inf = max(1 - p_y, max_{c != y} p_c)
    other_max = np.where(top1[:, None] == np.arange(C)[None, :], m2[:, None], m1[:, None])
    cost_by_class = np.maximum(1.0 - P, other_max)  # N x C
    cost = cost_by_class[:, labels]                  # N x N
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())


# ----------------------------------------------------------------------------
# MaNo (Xie et al., NeurIPS 2024): entry-wise L_p norm of the normalized logit matrix
# ----------------------------------------------------------------------------
MANO_P = 4
MANO_ETA = 5.0


def mano_phi(Z):
    """Selection criterion: mean negative log-softmax over all N*K entries (proportional, up to a
    constant, to the average KL divergence from the uniform distribution to the predictions)."""
    Z = np.asarray(Z, dtype=np.float64)
    m = Z.max(1, keepdims=True)
    lse = m + np.log(np.exp(Z - m).sum(1, keepdims=True))
    return float(-(Z - lse).mean())


def mano(Z, p=MANO_P, eta=MANO_ETA, mode="auto"):
    """MaNo score of a batch of logits Z [N, K]; higher => higher accuracy.

    S = ( 1/(N K) * sum_{i,k} sigma(z_i)_k^p )^(1/p), with sigma(z) = v(z) / ||v(z)||_1 and
    v = exp (softmax) if Phi(Z) > eta, else the 2nd-order Taylor polynomial v(z) = 1 + z + z^2/2
    (always positive). mode = "auto" (paper default, chosen per batch), "softmax" or "taylor".
    """
    Z = np.asarray(Z, dtype=np.float64)
    if mode == "auto":
        mode = "taylor" if mano_phi(Z) <= eta else "softmax"
    if mode == "taylor":
        v = 1.0 + Z + 0.5 * Z ** 2
    elif mode == "softmax":
        v = np.exp(Z - Z.max(1, keepdims=True))
    else:
        raise ValueError(mode)
    Q = v / v.sum(1, keepdims=True)
    return float(np.mean(Q ** p) ** (1.0 / p))


# ----------------------------------------------------------------------------
# Dispersion score (Xie et al., NeurIPS 2023): inter-class dispersion of penultimate features
# ----------------------------------------------------------------------------
def dispersion(H, yhat, K=None):
    """Dispersion score of a batch: log( sum_j m_j ||mu_bar - mu_j||^2 / (k - 1) ), where the
    clusters j are the predicted classes yhat, mu_j their feature centroids, m_j their sizes and
    mu_bar the mean feature of the batch (Eq. 4 of Xie et al. 2023; raw features, as in their
    Algorithm 1). k is the number of classes of the model (K) if given, else the number of
    predicted classes present. Higher => higher accuracy."""
    H = np.asarray(H, dtype=np.float64)
    yhat = np.asarray(yhat)
    mu = H.mean(0)
    cls, inv, m = np.unique(yhat, return_inverse=True, return_counts=True)
    sums = np.zeros((len(cls), H.shape[1]))
    np.add.at(sums, inv, H)
    cent = sums / m[:, None]
    k = K if K is not None else len(cls)
    num = float((m * ((cent - mu) ** 2).sum(1)).sum())
    return float(np.log(max(num, 1e-12) / max(k - 1, 1)))


def projection_matrix(d, k=256, seed=0):
    """Seeded Gaussian random projection R [d, k] with E||xR||^2 = ||x||^2 (Johnson-Lindenstrauss),
    used to store compact per-sample features for offline re-batching. Squared distances, and
    hence the Dispersion score's between-class term, are preserved in expectation."""
    return np.random.default_rng(seed).standard_normal((d, k)) / np.sqrt(k)
