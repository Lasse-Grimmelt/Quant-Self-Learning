
import numpy as np

def generate_pop_eigenvalues(n_assets=500, n_factors=6, p_market=0.3, p_factor=0.4):
    if p_market + p_factor > 1:
        raise ValueError("Sum of market and factor needs to be less than 1")

    pop_eigenvalues = np.zeros(n_assets)

    # Market mode
    pop_eigenvalues[0] = n_assets * p_market

    # Factor signals
    factors_idx = np.arange(1, n_factors + 1)
    decay_weights = 1.0 / (factors_idx ** 1.3)
    normalized_weights = decay_weights / np.sum(decay_weights)
    pop_eigenvalues[1:n_factors+1] = n_assets * p_factor * normalized_weights

    # Noise bulk
    noise_variance = n_assets * (1 - p_market - p_factor)
    remaining_count = n_assets - (n_factors + 1)
    pop_eigenvalues[n_factors+1:] = noise_variance / remaining_count

    return pop_eigenvalues


def generate_sp500_synthetic_norm(pop_eigenvalues, n_obs=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    data = rng.multivariate_normal(
        np.zeros(len(pop_eigenvalues)),
        np.diag(pop_eigenvalues),
        size=n_obs
    )

    sample_cov = np.cov(data, rowvar=False)
    sample_eigenvalues = np.linalg.eigvalsh(sample_cov)

    # Normalisation
    sample_eigenvalues *= (np.sum(pop_eigenvalues) / np.sum(sample_eigenvalues))
    return sample_eigenvalues[::-1]


def generate_sp500_synthetic_garch(pop_eigenvalues, n_obs=1000, alpha0=0.05, alpha1=0.1, beta1=0.85, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    p = len(pop_eigenvalues)

    epsilon = rng.standard_normal((n_obs, p)) * np.sqrt(pop_eigenvalues)

    r_t = np.zeros((n_obs, p))
    h_sq = np.zeros(n_obs + 1)

    h_sq[0] = alpha0 / (1 - alpha1 - beta1)

    for t in range(n_obs):
        h_t = np.sqrt(h_sq[t])
        r_t[t, :] = h_t * epsilon[t, :]

        shock = np.mean(r_t[t, :] ** 2)
        h_sq[t + 1] = alpha0 + alpha1 * shock + beta1 * h_sq[t]

    sample_cov = (r_t.T @ r_t) / n_obs
    sample_evs = np.linalg.eigvalsh(sample_cov)[::-1]

    # Normalisation
    sample_evs *= (np.sum(pop_eigenvalues) / np.sum(sample_evs))
    return sample_evs
