
import numpy as np
import importlib

def _load_func(module_name: str, func_name: str):
    mod = importlib.import_module(module_name)
    return getattr(mod, func_name)

def _is_garch(func_name: str):
    return "garch" in (func_name or "").lower()

def _single_draw(
    n_assets,
    n_observations,
    factor_range,
    p_market_range,
    p_factor_range,
    sample_generator,
    pop_generator,
    rng,
    alpha0_range,
    alpha1_range,
    beta1_range,
    sample_is_garch: bool,
):
    factor_min, factor_max = factor_range
    p_market_min, p_market_max = p_market_range
    p_factor_min, p_factor_max = p_factor_range

    n_factors = int(rng.integers(factor_min, factor_max + 1))

    attempt = 0
    while True:
        p_market = float(rng.uniform(p_market_min, p_market_max))
        p_factor = float(rng.uniform(p_factor_min, p_factor_max))
        if p_market + p_factor < 0.99:
            break
        attempt += 1
        if attempt > 100:
            raise RuntimeError("Unable to draw feasible p_market/p_factor within 100 attempts")

    pop_evs = pop_generator(
        n_assets=n_assets,
        n_factors=n_factors,
        p_market=p_market,
        p_factor=p_factor,
    )

    alpha0 = alpha1 = beta1 = np.nan
    if sample_is_garch:
        attempt = 0
        while True:
            alpha0 = float(rng.uniform(*alpha0_range))
            alpha1 = float(rng.uniform(*alpha1_range))
            beta1 = float(rng.uniform(*beta1_range))
            if alpha1 + beta1 < 0.99:
                break
            attempt += 1
            if attempt > 100:
                raise RuntimeError("Unable to draw feasible alpha1,beta1 within 100 attempts")

        sample_evs = sample_generator(
            pop_eigenvalues=pop_evs,
            n_obs=n_observations,
            alpha0=alpha0,
            alpha1=alpha1,
            beta1=beta1,
            rng=rng,
        )
    else:
        sample_evs = sample_generator(
            pop_eigenvalues=pop_evs,
            n_obs=n_observations,
            rng=rng,
        )

    pop_sorted = np.sort(pop_evs)[::-1]
    sample_sorted = np.sort(sample_evs)[::-1]

    meta_row = (n_factors, p_market, p_factor, alpha0, alpha1, beta1)
    return pop_sorted, sample_sorted, meta_row


def batch_synthetic_pairs_chunk_byname(args):
    (
        k,
        n_assets,
        n_observations,
        factor_range,
        p_market_range,
        p_factor_range,
        sample_module,
        sample_func,
        pop_module,
        pop_func,
        seed,
        alpha0_range,
        alpha1_range,
        beta1_range,
    ) = args

    sample_generator = _load_func(sample_module, sample_func)
    pop_generator = _load_func(pop_module, pop_func)

    sample_is_garch = _is_garch(sample_func)

    rng = np.random.default_rng(seed)

    pop_evs = np.empty((k, n_assets), dtype=float)
    sample_evs = np.empty((k, n_assets), dtype=float)

    n_factors_arr = np.empty(k, dtype=int)
    p_market_arr = np.empty(k, dtype=float)
    p_factor_arr = np.empty(k, dtype=float)
    alpha0_arr = np.full(k, np.nan, dtype=float)
    alpha1_arr = np.full(k, np.nan, dtype=float)
    beta1_arr = np.full(k, np.nan, dtype=float)

    for i in range(k):
        pop_i, sample_i, meta_row = _single_draw(
            n_assets=n_assets,
            n_observations=n_observations,
            factor_range=factor_range,
            p_market_range=p_market_range,
            p_factor_range=p_factor_range,
            sample_generator=sample_generator,
            pop_generator=pop_generator,
            rng=rng,
            alpha0_range=alpha0_range,
            alpha1_range=alpha1_range,
            beta1_range=beta1_range,
            sample_is_garch=sample_is_garch,
        )
        pop_evs[i] = pop_i
        sample_evs[i] = sample_i

        nf, pm, pf, a0, a1, b1 = meta_row
        n_factors_arr[i] = nf
        p_market_arr[i] = pm
        p_factor_arr[i] = pf
        alpha0_arr[i] = a0
        alpha1_arr[i] = a1
        beta1_arr[i] = a1
        beta1_arr[i] = b1

    meta = {
        "n_factors": n_factors_arr,
        "p_market": p_market_arr,
        "p_factor": p_factor_arr,
        "alpha0": alpha0_arr,
        "alpha1": alpha1_arr,
        "beta1": beta1_arr,
    }
    return pop_evs, sample_evs, meta
