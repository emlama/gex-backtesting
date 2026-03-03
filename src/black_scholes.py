"""Vectorized Black-Scholes pricing and implied volatility estimation.

This module provides numpy-vectorized implementations of core Black-Scholes
functions used by the backtesting pipeline. All functions operate on numpy
arrays rather than scalar values, enabling ~100x speedup vs row-by-row
pandas apply() when processing thousands of option trades per interval.

The primary consumers are:
- ``gex_calculator.py``: Uses ``calculate_greeks()`` to compute IV, delta,
  and gamma for GEX aggregation.
- ``metrics.py``: Uses ``calculate_gamma()`` indirectly via ``greeks.py``
  for higher-order Greek calculations.

Functions here compute first-order Greeks (delta, gamma) and implied
volatility. For higher-order Greeks (vomma, zomma, charm), see
``greeks.py`` which provides a class-based ``BlackScholesGreeks``
calculator.
"""

import numpy as np
from scipy.stats import norm


def calculate_d1(
    spot: np.ndarray,
    strike: np.ndarray,
    tte: np.ndarray,
    rate: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """Calculate the d1 term of the Black-Scholes formula.

    d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))

    Both ``tte`` and ``sigma`` are clamped to a minimum of 1e-10 to avoid
    division-by-zero when options are at or very near expiration.

    Args:
        spot: Underlying (SPX) price array.
        strike: Option strike price array.
        tte: Time to expiration in years (fractional).
        rate: Risk-free interest rate (annualized, e.g. 0.05 for 5%).
        sigma: Implied volatility array.

    Returns:
        Array of d1 values, same shape as inputs.
    """
    sqrt_tte = np.sqrt(np.maximum(tte, 1e-10))
    sigma_safe = np.maximum(sigma, 1e-10)
    return (np.log(spot / strike) + (rate + 0.5 * sigma_safe**2) * tte) / (sigma_safe * sqrt_tte)


def calculate_gamma(
    spot: np.ndarray,
    strike: np.ndarray,
    tte: np.ndarray,
    rate: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """Calculate option gamma (second derivative of price w.r.t. spot).

    Gamma = phi(d1) / (S * sigma * sqrt(T))

    Gamma is identical for calls and puts (put-call parity).  The result
    is clipped to [0, 1] to suppress numerical noise from near-zero
    denominators when TTE or sigma are extremely small.

    Args:
        spot: Underlying price array.
        strike: Strike price array.
        tte: Time to expiration in years.
        rate: Risk-free interest rate.
        sigma: Implied volatility array.

    Returns:
        Gamma values clipped to [0, 1].
    """
    sqrt_tte = np.sqrt(np.maximum(tte, 1e-10))
    sigma_safe = np.maximum(sigma, 1e-10)
    d1 = calculate_d1(spot, strike, tte, rate, sigma_safe)
    gamma = norm.pdf(d1) / (spot * sigma_safe * sqrt_tte)
    return np.clip(gamma, 0, 1)


def calculate_delta(
    spot: np.ndarray,
    strike: np.ndarray,
    tte: np.ndarray,
    rate: float,
    sigma: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    """Calculate option delta (first derivative of price w.r.t. spot).

    For calls: delta = N(d1)        (range [0, 1])
    For puts:  delta = N(d1) - 1    (range [-1, 0])

    Args:
        spot: Underlying price array.
        strike: Strike price array.
        tte: Time to expiration in years.
        rate: Risk-free interest rate.
        sigma: Implied volatility array.
        is_call: Boolean array -- True for calls, False for puts.

    Returns:
        Delta values for each option.
    """
    d1 = calculate_d1(spot, strike, tte, rate, sigma)
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1
    return np.where(is_call, call_delta, put_delta)


def estimate_iv_from_price(
    spot: np.ndarray,
    strike: np.ndarray,
    tte: np.ndarray,
    rate: float,
    price: np.ndarray,
    is_call: np.ndarray,
    iterations: int = 5,
) -> np.ndarray:
    """Estimate implied volatility from observed option prices using Newton-Raphson.

    Uses a vectorized Newton-Raphson iteration to invert the Black-Scholes
    formula for IV.  The algorithm:

    1. **Initial guess**: sigma_0 = price / (0.4 * S * sqrt(T)), clamped
       to [0.05, 2.0].  This Brenner-Subrahmanyam approximation provides
       a reasonable starting point for ATM options.
    2. **Iteration**: sigma_{n+1} = sigma_n - (BS(sigma_n) - price) / vega
       where vega = S * phi(d1) * sqrt(T).  Vega is floored at 1e-10 to
       prevent division by zero when options are deep OTM or near expiry.
    3. **Clamping**: After each iteration, sigma is clamped to [0.05, 3.0]
       to prevent divergence.

    Five iterations is typically sufficient for convergence to within ~0.01
    of true IV for liquid strikes.  Deep OTM options with very small prices
    may not converge precisely, but the clamp bounds keep results reasonable.

    Args:
        spot: Underlying price array.
        strike: Strike price array.
        tte: Time to expiration in years.
        rate: Risk-free interest rate.
        price: Observed option mid-price array.
        is_call: Boolean array -- True for calls, False for puts.
        iterations: Number of Newton-Raphson steps (default 5).

    Returns:
        Estimated implied volatility array, clamped to [0.05, 3.0].
    """
    sqrt_tte = np.sqrt(np.maximum(tte, 1e-10))
    sigma = np.clip(price / (0.4 * spot * sqrt_tte), 0.05, 2.0)

    for _ in range(iterations):
        d1 = calculate_d1(spot, strike, tte, rate, sigma)
        d2 = d1 - sigma * sqrt_tte

        call_price = spot * norm.cdf(d1) - strike * np.exp(-rate * tte) * norm.cdf(d2)
        put_price = strike * np.exp(-rate * tte) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        bs_price = np.where(is_call, call_price, put_price)

        vega = spot * norm.pdf(d1) * sqrt_tte
        vega_safe = np.maximum(vega, 1e-10)
        sigma = sigma - (bs_price - price) / vega_safe
        sigma = np.clip(sigma, 0.05, 3.0)

    return sigma


def calculate_greeks(
    spot: np.ndarray,
    strike: np.ndarray,
    tte: np.ndarray,
    rate: float,
    price: np.ndarray,
    is_call: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate first-order Greeks from observed option prices.

    Convenience function that chains IV estimation -> gamma -> delta in a
    single call.  This is the main entry point used by ``gex_calculator.py``
    when processing raw trade data that has prices but no pre-computed Greeks.

    Args:
        spot: Underlying price array.
        strike: Strike price array.
        tte: Time to expiration in years.
        rate: Risk-free interest rate.
        price: Observed option price array.
        is_call: Boolean array -- True for calls, False for puts.

    Returns:
        Dictionary with keys ``'iv'``, ``'delta'``, ``'gamma'``, each
        containing a numpy array of the same shape as the inputs.
    """
    iv = estimate_iv_from_price(spot, strike, tte, rate, price, is_call)
    gamma = calculate_gamma(spot, strike, tte, rate, iv)
    delta = calculate_delta(spot, strike, tte, rate, iv, is_call)
    return {"iv": iv, "delta": delta, "gamma": gamma}
