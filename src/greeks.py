"""Black-Scholes higher-order Greeks calculations.

Provides analytical closed-form formulas for gamma, vomma, zomma, and charm
-- the higher-order Greeks most relevant to understanding late-day 0DTE PUT
price explosions.  These Greeks capture the nonlinear feedback loops where
a move in spot or volatility amplifies itself:

- **Gamma**: d(delta)/d(S) -- delta acceleration, highest ATM near expiry.
- **Vomma** (volga): d(vega)/d(sigma) -- how vega itself changes with vol.
  High vomma means option prices explode during vol spikes.
- **Zomma**: d(gamma)/d(sigma) -- how gamma changes with vol.  Creates a
  feedback loop: vol spike -> gamma increase -> bigger hedging demand.
- **Charm**: d(delta)/d(t) -- delta decay, extreme for 0DTE near expiry.

The ``BlackScholesGreeks`` class is used by ``metrics.MetricCalculator``
for per-interval metric computation and by ``put_tracker.PutTracker``
for vomma-based strike selection.

For first-order Greeks (delta, gamma) and IV estimation from prices,
see ``black_scholes.py`` which provides standalone vectorized functions.
"""

import numpy as np
from scipy.stats import norm


class BlackScholesGreeks:
    """Calculator for Black-Scholes Greeks.

    Focuses on higher-order Greeks relevant to late-day PUT explosions:
    - Gamma: Rate of delta change (key for 0DTE)
    - Vomma: Rate of vega change w.r.t. volatility
    - Zomma: Rate of gamma change w.r.t. volatility
    - Charm: Rate of delta decay over time
    """

    def __init__(self, risk_free_rate: float = 0.05):
        """Initialize with risk-free rate.

        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        self.r = risk_free_rate

    def _d1_d2(
        self,
        S: float | np.ndarray,
        K: float | np.ndarray,
        T: float | np.ndarray,
        sigma: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate d1 and d2 for Black-Scholes.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility

        Returns:
            Tuple of (d1, d2) arrays
        """
        # Avoid division by zero
        T = np.maximum(T, 1e-10)
        sigma = np.maximum(sigma, 1e-10)

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2

    def gamma(
        self,
        S: float | np.ndarray,
        K: float | np.ndarray,
        T: float | np.ndarray,
        sigma: float | np.ndarray,
    ) -> np.ndarray:
        """Calculate gamma (d(delta)/d(S)).

        Gamma is highest ATM and explodes near expiration.
        For 0DTE, gamma can be 10-100x normal levels.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility

        Returns:
            Gamma value(s)
        """
        d1, _ = self._d1_d2(S, K, T, sigma)
        sqrt_T = np.sqrt(np.maximum(T, 1e-10))
        sigma = np.maximum(sigma, 1e-10)

        phi_d1 = norm.pdf(d1)
        return phi_d1 / (S * sigma * sqrt_T)

    def vomma(
        self,
        S: float | np.ndarray,
        K: float | np.ndarray,
        T: float | np.ndarray,
        sigma: float | np.ndarray,
    ) -> np.ndarray:
        """Calculate vomma (volga) = d(vega)/d(sigma).

        Vomma measures how vega changes with volatility.
        High vomma means option prices explode during vol spikes.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility

        Returns:
            Vomma value(s)
        """
        d1, d2 = self._d1_d2(S, K, T, sigma)
        sqrt_T = np.sqrt(np.maximum(T, 1e-10))
        sigma = np.maximum(sigma, 1e-10)

        phi_d1 = norm.pdf(d1)
        vega = S * phi_d1 * sqrt_T
        return vega * d1 * d2 / sigma

    def zomma(
        self,
        S: float | np.ndarray,
        K: float | np.ndarray,
        T: float | np.ndarray,
        sigma: float | np.ndarray,
    ) -> np.ndarray:
        """Calculate zomma = d(gamma)/d(sigma).

        Zomma measures how gamma changes with volatility.
        When volatility spikes, gamma can increase dramatically,
        creating a feedback loop for PUT explosions.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility

        Returns:
            Zomma value(s)
        """
        d1, d2 = self._d1_d2(S, K, T, sigma)
        sqrt_T = np.sqrt(np.maximum(T, 1e-10))
        sigma = np.maximum(sigma, 1e-10)

        phi_d1 = norm.pdf(d1)
        gamma = phi_d1 / (S * sigma * sqrt_T)
        return gamma * (d1 * d2 - 1) / sigma

    def charm(
        self,
        S: float | np.ndarray,
        K: float | np.ndarray,
        T: float | np.ndarray,
        sigma: float | np.ndarray,
        is_call: bool | np.ndarray = True,
    ) -> np.ndarray:
        """Calculate charm = d(delta)/d(t).

        Charm measures delta decay over time.
        For 0DTE, charm is extreme as options approach expiry.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility
            is_call: Whether option is a call (True) or put (False)

        Returns:
            Charm value(s)
        """
        d1, d2 = self._d1_d2(S, K, T, sigma)
        sqrt_T = np.sqrt(np.maximum(T, 1e-10))
        sigma = np.maximum(sigma, 1e-10)
        T = np.maximum(T, 1e-10)

        phi_d1 = norm.pdf(d1)
        charm_base = -phi_d1 * (2 * self.r * T - d2 * sigma * sqrt_T) / (
            2 * T * sigma * sqrt_T
        )

        return np.where(is_call, charm_base, -charm_base)

    def calculate_all(
        self,
        S: float | np.ndarray,
        K: float | np.ndarray,
        T: float | np.ndarray,
        sigma: float | np.ndarray,
        is_call: bool | np.ndarray = True,
    ) -> dict[str, np.ndarray]:
        """Calculate all four higher-order Greeks in a single call.

        Returns gamma, vomma, zomma, and charm.  This is the primary entry
        point used by ``MetricCalculator.calculate()`` for per-interval
        Greek computation.

        Args:
            S: Spot price (scalar or array).
            K: Strike price (scalar or array).
            T: Time to expiration in years.
            sigma: Implied volatility.
            is_call: Whether option is a call (affects charm sign only).

        Returns:
            Dictionary with keys ``'gamma'``, ``'vomma'``, ``'zomma'``,
            ``'charm'``, each containing numpy arrays.
        """
        return {
            "gamma": self.gamma(S, K, T, sigma),
            "vomma": self.vomma(S, K, T, sigma),
            "zomma": self.zomma(S, K, T, sigma),
            "charm": self.charm(S, K, T, sigma, is_call),
        }
