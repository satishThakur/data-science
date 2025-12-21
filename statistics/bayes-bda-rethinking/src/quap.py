"""
Quadratic Approximation (quap) for Bayesian Inference.

This module provides a fast approximation method for posterior distributions
using the Laplace approximation (second-order Taylor expansion around the MAP).

The key assumption: P(θ | data) ≈ MultivariateNormal(θ̂_MAP, Σ)

Usage:
    from src.quap import quap

    fit = quap(neg_log_posterior, initial_params, param_names)
    samples = fit.sample(n=10_000)
    fit.summary()
"""

import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm
from typing import Callable, Optional, List, Dict, Union


class QuapResult:
    """
    Result object from quadratic approximation (quap).

    Similar to R's rethinking::quap, provides easy interface for:
    - Sampling from posterior approximation
    - Computing credible intervals
    - Displaying summary statistics

    Attributes:
        mean (np.ndarray): MAP estimates (mode of posterior)
        cov (np.ndarray): Covariance matrix
        std (np.ndarray): Standard deviations (sqrt of diagonal of cov)
        param_names (List[str]): Parameter names
        corr (np.ndarray): Correlation matrix
        hessian (np.ndarray): Hessian matrix at MAP
        log_posterior_at_map (float): Log posterior density at MAP
        success (bool): Whether optimization converged
        fit_time (float): Time taken to fit (seconds)
    """

    def __init__(self,
                 mean: np.ndarray,
                 cov: np.ndarray,
                 param_names: List[str],
                 hessian: np.ndarray,
                 log_posterior_at_map: float,
                 success: bool,
                 fit_time: float):

        self.mean = mean
        self.cov = cov
        self.param_names = param_names
        self.hessian = hessian
        self.log_posterior_at_map = log_posterior_at_map
        self.success = success
        self.fit_time = fit_time

        # Create multivariate normal distribution
        self._dist = multivariate_normal(mean=mean, cov=cov)

        # Compute standard deviations
        self.std = np.sqrt(np.diag(cov))

        # Compute correlation matrix
        self.corr = self._compute_correlation()

    def _compute_correlation(self) -> np.ndarray:
        """Compute correlation matrix from covariance."""
        std_matrix = np.outer(self.std, self.std)
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = self.cov / std_matrix
            corr[~np.isfinite(corr)] = 0
        return corr

    def sample(self, n: int = 1000, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Sample from posterior approximation.

        Parameters:
            n (int): Number of samples to draw
            seed (int, optional): Random seed for reproducibility

        Returns:
            pd.DataFrame: Samples with columns for each parameter

        Example:
            >>> fit = quap(neg_log_post, [0, 1], ['mu', 'sigma'])
            >>> samples = fit.sample(n=10_000)
            >>> samples['mu'].mean()
        """
        samples = self._dist.rvs(size=n, random_state=seed)

        # Handle 1D case
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)

        return pd.DataFrame(samples, columns=self.param_names)

    def coef(self) -> Dict[str, float]:
        """
        Return MAP estimates as dictionary.

        Returns:
            Dict[str, float]: Parameter names mapped to MAP values
        """
        return dict(zip(self.param_names, self.mean))

    def vcov(self) -> pd.DataFrame:
        """
        Return covariance matrix as DataFrame.

        Returns:
            pd.DataFrame: Covariance matrix with parameter names
        """
        return pd.DataFrame(self.cov,
                          index=self.param_names,
                          columns=self.param_names)

    def corrcoef(self) -> pd.DataFrame:
        """
        Return correlation matrix as DataFrame.

        Returns:
            pd.DataFrame: Correlation matrix with parameter names
        """
        return pd.DataFrame(self.corr,
                          index=self.param_names,
                          columns=self.param_names)

    def credible_interval(self, prob: float = 0.89) -> pd.DataFrame:
        """
        Compute credible intervals for each parameter.

        Uses Normal approximation, so intervals are symmetric around mean.

        Parameters:
            prob (float): Probability mass (default 0.89, per McElreath)

        Returns:
            pd.DataFrame: DataFrame with 'lower' and 'upper' columns

        Example:
            >>> fit.credible_interval(prob=0.89)
                         lower     upper
            mu         153.456   156.789
            log_sigma    1.234     1.567
        """
        alpha = 1 - prob
        z = norm.ppf(1 - alpha/2)

        lower = self.mean - z * self.std
        upper = self.mean + z * self.std

        return pd.DataFrame({
            'lower': lower,
            'upper': upper
        }, index=self.param_names)

    def summary(self, prob: float = 0.89) -> pd.DataFrame:
        """
        Print and return summary statistics.

        Parameters:
            prob (float): Probability for credible intervals

        Returns:
            pd.DataFrame: Summary statistics for all parameters

        Example:
            >>> fit.summary()
            ======================================================================
            QUAP POSTERIOR APPROXIMATION
            ======================================================================
            Converged: True
            Time: 0.0234 seconds
            Log posterior at MAP: -1234.56

                          mean      std  89%_lower  89%_upper
            mu         155.123    0.823    153.789    156.457
            log_sigma    1.401    0.082      1.270      1.532
            ======================================================================
        """
        ci = self.credible_interval(prob)

        summary_df = pd.DataFrame({
            'mean': self.mean,
            'std': self.std,
            f'{prob:.0%}_lower': ci['lower'],
            f'{prob:.0%}_upper': ci['upper']
        }, index=self.param_names)

        # Print nice summary
        print("=" * 70)
        print("QUAP POSTERIOR APPROXIMATION")
        print("=" * 70)
        print(f"Converged: {self.success}")
        print(f"Time: {self.fit_time:.4f} seconds")
        print(f"Log posterior at MAP: {self.log_posterior_at_map:.2f}")
        print()
        print(summary_df.to_string(float_format=lambda x: f'{x:.4f}'))
        print("=" * 70)

        return summary_df

    def __repr__(self):
        return f"QuapResult({len(self.param_names)} parameters: {', '.join(self.param_names)})"

    def __str__(self):
        return self.__repr__()


def quap(neg_log_posterior: Callable,
         initial_params: Union[np.ndarray, List[float]],
         param_names: Optional[List[str]] = None,
         method: str = 'BFGS',
         hessian_eps: float = 1e-5) -> QuapResult:
    """
    Quadratic approximation (quap) of posterior distribution.

    Approximates P(θ | data) ≈ MultivariateNormal(θ̂_MAP, Σ) where:
    - θ̂_MAP = Maximum A Posteriori estimate (mode)
    - Σ = Inverse Hessian at MAP (covariance)

    This is the Laplace approximation: a second-order Taylor expansion
    of log P(θ | data) around its maximum.

    Parameters:
        neg_log_posterior (callable): Function computing -log P(θ | data).
            Should accept array of parameters, return scalar.
        initial_params (array-like): Starting values for optimization.
        param_names (list of str, optional): Parameter names.
            Default: ['param_0', 'param_1', ...]
        method (str): Optimization method (default: 'BFGS').
            Other options: 'Nelder-Mead', 'Powell', 'CG', 'Newton-CG'
        hessian_eps (float): Step size for finite difference Hessian.
            Default: 1e-5

    Returns:
        QuapResult: Object with methods:
            - .sample(n): Draw samples from posterior
            - .summary(): Display summary statistics
            - .coef(): Get MAP estimates
            - .vcov(): Get covariance matrix
            - .credible_interval(prob): Get credible intervals

    Example:
        >>> def neg_log_post(params):
        ...     mu, log_sigma = params
        ...     sigma = np.exp(log_sigma)
        ...     log_lik = np.sum(norm.logpdf(data, mu, sigma))
        ...     log_prior = norm.logpdf(mu, 0, 10) + norm.logpdf(sigma, 0, 10)
        ...     return -(log_lik + log_prior)
        ...
        >>> fit = quap(neg_log_post, [0, 0], ['mu', 'log_sigma'])
        >>> samples = fit.sample(n=10_000)
        >>> fit.summary()

    Notes:
        - Assumes posterior is approximately multivariate normal
        - Works best with large sample sizes (CLT)
        - Use unconstrained parameterizations (e.g., log(σ) not σ)
        - For bounded parameters, transform to unbounded space first
        - Compare to grid/MCMC to validate approximation
    """
    initial_params = np.asarray(initial_params)

    if param_names is None:
        param_names = [f'param_{i}' for i in range(len(initial_params))]

    # Step 1: Find MAP (Maximum A Posteriori)
    start_time = time.time()

    result = minimize(
        neg_log_posterior,
        x0=initial_params,
        method=method,
        options={'disp': False}
    )

    if not result.success:
        print(f"⚠️  Warning: Optimization may not have converged!")
        print(f"   Message: {result.message}")

    theta_map = result.x

    # Step 2: Compute Hessian via finite differences
    def compute_hessian(f, x, eps=hessian_eps):
        """Compute Hessian matrix via central finite differences."""
        n = len(x)
        hess = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):  # Only compute upper triangle
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()

                x_pp[i] += eps
                x_pp[j] += eps

                x_pm[i] += eps
                x_pm[j] -= eps

                x_mp[i] -= eps
                x_mp[j] += eps

                x_mm[i] -= eps
                x_mm[j] -= eps

                val = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps * eps)
                hess[i, j] = val
                hess[j, i] = val  # Symmetric

        return hess

    hessian = compute_hessian(neg_log_posterior, theta_map)

    # Step 3: Covariance = inverse Hessian
    try:
        cov_matrix = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        print("⚠️  Warning: Hessian is singular! Using pseudo-inverse.")
        cov_matrix = np.linalg.pinv(hessian)

    fit_time = time.time() - start_time

    # Step 4: Return QuapResult object
    return QuapResult(
        mean=theta_map,
        cov=cov_matrix,
        param_names=param_names,
        hessian=hessian,
        log_posterior_at_map=-result.fun,
        success=result.success,
        fit_time=fit_time
    )
