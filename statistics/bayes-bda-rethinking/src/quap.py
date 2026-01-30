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

        # Parameter transformations
        self._transformations = {}  # {old_name: {'new_name': str, 'transform': callable}}
        self._original_param_names = list(param_names)  # Keep original names

    def _compute_correlation(self) -> np.ndarray:
        """Compute correlation matrix from covariance."""
        std_matrix = np.outer(self.std, self.std)
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = self.cov / std_matrix
            corr[~np.isfinite(corr)] = 0
        return corr

    def transform_param(self,
                       param_name: str,
                       new_name: str,
                       transform: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Register a parameter transformation to be applied to all outputs.

        This allows you to work with constrained parameters (e.g., σ > 0) while
        QUAP internally uses unconstrained parameterizations (e.g., log(σ)).

        After calling this method, all QUAP methods will show the transformed
        parameter instead of the original.

        Parameters:
            param_name (str): Name of parameter to transform (e.g., 'log_sigma')
            new_name (str): Name for transformed parameter (e.g., 'sigma')
            transform (callable): Transformation function (e.g., np.exp)

        Example:
            >>> fit = quap(neg_log_post, [0, 0], ['mu', 'log_sigma'])
            >>> fit.transform_param('log_sigma', 'sigma', np.exp)
            >>> fit.summary()  # Now shows 'sigma' instead of 'log_sigma'

        Notes:
            - Transformations are applied via sampling, so statistics reflect
              the true distribution (e.g., σ ~ Log-Normal if log(σ) ~ Normal)
            - This is more accurate than transforming means/SDs directly
            - Credible intervals will be asymmetric for asymmetric transforms
        """
        if param_name not in self._original_param_names:
            raise ValueError(
                f"Parameter '{param_name}' not found. "
                f"Available: {self._original_param_names}"
            )

        if param_name in self._transformations:
            old_new_name = self._transformations[param_name]['new_name']
            print(f"⚠️  Warning: Overwriting existing transformation "
                  f"'{param_name}' -> '{old_new_name}' with '{new_name}'")

        self._transformations[param_name] = {
            'new_name': new_name,
            'transform': transform
        }

        print(f"✓ Registered transformation: {param_name} -> {new_name}")

    def _get_display_names(self) -> List[str]:
        """Get parameter names with transformations applied."""
        names = []
        for name in self._original_param_names:
            if name in self._transformations:
                names.append(self._transformations[name]['new_name'])
            else:
                names.append(name)
        return names

    def _apply_transformations(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Apply registered transformations to samples."""
        result = samples.copy()

        # Apply each transformation
        for old_name, trans_info in self._transformations.items():
            new_name = trans_info['new_name']
            transform_fn = trans_info['transform']

            # Transform the column
            result[new_name] = transform_fn(result[old_name].values)

            # Drop the old column
            result = result.drop(columns=[old_name])

        # Reorder columns to match original order
        display_names = self._get_display_names()
        return result[display_names]

    def sample(self, n: int = 1000, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Sample from posterior approximation.

        If parameter transformations have been registered via transform_param(),
        they will be applied to the samples.

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

        df = pd.DataFrame(samples, columns=self._original_param_names)

        # Apply transformations if any
        if self._transformations:
            df = self._apply_transformations(df)

        return df

    def coef(self) -> Dict[str, float]:
        """
        Return parameter estimates as dictionary.

        For untransformed parameters, returns MAP estimates.
        For transformed parameters, returns posterior mean via sampling
        (more accurate than transforming the MAP for asymmetric transforms).

        Returns:
            Dict[str, float]: Parameter names mapped to estimates

        Notes:
            If transformations are registered, this samples 10,000 times
            to compute posterior means for transformed parameters.
        """
        if not self._transformations:
            # No transformations: return MAP directly
            return dict(zip(self._original_param_names, self.mean))

        # With transformations: compute via sampling
        samples = self.sample(n=10000, seed=42)
        return {name: samples[name].mean() for name in samples.columns}

    def vcov(self) -> pd.DataFrame:
        """
        Return covariance matrix as DataFrame.

        For untransformed parameters, returns analytical covariance.
        For transformed parameters, computes sample covariance.

        Returns:
            pd.DataFrame: Covariance matrix with parameter names
        """
        if not self._transformations:
            return pd.DataFrame(self.cov,
                              index=self._original_param_names,
                              columns=self._original_param_names)

        # With transformations: compute via sampling
        samples = self.sample(n=10000, seed=42)
        cov_matrix = samples.cov()
        return cov_matrix

    def corrcoef(self) -> pd.DataFrame:
        """
        Return correlation matrix as DataFrame.

        For untransformed parameters, returns analytical correlation.
        For transformed parameters, computes sample correlation.

        Returns:
            pd.DataFrame: Correlation matrix with parameter names
        """
        if not self._transformations:
            return pd.DataFrame(self.corr,
                              index=self._original_param_names,
                              columns=self._original_param_names)

        # With transformations: compute via sampling
        samples = self.sample(n=10000, seed=42)
        corr_matrix = samples.corr()
        return corr_matrix

    def credible_interval(self, prob: float = 0.89) -> pd.DataFrame:
        """
        Compute credible intervals for each parameter.

        For untransformed parameters, uses analytical Normal quantiles.
        For transformed parameters, computes intervals via sampling
        (correctly handles asymmetric intervals from asymmetric transforms).

        Parameters:
            prob (float): Probability mass (default 0.89, per McElreath)

        Returns:
            pd.DataFrame: DataFrame with 'lower' and 'upper' columns

        Example:
            >>> fit.credible_interval(prob=0.89)
                         lower     upper
            mu         153.456   156.789
            sigma        0.234     0.567
        """
        if not self._transformations:
            # No transformations: analytical formula
            alpha = 1 - prob
            z = norm.ppf(1 - alpha/2)

            lower = self.mean - z * self.std
            upper = self.mean + z * self.std

            return pd.DataFrame({
                'lower': lower,
                'upper': upper
            }, index=self._original_param_names)

        # With transformations: compute via sampling
        samples = self.sample(n=10000, seed=42)
        alpha = 1 - prob

        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        intervals = []
        for col in samples.columns:
            lower = np.percentile(samples[col], lower_percentile)
            upper = np.percentile(samples[col], upper_percentile)
            intervals.append({'lower': lower, 'upper': upper})

        return pd.DataFrame(intervals, index=samples.columns)

    def summary(self, prob: float = 0.89) -> pd.DataFrame:
        """
        Print and return summary statistics.

        If transformations are registered, statistics are computed via sampling
        to correctly reflect the transformed distributions.

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
            mu      155.123    0.823    153.789    156.457
            sigma     0.823    0.059      0.726      0.923
            ======================================================================
        """
        ci = self.credible_interval(prob)

        if not self._transformations:
            # No transformations: use analytical moments
            summary_df = pd.DataFrame({
                'mean': self.mean,
                'std': self.std,
                f'{prob:.0%}_lower': ci['lower'],
                f'{prob:.0%}_upper': ci['upper']
            }, index=self._original_param_names)
        else:
            # With transformations: compute via sampling
            samples = self.sample(n=10000, seed=42)
            means = samples.mean()
            stds = samples.std()

            summary_df = pd.DataFrame({
                'mean': means,
                'std': stds,
                f'{prob:.0%}_lower': ci['lower'],
                f'{prob:.0%}_upper': ci['upper']
            })

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
        display_names = self._get_display_names()
        return f"QuapResult({len(display_names)} parameters: {', '.join(display_names)})"

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
