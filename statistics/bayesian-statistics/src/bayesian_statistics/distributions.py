"""Functions for working with distributions in Bayesian statistics."""

import numpy as np
from scipy import stats


def beta_mean_credible_interval(alpha, beta, ci=0.95):
    """
    Calculate the mean and credible interval for a Beta distribution.
    
    Parameters
    ----------
    alpha : float
        Alpha parameter of the Beta distribution
    beta : float
        Beta parameter of the Beta distribution
    ci : float, default=0.95
        Credible interval (between 0 and 1)
        
    Returns
    -------
    dict
        Dictionary containing the mean, mode, and credible interval
    """
    mean = alpha / (alpha + beta)
    
    # Mode is only defined when alpha and beta are both > 1
    if alpha > 1 and beta > 1:
        mode = (alpha - 1) / (alpha + beta - 2)
    else:
        mode = None
    
    # Calculate credible interval
    lower_bound = stats.beta.ppf((1-ci)/2, alpha, beta)
    upper_bound = stats.beta.ppf(1-(1-ci)/2, alpha, beta)
    
    return {
        'mean': mean,
        'mode': mode,
        'credible_interval': (lower_bound, upper_bound)
    }


def update_beta_binomial(prior_alpha, prior_beta, n, k):
    """
    Update a Beta prior with Binomial likelihood.
    
    Parameters
    ----------
    prior_alpha : float
        Alpha parameter of the prior Beta distribution
    prior_beta : float
        Beta parameter of the prior Beta distribution
    n : int
        Number of trials in the Binomial likelihood
    k : int
        Number of successes in the Binomial likelihood
        
    Returns
    -------
    tuple
        (posterior_alpha, posterior_beta)
    """
    posterior_alpha = prior_alpha + k
    posterior_beta = prior_beta + n - k
    
    return posterior_alpha, posterior_beta