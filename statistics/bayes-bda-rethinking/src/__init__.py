"""
Bayesian Data Analysis utilities for Statistical Rethinking.

This package contains reusable implementations of statistical methods
used throughout the course.
"""

from .quap import quap, QuapResult

__all__ = ['quap', 'QuapResult']
