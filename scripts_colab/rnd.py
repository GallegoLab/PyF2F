#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python functions describing the pdf and cdf of a distribution
for distances in 2D, following the Eq. (4) in Churchman et al. 2006
"""

import numpy as np
from scipy.special import iv as besselI


# probability density function for distances in 2D. Eq.(4) in Churcham et al. 2006
def pdf(l, mu, sigma):
    """
    Returns the list of probabilities for each value in "l" (dataset)
    given the function parameters "mu" (true distance) and "sigma"
    (variance of the distribution).
    Parameters
    ----------
    l
    mu
    sigma

    Returns
    -------

    """
    # the equation defining the non-gaussian distribution of the values
    def f(r, mu, sigma):

        return (r / sigma ** 2) * np.exp(- (mu ** 2 + r ** 2) / (2 * sigma ** 2)) * besselI(0, r * mu / sigma ** 2)

    # output in function of the input length
    if np.size(l) > 1:
        return [f(x, mu, sigma) for x in l]
    elif np.size(l) == 1:
        return f(l, mu, sigma)

    # comulative density function approx of pdf


def cdf(mu, sigma, dx=0.01, x0=0, x1=2 * 1E2):
    """
    Return the cumulative probability density function
    of the data given the function in churchman
    Parameters
    ----------
    mu
    sigma
    dx: bin size
    x0
    x1

    Returns
    -------

    """
    l = [i for i in np.arange(x0, x1, dx)]  # 20000 values between 0 and 200
    return [l, dx * np.cumsum(pdf(l, mu, sigma))]


# random values from pdf, n values, given the cdf c , adding noise datapoint
# of noise, which are datapoint not obeying the pdf. These data points are
# those that should be removed by the bootstrapping
def rf(n, c, noise=0, noise_mean=np.nan, noise_std=np.nan):
    """
    Method to create dataset of distances (with and without outliers) obeying
    the cdf of Churchman using a random seed of values uniformly distributed
    Parameters
    ----------
    n: size of dataset
    c: cdf
    noise
    noise_mean
    noise_std

    Returns
    -------
    random dataset with and without noise
    """
    # define the vectors to store the distances
    x = np.zeros(n)
    xn = np.zeros(n + noise)

    # generate n random values to be used with the cdf to 
    # determine semi-random distance values obeying the pdf
    y = np.random.rand(n)  # uniform distribution (values [0 - 1])

    for i in range(n):
        # find the cdf value c that is the closest to y
        tmp = min([j for j in c[1] if j > y[i]])
        # map it to its corresponding distance, and store it in x
        x[i] = c[0][c[1].tolist().index(tmp)]

    # if noise mean and std are nan, create default values
    if noise_mean != noise_mean:
        noise_mean = max(x)
    if noise_std != noise_std:
        noise_std = noise_mean / 3

    xn[: i + 1] = x
    xn[i + 1:] = np.abs(np.random.normal(
        loc=noise_mean,
        scale=noise_std,
        size=noise))  # abs is to avoid possible neg values. Neg distances do not exist

    # shuffle and return xn so that the noisy 
    # values are not only at the end of the vector
    np.random.shuffle(xn)
    return x, xn.tolist()
