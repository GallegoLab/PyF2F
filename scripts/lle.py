#!/usr/bin/python3.7
# coding=utf-8
"""
Python functions to run a Log Likelihood Estimate on the data.
"""
import time
from rnd import pdf, cdf, rf
from scipy.optimize import minimize
import numpy as np
import warnings
import logging

c = cdf(20, 10)
d = rf(100, c)


def LL(x, d):
    """
    Log likelihood
    Parameters
    ----------
    x: initial values for mu and sigma
    d: list of distances

    Returns
    -------
    min of the LL
    """
    # IMPORTANT NOTE: LL is defined as minus the log likelihood because its values
    # will be optimised through a minimisation process. Hence, the values that maximise
    # the log likelihood are those that minimise LL. This note will be important
    # in the bootstrap method, as the outliers rejected are those whose rejection
    # increases the log likelihood, thus decreases LL
    pp = pdf(d, x[0], x[1])
    return -np.nansum(np.log(pp))


# define the Shannon entropy scoring function
def S(mus, sigmas):
    """
    Shannon's entropy scoring function
    Parameters
    ----------
    mus: distribution of mus to get the scoring
    sigmas: distribution of sigmas to get the scoring

    Returns
    -------
    Score according to each mu and sigma
    """
    N = len(mus)
    M = len(sigmas)

    # convert list to numpy array
    mus = np.array([mus[i][0] for i in range(N)])
    sigmas = np.array([sigmas[i][0] for i in range(M)])

    # conpute the inversion of the deltas. See point 6 in the section
    # Quantification and Statistical Analysis - Image processing and 
    # distance measurements, in Picco et al. Cell 2018
    np.seterr(divide='ignore', invalid='ignore')
    Im = 1 / np.abs(mus[1:] - mus[:-1])
    Is = 1 / np.abs(sigmas[1:] - sigmas[:-1])

    p_m = Im / np.nansum(Im)
    p_s = Is / np.nansum(Is)

    return - p_m * np.log(p_m) - p_s * np.log(p_s)


def optim(LL, x0, d, verbose=False):
    """
    Compute the minimization catching warnings for the initial
    method choice that might not compel with the hess = True
    option
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # optimize the function by minimization
        # LL --> function to minimize
        # x0 --> initial guess for "mu" and "sigma"
        # bnds = ((100, 200), (10, 20))
        opt = minimize(LL, x0, args=d, method="BFGS", bounds=None)

    # extract the parameter estimate and compute their standard
    # error estimate
    x = opt['x']
    se = np.sqrt(np.diagonal(opt['hess_inv']))

    # formatting and output
    mu = [x[0], se[0]]
    sigma = [x[1], se[1]]

    if verbose:  # print results
        logging.info("\n###############################\n"
                     "'-----OPTIM-----'\n"
                     'mu : '
                     + str(np.round(mu[0], 2))
                     + ' +/- '
                     + str(np.round(mu[1], 2))
                     + "\n"
                       'sigma : '
                     + str(np.round(sigma[0], 2))
                     + ' +/- '
                     + str(np.round(sigma[1], 2))
                     )
        print('-----OPTIM-----')
        print('mu : '
              + str(np.round(mu[0], 2))
              + ' +/- '
              + str(np.round(mu[1], 2)))
        print('sigma : '
              + str(np.round(sigma[0], 2))
              + ' +/- '
              + str(np.round(sigma[1], 2)))

    return mu, sigma


def bootstrap(results_dir, LL, x0, d, cutoff=np.nan, reject_lower=2):
    """
    Boostrap to reject possible contaminants in the dataset.

    The assumption is that, if some outliers/contaminants are
    present in the distribution of distances, those will come
    from the tail of the distribution (e.g, largest distances).

    1) Sort distances to look into the right tail.
    2) Assume that 1/3 of the data could have outliers (right tail) --> cutoff
    3) Initial guess for mu & sigma
    - Run Log Likelihood minimization to find mu and sigma close to the optim.
    4) Outlier Search
     - Search mu and sigma more probable according to the distribution by
     minimizing the log likelihood each time a distance is removed from the
     dataset.
     - The values of mu & sigma that minimize the log likelihood are those
     that maximize the likelihood.
     5) Scoring function: Shannon Entropy
     - Use Sh as scoring function to evaluate mu and sigma.
     - Search for mu and sigma that have max in the Sh scoring.

    Parameters
    ----------
    LL: Log likelihood function to minimize.
    x0: initial values for mu and sigma
    d: list of distances (initial data).
    cutoff: to search on the data
    reject_lower: reject high-scored mu lower than this distance (in nm)

    Returns
    -------
    mu, sigma, sh, i_min, i_max, n
    """
    d.sort()
    # sorting of the distances is important, because it will ease the
    # min LL analysis by focusing only on the LL values past the LL max
    # thus on those values coming from the tail of the distribution
    # (i.e. the largest distances)
    if cutoff != cutoff:
        cutoff = 2 * len(d) / 3  # 5 in the R code, but I think it is safe to assume
        # that more than half of the data area good
    else:
        cutoff = cutoff * len(d)

    # order the distance values. Important outliers are in the right tail, 
    # so it is convenient to start removing those
    m, s = optim(LL, x0, d, verbose=True)

    # storage for the incremental changes in the 
    # estimate of mu (m), sigma (s), and the 
    # Shannon entropy output (sh)
    mu = [m]
    sigma = [s]
    dd = [d]

    # start the outliers search
    search = True
    m_tmp = x0[0]  # the m and s for the LL to determine which
    s_tmp = x0[1]  # d is 'more likely' do be rejected

    print("\nStarting search of outliers:\n"
          f"\n\tNumber of distances: {len(d)}\n"
          f"\tCutoff: Explore {100 - (cutoff * 100 / len(d))}% of distance distribution\n"
          f"\tReject lower: {reject_lower} nm\n\n...\n")
    while search:

        # the number of distance measurements left
        n = len(dd[-1])
        # storage vector for the LL estimates when removing a distance
        l = []
        # storage vector for the distance dataset with each removed distance
        dtmp = []
        # Here it computes the LL of all sets lacking a given distance
        for i in range(n):
            dtmp.append([dd[-1][j] for j in range(n) if j != i])  # store all minus distance j
            l.append(LL([m_tmp, s_tmp], dtmp[i]))  # compute LL

        # keep the d that is 'more likely' to belong to the dataset (i.e. max likelihood )
        # NOTE that LL has been defined with a minus for optim to search for a minimum. Thus,
        # here we need to search for a max instead of a minimum, and for a minimum, instead of a max
        l_max = l.index(max(l))  # that is the index of the value most pertinent to the pdf, whose
        # removal maximises the Likelihood ( i.e. minimises the log likelihood)
        ll = l[l_max:]
        i_sel = l_max + ll.index(min(ll))

        new_dataset = dtmp[i_sel]
        # compute a new optimisation on the dataset without the 'less likely' distance
        m, s = optim(LL, x0=x0, d=new_dataset, verbose=True)

        # store the mu and sigma values
        mu.append(m)
        sigma.append(s)
        dd.append(new_dataset)

        if len(new_dataset) < cutoff:
            search = False
            m_tmp = m[0]
            s_tmp = s[0]
        else:
            m_tmp = m[0]
            s_tmp = s[0]

    # Absolute value for mu: because the BesselI function is symmetric, one can
    # get two possible values (negative or positive). We apply the absolute value
    # to avoid getting negative distances. We also impose a max error for the mu
    # estimation to filter out bad estimations

    mu = [np.abs(m) for m in mu]
    sigma = [np.abs(sig) for sig in sigma]

    # compute and store the shannon entropy 
    sh = S(mu, sigma).tolist()
    # print(tuple(zip(sh, mu, sigma, dd))[0])
    # exit()
    nested_results = sorted(tuple(zip(sh, mu, sigma, dd)), key=lambda x: x[0], reverse=True)
    # Filter bad fittings
    max_error = 10  # good mu estimations cannot overpass this error

    nested_results = [result for result in nested_results if result[1][1] < max_error]
    sh, mu, sigma, dd = list(zip(*nested_results))

    # Get max/min scored distribution
    i_max = sh.index(nested_results[0][0])
    i_min = sh.index(nested_results[-1][0])
    high_scored_mu = mu[0][0]
    # print([(i[0], i[1]) for i in nested_results])

    # Check if distance < reject_lower cutoff:
    if high_scored_mu < reject_lower:
        i = 0
        try:
            while high_scored_mu < reject_lower:
                i += 1
                high_scored_mu = nested_results[i][1][0]
                i_max = sh.index(nested_results[i][0])
        except IndexError:
            # NOTICE: the distance distribution seems a bit noisy, this could be a bit fishy
            # It clashed with cutoff without any scored distance on the right side
            # Increasing the cutoff to 2/5 and searching again.
            print('\n\n\tThe distance distribution seems a bit noisy, this step could be a bit fishy\n'
                  f'Searching again with a cutoff of {cutoff / 2} instead of {cutoff}\n')
            time.sleep(3)
            return bootstrap(results_dir, LL, x0, d, cutoff=2 * len(d) / 5, reject_lower=10)
    selected_distribution = dd[i_max]
    num_dist = len(selected_distribution)
    print("\n\n#-#-#-#-#-#-#-#-#-#-#-#-\n"
          "Number of initial distances before fitting: {}\n"
          "Number of distances rejected: {}\n"
          "Final number of distances: {}\n"
          "#-#-#-#-#-#-#-#-#-#-#-#-\n\n".format(len(d), len(d) - num_dist, num_dist))

    print('-----BOOTSTRAP-----')
    print('max Sh: ' + str(np.nanmax(sh)))
    print('mu = ' + str(mu[i_max][0]) + ' +/- ' + str(mu[i_max][1]))
    print('sigma = ' + str(sigma[i_max][0]) + ' +/- ' + str(sigma[i_max][1]))
    print('n = ' + str(num_dist))
    print('min Sh: ' + str(np.nanmin(sh)))
    print('mu = ' + str(mu[i_min][0]) + ' +/- ' + str(mu[i_min][1]))
    print('sigma = ' + str(sigma[i_min][0]) + ' +/- ' + str(sigma[i_min][1]))

    return mu, sigma, sh, i_min, i_max, num_dist, selected_distribution
