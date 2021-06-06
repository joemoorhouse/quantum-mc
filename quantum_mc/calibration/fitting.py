
from . import time_series as ts
from scipy.stats import norm
import numpy as np

def get_cdf_data(ticker):
    rets = ts.returns(ticker) # 10 day log returns
    rets = rets - np.mean(rets)      
    # normalize returns into units of (maximum-likelihood-estimated) standard deviations
    sig = np.std(rets)
    rets = rets / sig
    return (ts.ecdf(rets), sig)

def get_std(ticker):
    rets = ts.returns(ticker) # 10 day log returns
    sig = np.std(rets)
    return sig

def get_fit_data(ticker, norm_to_rel = True):
    ((x, cdf), sigma) = get_cdf_data(ticker)
    if norm_to_rel:
        x = (np.exp(sigma * x) - 1) / sigma
    return (norm.ppf(cdf), x)

def fit_piecewise_linear(x, y):
    """ Fit to piecewise linear
    x contains returns in units of standard deviation and y the transformed returns (also as number of standard deviations)
    """
    from sklearn.linear_model import LinearRegression
    from scipy import optimize
    import math

    def piecewise_linear3(x, x0, y0, rdx0, k0, k1, k2):
        dx0 = rdx0 * rdx0 # use square root of delta as a parameter to keep dx0 > 0
        x1 = x0 + dx0
        y1 = y0 + k1 * dx0 
        return np.piecewise(x, [x < x0, (x >= x0) & (x < x1), x >= x1], [lambda x:k0 * x + y0 - k0 * x0, lambda x:k1 * x + y0 - k1 * x0, lambda x:k2 * x + y1 - k2 * x1])

    p, *_ = optimize.curve_fit(piecewise_linear3, x, y)     
    return lambda z : piecewise_linear3(z, *p)

def fit_piecewise_cubic(x, y):
    """ Fit to piecewise cubic splines, mainly to demonstrate that if we want a smooth CDF and PDF we can get one 
    x contains returns in units of standard deviation and y the transformed returns (also as number of standard deviations)
    """
    import scipy
    coeffs = scipy.interpolate.splrep(x, y, task = -1, t = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])) #, xb = -4, xe = 4)
    return lambda z : scipy.interpolate.splev(z, coeffs)


