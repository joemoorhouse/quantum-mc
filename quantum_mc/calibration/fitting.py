
from numpy.lib.function_base import _i0_1
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

def get_correl(ticker1, ticker2):
    rets1 = ts.returns(ticker1)
    rets2 = ts.returns(ticker2)
    return np.corrcoef(rets1, rets2)

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

    (x0, y0, rdx0, k0, k1, k2) = p
    dx0 = rdx0 * rdx0
    x1 = x0 + dx0
    coeffs = (x0, y0, x1, k0, k1, k2)

    return (lambda z : piecewise_linear3(z, *p), coeffs)

def fit_piecewise_cubic(x, y):
    """ Fit to piecewise cubic splines, mainly to demonstrate that if we want a smooth CDF and PDF we can get one 
    x contains returns in units of standard deviation and y the transformed returns (also as number of standard deviations)
    """
    import scipy
    coeffs = scipy.interpolate.splrep(x, y, task = -1, t = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])) #, xb = -4, xe = 4)
    return lambda z : scipy.interpolate.splev(z, coeffs)


def convert_to_integer(trans, coeffs, x_min = -4.0, x_max = 4.0, y_min = None, y_max = None):
    """ Possible way to convert from float to integer arithmetic """
    
    (x0, y0, x1, k0, k1, k2) = coeffs

    nbits1 = 3 # number of qubits for normal distribution 
    nbits2 = 2 # number of qubits added for transform

    if not y_min:
        y_min = trans(x_min)
    
    if not y_max:
        y_max = trans(x_max)

    scale_x = (x_max - x_min) / (2**nbits1 - 1)
    scale_y = (y_max - y_min) / (2**(nbits1 + nbits2) - 1)

    def grad(a):
        return round(a * scale_x / scale_y)

    i = np.arange(0, 2**nbits1)

    def i_to_x(i):
        return scale_x * i + x_min

    def x_to_i(x):
        return round((x - x_min) / scale_x)

    def j_to_y(j):
        return scale_y * j + y_min

    def y_to_j(y):
        return round((y - y_min) / scale_y)

    i_0 = x_to_i(x0)
    i_1 = x_to_i(x1)
    a0 = grad(k0)
    a1 = grad(k1)
    a2 = grad(k2)
    j_0 = y_to_j(y0)
    j_1 = a1 * (i_1 - i_0) + j_0
    b0 = j_0 - a0 * i_0
    b1 = j_1 - a1 * i_1
    b2 = j_1 - a2 * i_1

    def i_to_j(i):
        if i <= i_0:
            return a0 * i + b0
        elif i <= i_1:
            return a1 * i + b1
        else:
            return a2 * i + b2

    return (i_0, i_1, a0, a1, a2, b0, b1, b2, i_to_j, i_to_x, j_to_y)
