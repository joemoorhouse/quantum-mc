import requests as req
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np

def source(ticker):
    # proxies = {'https': 'http://127.0.0.1:8888'}
    # e.g. AAPL and MSFT
    res = req.get('https://uk.finance.yahoo.com/quote/MSFT/history?period1=1490918400&period2=1617235200&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true', verify=False) #, proxies=proxies)

def get_data(ticker):
    data = pd.read_csv("data/" + ticker + ".csv")
    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d") 
    data.set_index("Date", inplace=True)
    return data

def returns(ticker):
    data = get_data(ticker)

    # align to business days; reindex such that missing dates become NaNs
    dates = pd.date_range(data.index[0], data.index[-1], freq=BDay())
    aligned = np.array(data.Close.reindex(dates))

    # calculate 10 day log returns
    days = 10
    rets = np.log(aligned[days:] / aligned[0:-days]) 
    return rets[~np.isnan(rets)] 

def normalize_sort(rets):  
    norm = np.sort(rets)
    # zero mean; historical drift not taken to be good predictor
    norm = norm - np.mean(norm)      
    # normalize returns into units of (maximum-likelihood-estimated) standard deviations
    sig = np.std(norm)
    norm = norm / sig

def ecdf(data, remove_prob_1_point = True):
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n + 1) / n
    if remove_prob_1_point:
        x = x[0:-1]
        y = y[0:-1]
    return(x, y)

def correl(rets1, rets2):
    from scipy.stats import pearsonr
    corr, _ = pearsonr(rets1, rets2) 
    return corr
    #rets1_demean = ret1 - np.mean(rets_AAPL)
    #rets2_demean = rets2 - np.mean(rets_MSFT)
    #np.cov(rets1, rets2, ddof = 0)[0,1] / (np.std(rets1) * np.std(rets2))



