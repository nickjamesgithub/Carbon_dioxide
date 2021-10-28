import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import scipy.optimize

# Write Important Functions
def calculate_portfolio_sharpe(weights, returns, covariance):
    # Calculate Negative Sharpe Ratio
    weights = np.matrix(weights)
    returns = np.dot(weights, returns)
    risk = np.sqrt(np.dot(np.dot(weights, covariance), np.transpose(weights)))
    return - returns/risk

def minimize_returns(weights, returns, covariance):
    # Calculate Negative Risk
    weights = np.matrix(weights)
    returns = np.dot(weights, returns)
    risk = np.sqrt(np.dot(np.dot(weights, covariance), np.transpose(weights)))
    return - risk

def calculate_returns(weights, returns):
    weights = np.matrix(weights)
    returns = np.dot(weights, returns)
    return - returns

def rand_weights(n):
    k = np.random.rand(n)
    return k / sum(k)

def sharpe_optimiser(data, window, counter, lb, ub):
    returns = [np.mean(data[window:counter,0]), np.mean(data[window:counter,1]), np.mean(data[window:counter,2]), np.mean(data[window:counter,3])]
    data = np.transpose(data[:,[0,1,2,3]].astype("float"))
    cov = np.cov(data)
    rf_rate = 0
    w0 = [1/4, 1/4, 1/4, 1/4]

    bnds = ((lb, ub), (lb, ub), (lb, ub), (lb, ub))
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1.0})
    res= scipy.optimize.minimize(calculate_portfolio_sharpe, w0, args=(returns, cov), method='SLSQP',constraints=cons, bounds=bnds)
    w_g = res.x
    # portfolio_returns = np.dot(w_g,returns)
    # portfolio_std = np.sqrt(np.dot(np.dot(w_g,cov),w_g))
    return w_g

def generate_piecewise_time_series(n, cp_frequency, mean_1, sd_1, mean_2, sd_2):
    samples_1 = []
    changepoints = []
    flags_stored = []
    flag = 0
    counter = 0
    while len(samples_1) < n:
        if flag == 0:
            samples_1.append(sp.norm.rvs(mean_1,sd_1,1))
            flags_stored.append(0)
        if flag == 1:
            samples_1.append(sp.norm.rvs(mean_2,sd_2,1))
            flags_stored.append(1)
        counter += 1
        if counter % cp_frequency == 0 and flags_stored[-1] == 0:
            flag = 1
            changepoints.append(counter)
        if counter % cp_frequency == 0 and flags_stored[-1] == 1:
            flag = 0
            changepoints.append(counter)
    return samples_1, changepoints

# Dynamic Optimisation backtest
def backtest_function(data, rebalancing_freq, lb, ub):
    equity_weight_historic = []
    dff_weight_historic = []
    arf_weight_historic = []
    pairs_weight_historic = []
    dynamic_optimiser_cumulative_returns = [100]
    dynamic_opt_returns = []

    counter = 0
    for i in np.arange(6, len(data), 1):
        if counter % rebalancing_freq == 0:
            equity_weight, arf_weight, dff_weight, pairs_weight = sharpe_optimiser(data, i - 6, i, lb, ub)

        equity_weighted_return = equity_weight * (1 + data[i, 0])
        arf_weighted_return = arf_weight * (1 + data[i, 1])
        dff_weighted_return = dff_weight * (1 + data[i, 2])
        pairs_weighted_return = pairs_weight * (1 + data[i, 3])
        total = equity_weighted_return + arf_weighted_return + dff_weighted_return + pairs_weighted_return
        dynamic_opt_returns.append(total - 1)
        dynamic_optimiser_cumulative_returns.append(dynamic_optimiser_cumulative_returns[-1] * total)

        # Append weights to determine allocation over time
        equity_weight_historic.append(equity_weight)
        dff_weight_historic.append(arf_weight)
        arf_weight_historic.append(dff_weight)
        pairs_weight_historic.append(pairs_weight)

        counter += 1

    dynamic_sharpe_ratio = np.mean(dynamic_opt_returns / np.std(dynamic_opt_returns))

    return dynamic_opt_returns, dynamic_optimiser_cumulative_returns, dynamic_sharpe_ratio, \
           equity_weight_historic, dff_weight_historic, arf_weight_historic, pairs_weight_historic
