import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
from scipy.stats import norm
import seaborn as sns
import yfinance as yf

# Tickers and initial portfolio value
tickers = ['AAPL', 'MSFT', 'IBM', 'TSLA', 'QCOM']
num_stocks = len(tickers)
portfolio = 500000

# Generate random weights and normalize
weights = np.random.random(num_stocks)
weights /= np.sum(weights)

# Download stock data
data = yf.download(tickers, start='2018-01-01')['Adj Close']
print(data.head())

# Calculate log returns
logr = np.log(1 + data.pct_change()[1:])

# Mean and variance of log returns
m = logr.mean()
var = logr.var()

# Drift calculation
drift = m - (0.5 * var)

# Covariance and standard deviation of log returns
covar = logr.cov()
stdev = logr.std()

# Simulation parameters
trials = 1000
days = 100

# Initialize simulation matrix
simulations = np.zeros((days, trials))

#The Cholesky decomposition transforms independent standard normal random variables into correlated variables 
#by using the L in the cholesky decomposition (LL^T), or square root of the matrix. 
#This method ensures the generated variables maintain the desired covariance structure. 
#It is crucial in our Monte Carlo simulations for accurately modeling dependencies among asset returns. 
#By multiplying the Cholesky matrix with independent random variables, we create correlated samples that reflect real-world relationships, enhancing the realism of simulations.

# Cholesky decomposition
chol = np.linalg.cholesky(covar)

# Simulation loop
for i in range(trials):
    # Generate random variables
    Z = norm.ppf(np.random.rand(days, num_stocks))
    daily_returns = np.exp(drift.values + np.dot(Z, chol.T) * stdev.values)
    # Set the first value to portfolio
    simulations[0, i] = portfolio
    # Calculate cumulative returns
    for t in range(1, days):
        simulations[t, i] = simulations[t-1, i] * np.dot(weights, daily_returns[t, :])

# Plot results
plt.figure(figsize=(15, 8))
plt.plot(simulations)
plt.ylabel('Portfolio value')
plt.xlabel('Days')
plt.title(f'Monte Carlo simulation for {portfolio}€\n{tickers}\n{np.round(weights*100,2)}%')
plt.show()

# Histogram of final portfolio values
sns.histplot(simulations[-1], kde=True)
plt.ylabel('Frequency')
plt.xlabel('Portfolio Value')
plt.title('Distribution of Final Portfolio Values')
plt.show()
