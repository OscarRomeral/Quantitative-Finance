import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib import style
from scipy.stats import norm
import yfinance as yf

style.use('seaborn-v0_8')

ticker = 'AAPL'
data = pd.DataFrame()
data[ticker] = yf.download(ticker, start='2012-01-01')['Adj Close']

log_returns = np.log(1+data.pct_change())
u = log_returns.mean()
var = log_returns.var()
drift = u - 0.5*var
#Can be used like this as we are treating the full population and not a sample
stdev = log_returns.std()

#Days that we want to project the simulation
days = 100
#The greater, the specific and better the previssions (different paths)
trials = 10000

#Randomness of the experiment, random variables of the number of days and trials, in order to have as many different options as trials per each day
#It is a normal distribution (Z)
Z = norm.ppf(np.random.rand(days, trials))
#Log normal distribution
daily_returns = np.exp(drift.values + stdev.values * Z)
#Initialize matrix of paths values, as 0 values
prices_path = np.zeros_like(daily_returns)
#Fix last proie (current) as we start at the last one in order to predict future
prices_path[0] = data.iloc[-1]
print(prices_path[0])
#Array to store the mean of all possible paths
mean_paths = np.zeros(days)
mean_paths[0] = data.iloc[-1]

#Price_today = Price_yesterday*exp(drift + stdev * Z)
for t in range(1, days):
    prices_path[t] = prices_path[t-1]*daily_returns[t]
    mean_paths[t] = prices_path[t].mean()




#Plotting
plt.figure(figsize=(15,6))
plt.plot(pd.DataFrame(prices_path))
plt.plot(pd.DataFrame(mean_paths), color="black")
plt.xlabel("Number of days")
plt.ylabel("Price of " + ticker)
#Second plot with the frequencies (hystogram) of the price when we reach the last day
sns.displot(pd.DataFrame(prices_path).iloc[-1])
plt.xlabel("Price at " + str(days) + "days")
plt.ylabel("Frequency")
plt.show()
