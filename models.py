import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
table = []
tickers = ["AAPL", "NVDA", "GOOGL", "TSLA", "MSFT", "AMZN", "INTC", "VOO", "SPY", "META"];
market_ticker = "^GSPC" # S&P 500
rf_ticker = "^IRX" # Risk Free Rate (Treasury Bill)
data = yf.download(tickers,start="2024-01-01",end="2025-10-17",auto_adjust=True,threads=False)

adj_close = data["Close"]
pd.set_option('display.max_columns', None)
print(adj_close.head())
