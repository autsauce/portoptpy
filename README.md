# portoptpy
Python library for interfacing with and conducting backtests using the Portfolio Optimizer API: https://docs.portfoliooptimizer.io/

### Installation
```
pip install portoptpy
```

### Authentication
An API key is not required to use the Portfolio Optimizer API, however, authenticated users get full access to all endpoints more favorable API limits. Using this library for backtesting purposes will likely require an API key which can be obtained here: https://www.buymeacoffee.com/portfolioopt

### Usage
```
from portoptpy import PortfolioOptimizer

po = PortfolioOptimizer(api_key = 'YOUR_API_KEY')
```

Performing a single minimum variance portfolio optimization using a 63 day lookback period for calculating the covaraince matrix:
```
portfolio = po.construct_minimum_variance_portfolio(symbols = ['SPY','TLT','GLD','BTC-USD'], lookback = 63)
```

Backtesting an equal risk contributions portfolio using an exponentially weighted covariance matrix with decay factor of 0.95:
```
backtest = po.backtest_equal_risk_contributions_portfolio(symbols = ['SPY','TLT','GLD','BTC-USD'], lookback = 63, covariance_type = 'exponential', decay_factor = 0.95)

backtest[['portfolio_equity_curve','benchmark_equity_curve']].plot()
```
