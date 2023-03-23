from typing import List, Dict, Any
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import requests
import aiohttp
import asyncio

class PortfolioOptimizer:
    """A class to handle portfolio optimization using the Portfolio Optimizer API."""

    def __init__(self, api_key: str = None):
        """
        Initializes the PortfolioOptimizer class.
        
        :param api_key: The API key for Portfolio Optimizer API, defaults to None.
        :type api_key: str, optional
        """
        self._api_key = api_key
        self._base_url = 'https://api.portfoliooptimizer.io/v1'
        self._returns = pd.DataFrame()
        self._headers = {}
        if self._api_key:
            self._headers["X-API-Key"] = self._api_key

    def get_api_limits(self) -> Dict[str, Any]:
        """
        Gets the limits of the Portfolio Optimizer API.
        
        :return: A dictionary containing the API status headers.
        :rtype: Dict[str, Any]
        """

        response = requests.request(
            method='GET', url=self._base_url, headers=self._headers)
        return dict(response.headers)

    def _make_request(self, endpoint: str, method: str = "GET", params: Dict[str, Any] = None, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Makes an HTTP request to the Portfolio Optimizer API.
        
        :param endpoint: The API endpoint to call.
        :type endpoint: str
        :param method: The HTTP method to use, defaults to "GET".
        :type method: str, optional
        :param params: URL parameters for the API call, defaults to None.
        :type params: Dict[str, Any], optional
        :param data: JSON data to send with the request, defaults to None.
        :type data: Dict[str, Any], optional
        :return: The API response as a dictionary.
        :rtype: Dict[str, Any]
        """

        url = f"{self._base_url}{endpoint}"
        response = requests.request(
            method, url, headers=self._headers, params=params, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            print(response.text)
            response.raise_for_status()

    def _get_asset_prices(self, symbols: List[str]) -> pd.DataFrame:
        """
        Retrieves historical prices for the given symbols.
        
        :param symbols: List of stock symbols.
        :type symbols: List[str]
        :return: DataFrame containing historical prices.
        :rtype: pd.DataFrame
        """
        prices = yf.download(tickers=symbols, ignore_tz=True,
                             progress=False, auto_adjust=True)["Close"]
        equities = [s for s in symbols if '-' not in s]

        if equities:
            prices.dropna(axis=0, how='all', subset=equities, inplace=True)

        if prices.iloc[-1].isna().sum() != 0:
            prices = prices.iloc[:-1, :]

        prices.index = pd.to_datetime(prices.index).rename("date")
        return prices

    def _get_asset_returns(self, symbols: List[str], lookback: int) -> pd.DataFrame:
        """
        Retrieves historical returns for the given symbols.

        :param symbols: List of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :return: DataFrame containing historical returns.
        :rtype: pd.DataFrame
        """
        if not all(s in self._returns.columns for s in symbols) or any(self._returns[s].tail(lookback).isna().sum() >= lookback for s in symbols):
            self._returns = self._get_asset_prices(symbols).pct_change(1)
        
        nan_columns = self._returns.columns[self._returns.tail(lookback).isna().any()].tolist()
        if nan_columns:
            raise ValueError(f'''Yahoo Finance did not have enough historical data available based on the specified lookback period for the following symbols: {nan_columns}.
            Try the following options to correct this issue:
            1. Decrease the lookback period.
            2. Verify that the provided symbols are correct and supported by Yahoo.
            3. Provide a custom returns DataFrame with enough data for your desired lookback period.''')
            
        return self._returns

    def _validate_returns_format(self, returns: pd.DataFrame, symbols: List[str], lookback: int) -> bool:
        """
        Validates the provided returns DataFrame.

        :param returns: DataFrame containing historical returns.
        :type returns: pd.DataFrame
        :param symbols: List of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :return: True if the returns DataFrame is valid, False otherwise.
        :rtype: bool
        """
        if returns is None:
            return False

        if not isinstance(returns, pd.DataFrame):
            warnings.warn('The data passed for returns is not a pandas DataFrame. Defaulting to Yahoo Finance for returns data.')
            return False

        if not set(symbols).issubset(set(returns.columns)):
            warnings.warn('The symbols passed do not all exist in the DataFrame columns that was provided for returns. Defaulting to Yahoo Finance for returns data.')
            return False

        if len(returns) < lookback:
            warnings.warn('The DataFrame that was provided for returns does not have enough rows for the specified lookback period. Defaulting to Yahoo Finance for returns data.')
            return False

        for col in returns.columns.tolist():
            non_nan_values = returns[col].dropna()
            if len(non_nan_values) < lookback:
                warnings.warn('The DataFrame provided for returns does not have enough non-NaN values in each column for the specified lookback period. Defaulting to Yahoo Finance for returns data.')
                return False

        return True


    def _get_position_history(self, symbols: List[str], lookback: int, frequency: str = 'month_start', method: str = 'any', returns: pd.DataFrame = None) -> pd.DataFrame:
                    """
                    Returns a DataFrame containing the position history of the given symbols, filtered by lookback and returns DataFrame.

                    :param symbols: List of stock symbols.
                    :type symbols: List[str]
                    :param lookback: The number of rows to consider from the returns DataFrame.
                    :type lookback: int
                    :param returns: DataFrame containing historical returns of the symbols.
                    :type returns: pd.DataFrame
                    :return: DataFrame containing the position history of the given symbols.
                    :rtype: pd.DataFrame
                    """
                    if not self._validate_returns_format(returns, symbols, lookback):
                        returns = self._get_asset_returns(symbols, lookback)

                    returns = returns[symbols].copy()

                    df = []
                    for date in returns.index:
                        valid_symbols = [s for s in symbols if pd.notna(returns.loc[:date, s]).tail(lookback + 1).all()]
                        df.append([date] + valid_symbols)

                    df = pd.DataFrame(df)
                    df.columns = ['date'] + [f'position_{i + 1}' for i in range(df.shape[1] - 1)]
                    df.set_index('date', inplace=True)

                    if method == 'any':
                        df = df.loc[df.T.notna().sum()[df.T.notna().sum() > 1].index].copy()
                    elif method == 'all':
                        df.dropna(inplace=True)

                    frequencies = {'week_start': 'W-MON',
                                    'week_end': 'W-FRI',
                                    'month_start': 'MS',
                                    'month_end': 'M',
                                    'quarter_start': 'QS-JAN',
                                    'quarter_end': 'QS-DEC',
                                    'year_start': 'AS-JAN',
                                    'year_end': 'AS-DEC'}

                    freq = frequencies.get(frequency)

                    if frequency == 'day':
                        rebalance_days = df.index.tolist()

                    elif 'end' in frequency:
                        date_range = pd.date_range(df.index.min(),df.index.max(),freq=freq)
                        rebalance_days = []

                        for i in range(0,len(date_range)):
                            rebalance_days.append(df.loc[:date_range[i]].index.max())

                    elif 'start' in frequency:
                        date_range = pd.date_range(df.index.min(),df.index.max(),freq=freq)
                        rebalance_days = []

                        for i in range(0,len(date_range)):
                            rebalance_days.append(df.loc[date_range[i]:].index.min())

                    else:
                        raise ValueError('Invalid frequency. Valid frequencies are: day, week_start, week_end, month_start, month_end, quarter_start, quarter_end, year_start, year_end.')

                    df = df.loc[rebalance_days]

                    return df

    def _get_momentum_position_history(self, symbols: List[str], mom_lookback: List[int], num_positions: int, frequency: str = 'month_start', method: str = 'any', returns: pd.DataFrame = None) -> pd.DataFrame:
                    """
                    Returns a DataFrame containing the position history of the given symbols, filtered by lookback and returns DataFrame.

                    :param symbols: List of stock symbols.
                    :type symbols: List[str]
                    :param mom_lookback: A list of lookback periods (in months) to use for calculating the momentum.
                    :type mom_lookback: List[int]
                    :param num_positions: The number of positions to hold at each rebalance.
                    :type num_positions: int
                    :param returns: DataFrame containing historical returns of the symbols.
                    :type returns: pd.DataFrame
                    :return: DataFrame containing the position history of the given symbols.
                    :rtype: pd.DataFrame
                    """
                    # Get asset prices
                    prices = self._get_asset_prices(symbols)

                    # Calculate momentum weights
                    r = range(1, len(mom_lookback) + 1)
                    mom_weights = [(i / sum(r)) * 10 for i in r]
                    mom_weights.reverse()

                    # Calculate momentum percentage changes
                    pcts = [prices.shift(1).pct_change(int(p * 21)) for p in mom_lookback]
                    pcts = sum([pcts[i] * mom_weights[i] for i in range(len(mom_lookback))]) / 10
                    momentums = pcts[symbols]

                    # Create position history DataFrame
                    if method == 'any':
                        df = [[index] + row.dropna().sort_values(ascending=False).head(num_positions).index.tolist() for index, row in momentums.iterrows() if row.dropna().shape[0] >= num_positions]
                    elif method == 'all':
                        df = [[index] + row.dropna().sort_values(ascending=False).head(num_positions).index.tolist() for index, row in momentums.iterrows() if row.dropna().shape[0] == momentums.shape[1]]
                    df = pd.DataFrame(df)
                    df.columns = ['date'] + [f'position_{i + 1}' for i in range(df.shape[1] - 1)]
                    df.set_index('date', inplace=True)

                    frequencies = {'week_start': 'W-MON',
                                    'week_end': 'W-FRI',
                                    'month_start': 'MS',
                                    'month_end': 'M',
                                    'quarter_start': 'QS-JAN',
                                    'quarter_end': 'QS-DEC',
                                    'year_start': 'AS-JAN',
                                    'year_end': 'AS-DEC'}

                    freq = frequencies.get(frequency)

                    if frequency == 'day':
                        rebalance_days = df.index.tolist()

                    elif 'end' in frequency:
                        date_range = pd.date_range(df.index.min(),df.index.max(),freq=freq)
                        rebalance_days = []

                        for i in range(0,len(date_range)):
                            rebalance_days.append(df.loc[:date_range[i]].index.max())

                    elif 'start' in frequency:
                        date_range = pd.date_range(df.index.min(),df.index.max(),freq=freq)
                        rebalance_days = []

                        for i in range(0,len(date_range)):
                            rebalance_days.append(df.loc[date_range[i]:].index.min())

                    else:
                        raise ValueError('Invalid frequency. Valid frequencies are: day, week_start, week_end, month_start, month_end, quarter_start, quarter_end, year_start, year_end.')

                    df = df.loc[rebalance_days]

                    return df

    def _generate_portfolio_equity_curve(self,weights_history: pd.DataFrame, returns: pd.DataFrame, start_equity: float = 100000, fee: float = 0.0) -> pd.DataFrame:
        """
        Generates the equity curve for a portfolio given its weights history and returns data.

        Parameters:
        weights_history (pd.DataFrame): A dataframe containing the weights history for each asset in the portfolio.
        returns (pd.DataFrame): A dataframe containing the returns data for each asset in the portfolio.
        start_equity (float): The starting equity of the portfolio (default is 100000).
        fee (float): The trading fee for the portfolio (default is 0.0).

        Returns:
        pd.DataFrame: A dataframe containing the equity curve for the portfolio.
        """
        
        n = weights_history[[c for c in weights_history.columns if c.startswith('position_')]].shape[1]
        rebalance_days = weights_history.index.tolist()

        df = returns.index.to_frame().set_index('date').join(weights_history,how='left').loc[weights_history.index.min():].ffill()

        positions_columns = [f'position_{i + 1}' for i in range(n)]
        weights_columns = [f'weight_{i + 1}' for i in range(n)]
        values_columns = [f'value_{i + 1}' for i in range(n)]
        returns_columns = [f'returns_{i + 1}' for i in range(n)]

        for i in range(0,len(df.index.values)):
            today = df.index[i]
            cp = df.loc[today,positions_columns].dropna().tolist()
            if today == df.index.min():
                e = start_equity
                w = df.loc[today,weights_columns].values
                r = returns.loc[today,cp].values
                s = n - r.size
                if s > 0:
                    r = np.pad(r,(0,s))
                df.loc[today,returns_columns] = r
                v = (e * w * (r+1))
                df.loc[today,values_columns] = v
                df.loc[today,'portfolio_equity_curve'] = sum(v)

            elif today in rebalance_days:
                yday = df.index[i-1]
                e = df.loc[yday,'portfolio_equity_curve'] * (1-(fee/12))
                w = df.loc[today,weights_columns].values
                r = returns.loc[today,cp].values
                s = n - r.size
                if s > 0:
                    r = np.pad(r,(0,s))
                df.loc[today,returns_columns] = r
                v = (e * w * (r+1))
                df.loc[today,values_columns] = v
                df.loc[today,'portfolio_equity_curve'] = sum(v)
                
            else:
                yday = df.index[i-1]
                r = returns.loc[today,cp].values
                s = n - r.size
                if s > 0:
                    r = np.pad(r,(0,s))
                df.loc[today,returns_columns] = r
                v = (df.loc[yday,values_columns] * (r+1))
                df.loc[today,values_columns] = v
                df.loc[today,'portfolio_equity_curve'] = sum(v)

        return df

    def _generate_benchmark_equity_curve(self,
        benchmark: str,
        start_equity: float,
        start_date: str,
        end_date: str
    ) -> pd.Series:
        """
        Generates a benchmark equity curve based on a provided ticker symbol and date range.

        :param benchmark: Ticker symbol for the benchmark index or stock.
        :type benchmark: str
        :param start_equity: Starting equity value.
        :type start_equity: float
        :param start_date: Start date for the equity curve in the format "YYYY-MM-DD".
        :type start_date: str
        :param end_date: End date for the equity curve in the format "YYYY-MM-DD".
        :type end_date: str
        :return: Cumulative return of the benchmark equity curve.
        :rtype: pd.Series
        """
        returns = yf.download(
            tickers=benchmark,
            ignore_tz=True,
            auto_adjust=True,
            progress=False
        )['Close'].pct_change().dropna() + 1
        
        equity_curve = (returns.loc[start_date:end_date].cumprod() * start_equity).rename('benchmark_equity_curve')
        equity_curve.index = equity_curve.index.rename('date')
        
        return equity_curve

    def _get_covariance_matrix(self, symbols: List[str], lookback: int, returns: pd.DataFrame = None) -> pd.DataFrame:
        """
        Retrieves the covariance matrix of the given symbols.
        
        :param symbols: List of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :param returns: Optional user-provided returns DataFrame, defaults to None.
        :type returns: pd.DataFrame, optional
        :return: The covariance matrix as a DataFrame.
        :rtype: pd.DataFrame
        """
        if not self._validate_returns_format(returns, symbols, lookback):
            returns = self._get_asset_returns(symbols,lookback)

        returns = returns[symbols].tail(lookback)
        matrix = returns.cov()

        return matrix

    def _get_exponential_covariance_matrix(self, symbols: List[str], lookback: int, decay_factor: float, returns: pd.DataFrame = None) -> pd.DataFrame:
        """
        Retrieves the exponentially weighted covariance matrix for a list of assets.

        :param symbols: List of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :param decay_factor: The decay factor to use in the exponential weighting, between 0 and 1.
        :type decay_factor: float
        :param returns: Optional user-provided returns DataFrame, defaults to None.
        :type returns: pd.DataFrame, optional
        :return: The exponentially weighted covariance matrix as a DataFrame.
        :rtype: pd.DataFrame
        """
        if not self._validate_returns_format(returns, symbols, lookback):
            returns = self._get_asset_returns(symbols,lookback)

        returns = returns[symbols].tail(lookback)

        data = {
            "assets": [
                {"assetReturns": returns[col].tolist()}
                for col in returns.columns
            ],
            "decayFactor": decay_factor
        }

        response = self._make_request(
            '/assets/covariance/matrix/exponentially-weighted', method='POST', data=data)
        matrix = pd.DataFrame(
            response['assetsCovarianceMatrix'], index=symbols, columns=symbols)

        return matrix

    def construct_equal_risk_contributions_portfolio(self,
                                           symbols: List[str],
                                           lookback: int,
                                           covariance_type: str = 'standard',
                                           decay_factor: float = 0.94,
                                           returns: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculates an equal risk contributions portfolio for the given symbols using the Portfolio Optimizer API.

        :param symbols: List of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :param covariance_type: The type of covariance matrix to use. Either 'standard' or 'exponential'.
        :type covariance_type: str, optional
        :param decay_factor: The decay factor to use in the exponential weighting, between 0 and 1. Only used if covariance_type='exponential'.
        :type decay_factor: float, optional
        :param returns: Optional user-provided returns DataFrame, defaults to None.
        :type returns: pd.DataFrame, optional
        :return: The optimized portfolio weights as a Pandas DataFrame.
        :rtype: pd.DataFrame
        """
        if not self._validate_returns_format(returns, symbols, lookback):
            returns = self._get_asset_returns(symbols,lookback)

        if covariance_type == 'standard':
            matrix = self._get_covariance_matrix(symbols, lookback, returns)
        elif covariance_type == 'exponential':
            matrix = self._get_exponential_covariance_matrix(symbols, lookback, decay_factor, returns)
        else:
            raise ValueError("covariance_type must be either 'standard' or 'exponential'")

        data = {
            "assets": len(symbols),
            "assetsCovarianceMatrix": matrix.values.tolist(),
            "constraints": {}
        }

        endpoint = "/portfolio/optimization/equal-risk-contributions"
        response = self._make_request(endpoint, method="POST", data=data)

        weights = response['assetsWeights']
        df = pd.DataFrame({'symbol': symbols, 'weight': weights}).set_index('symbol')

        return df

    def construct_hierarchical_risk_parity_portfolio(self, symbols: List[str],
                                           lookback: int,
                                           covariance_type: str = 'standard',
                                           decay_factor: float = 0.94,
                                           returns: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculates a hierarchical risk parity portfolio for the given symbols using the Portfolio Optimizer API.

        :param symbols: List of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :param covariance_matrix_type: Optional type of covariance matrix to use, either 'standard' or 'exponential', defaults to 'standard'.
        :type covariance_matrix_type: str, optional
        :param decay_factor: Optional decay factor for the exponentially weighted covariance matrix, defaults to None.
        :type decay_factor: float, optional
        :param returns: Optional user-provided returns DataFrame, defaults to None.
        :type returns: pd.DataFrame, optional
        :return: The optimized portfolio weights as a Pandas DataFrame.
        :rtype: pd.DataFrame
        """
        if not self._validate_returns_format(returns, symbols, lookback):
            returns = self._get_asset_returns(symbols,lookback)

        if covariance_type == 'standard':
            matrix = self._get_covariance_matrix(symbols, lookback, returns)
        elif covariance_type == 'exponential':
            matrix = self._get_exponential_covariance_matrix(symbols, lookback, decay_factor, returns)
        else:
            raise ValueError("covariance_type must be either 'standard' or 'exponential'")

        data = {
            "assets": len(symbols),
            "assetsCovarianceMatrix": matrix.values.tolist(),
            "constraints": {}
        }

        endpoint = "/portfolio/optimization/hierarchical-risk-parity"
        response = self._make_request(endpoint, method="POST", data=data)

        weights = response['assetsWeights']
        df = pd.DataFrame({'symbol': symbols, 'weight': weights}).set_index('symbol')

        return df

    def construct_hierarchical_risk_parity_cluster_based(self, symbols: List[str], lookback: int,
                                               covariance_type: str = 'standard', decay_factor: float = 0.94,
                                               returns: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculates a hierarchical risk parity clustering-based portfolio for the given symbols using the Portfolio Optimizer API.

        :param symbols: List of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :param covariance_type: The type of covariance matrix to use, either 'standard' or 'exponential', defaults to 'standard'.
        :type covariance_type: str, optional
        :param decay_factor: The decay factor to use in the exponential weighting, between 0 and 1, defaults to None.
        :type decay_factor: float, optional
        :param returns: Optional user-provided returns DataFrame, defaults to None.
        :type returns: pd.DataFrame, optional
        :return: The optimized portfolio weights as a Pandas DataFrame.
        :rtype: pd.DataFrame
        """
        if not self._validate_returns_format(returns, symbols, lookback):
            returns = self._get_asset_returns(symbols,lookback)

        if covariance_type == 'standard':
            matrix = self._get_covariance_matrix(symbols, lookback, returns)
        elif covariance_type == 'exponential':
            matrix = self._get_exponential_covariance_matrix(symbols, lookback, decay_factor, returns)
        else:
            raise ValueError("covariance_type must be either 'standard' or 'exponential'")

        data = {
            "assets": len(symbols),
            "assetsCovarianceMatrix": matrix.values.tolist(),
            "constraints": {}
        }

        endpoint = "/portfolio/optimization/hierarchical-risk-parity/clustering-based"
        response = self._make_request(endpoint, method="POST", data=data)

        weights = response['assetsWeights']
        df = pd.DataFrame(
            {'symbol': symbols, 'weight': weights}).set_index('symbol')

        return df

    def construct_most_diversified_portfolio(self, symbols: List[str], lookback: int,
                                covariance_type: str = 'standard',
                                decay_factor: float = 0.94,
                                returns: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculates the most diversified portfolio for the given symbols using the Portfolio Optimizer API.

        :param symbols: List of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :param covariance_type: The type of covariance matrix to use, either 'standard' or 'exponential', defaults to 'standard'.
        :type covariance_type: str, optional
        :param decay_factor: The decay factor to use in the exponential weighting, between 0 and 1. Only used if covariance_type='exponential'.
        :type decay_factor: float, optional
        :param returns: Optional user-provided returns DataFrame, defaults to None.
        :type returns: pd.DataFrame, optional
        :return: The optimized portfolio weights as a Pandas DataFrame.
        :rtype: pd.DataFrame
        """
        if not self._validate_returns_format(returns, symbols, lookback):
            returns = self._get_asset_returns(symbols, lookback)

        if covariance_type == 'standard':
            matrix = self._get_covariance_matrix(symbols, lookback, returns)
        elif covariance_type == 'exponential':
            matrix = self._get_exponential_covariance_matrix(symbols, lookback, decay_factor, returns)
        else:
            raise ValueError("covariance_type must be either 'standard' or 'exponential'")

        data = {
            "assets": len(symbols),
            "assetsCovarianceMatrix": matrix.values.tolist()
        }

        endpoint = "/portfolio/optimization/most-diversified"
        response = self._make_request(endpoint, method="POST", data=data)

        weights = response['assetsWeights']
        df = pd.DataFrame({'symbol': symbols, 'weight': weights}).set_index('symbol')

        return df

    def construct_minimum_variance_portfolio(self, symbols: List[str], lookback: int,
                                covariance_type: str = 'standard',
                                decay_factor: float = 0.94,
                                returns: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculates the minimum variance portfolio for the given symbols using the Portfolio Optimizer API.

        :param symbols: List of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :param covariance_type: The type of covariance matrix to use, either 'standard' or 'exponential', defaults to 'standard'.
        :type covariance_type: str, optional
        :param decay_factor: The decay factor to use in the exponential weighting, between 0 and 1. Only used if covariance_type='exponential'.
        :type decay_factor: float, optional
        :param returns: Optional user-provided returns DataFrame, defaults to None.
        :type returns: pd.DataFrame, optional
        :return: The optimized portfolio weights as a Pandas DataFrame.
        :rtype: pd.DataFrame
        """
        if not self._validate_returns_format(returns, symbols, lookback):
            returns = self._get_asset_returns(symbols, lookback)

        if covariance_type == 'standard':
            matrix = self._get_covariance_matrix(symbols, lookback, returns)
        elif covariance_type == 'exponential':
            matrix = self._get_exponential_covariance_matrix(symbols, lookback, decay_factor, returns)
        else:
            raise ValueError("covariance_type must be either 'standard' or 'exponential'")

        data = {
            "assets": len(symbols),
            "assetsCovarianceMatrix": matrix.values.tolist()
        }

        endpoint = "/portfolio/optimization/minimum-variance"
        response = self._make_request(endpoint, method="POST", data=data)

        weights = response['assetsWeights']
        df = pd.DataFrame({'symbol': symbols, 'weight': weights}).set_index('symbol')

        return df

    def _async_get_covariance_matrix(self, symbols: List[str], lookback: int, returns: pd.DataFrame = None) -> pd.DataFrame:
        """
        Retrieves the covariance matrix of the given symbols.

        :param symbols: List of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :param returns: Optional user-provided returns DataFrame, defaults to None.
        :type returns: pd.DataFrame, optional
        :return: The covariance matrix as a DataFrame.
        :rtype: pd.DataFrame
        """

        returns = returns[symbols].tail(lookback)
        matrix = returns.cov()

        return matrix

    async def _async_get_exponential_covariance_matrix(self, symbols: List[str], lookback: int, decay_factor: float, returns: pd.DataFrame = None) -> pd.DataFrame:
        """
        Retrieves the exponentially weighted covariance matrix for a list of assets.

        :param symbols: List of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :param decay_factor: The decay factor to use in the exponential weighting, between 0 and 1.
        :type decay_factor: float
        :param returns: Optional user-provided returns DataFrame, defaults to None.
        :type returns: pd.DataFrame, optional
        :return: The exponentially weighted covariance matrix as a DataFrame.
        :rtype: pd.DataFrame
        """

        returns = returns[symbols].tail(lookback)

        data = {
            "assets": [
                {"assetReturns": returns[col].tolist()}
                for col in returns.columns
            ],
            "decayFactor": decay_factor
        }

        endpoint = '/assets/covariance/matrix/exponentially-weighted'

        async with aiohttp.ClientSession() as session:
            async with session.post(url=self._base_url + endpoint, headers=self._headers, json=data) as r:

                try:
                    r.raise_for_status()
                except:
                    print(f"Response text: {await r.text()}")
                    r.raise_for_status()
                    
                response = await r.json()

        matrix = pd.DataFrame(response['assetsCovarianceMatrix'], index=symbols, columns=symbols)

        return matrix


    async def backtest_equal_risk_contributions_portfolio(self,
                                                         symbols: List[str],
                                                         lookback: int, 
                                                         covariance_type: str = 'standard', 
                                                         decay_factor: float = 0.94,
                                                         frequency: str = 'month_start',
                                                         method='any', 
                                                         start_equity: float = 100000,
                                                         benchmark: str = 'SPY',
                                                         fee: float = 0.0,
                                                         returns: pd.DataFrame = None,
                                                         position_history: pd.DataFrame = None) -> pd.DataFrame:
        """
        Backtests an equal risk contributions portfolio over a specified period.

        :param symbols: A list of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :param frequency: The frequency of the returns data, either 'daily', 'weekly', or 'monthly'.
        :type frequency: str
        :param covariance_type: The type of covariance matrix to use in optimization, either 'standard' or 'exponential'. Defaults to 'standard'.
        :type covariance_type: str, optional
        :param decay_factor: The decay factor to use in the exponential weighting. Defaults to 0.94.
        :type decay_factor: float, optional
        :param frequency: The rebalance frequency for the backtest, either 'day', 'week_start', 'week_end', 'month_start', 'month_end', 'quarter_start', 'quarter_end', 'year_start', or 'year_end'. Defaults to 'month_start'.
        :type frequency: str, optional
        :param method: Whether to start the backtest when 'all' symbols have sufficient lookback data, or when 'any' symbol has sufficient lookback data. Defaults to 'any'.
        :type method: str, optional
        :param returns: User-provided returns DataFrame. Defaults to None.
        :type returns: pd.DataFrame, optional
        :param date_asset_pair_list: A list of tuples containing dates and their respective asset lists. Defaults to None.
        :type date_asset_pair_list: List[Tuple[List[str], str]], optional
        :param position_history: A DataFrame containing the position history of the portfolio. Defaults to None.
        :type position_history: pd.DataFrame, optional
        :return: DataFrame with portfolio data for each period.
        :rtype: pd.DataFrame
        """

        async def _async_single_portfolio_optimization(
                                                symbols: List[str],
                                                lookback: int,
                                                covariance_type: str = 'standard',
                                                decay_factor: float = 0.94,
                                                returns: pd.DataFrame = None) -> pd.DataFrame:
            
            if covariance_type == 'standard':
                matrix = self._async_get_covariance_matrix(symbols, lookback, returns)
            elif covariance_type == 'exponential':
                matrix = await self._async_get_exponential_covariance_matrix(symbols, lookback, decay_factor, returns)
            else:
                raise ValueError("covariance_type must be either 'standard' or 'exponential'")

            data = {
                "assets": len(symbols),
                "assetsCovarianceMatrix": matrix.values.tolist(),
                "constraints": {}
            }

            endpoint = "/portfolio/optimization/equal-risk-contributions"

            async with aiohttp.ClientSession() as session:
                async with session.post(url=self._base_url + endpoint, headers=self._headers, json=data) as r:
                    if r.raise_for_status():
                        print(f"Error retrieving equal risk contributions portfolio: {await r.text()}")
                    else:
                        response = await r.json()

            weights = response['assetsWeights']

            return weights

        async def _call_optimization_function(symbols: List[str], returns: pd.DataFrame = None) -> pd.DataFrame:
            return await _async_single_portfolio_optimization(symbols=symbols, 
                                                              lookback=lookback, 
                                                              covariance_type=covariance_type, 
                                                              decay_factor=decay_factor, 
                                                              returns=returns)

        if not self._validate_returns_format(returns, symbols, lookback):
            returns = self._get_asset_returns(symbols, lookback)

        if position_history is None:
            position_history = self._get_position_history(symbols, lookback, frequency, method)

        n = position_history.shape[1]

        date_symbol_pair_list = [(date,symbols.dropna().tolist()) for date,symbols in position_history.iterrows()]

        tasks = [_call_optimization_function(symbols, returns.shift(1).loc[:date]) for date, symbols in date_symbol_pair_list]
        results = await asyncio.gather(*tasks)

        weights = pd.DataFrame(results, index=position_history.index).fillna(0.0)
        weights.columns = [f'weight_{i + 1}' for i in range(n)]
        weights_history = pd.concat([position_history,weights],axis=1)

        backtest = self._generate_portfolio_equity_curve(weights_history, returns, start_equity, fee)
        benchmark_equity_curve = self._generate_benchmark_equity_curve(benchmark,start_equity,backtest.index.min(),backtest.index.max())
        backtest = backtest.join(benchmark_equity_curve.to_frame(),how='left')
        backtest['benchmark_equity_curve'] = backtest['benchmark_equity_curve'].ffill().bfill()

        return backtest
    
    async def backtest_hierarchical_risk_parity_portfolio(self,
                                                        symbols: List[str],
                                                        lookback: int, 
                                                        covariance_type: str = 'standard', 
                                                        decay_factor: float = 0.94,
                                                        frequency: str = 'month_start',
                                                        method='any', 
                                                        start_equity: float = 100000,
                                                        benchmark: str = 'SPY',
                                                        fee: float = 0.0,
                                                        returns: pd.DataFrame = None,
                                                        position_history: pd.DataFrame = None) -> pd.DataFrame:
        """
        Backtests a hierarchical risk parity portfolio over a specified period.

        :param symbols: A list of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :param frequency: The frequency of the returns data, either 'daily', 'weekly', or 'monthly'.
        :type frequency: str
        :param covariance_type: The type of covariance matrix to use in optimization, either 'standard' or 'exponential'. Defaults to 'standard'.
        :type covariance_type: str, optional
        :param decay_factor: The decay factor to use in the exponential weighting. Defaults to 0.94.
        :type decay_factor: float, optional
        :param frequency: The rebalance frequency for the backtest, either 'day', 'week_start', 'week_end', 'month_start', 'month_end', 'quarter_start', 'quarter_end', 'year_start', or 'year_end'. Defaults to 'month_start'.
        :type frequency: str, optional
        :param method: Whether to start the backtest when 'all' symbols have sufficient lookback data, or when 'any' symbol has sufficient lookback data. Defaults to 'any'.
        :type method: str, optional
        :param returns: User-provided returns DataFrame. Defaults to None.
        :type returns: pd.DataFrame, optional
        :param date_asset_pair_list: A list of tuples containing dates and their respective asset lists. Defaults to None.
        :type date_asset_pair_list: List[Tuple[List[str], str]], optional
        :param position_history: A DataFrame containing the position history of the portfolio. Defaults to None.
        :type position_history: pd.DataFrame, optional
        :return: DataFrame with portfolio data for each period.
        :rtype: pd.DataFrame
        """

        async def _async_single_portfolio_optimization(
                                                symbols: List[str],
                                                lookback: int,
                                                covariance_type: str = 'standard',
                                                decay_factor: float = 0.94,
                                                returns: pd.DataFrame = None) -> pd.DataFrame:
            
            if covariance_type == 'standard':
                matrix = self._async_get_covariance_matrix(symbols, lookback, returns)
            elif covariance_type == 'exponential':
                matrix = await self._async_get_exponential_covariance_matrix(symbols, lookback, decay_factor, returns)
            else:
                raise ValueError("covariance_type must be either 'standard' or 'exponential'")

            data = {
                "assets": len(symbols),
                "assetsCovarianceMatrix": matrix.values.tolist(),
                "constraints": {}
            }

            endpoint = "/portfolio/optimization/hierarchical-risk-parity"

            async with aiohttp.ClientSession() as session:
                async with session.post(url=self._base_url + endpoint, headers=self._headers, json=data) as r:
                    if r.raise_for_status():
                        print(f"Error retrieving equal risk contributions portfolio: {await r.text()}")
                    else:
                        response = await r.json()

            weights = response['assetsWeights']

            return weights

        async def _call_optimization_function(symbols: List[str], returns: pd.DataFrame = None) -> pd.DataFrame:
            return await _async_single_portfolio_optimization(symbols=symbols, 
                                                                lookback=lookback, 
                                                                covariance_type=covariance_type, 
                                                                decay_factor=decay_factor, 
                                                                returns=returns)

        if not self._validate_returns_format(returns, symbols, lookback):
            returns = self._get_asset_returns(symbols, lookback)

        if position_history is None:
            position_history = self._get_position_history(symbols, lookback, frequency, method)

        n = position_history.shape[1]

        date_symbol_pair_list = [(date,symbols.dropna().tolist()) for date,symbols in position_history.iterrows()]

        tasks = [_call_optimization_function(symbols, returns.shift(1).loc[:date]) for date, symbols in date_symbol_pair_list]
        results = await asyncio.gather(*tasks)

        weights = pd.DataFrame(results, index=position_history.index).fillna(0.0)
        weights.columns = [f'weight_{i + 1}' for i in range(n)]
        weights_history = pd.concat([position_history,weights],axis=1)

        backtest = self._generate_portfolio_equity_curve(weights_history, returns, start_equity, fee)
        benchmark_equity_curve = self._generate_benchmark_equity_curve(benchmark,start_equity,backtest.index.min(),backtest.index.max())
        backtest = backtest.join(benchmark_equity_curve.to_frame(),how='left').ffill().bfill()

        return backtest
    
    async def backtest_hierarchical_risk_parity_cluster_based_portfolio(self,
                                                    symbols: List[str],
                                                    lookback: int, 
                                                    covariance_type: str = 'standard', 
                                                    decay_factor: float = 0.94,
                                                    frequency: str = 'month_start',
                                                    method='any', 
                                                    start_equity: float = 100000,
                                                    benchmark: str = 'SPY',
                                                    fee: float = 0.0,
                                                    returns: pd.DataFrame = None,
                                                    position_history: pd.DataFrame = None) -> pd.DataFrame:
        """
        Backtests a hierarchical risk parity clustering-based portfolio over a specified period.

        :param symbols: A list of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :param frequency: The frequency of the returns data, either 'daily', 'weekly', or 'monthly'.
        :type frequency: str
        :param covariance_type: The type of covariance matrix to use in optimization, either 'standard' or 'exponential'. Defaults to 'standard'.
        :type covariance_type: str, optional
        :param decay_factor: The decay factor to use in the exponential weighting. Defaults to 0.94.
        :type decay_factor: float, optional
        :param frequency: The rebalance frequency for the backtest, either 'day', 'week_start', 'week_end', 'month_start', 'month_end', 'quarter_start', 'quarter_end', 'year_start', or 'year_end'. Defaults to 'month_start'.
        :type frequency: str, optional
        :param method: Whether to start the backtest when 'all' symbols have sufficient lookback data, or when 'any' symbol has sufficient lookback data. Defaults to 'any'.
        :type method: str, optional
        :param returns: User-provided returns DataFrame. Defaults to None.
        :type returns: pd.DataFrame, optional
        :param date_asset_pair_list: A list of tuples containing dates and their respective asset lists. Defaults to None.
        :type date_asset_pair_list: List[Tuple[List[str], str]], optional
        :param position_history: A DataFrame containing the position history of the portfolio. Defaults to None.
        :type position_history: pd.DataFrame, optional
        :return: DataFrame with portfolio data for each period.
        :rtype: pd.DataFrame
        """

        async def _async_single_portfolio_optimization(
                                                symbols: List[str],
                                                lookback: int,
                                                covariance_type: str = 'standard',
                                                decay_factor: float = 0.94,
                                                returns: pd.DataFrame = None) -> pd.DataFrame:
            
            if covariance_type == 'standard':
                matrix = self._async_get_covariance_matrix(symbols, lookback, returns)
            elif covariance_type == 'exponential':
                matrix = await self._async_get_exponential_covariance_matrix(symbols, lookback, decay_factor, returns)
            else:
                raise ValueError("covariance_type must be either 'standard' or 'exponential'")

            data = {
                "assets": len(symbols),
                "assetsCovarianceMatrix": matrix.values.tolist(),
                "constraints": {}
            }

            endpoint = "/portfolio/optimization/hierarchical-risk-parity/clustering-based"

            async with aiohttp.ClientSession() as session:
                async with session.post(url=self._base_url + endpoint, headers=self._headers, json=data) as r:
                    if r.raise_for_status():
                        print(f"Error retrieving equal risk contributions portfolio: {await r.text()}")
                    else:
                        response = await r.json()

            weights = response['assetsWeights']

            return weights

        async def _call_optimization_function(symbols: List[str], returns: pd.DataFrame = None) -> pd.DataFrame:
            return await _async_single_portfolio_optimization(symbols=symbols, 
                                                                lookback=lookback, 
                                                                covariance_type=covariance_type, 
                                                                decay_factor=decay_factor, 
                                                                returns=returns)

        if not self._validate_returns_format(returns, symbols, lookback):
            returns = self._get_asset_returns(symbols, lookback)

        if position_history is None:
            position_history = self._get_position_history(symbols, lookback, frequency, method)

        n = position_history.shape[1]

        date_symbol_pair_list = [(date,symbols.dropna().tolist()) for date,symbols in position_history.iterrows()]

        tasks = [_call_optimization_function(symbols, returns.shift(1).loc[:date]) for date, symbols in date_symbol_pair_list]
        results = await asyncio.gather(*tasks)

        weights = pd.DataFrame(results, index=position_history.index).fillna(0.0)
        weights.columns = [f'weight_{i + 1}' for i in range(n)]
        weights_history = pd.concat([position_history,weights],axis=1)

        backtest = self._generate_portfolio_equity_curve(weights_history, returns, start_equity, fee)
        benchmark_equity_curve = self._generate_benchmark_equity_curve(benchmark,start_equity,backtest.index.min(),backtest.index.max())
        backtest = backtest.join(benchmark_equity_curve.to_frame(),how='left').ffill().bfill()

        return backtest
    
    async def backtest_most_diversified_portfolio(self,
                                                    symbols: List[str],
                                                    lookback: int, 
                                                    covariance_type: str = 'standard', 
                                                    decay_factor: float = 0.94,
                                                    frequency: str = 'month_start',
                                                    method='any', 
                                                    start_equity: float = 100000,
                                                    benchmark: str = 'SPY',
                                                    fee: float = 0.0,
                                                    returns: pd.DataFrame = None,
                                                    position_history: pd.DataFrame = None) -> pd.DataFrame:
        """
        Backtests a most diversified portfolio over a specified period.

        :param symbols: A list of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :param frequency: The frequency of the returns data, either 'daily', 'weekly', or 'monthly'.
        :type frequency: str
        :param covariance_type: The type of covariance matrix to use in optimization, either 'standard' or 'exponential'. Defaults to 'standard'.
        :type covariance_type: str, optional
        :param decay_factor: The decay factor to use in the exponential weighting. Defaults to 0.94.
        :type decay_factor: float, optional
        :param frequency: The rebalance frequency for the backtest, either 'day', 'week_start', 'week_end', 'month_start', 'month_end', 'quarter_start', 'quarter_end', 'year_start', or 'year_end'. Defaults to 'month_start'.
        :type frequency: str, optional
        :param method: Whether to start the backtest when 'all' symbols have sufficient lookback data, or when 'any' symbol has sufficient lookback data. Defaults to 'any'.
        :type method: str, optional
        :param returns: User-provided returns DataFrame. Defaults to None.
        :type returns: pd.DataFrame, optional
        :param date_asset_pair_list: A list of tuples containing dates and their respective asset lists. Defaults to None.
        :type date_asset_pair_list: List[Tuple[List[str], str]], optional
        :param position_history: A DataFrame containing the position history of the portfolio. Defaults to None.
        :type position_history: pd.DataFrame, optional
        :return: DataFrame with portfolio data for each period.
        :rtype: pd.DataFrame
        """

        async def _async_single_portfolio_optimization(
                                                symbols: List[str],
                                                lookback: int,
                                                covariance_type: str = 'standard',
                                                decay_factor: float = 0.94,
                                                returns: pd.DataFrame = None) -> pd.DataFrame:
            
            if covariance_type == 'standard':
                matrix = self._async_get_covariance_matrix(symbols, lookback, returns)
            elif covariance_type == 'exponential':
                matrix = await self._async_get_exponential_covariance_matrix(symbols, lookback, decay_factor, returns)
            else:
                raise ValueError("covariance_type must be either 'standard' or 'exponential'")

            data = {
                "assets": len(symbols),
                "assetsCovarianceMatrix": matrix.values.tolist(),
                "constraints": {}
            }

            endpoint = "/portfolio/optimization/most-diversified"

            async with aiohttp.ClientSession() as session:
                async with session.post(url=self._base_url + endpoint, headers=self._headers, json=data) as r:
                    if r.raise_for_status():
                        print(f"Error retrieving equal risk contributions portfolio: {await r.text()}")
                    else:
                        response = await r.json()

            weights = response['assetsWeights']

            return weights

        async def _call_optimization_function(symbols: List[str], returns: pd.DataFrame = None) -> pd.DataFrame:
            return await _async_single_portfolio_optimization(symbols=symbols, 
                                                                lookback=lookback, 
                                                                covariance_type=covariance_type, 
                                                                decay_factor=decay_factor, 
                                                                returns=returns)

        if not self._validate_returns_format(returns, symbols, lookback):
            returns = self._get_asset_returns(symbols, lookback)

        if position_history is None:
            position_history = self._get_position_history(symbols, lookback, frequency, method)

        n = position_history.shape[1]

        date_symbol_pair_list = [(date,symbols.dropna().tolist()) for date,symbols in position_history.iterrows()]

        tasks = [_call_optimization_function(symbols, returns.shift(1).loc[:date]) for date, symbols in date_symbol_pair_list]
        results = await asyncio.gather(*tasks)

        weights = pd.DataFrame(results, index=position_history.index).fillna(0.0)
        weights.columns = [f'weight_{i + 1}' for i in range(n)]
        weights_history = pd.concat([position_history,weights],axis=1)

        backtest = self._generate_portfolio_equity_curve(weights_history, returns, start_equity, fee)
        benchmark_equity_curve = self._generate_benchmark_equity_curve(benchmark,start_equity,backtest.index.min(),backtest.index.max())
        backtest = backtest.join(benchmark_equity_curve.to_frame(),how='left').ffill().bfill()

        return backtest
    
    async def backtest_minimum_variance_portfolio(self,
                                                    symbols: List[str],
                                                    lookback: int, 
                                                    covariance_type: str = 'standard', 
                                                    decay_factor: float = 0.94,
                                                    frequency: str = 'month_start',
                                                    method='any', 
                                                    start_equity: float = 100000,
                                                    benchmark: str = 'SPY',
                                                    fee: float = 0.0,
                                                    returns: pd.DataFrame = None,
                                                    position_history: pd.DataFrame = None) -> pd.DataFrame:
        """
        Backtests a minimum variance portfolio over a specified period.

        :param symbols: A list of stock symbols.
        :type symbols: List[str]
        :param lookback: The number of rows to consider from the returns DataFrame.
        :type lookback: int
        :param frequency: The frequency of the returns data, either 'daily', 'weekly', or 'monthly'.
        :type frequency: str
        :param covariance_type: The type of covariance matrix to use in optimization, either 'standard' or 'exponential'. Defaults to 'standard'.
        :type covariance_type: str, optional
        :param decay_factor: The decay factor to use in the exponential weighting. Defaults to 0.94.
        :type decay_factor: float, optional
        :param frequency: The rebalance frequency for the backtest, either 'day', 'week_start', 'week_end', 'month_start', 'month_end', 'quarter_start', 'quarter_end', 'year_start', or 'year_end'. Defaults to 'month_start'.
        :type frequency: str, optional
        :param method: Whether to start the backtest when 'all' symbols have sufficient lookback data, or when 'any' symbol has sufficient lookback data. Defaults to 'any'.
        :type method: str, optional
        :param returns: User-provided returns DataFrame. Defaults to None.
        :type returns: pd.DataFrame, optional
        :param date_asset_pair_list: A list of tuples containing dates and their respective asset lists. Defaults to None.
        :type date_asset_pair_list: List[Tuple[List[str], str]], optional
        :param position_history: A DataFrame containing the position history of the portfolio. Defaults to None.
        :type position_history: pd.DataFrame, optional
        :return: DataFrame with portfolio data for each period.
        :rtype: pd.DataFrame
        """

        async def _async_single_portfolio_optimization(
                                                symbols: List[str],
                                                lookback: int,
                                                covariance_type: str = 'standard',
                                                decay_factor: float = 0.94,
                                                returns: pd.DataFrame = None) -> pd.DataFrame:
            
            if covariance_type == 'standard':
                matrix = self._async_get_covariance_matrix(symbols, lookback, returns)
            elif covariance_type == 'exponential':
                matrix = await self._async_get_exponential_covariance_matrix(symbols, lookback, decay_factor, returns)
            else:
                raise ValueError("covariance_type must be either 'standard' or 'exponential'")

            data = {
                "assets": len(symbols),
                "assetsCovarianceMatrix": matrix.values.tolist(),
                "constraints": {}
            }

            endpoint = "/portfolio/optimization/minimum-variance"

            async with aiohttp.ClientSession() as session:
                async with session.post(url=self._base_url + endpoint, headers=self._headers, json=data) as r:
                    if r.raise_for_status():
                        print(f"Error retrieving equal risk contributions portfolio: {await r.text()}")
                    else:
                        response = await r.json()

            weights = response['assetsWeights']

            return weights

        async def _call_optimization_function(symbols: List[str], returns: pd.DataFrame = None) -> pd.DataFrame:
            return await _async_single_portfolio_optimization(symbols=symbols, 
                                                                lookback=lookback, 
                                                                covariance_type=covariance_type, 
                                                                decay_factor=decay_factor, 
                                                                returns=returns)

        if not self._validate_returns_format(returns, symbols, lookback):
            returns = self._get_asset_returns(symbols, lookback)

        if position_history is None:
            position_history = self._get_position_history(symbols, lookback, frequency, method)

        n = position_history.shape[1]

        date_symbol_pair_list = [(date,symbols.dropna().tolist()) for date,symbols in position_history.iterrows()]

        tasks = [_call_optimization_function(symbols, returns.shift(1).loc[:date]) for date, symbols in date_symbol_pair_list]
        results = await asyncio.gather(*tasks)

        weights = pd.DataFrame(results, index=position_history.index).fillna(0.0)
        weights.columns = [f'weight_{i + 1}' for i in range(n)]
        weights_history = pd.concat([position_history,weights],axis=1)

        backtest = self._generate_portfolio_equity_curve(weights_history, returns, start_equity, fee)
        benchmark_equity_curve = self._generate_benchmark_equity_curve(benchmark,start_equity,backtest.index.min(),backtest.index.max())
        backtest = backtest.join(benchmark_equity_curve.to_frame(),how='left').ffill().bfill()

        return backtest