'''this script generates reports about the historical company data and also historical time series 
performance analysis of all stocks in the user's watch list'''

from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import os
import pandas as pd
import yfinance as yf

load_dotenv()

PROJECT_ROOT_DIRECTORY = os.getenv('PROJECT_ROOT_DIRECTORY')
PROJECT_FILE_DROP_FOLDER = os.getenv('PROJECT_FILE_DROP_FOLDER')
FILE_DROP_DIR_PATH = f'{PROJECT_FILE_DROP_FOLDER}/stocks'

USER_STOCK_WATCH_LIST = os.getenv('USER_STOCK_WATCH_LIST', 'None').split(',')  # Comma separated list of stock symbols
valid_tickers = [ticker for ticker in USER_STOCK_WATCH_LIST if ticker != 'None']

# create the file drop folder if it doesn't exist
if not os.path.exists(FILE_DROP_DIR_PATH):
    os.makedirs(FILE_DROP_DIR_PATH)

now = datetime.now()

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 150)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 35)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 1)  

class Stocks:
    def __init__(self, tickers):
        self.tickers = tickers
        self.historical_data, self.stock_info = self.fetch_all_stock_data()
        self.comprehensive_summary = self.create_comprehensive_summary()

    def fetch_all_stock_data(self):
        # Fetch historical data for all tickers
        historical_data = yf.download(self.tickers, period="max", group_by='ticker')
        # for historical_data, filter out all rows where 'Date' is before 2010-01-01
        historical_data = historical_data[historical_data.index > '2010-01-01']

        # Save historical data to CSV
        historical_data.to_csv(f'{FILE_DROP_DIR_PATH}/stock_data_historical_{now}.csv')
        print(f"Stock data historical saved to {FILE_DROP_DIR_PATH}/stock_data_historical_{now}.csv")

        # Fetch additional stock information and save to CSV
        stock_info_list = []
        for ticker in self.tickers:
            stock = yf.Ticker(ticker)
            info = stock.info
            info['symbol'] = ticker  # Ensure the symbol is part of the info
            stock_info_list.append(info)

        # Convert list of dictionaries to DataFrame and save to CSV
        stock_info_df = pd.DataFrame(stock_info_list)
        # Set the stock's name (ticker symbol) to be the index of the stock_info_df DataFrame
        stock_info_df.set_index('symbol', inplace=True)
        stock_info_df.to_csv(f'{FILE_DROP_DIR_PATH}/stock_data_all_info_{now}.csv')
        print(f"Stock data all info saved to {FILE_DROP_DIR_PATH}/stock_data_all_info_{now}.csv")

        return historical_data, stock_info_df
    
    def load_historical_data(self, filepath):
        return pd.read_csv(filepath, header=[0, 1], index_col=0, parse_dates=True)

    def calculate_change(self, data, days):
        if len(data) >= days:
            return data.iloc[-1] - data.iloc[-days], ((data.iloc[-1] - data.iloc[-days]) / data.iloc[-days]) * 100
        return np.nan, np.nan
        
    def calculate_moving_average(self, data, window):
        return data.rolling(window=window).mean().iloc[-1]

    def calculate_rsi(self, data, window=14):
        delta = data.diff()
        gain = (delta.clip(lower=0)).rolling(window=window).mean()
        loss = (-delta.clip(upper=0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def calculate_volume_trend(self, data, days):
        return data.iloc[-days:].mean()

    def calculate_macd(self, data):
        ema12 = data.ewm(span=12, adjust=False).mean()
        ema26 = data.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd.iloc[-1], signal.iloc[-1]

    def calculate_bollinger_bands(self, data):
        sma20 = data.rolling(window=20).mean()
        std20 = data.rolling(window=20).std()
        upper_band = sma20 + (std20 * 2)
        lower_band = sma20 - (std20 * 2)
        return upper_band.iloc[-1], lower_band.iloc[-1]

    def calculate_earnings_per_share(self, info):
        net_income = info.get('netIncome', np.nan)
        outstanding_shares = info.get('sharesOutstanding', np.nan)
        if not np.isnan(net_income) and not np.isnan(outstanding_shares) and outstanding_shares != 0:
            return net_income / outstanding_shares
        return np.nan

    def calculate_pe_ratio(self, info, current_price):
        earnings_per_share = self.calculate_earnings_per_share(info)
        if not np.isnan(earnings_per_share) and earnings_per_share != 0:
            return current_price / earnings_per_share
        return np.nan

    def calculate_dividend_yield(self, info, current_price):
        dividend_rate = info.get('dividendRate', 0)

        try:
            dividend_rate = float(dividend_rate)
        except (TypeError, ValueError):
            dividend_rate = 0.0

        try:
            current_price = float(current_price)
        except (TypeError, ValueError):
            current_price = 0.0

        # Check if current_price is not zero to avoid division by zero
        if current_price != 0:
            return dividend_rate / current_price
        return 0  # Return 0 as dividend yield if current_price is 0 or not a valid number

    def calculate_atr(self, data, window):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr.iloc[-1]

    def determine_signal(self, macd, signal, rsi):
        if macd > signal and rsi < 70:
            return "Buy"
        elif macd < signal and rsi > 30:
            return "Sell"
        else:
            return "Hold"
        
    def create_summary_row(self, stock_data, symbol, info):
        changes = {}
        for days in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 30, 60, 90, 180, 365, 730, 1095, 1460, 1825, 3650]:
            changes[f'change_{days}d_$'], changes[f'change_{days}d_%'] = self.calculate_change(stock_data['Close'], days)

        macd, signal = self.calculate_macd(stock_data['Close'])
        upper_band, lower_band = self.calculate_bollinger_bands(stock_data['Close'])
        rsi = self.calculate_rsi(stock_data['Close'])
        signal = self.determine_signal(macd, signal, rsi)
        current_price = stock_data['Close'].iloc[-1]
        eps = self.calculate_earnings_per_share(info)
        pe_ratio = self.calculate_pe_ratio(info, current_price)
        dividend_yield = self.calculate_dividend_yield(info, current_price)
        # volume_stats = self.calculate_volume_stats(stock_data)
        
        row = {
            'symbol': symbol,
            'trade_signal': signal,
            'macd_signal': signal,
            'rsi': rsi,
            'macd': macd,
            'last closing_price': stock_data['Close'].iloc[-1],
            'last_opening_price': stock_data['Open'].iloc[-1],
            'bollinger_upper': upper_band,
            'bollinger_lower': lower_band,
            'last_trading_day_change_$': stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2],
            'last_trading_day_change_%': ((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2]) * 100,
            **changes,
            'last_trading_day_high': stock_data['High'].iloc[-1],
            'last_trading_day_low': stock_data['Low'].iloc[-1],
            '3_day_high': stock_data['High'].tail(3).max(),
            '3_day_low': stock_data['Low'].tail(3).min(),
            '4_day_high': stock_data['High'].tail(4).max(),
            '4_day_low': stock_data['Low'].tail(4).min(),
            '5_day_high': stock_data['High'].tail(5).max(),
            '5_day_low': stock_data['Low'].tail(5).min(),
            '6_day_high': stock_data['High'].tail(6).max(),
            '6_day_low': stock_data['Low'].tail(6).min(),
            '7_day_high': stock_data['High'].tail(7).max(),
            '7_day_low': stock_data['Low'].tail(7).min(),
            '8_day_high': stock_data['High'].tail(8).max(),
            '8_day_low': stock_data['Low'].tail(8).min(),
            '9_day_high': stock_data['High'].tail(9).max(),
            '9_day_low': stock_data['Low'].tail(9).min(),
            '10_day_high': stock_data['High'].tail(10).max(),
            '10_day_low': stock_data['Low'].tail(10).min(),
            '11_day_high': stock_data['High'].tail(11).max(),
            '11_day_low': stock_data['Low'].tail(11).min(),
            '12_day_high': stock_data['High'].tail(12).max(),
            '12_day_low': stock_data['Low'].tail(12).min(),
            '13_day_high': stock_data['High'].tail(13).max(),
            '13_day_low': stock_data['Low'].tail(13).min(),
            '14_day_high': stock_data['High'].tail(14).max(),
            '14_day_low': stock_data['Low'].tail(14).min(),
            '21_day_high': stock_data['High'].tail(21).max(),
            '21_day_low': stock_data['Low'].tail(21).min(),
            '30_day_high': stock_data['High'].tail(30).max(),
            '30_day_low': stock_data['Low'].tail(30).min(),
            '60_day_high': stock_data['High'].tail(60).max(),
            '60_day_low': stock_data['Low'].tail(60).min(),
            '90_day_high': stock_data['High'].tail(90).max(),
            '90_day_low': stock_data['Low'].tail(90).min(),
            '180_day_high': stock_data['High'].tail(180).max(),
            '180_day_low': stock_data['Low'].tail(180).min(),
            '365_day_high': stock_data['High'].tail(365).max(),
            '365_day_low': stock_data['Low'].tail(365).min(),
            '730_day_high': stock_data['High'].tail(730).max(),
            '730_day_low': stock_data['Low'].tail(730).min(),
            '1095_day_high': stock_data['High'].tail(1095).max(),
            '1095_day_low': stock_data['Low'].tail(1095).min(),
            '1460_day_high': stock_data['High'].tail(1460).max(),
            '1460_day_low': stock_data['Low'].tail(1460).min(),
            '1825_day_high': stock_data['High'].tail(1825).max(),
            '1825_day_low': stock_data['Low'].tail(1825).min(),
            '3650_day_high': stock_data['High'].tail(3650).max(),
            '3650_day_low': stock_data['Low'].tail(3650).min(),
            'moving_average_03': self.calculate_moving_average(stock_data['Close'], 3),
            'moving_average_04': self.calculate_moving_average(stock_data['Close'], 4),
            'moving_average_05': self.calculate_moving_average(stock_data['Close'], 5),
            'moving_average_06': self.calculate_moving_average(stock_data['Close'], 6),
            'moving_average_07': self.calculate_moving_average(stock_data['Close'], 7),
            'moving_average_08': self.calculate_moving_average(stock_data['Close'], 8),
            'moving_average_09': self.calculate_moving_average(stock_data['Close'], 9),
            'moving_average_10': self.calculate_moving_average(stock_data['Close'], 10),
            'moving_average_11': self.calculate_moving_average(stock_data['Close'], 11),
            'moving_average_12': self.calculate_moving_average(stock_data['Close'], 12),
            'moving_average_13': self.calculate_moving_average(stock_data['Close'], 13),
            'moving_average_14': self.calculate_moving_average(stock_data['Close'], 14),
            'moving_average_21': self.calculate_moving_average(stock_data['Close'], 21),
            'moving_average_30': self.calculate_moving_average(stock_data['Close'], 30),
            'moving_average_40': self.calculate_moving_average(stock_data['Close'], 40),
            'moving_average_50': self.calculate_moving_average(stock_data['Close'], 50),
            'moving_average_60': self.calculate_moving_average(stock_data['Close'], 60),
            'moving_average_100': self.calculate_moving_average(stock_data['Close'], 100),
            'moving_average_200': self.calculate_moving_average(stock_data['Close'], 200),
            'moving_average_300': self.calculate_moving_average(stock_data['Close'], 300),
            'volume_trend_03d': self.calculate_volume_trend(stock_data['Volume'], 3),
            'volume_trend_04d': self.calculate_volume_trend(stock_data['Volume'], 4),
            'volume_trend_05d': self.calculate_volume_trend(stock_data['Volume'], 5),
            'volume_trend_06d': self.calculate_volume_trend(stock_data['Volume'], 6),
            'volume_trend_07d': self.calculate_volume_trend(stock_data['Volume'], 7),
            'volume_trend_08d': self.calculate_volume_trend(stock_data['Volume'], 8),
            'volume_trend_09d': self.calculate_volume_trend(stock_data['Volume'], 9),
            'volume_trend_10d': self.calculate_volume_trend(stock_data['Volume'], 10),
            'volume_trend_11d': self.calculate_volume_trend(stock_data['Volume'], 11),
            'volume_trend_12d': self.calculate_volume_trend(stock_data['Volume'], 12),
            'volume_trend_13d': self.calculate_volume_trend(stock_data['Volume'], 13),
            'volume_trend_14d': self.calculate_volume_trend(stock_data['Volume'], 14),
            'volume_trend_21d': self.calculate_volume_trend(stock_data['Volume'], 21),
            'volume_trend_30d': self.calculate_volume_trend(stock_data['Volume'], 30),
            'volume_trend_40d': self.calculate_volume_trend(stock_data['Volume'], 40),
            'volume_trend_50d': self.calculate_volume_trend(stock_data['Volume'], 50),
            'volume_trend_60d': self.calculate_volume_trend(stock_data['Volume'], 60),
            'volume_trend_90d': self.calculate_volume_trend(stock_data['Volume'], 90),
            'atr_2d': self.calculate_atr(stock_data, 2), 
            'atr_3d': self.calculate_atr(stock_data, 3), 
            'atr_4d': self.calculate_atr(stock_data, 4),
            'atr_5d': self.calculate_atr(stock_data, 5),  
            'atr_6d': self.calculate_atr(stock_data, 6), 
            'atr_7d': self.calculate_atr(stock_data, 7),
            'atr_8d': self.calculate_atr(stock_data, 8),
            'atr_9d': self.calculate_atr(stock_data, 9),
            'atr_10d': self.calculate_atr(stock_data, 10),
            'atr_11d': self.calculate_atr(stock_data, 11),
            'atr_12d': self.calculate_atr(stock_data, 12),
            'atr_13d': self.calculate_atr(stock_data, 13),
            'atr_14d': self.calculate_atr(stock_data, 14),
            'atr_21': self.calculate_atr(stock_data, 21),
            'atr_30d': self.calculate_atr(stock_data, 30),
            'atr_60d': self.calculate_atr(stock_data, 60),
            'atr_90d': self.calculate_atr(stock_data, 90),
            'atr_180d': self.calculate_atr(stock_data, 180),
            # ... Add other calculations as needed ...
        }
        row.update({
            'eps': eps,
            'pe_ratio': pe_ratio,
            'dividend_yield': dividend_yield,
            # **volume_stats
        })
        for key, value in row.items():
            if isinstance(value, (int, float)):
                row[key] = round(value, 2)
        return row

    def create_comprehensive_summary(self):
        # Adapted to utilize the instance variables
        summary_rows = []
        for symbol in self.tickers:
            stock_data = self.historical_data[symbol]
            info = self.stock_info.loc[symbol].to_dict()
            summary_row = self.create_summary_row(stock_data, symbol, info)
            summary_rows.append(summary_row)

        comprehensive_summary = pd.DataFrame(summary_rows)
        comprehensive_summary.to_csv(f'{FILE_DROP_DIR_PATH}/stock_data_stats_{now}.csv', index=False)
        print(f"Stock data stats summary saved to {FILE_DROP_DIR_PATH}/stock_data_stats_{now}.csv")
        return comprehensive_summary

# this script used to end like this before writing the test:
# Stocks(valid_tickers)

# and it's since been changed to this:
if __name__ == '__main__':
    stocks = Stocks(valid_tickers)
