from chatbot_data_engines.stock_data import Stocks
import unittest
import os
import shutil
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env
load_dotenv()

class TestStocksData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up file paths and stock list for testing
        cls.file_drop_dir = os.getenv('PROJECT_FILE_DROP_FOLDER') + '/stocks'
        cls.user_stock_watch_list = os.getenv('USER_STOCK_WATCH_LIST', '').split(',')
        cls.valid_tickers = [ticker for ticker in cls.user_stock_watch_list if ticker]

        # Get the archive directory from the environment variable
        cls.archive_dir = os.getenv('PROJECT_ARCHIVE_DIRECTORY')

        # Ensure the archive directory exists
        if not os.path.exists(cls.archive_dir):
            os.makedirs(cls.archive_dir)

        # Move existing stock data files to archive instead of deleting
        cls.move_to_archive()

        # Initialize the Stocks class with valid tickers
        cls.stocks = Stocks(cls.valid_tickers)

    @classmethod
    def move_to_archive(cls):
        """Move existing files in the file drop directory to the archive."""
        for file_name in os.listdir(cls.file_drop_dir):
            file_path = os.path.join(cls.file_drop_dir, file_name)
            if os.path.isfile(file_path):
                shutil.move(file_path, cls.archive_dir)

    @staticmethod
    def get_most_recent_file(directory, prefix):
        """Find the most recent file in the directory that matches the given prefix."""
        files = [f for f in os.listdir(directory) if f.startswith(prefix)]
        if not files:
            return None
        most_recent_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
        return os.path.join(directory, most_recent_file)

    def test_historical_data_fetched(self):
        """Test that historical data is fetched and saved as a CSV."""
        historical_file = self.get_most_recent_file(self.file_drop_dir, 'stock_data_historical')
        self.assertIsNotNone(historical_file, "Historical data CSV was not created.")
        historical_data = pd.read_csv(historical_file)
        self.assertFalse(historical_data.empty, "Historical data CSV is empty.")

    def test_stock_info_fetched(self):
        """Test that stock info is fetched and saved as a CSV."""
        info_file = self.get_most_recent_file(self.file_drop_dir, 'stock_data_all_info')
        self.assertIsNotNone(info_file, "Stock info CSV was not created.")
        stock_info_data = pd.read_csv(info_file)
        self.assertFalse(stock_info_data.empty, "Stock info CSV is empty.")

    def test_comprehensive_summary_generated(self):
        """Test that the comprehensive summary is generated and saved as a CSV."""
        summary_file = self.get_most_recent_file(self.file_drop_dir, 'stock_data_stats')
        self.assertIsNotNone(summary_file, "Comprehensive summary CSV was not created.")
        summary_data = pd.read_csv(summary_file)
        self.assertFalse(summary_data.empty, "Comprehensive summary CSV is empty.")

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        historical_data = self.stocks.historical_data[self.valid_tickers[0]]['Close']
        rsi = self.stocks.calculate_rsi(historical_data)
        self.assertIsInstance(rsi, float, "RSI should return a float.")

    def test_calculate_macd(self):
        """Test MACD calculation."""
        historical_data = self.stocks.historical_data[self.valid_tickers[0]]['Close']
        macd, signal = self.stocks.calculate_macd(historical_data)
        self.assertIsInstance(macd, float, "MACD should return a float.")
        self.assertIsInstance(signal, float, "Signal line should return a float.")

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        historical_data = self.stocks.historical_data[self.valid_tickers[0]]['Close']
        upper_band, lower_band = self.stocks.calculate_bollinger_bands(historical_data)
        self.assertIsInstance(upper_band, float, "Upper Bollinger Band should return a float.")
        self.assertIsInstance(lower_band, float, "Lower Bollinger Band should return a float.")

if __name__ == '__main__':
    unittest.main()
