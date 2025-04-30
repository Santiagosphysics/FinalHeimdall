import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

from dotenv import load_dotenv


from binance.client import Client
from prophet import Prophet
import  requests


import os  

class get_data_crypto:
    def download_data(self, start_time, end_time, crypto, time):

        data_time = {'minutes':Client.KLINE_INTERVAL_1MINUTE,'seconds':Client.KLINE_INTERVAL_1SECOND,'hours':Client.KLINE_INTERVAL_1HOUR,}
        if time not in data_time:
            raise ValueError(f'El parámetro {time} no está en las opciones {data_time}')
        
        load_dotenv()
        
        api_key  = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

        client = Client(api_key, api_secret)
        start_time_ms = int(pd.Timestamp(start_time).timestamp()*1000)
        end_time_ms = int(pd.Timestamp(end_time).timestamp()*1000)

        all_candles = []

        current_start_time = start_time_ms

        while current_start_time < end_time_ms:
            
            candles = client.get_klines(symbol=crypto, interval = data_time[time], startTime=current_start_time, endTime=end_time_ms, limit = 1000)
            if not candles:
                break

            all_candles.extend(candles)
            current_start_time = candles[-1][6] + 1 

        columns = [ 'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        
        df = pd.DataFrame(all_candles, columns=columns)    
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')    
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        df['close'] = df['close'].astype(float)
        df['open'] = df['open'].astype(float)

        return df



    def download_data_cloud(self, start_time, end_time, crypto, time):

        data_time = {'minutes':'minute', 'hours':'hour'}

        if time not in data_time:
            raise ValueError(f'The value {time} doesnt have in options: {data_time}')
        
        start_time_timestamp = int(pd.Timestamp(start_time).timestamp())
        end_time_timestamp = int(pd.Timestamp(end_time).timestamp())

        url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart/range"
        params = {'vs_currency':'usd', 'from':start_time_timestamp, 'to':end_time_timestamp}
        response = requests.get(url, params=params)

        if response.status_code != 200:
            raise Exception(f"Error in the application of CoinGecko {response.status_code} ")

        data = response.json()

        if 'prices' not in data:
            raise Exception('Couldnt find data prices in the dataset')

        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp','price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df
    

    

class models:
    def prophet_model(data, time):
        if len(data.columns) > 2:
            raise ValueError('Your dataset has more than two columns {data.columns}')
        data.columns = ['ds','y']

        model = Prophet(daily_seasonality=True)
        model.fit(data)

        future = model.make_future_dataframe(periods=24*60, freq = time)

        predict = model.predict(future)

        fig1 = model.plot(predict)
        plt.title('Prediction for the price for the next 24 hours')
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.show()

        return predict

