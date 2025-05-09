import pandas as od 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np 

from dotenv import load_dotenv

from binance.client import Client

from prophet import Prophet
import xgboost  as xgb

import  requests

from utils import get_data_crypto

class XGBoost:
    def XGBoost_model(self, data, time ):
        data.columns = ['ds', 'y']
        df = pd.DataFrame()

        if time == 'S':
            df['second']= data['ds'].dt.second
        
        df['minute'] = data['ds'].dt.minute
        df['hour'] = data['ds'].dt.hour
        df['dayofweek'] = data['ds'].dt.dayofweek
        df['day'] = data['ds'].dt.day

        df['y'] = round(data['y'], 3)
        df['ds'] = data['ds']

        X = df[['minute', 'hour', 'day', 'dayofweek']]
        y = df['y']

        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
        model.fit(X, y)

        if time == 'min':
            df_pred = pd.date_range(start=df['ds'].max() + pd.Timedelta(minutes=1), periods=24*60, freq='min')
        elif time == 'S':
            df_pred = pd.date_range(start=df['ds'].max() + pd.Timedelta(seconds=1), periods=24*60*60, freq='S')
        else:
            raise ValueError('Please write a correct option (min, S) ')
        

        df_final = pd.DataFrame({'ds':df_pred})

        df_final['minute'] =    df_final['ds'].dt.minute
        df_final['hour'] =      df_final['ds'].dt.hour
        df_final['dayofweek'] = df_final['ds'].dt.dayofweek
        df_final['day'] =       df_final['ds'].dt.day

        if time == 's':
            df_final['second'] = df_final['ds'].dt.second
            X_final = df_final[['second', 'minute', 'hour', 'dayofweek', 'day']]
        else:
            X_final = df_final[['minute', 'hour', 'day', 'dayofweek']]
        response = model.predict(X_final)

        df_final['y'] = response

        df_real = pd.DataFrame({'ds':df['ds'],'y':df['y']})
        df_result = pd.concat([df_real, df_final], ignore_index=True)
        df_result = df_result[['ds', 'y']]

        return df, df_final
    

    def XGBoost_final(self, data, time ):

        data = data[['close_time','close', 'volume']]

        df = pd.DataFrame()

        df['minute'] = data['close_time'].dt.minute
        df['hour'] = data['close_time'].dt.hour
        df['dayofweek'] = data['close_time'].dt.dayofweek
        df['day'] = data['close_time'].dt.day

        df['price'] = data['close'].astype(float)
        df['volume'] = data['volume']

        df['close_time'] = data['close_time']

        if time == 'S' or time == 's':
            df['second'] = data['close_time'].dt.second        
            X = df[['second', 'minute', 'hour', 'dayofweek','day']]
        
        else:
            X = df[['minute', 'hour', 'dayofweek', 'day']]
        

        y = df[['price']]

        model = xgb.XGBRegressor(objective= 'reg.squarederror', n_estimators=100, learning_rate=0.1)
        model.fit(X,y)

        if time == 'min':
            df_pred = pd.date_range(start=df['close_time'].max() + pd.Timedelta(minutes=1), periods=24*60, freq='min')
        else:
            df_pred = pd.date_range(start=df['close_time'].max() + pd.Timedelta(seconds=1), periods=24*60*60, freq='S')

        df_final = pd.DataFrame({'close_time':df_pred})



        if time == 's' or time == 'S':
            df_final['second'] = df_final['ds'].dt.second
            X_final = df_final[['second', 'minute', 'hour', 'dayofweek', 'day']]
        else:
            df_final['minute'] =    df_final['ds'].dt.minute
            df_final['hour'] =      df_final['ds'].dt.hour
            df_final['dayofweek'] = df_final['ds'].dt.dayofweek
            df_final['day'] =       df_final['ds'].dt.day   
            X_final = df_final[['minute', 'hour', 'day', 'dayofweek']] 

        df_final['Pred Price'] = model.predict(X_final)    

        return data, df_final


start_time = '2025-03-15 00:00:00'
end_time = '2025-03-15 01:00:00'

days_fine_pred = 1
days_pred = 2
crypto='ETHUSDT'
time='S'

data = get_data_crypto().download_data_volume(start_time=start_time, end_time=end_time, crypto=crypto, time=time)
test_1 = XGBoost().XGBoost_final(data=data, time=time)