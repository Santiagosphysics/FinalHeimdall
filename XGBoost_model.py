import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.dates as mdates

import xgboost  as xgb


from utils import get_data_crypto
from volume import VolumeXGBoost


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
        elif time == 'S' or time == 's':
            df_pred = pd.date_range(start=df['ds'].max() + pd.Timedelta(seconds=1), periods=24*60*60, freq='s')
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

    


    def XGBoost_final(self, data, crypto, time ):
        data = data[['close_time', 'close', 'volume']].rename(columns={'close_time':'ds', 'close':'y'})

        df = pd.DataFrame()
        
        df['minute'] = data['ds'].dt.minute
        df['hour'] = data['ds'].dt.hour
        df['dayofweek'] = data['ds'].dt.dayofweek
        df['day'] = data['ds'].dt.day

        df['y'] = round(data['y'], 3)
        df['ds'] = data['ds']
        df['volume'] = data['volume'].astype(float)

        if time == 'S':
            df['second']= data['ds'].dt.second
            X = df[['second', 'minute', 'hour', 'dayofweek', 'day', 'volume']]

        else:        
            X = df[['minute', 'hour', 'day', 'dayofweek', 'volume']]

        y = df['y']

        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
        model.fit(X, y)

        if time == 'min':
            df_pred = pd.date_range(start=df['ds'].max() + pd.Timedelta(minutes=1), periods=24*60, freq='min')
        elif time == 'S':
            df_pred = pd.date_range(start=df['ds'].max() + pd.Timedelta(seconds=1), periods=24*60*60, freq='S')
        else:
            raise ValueError('Please write a correct option (min, S, s) ')
        
        df_final = pd.DataFrame({'ds':df_pred})

        df_final['minute'] =    df_final['ds'].dt.minute
        df_final['hour'] =      df_final['ds'].dt.hour
        df_final['dayofweek'] = df_final['ds'].dt.dayofweek
        df_final['day'] =       df_final['ds'].dt.day


        volume_pred = VolumeXGBoost().model_volume(end_time=df['ds'].max(), days_fine_pred=7, crypto=crypto, time=time) #---------------------------------------------0.1 for testing, fot another exercise remember change it fo at least 7

        df_final['volume'] = volume_pred['Fine Vol']

        if time == 'S' or time == 's':
            df_final['second'] = df_final['ds'].dt.second
            X_final = df_final[['second', 'minute', 'hour', 'dayofweek', 'day', 'volume']]
        else:
            X_final = df_final[['minute', 'hour', 'day', 'dayofweek', 'volume']]

        df_final['Pred Price'] = model.predict(X_final)        

        return df, df_final


