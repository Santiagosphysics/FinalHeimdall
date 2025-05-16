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


        volume_pred = VolumeXGBoost().model_volume(end_time=df['ds'].max(), days_fine_pred=10, days_pred=1, crypto=crypto, time=time) #No funciona la predicción gruesa
        real_data = get_data_crypto().download_data(start_time=df_final['ds'].min(), end_time=df_final['ds'].max(), crypto=crypto, time=time)
        real_data = real_data.rename(columns={'close_time':'ds', 'close':'Real Price'})

        df_final = pd.merge(left=df_final, right=real_data, how='left', on='ds')

        df_final['volume'] = volume_pred['Fine Vol']
        df_final = df_final.dropna(axis=0, ignore_index=True)

        if time == 'S' or time == 's':
            df_final['second'] = df_final['ds'].dt.second
            X_final = df_final[['second', 'minute', 'hour', 'dayofweek', 'day', 'volume']]
        else:
            X_final = df_final[['minute', 'hour', 'day', 'dayofweek', 'volume']]

        df_final['Pred Price'] = model.predict(X_final)

        df_final['Diff Fine'] = np.absolute(df_final['Real Price'] - df_final['Pred Price'])
  
        difference_fine = round(np.trapz(df_final['Diff Fine'], x=mdates.date2num(df_final['ds'])), 2)

        # plt.plot(df_final['ds'], df_final['Pred Price'], label='Pred Price')
        # plt.plot(df_final['ds'], df_final['Real Price'], c='green', label='Real Price')

        # plt.fill_between(df_final['ds'], df_final['Real Price'], df_final['Pred Price'], alpha=0.3)
        # plt.title(f"Diff Fine: {difference_fine}" )
        # plt.ylabel('Price')
        # plt.xlabel(f"Prediction since {df_final['ds'].min()} until {df_final['ds'].max()}")
        # plt.legend()

        # plt.show()

        return df, df_final


# start_time = '2025-03-14 00:00:00'
# end_time = '2025-03-15 01:00:00'

# days_fine_pred = 1
# days_pred = 2
# crypto='ETHUSDT'
# time='s'

# data = get_data_crypto().download_data_volume(start_time=start_time, end_time=end_time, crypto=crypto, time=time)
# test_1 = XGBoost().XGBoost_final(data=data, crypto=crypto, time=time)

# data = get_data_crypto().download_data(start_time=start_time, end_time=end_time, crypto=crypto, time=time)
# test_2 = XGBoost().XGBoost_model(data=data, time=time)