import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np 

from dotenv import load_dotenv

from binance.client import Client

from prophet import Prophet
import xgboost  as xgb

import  requests


import os  

class get_data_crypto:
    def download_data(self, start_time, end_time, crypto, time):

        data_time = {'min':Client.KLINE_INTERVAL_1MINUTE,'S':Client.KLINE_INTERVAL_1SECOND,'hours':Client.KLINE_INTERVAL_1HOUR, 's':Client.KLINE_INTERVAL_1SECOND}
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

        df = df[['close_time', 'close']]
        df = df.rename({'close_time':'ds', 'close':'y'})

        return df

    def download_data_volume(self, start_time, end_time, crypto, time):

        data_time = {'min':Client.KLINE_INTERVAL_1MINUTE,'S':Client.KLINE_INTERVAL_1SECOND,'hours':Client.KLINE_INTERVAL_1HOUR,}
        if time not in data_time:
            raise ValueError(f'El parámetro {time} no está en las opciones {data_time}')
        
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


        return df

    def download_data_cloud(self, start_time, end_time, crypto, time):

        data_time = {'min':'min', 'S':'S'}

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
    def prophet_model(self, data, time):
        if len(data.columns) > 2:
            raise ValueError('Your dataset has more than two columns {data.columns}')
        
        data.columns = ['ds','y']

        model = Prophet(daily_seasonality=True)
        model.fit(data)

        if time == 'min':
            future = model.make_future_dataframe(periods=24*60, freq = time)
        elif time == 'S':
            future = model.make_future_dataframe(periods=24*60*60, freq = time)
        else:
            raise ValueError('Please write a correct option for time ("S" or "min")')

        predict = model.predict(future)

        fig1 = model.plot(predict)
        plt.title('Prediction for the price for the next 24 hours')
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.show()

        return data, predict



    def XGBoost_plot(self, data, time ):
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

        plt.figure(figsize=(10,6))
        plt.plot(df_final['ds'], response, label='Predicted future labels', color='r')
        plt.plot(df['ds'], df['y'], label = 'Real price')
        plt.title(f'Prediction since using XGBoost')
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.grid(True)
        plt.show()

        df_real = pd.DataFrame({'ds':df['ds'],'y':df['y']})

        df_result = pd.concat([df_real, df_final], ignore_index=True)

        df_result = df_result[['ds', 'y']]

        return df, df_final
    

    def XGBoost_model(self, data, time ):
        data.columns = ['ds', 'y']
        df = pd.DataFrame()
        
        df['minute'] = data['ds'].dt.minute
        df['hour'] = data['ds'].dt.hour
        df['dayofweek'] = data['ds'].dt.dayofweek
        df['day'] = data['ds'].dt.day

        df['y'] = round(data['y'], 3)
        df['ds'] = data['ds']

        if time == 'S':
            df['second']= data['ds'].dt.second
            X = df[['second', 'minute', 'hour', 'dayofweek', 'day']]

        else:        
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





        

        if time == 'S' or time == 's':
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



    def XGBoost_model_bootstrap(self, data, time):
        data.columns = ['ds', 'y']
        df = pd.DataFrame()

        # Preprocesamiento de características temporales
        if time == 's':
            df['second'] = data['ds'].dt.second

        df['minute'] = data['ds'].dt.minute
        df['hour'] = data['ds'].dt.hour
        df['dayofweek'] = data['ds'].dt.dayofweek
        df['day'] = data['ds'].dt.day
        df['y'] = round(data['y'], 3)
        df['ds'] = data['ds']

        X = df[['minute', 'hour', 'day', 'dayofweek']]
        y = df['y']

        # Entrenar modelo XGBoost
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
        model.fit(X, y)

        # Generar fechas futuras
        if time == 'm':
            df_pred = pd.date_range(start=df['ds'].max() + pd.Timedelta(minutes=1), periods=24*60, freq='min')
        elif time == 's':
            df_pred = pd.date_range(start=df['ds'].max() + pd.Timedelta(seconds=1), periods=24*60*60, freq='S')
        else:
            raise ValueError('Please write a correct option (m, s)')

        df_final = pd.DataFrame({'ds': df_pred})
        df_final['minute'] = df_final['ds'].dt.minute
        df_final['hour'] = df_final['ds'].dt.hour
        df_final['dayofweek'] = df_final['ds'].dt.dayofweek
        df_final['day'] = df_final['ds'].dt.day

        if time == 's':
            df_final['second'] = df_final['ds'].dt.second
            X_final = df_final[['second', 'minute', 'hour', 'dayofweek', 'day']]
        else:
            X_final = df_final[['minute', 'hour', 'day', 'dayofweek']]



        # Predecir y simular intervalo de confianza (ejemplo con bootstrapping)
        response = model.predict(X_final)
        n_bootstrap = 100
        bootstrap_preds = np.zeros((n_bootstrap, len(response)))

        for i in range(n_bootstrap):
            sample_idx = np.random.choice(len(X), size=len(X), replace=True)
            model_i = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.2)
            model_i.fit(X.iloc[sample_idx], y.iloc[sample_idx])
            bootstrap_preds[i] = model_i.predict(X_final)

        ci_lower = np.percentile(bootstrap_preds, 2.5, axis=0)  # Percentil 2.5%
        ci_upper = np.percentile(bootstrap_preds, 97.5, axis=0)  # Percentil 97.5%

        # Gráfica
        plt.figure(figsize=(12, 6))
        plt.scatter(df['ds'], df['y'], color='black', label='Datos históricos',alpha=0.6,s=15)    
        plt.plot(df['ds'], df['y'])    
        plt.plot(  df_final['ds'], response,   color='red',   label='Predicción XGBoost',  linewidth=2)  
        # 3. Intervalo de confianza (área azul)
        plt.fill_between(df_final['ds'], ci_lower, ci_upper, color='blue', alpha=0.2,label='Intervalo de confianza (95%)' )
        
        # Personalización
        plt.title(f'Predicción desde {df["ds"].max()} con XGBoost', fontsize=14)
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Precio', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # DataFrame de salida
        df_final['y'] = response
        df_real = pd.DataFrame({'ds': df['ds'], 'y': df['y']})
        df_result = pd.concat([df_real, df_final], ignore_index=True)

        return df_result[['ds', 'y']]


class meassures():
    def data_predict(self, model_pred, time, crypto ):

        model = model_pred.copy()
        model = model.rename(columns={'y':'yhat'})
        model = model[['ds', 'yhat']]

        model = model[model['ds']>=model['ds'].max() - pd.Timedelta(minutes=24*60)]
        data = get_data_crypto().download_data(start_time =model['ds'].max() - pd.Timedelta(minutes=24*60) , end_time=model['ds'].max(), crypto=crypto, time=time)
        data.columns = ['ds','real price']

        if time == 'min':
            model['ds'] = model['ds'].dt.round(time)
            data['ds'] = data['ds'].dt.round(time)
            
        elif time == 'S':
            model['ds'] = model['ds'].dt.round(time)
            data['ds'] = data['ds'].dt.round(time)
        else:
            raise ValueError(f'Please write a correct option for the time {time}')

        df = pd.merge(left=model, right=data, on='ds', how='left')
        df = df.dropna(axis=0).reset_index(drop=True)

        df['difference'] = np.absolute(df['yhat'] - df['real price'])
        df['mismatch'] = df['difference']/df['real price']

        difference_shadow = round(np.trapz(df['difference'], x=mdates.date2num(df['ds'])), 2)
        mismatch_percent = round(np.trapz(df['mismatch'], x=mdates.date2num(df['ds'])), 4)
        percent_rent = round( df['yhat'][df['ds'] == df['ds'].min()][0]*2/100  ,2)

        plt.figure(figsize=(10,6))
        plt.plot(df['ds'], df['yhat'], c='r', label = 'Predict Price' )
        plt.plot(df['ds'], df['real price'], c='g', label='Real Price')

        plt.fill_between(df['ds'], df['real price'], df['yhat'], alpha=0.3, label='Area betwen')

        plt.title(f' Accuracy {1-mismatch_percent}, Min Profit: {percent_rent}\n  Mismatch in dollars: {difference_shadow} Error percent: {mismatch_percent}')
        plt.ylabel(f'Price of {crypto}')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.show()

        
        return df   
