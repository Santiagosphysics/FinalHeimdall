import matplotlib.pyplot as plt
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



    def XGBoost_model(data, time ):
        data.columns = ['ds', 'y']
        df = pd.DataFrame()

        if time == 's':
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

        if time == 'm':
            df_pred = pd.date_range(start=df['ds'].max() + pd.Timedelta(minutes=1), periods=24*60, freq='min')
        elif time == 's':
            df_pred = pd.date_range(start=df['ds'].max() + pd.Timedelta(seconds=1), periods=24*60*60, freq='S')
        else:
            raise ValueError('Please write a correct option (m, s) ')
        

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

        plt.figure(figsize=(10,10))
        plt.plot(df_final['ds'], response, label='Predicted future labels', color='r')
        plt.plot(df['ds'], df['y'], label = 'Real price', color='b')
        plt.title(f'Prediction since {df['ds'].max()} using XGBoost')
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.show()

        df_real = pd.DataFrame({'ds':df['ds'],'y':df['y']})

        df_result = pd.concat([df_real, df_final], ignore_index=True)

        return df_result[['ds', 'y']]



    def XGBoost_model_bootstrap(data, time):
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
            model_i = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
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
