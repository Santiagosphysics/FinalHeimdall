import pandas as pd
import xgboost  as xgb

from volume import VolumeXGBoost

from sklearn.metrics  import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


class XGBoost:    

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
    

    
    # def XGBoostMetrcis(self, data, crypto, time ):
    #     data = data[['close_time', 'close', 'volume']].rename(columns={'close_time':'ds', 'close':'y'})
    #     print(data.columns)

    #     df = pd.DataFrame()
        
    #     df['minute'] = data['ds'].dt.minute
    #     df['hour'] = data['ds'].dt.hour
    #     df['dayofweek'] = data['ds'].dt.dayofweek
    #     df['day'] = data['ds'].dt.day

    #     df['y'] = round(data['y'], 3)
    #     df['ds'] = data['ds']
    #     df['volume'] = data['volume'].astype(float)

    #     if time == 'S':
    #         df['second']= data['ds'].dt.second
    #         X = df[['second', 'minute', 'hour', 'dayofweek', 'day', 'volume']]

    #     else:        
    #         X = df[['minute', 'hour', 'day', 'dayofweek', 'volume']]

    #     y = df['y']

    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size= 0.8, random_state=42 )

    #     model = xgb.XGBRegressor(objective='reg:squarederror', 
    #                              n_estimators=2000,
    #                              learning_rate=0.02,
    #                              alpha=50, 
    #                              reg_lambda = 50,                                 
    #                              max_depth = 3,
    #                              random_state = 42,
    #                              early_stopping_rounds=15,
    #                              subsample=0.6,
    #                              colsample_bytree=0.6,
                                 
    #                              )

        
    #     model.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)], verbose=False )

    #     evals_result = model.evals_result()


    #     train_error = evals_result['validation_0']['rmse']
    #     valid_error = evals_result['validation_1']['rmse']

    #     # plt.figure(figsize=(10,6))
    #     # plt.plot(train_error, label='Train')
    #     # plt.plot(valid_error, label ='Validation')
    #     # plt.xlabel('Iteración')
    #     # plt.ylabel('RMSE')
    #     # plt.legend()
    #     # plt.grid(True)
    #     # plt.show()

        
    #     if time == 'min':
    #         df_pred = pd.date_range(start=df['ds'].max() + pd.Timedelta(minutes=1), periods=24*60, freq='min')
    #     elif time == 'S':
    #         df_pred = pd.date_range(start=df['ds'].max() + pd.Timedelta(seconds=1), periods=24*60*60, freq='S')
    #     else:
    #         raise ValueError('Please write a correct option (min, S, s) ')
        
    #     df_final = pd.DataFrame({'ds':df_pred})

    #     df_final['minute'] =    df_final['ds'].dt.minute
    #     df_final['hour'] =      df_final['ds'].dt.hour
    #     df_final['dayofweek'] = df_final['ds'].dt.dayofweek
    #     df_final['day'] =       df_final['ds'].dt.day


    #     volume_pred = VolumeXGBoost().model_volume(end_time=df['ds'].max(), days_fine_pred=7, crypto=crypto, time=time) #---------------------------------------------0.1 for testing, fot another exercise remember change it fo at least 7

    #     df_final['volume'] = volume_pred['Fine Vol']

    #     if time == 'S' or time == 's':
    #         df_final['second'] = df_final['ds'].dt.second
    #         X_final = df_final[['second', 'minute', 'hour', 'dayofweek', 'day', 'volume']]
    #     else:
    #         X_final = df_final[['minute', 'hour', 'day', 'dayofweek', 'volume']]

    #     df_final['Pred Price'] = model.predict(X_final)     
        


    #     y_pred = model.predict(X_test)        

    #     mae = mean_absolute_error(y_test, y_pred)
    #     mse = mean_squared_error(y_test, y_pred)
    #     rmse = root_mean_squared_log_error(y_test, y_pred)
    #     r2 = r2_score(y_test, y_pred)

    #     metrics = {'MeanAbsoluteError':[mae], 'RootMeanSquaredError':[rmse], 'MeanSquaredError':[mse], 'r2':[r2]}

    #     metrics = pd.DataFrame(metrics)

    #     return metrics, df_final



    def XGBoostMetrcis(self, data, crypto, time ):
        data = data[['close_time', 'close', 'open', 'high', 'low', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume','taker_buy_quote_asset_volume', 'volume']].rename(columns={'close_time':'ds', 'close':'y'})

        df = pd.DataFrame()
        
        df['minute'] = data['ds'].dt.minute
        df['hour'] = data['ds'].dt.hour
        df['dayofweek'] = data['ds'].dt.dayofweek
        df['day'] = data['ds'].dt.day

        df['y'] = data['y']
        df['ds'] = data['ds']
        df[['open', 'high', 'low', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'volume']] = data[['open', 'high', 'low', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'volume']].astype(float)

        if time == 'S':
            df['second']= data['ds'].dt.second
            X = df[['second', 'minute', 'hour', 'dayofweek', 'day', 'volume']]

        else:        
            X = df[['minute', 'hour', 'day', 'dayofweek', 'volume']]

        y = df['y']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size= 0.8, random_state=42 )

        model = xgb.XGBRegressor(objective='reg:squarederror', 
                                 n_estimators=2000,
                                 learning_rate=0.02,
                                 alpha=50, 
                                 reg_lambda = 50,                                 
                                 max_depth = 3,
                                 random_state = 42,
                                 early_stopping_rounds=15,
                                 subsample=0.6,
                                 colsample_bytree=0.6,
                                 
                                 )

        
        model.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)], verbose=False )

        evals_result = model.evals_result()


        # train_error = evals_result['validation_0']['rmse']
        # valid_error = evals_result['validation_1']['rmse']

        # plt.figure(figsize=(10,6))
        # plt.plot(train_error, label='Train')
        # plt.plot(valid_error, label ='Validation')
        # plt.xlabel('Iteración')
        # plt.ylabel('RMSE')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        
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

        volume_pred, vol_metrics = VolumeXGBoost().VolumeModel(data= df[['ds', 'volume']] , time=time) #-------------------------------------------
        df_final['volume'] = volume_pred['volume']

        if time == 'S' or time == 's':
            df_final['second'] = df_final['ds'].dt.second
            X_final = df_final[['second', 'minute', 'hour', 'dayofweek', 'day', 'volume']]
        else:
            X_final = df_final[['minute', 'hour', 'day', 'dayofweek', 'volume']]

        df_final['Pred Price'] = model.predict(X_final)     

        y_pred = model.predict(X_test)        

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {'MeanAbsoluteError':[mae], 'RootMeanSquaredError':[rmse], 'MeanSquaredError':[mse], 'r2':[r2]}

        metrics = pd.DataFrame(metrics)

        return metrics, vol_metrics, df_final

