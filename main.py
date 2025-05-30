
from indicators import ModelIndicators
 
end_time = '2025-05-24 13:00:00'
 
crypto='BNBUSDT'
time='min'

FirstTime= 2
SecondTime=4
ThirdTime =6
FourthTime=8
FifthTime= 10

RealPrice = 'off'
ShowImage = 'on'

metrics, data = ModelIndicators().XGBoostTimeReduce(end_time, FirstTime, SecondTime, ThirdTime, FourthTime, FifthTime, crypto, time)
plot_1 = ModelIndicators().CreateImagesFivePreds(data=data, RealPrice=RealPrice, ShowImage=ShowImage, crypto=crypto)
print(metrics)


    # def XGBoostMetrcis(self, data, crypto, time ):
    #     data = data[['close_time', 'close', 'volume']].rename(columns={'close_time':'ds', 'close':'y'})

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
    #                              n_estimators=2000, # Best development with 500 trees
    #                              learning_rate=0.15,
    #                              alpha=1, # Best development with 1
    #                              reg_lambda = 1,
    #                              random_state = 42
    #                              )
    #     model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20)
        

    #     y_pred = model.predict(X_test)        

    #     mae = mean_absolute_error(y_test, y_pred)
    #     mse = mean_squared_error(y_test, y_pred)
    #     r2 = r2_score(y_test, y_pred)

    #     metrics = {'MeanAbsoluteError':[mae], 'MeanSquaredError':[mse], 'r2':[r2]}

    #     metrics = pd.DataFrame(metrics)

    #     return metrics
