from utils import get_data_crypto, models
import matplotlib.pyplot as plt 
import pandas as pd 

class volume:
    def volume_all_variables(self, start_time, end_time, crypto, time):

        data = get_data_crypto().download_data_volume(start_time=start_time, end_time=end_time, crypto=crypto, time=time)
        data = data[[ 'close_time', 'close', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume','taker_buy_quote_asset_volume']]
        return data




    def model_volume(self, end_time, days_fine_pred, days_pred, crypto, time):
        end_time = pd.to_datetime(end_time)

        midle_time = end_time - pd.Timedelta(minutes=days_fine_pred*24*60)
        start_time = end_time - pd.Timedelta(minutes = days_pred*24*60)

        future_time = end_time + pd.Timedelta(minutes = 24*60)

        first_data = get_data_crypto().download_data_volume(start_time=midle_time, end_time=end_time, crypto=crypto, time=time)
        second_data = get_data_crypto().download_data_volume(start_time=start_time, end_time=end_time, crypto=crypto, time=time)
        third_data = get_data_crypto().download_data_volume(start_time=end_time, end_time= future_time, crypto=crypto, time=time)


        first_data = first_data[['close_time', 'volume']]
        second_data = second_data[['close_time', 'volume']]
        third_data = third_data[['close_time', 'volume']]

        df1, first_pred = models().XGBoost_model(data=first_data, time=time)
        df2, second_pred = models().XGBoost_model(data=second_data, time=time)

        first_pred = first_pred[['ds', 'y']].rename(columns={'ds':'close_time','y':'Fine Vol'})
        second_pred = second_pred[['ds','y']].rename(columns={'ds':'close_time','y':'Large Vol'})

        data = pd.merge(left=third_data, right=first_pred, how='left', on='close_time')
        data = pd.merge(left=data, right=second_pred, how='left', on='close_time')

        data = data.dropna(subset=['volume'], ignore_index=True)

        
        plt.plot(data['close_time'], data['volume'])
        plt.plot(data['close_time'], data['Fine Vol'], c='green')
        plt.plot(data['close_time'], data['Large Vol'], c='black')
        

        plt.show()

        print(data['volume'])


    


end_time = '2025-03-15 01:00:00'

days_fine_pred = 1
days_pred = 2
crypto='ETHUSDT'
time='min'

test_1 = volume().model_volume(end_time, days_fine_pred, days_pred, crypto, time)
print(test_1)