from utils import models, get_data_crypto, meassures

# data = get_data_crypto().download_data_cloud(start_time='2025-04-25', end_time='2025-04-30 13:00:00', crypto="BNBUSDT", time='S')
# prophet_model = models.prophet_model(data=data, time='S')

import pandas as pd 
import datetime
import matplotlib.pyplot as plt 


class indicators:
    def predictions(self, days_fine_pred, days_pred, crypto, time):

        start_time = datetime.datetime.now()
        midle_time = start_time - pd.Timedelta(minutes=days_fine_pred*24*60)
        end_time = start_time - pd.Timedelta(minutes = days_pred*24*60)

        first_data = get_data_crypto().download_data(start_time=midle_time, end_time=start_time, crypto=crypto, time=time)
        second_data = get_data_crypto().download_data(start_time=end_time, end_time=start_time, crypto=crypto, time=time)

        df1, first_pred = models().XGBoost_model(data=first_data, time=time)
        df2, second_pred = models().XGBoost_model(data=second_data, time=time)

        first_pred = first_pred[['ds', 'y']]
        second_pred = second_pred[['ds', 'y']]

        plt.plot(first_pred['ds'], first_pred['y'], label='Fine prediction')
        plt.plot(second_pred['ds'], second_pred['y'], label = 'Large prediction', c='r')
        plt.title('Comparative  between two trains')
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True)
        plt.show()

        return 

    def development_model(self, end_time, days_fine_pred, days_pred, crypto, time):

        end_time = pd.to_datetime(end_time)

        midle_time = end_time - pd.Timedelta(minutes=days_fine_pred*24*60)
        start_time = end_time - pd.Timedelta(minutes = days_pred*24*60)

        future_time = end_time + pd.Timedelta(minutes = 24*60)

        first_data = get_data_crypto().download_data(start_time=midle_time, end_time=end_time, crypto=crypto, time=time)
        second_data = get_data_crypto().download_data(start_time=start_time, end_time=end_time, crypto=crypto, time=time)
        third_data = get_data_crypto().download_data(start_time=end_time, end_time= future_time, crypto=crypto, time=time)
        third_data = third_data.rename(columns={'close_time':'ds','close':'y'})

        df1, first_pred = models().XGBoost_model(data=first_data, time=time)
        df2, second_pred = models().XGBoost_model(data=second_data, time=time)

        first_pred = first_pred.rename(columns={'y':'Fine Pred'})

        first_pred = first_pred[['ds', 'y']]
        second_pred = second_pred[['ds', 'y']]

        data = pd.merge(left=third_data, right = second_pred, on='ds',how='left')

        plt.figure(figsize=(10,6))
        plt.plot(first_pred['ds'], first_pred['y'], label='Fine prediction', c='black')
        plt.plot(second_pred['ds'], second_pred['y'], label = 'Large prediction', c='r')
        plt.plot(third_data['ds'], third_data['y'], label = 'Real Price', c='green')

        plt.fill_between(third_data['ds'], second_pred['y'], third_data['y'], label='Mismatch Large P')
        plt.fill_between(third_data['ds'], first_pred['y'], third_data['y'], label = 'Mismatch Fine P')

        plt.title(f"Prediction since {third_data['ds'].min()} until {third_data['ds'].max()}")
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True)
        plt.show()

        return 
    


end_time = '2025-03-15 00:00:00'

days_fine_pred = 1
days_pred = 5
crypto='ETHUSDT'
time='min'


test_1 = indicators().development_model(end_time = end_time, days_fine_pred=days_fine_pred, days_pred = days_pred, crypto = crypto, time = time)