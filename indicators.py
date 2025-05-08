from utils import models, get_data_crypto, meassures


import pandas as pd 
import datetime
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import numpy as np  


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

        third_data = third_data.rename(columns={'close_time':'ds','close':'Real Price'})

        df1, first_pred = models().XGBoost_model(data=first_data, time=time)
        df2, second_pred = models().XGBoost_model(data=second_data, time=time)

        first_pred = first_pred.rename(columns={'y':'Fine Pred'})
        second_pred = second_pred.rename(columns= {'y':'Large Pred'})


        first_pred = first_pred[['ds', 'Fine Pred']]
        second_pred = second_pred[['ds', 'Large Pred']] 

        data = pd.merge(left=third_data, right = second_pred, on='ds',how='left')
        data = pd.merge(left=data, right=first_pred, on='ds', how='left')


        data['Diff Fine'] = np.absolute(data['Real Price'] - data['Fine Pred'])
        data['Diff Large'] = np.absolute(data['Real Price'] - data['Large Pred'])

        data = data.dropna(subset=['Diff Fine', 'Diff Large'], ignore_index=True)

        print(np.mean(data['Diff Fine']), ' ',np.mean(data['Diff Large']) )


        difference_fine = round(np.trapezoid(data['Diff Fine'], x=mdates.date2num(data['ds'])), 2)
        difference_large = round(np.trapezoid(data['Diff Large'], x=mdates.date2num(data['ds'])), 2)
        
        min_price = round(third_data['Real Price'].min(), 2)*2/100

        plt.figure(figsize=(10,6))
        plt.plot(data['ds'], data['Fine Pred'], label='Fine prediction', c='black')
        plt.plot(data['ds'], data['Large Pred'], label = 'Large prediction', c='r')
        plt.plot(data['ds'], data['Real Price'], label = 'Real Price', c='green')

        plt.fill_between(data['ds'], data['Fine Pred'], data['Real Price'], label='Diff Real VS Fine P', alpha=0.2)
        plt.fill_between(data['ds'], data['Large Pred'], data['Real Price'], label='Diff Real VS Large P', alpha = 0.2)

        plt.title(f"Diff Fine: {difference_fine}, Diff Large: {difference_large}\n Min Profit {min_price}" )
        plt.ylabel('Price')
        plt.xlabel(f"Prediction since {third_data['ds'].min()} until {third_data['ds'].max()}")
        plt.legend()
        plt.grid(True)
        plt.show()

        return 
    


end_time = '2025-03-15 00:00:00'

days_fine_pred = 5
days_pred = 10
crypto='ETHUSDT'
time='S'


test_1 = indicators().development_model(end_time = end_time, days_fine_pred=days_fine_pred, days_pred = days_pred, crypto = crypto, time = time)
#test_1 = indicators().predictions(days_fine_pred=days_fine_pred, days_pred=days_pred, crypto=crypto, time=time)