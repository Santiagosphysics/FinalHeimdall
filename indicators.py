
import matplotlib
matplotlib.use('TkAgg')

from utils import models, get_data_crypto, meassures
from XGBoost_model import XGBoost

import pandas as pd 
import datetime
import numpy as np  
import math

import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvas

from PIL import Image

class ModelIndicators:
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


        difference_fine = round(np.trapz(data['Diff Fine'], x=mdates.date2num(data['ds'])), 2)
        difference_large = round(np.trapz(data['Diff Large'], x=mdates.date2num(data['ds'])), 2)
        
        min_price = round(third_data['Real Price'].min()*2/100 , 2) #Revisar problema ------------------------------------------------

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
    

    def development_XGBoost_final(self, end_time, days_fine_pred, days_pred, crypto, time):

        end_time = pd.to_datetime(end_time)

        midle_time = end_time - pd.Timedelta(minutes=days_fine_pred*24*60)
        start_time = end_time - pd.Timedelta(minutes = days_pred*24*60)

        future_time = end_time + pd.Timedelta(minutes = 24*60)

        first_data = get_data_crypto().download_data_volume(start_time=midle_time, end_time=end_time, crypto=crypto, time=time)
        second_data = get_data_crypto().download_data_volume(start_time=start_time, end_time=end_time, crypto=crypto, time=time)
        third_data = get_data_crypto().download_data(start_time=end_time, end_time= future_time, crypto=crypto, time=time)

        third_data = third_data.rename(columns={'close_time':'ds','close':'Real Price'})

        df1, first_pred = XGBoost().XGBoost_final(data=first_data, time=time, crypto=crypto)
        df2, second_pred = XGBoost().XGBoost_final(data=second_data, time=time, crypto=crypto)

        first_pred = first_pred.rename(columns={'Pred Price':'Fine Pred'})
        second_pred = second_pred.rename(columns= {'Pred Price':'Large Pred'})


        first_pred = first_pred[['ds', 'Fine Pred']]
        second_pred = second_pred[['ds', 'Large Pred']] 

        data = pd.merge(left=third_data, right = second_pred, on='ds',how='left')
        data = pd.merge(left=data, right=first_pred, on='ds', how='left')


        data['Diff Fine'] = np.absolute(data['Real Price'] - data['Fine Pred'])
        data['Diff Large'] = np.absolute(data['Real Price'] - data['Large Pred'])

        data = data.dropna(subset=['Diff Fine', 'Diff Large'], ignore_index=True)

        #print(np.mean(data['Diff Fine']), ' ',np.mean(data['Diff Large']) )


        #difference_fine = round(np.trapz(data['Diff Fine'], x=mdates.date2num(data['ds'])), 2)
        #difference_large = round(np.trapz(data['Diff Large'], x=mdates.date2num(data['ds'])), 2)
        
        #min_price = round(third_data['Real Price'].min()*2/100 , 2) #Revisar problema ------------------------------------------------

        #plt.figure(figsize=(10,6))
        #plt.plot(data['ds'], data['Fine Pred'], label='Fine prediction', c='black')
        #plt.plot(data['ds'], data['Large Pred'], label = 'Large prediction', c='r')
        #plt.plot(data['ds'], data['Real Price'], label = 'Real Price', c='green')

        #plt.fill_between(data['ds'], data['Fine Pred'], data['Real Price'], label='Diff Real VS Fine P', alpha=0.2)
        #plt.fill_between(data['ds'], data['Large Pred'], data['Real Price'], label='Diff Real VS Large P', alpha = 0.2)

        #plt.title(f"Diff Fine: {difference_fine}, Diff Large: {difference_large}\n Min Profit {min_price}" )
        #plt.ylabel('Price')
        #plt.xlabel(f"Prediction since {third_data['ds'].min()} until {third_data['ds'].max()}")
        #plt.legend()
        #plt.grid(True)
        #plt.show()

        return data
    
    def CreateImages(self, data):
        difference_fine = round(np.trapz(data['Diff Fine'], x=mdates.date2num(data['ds'])), 2)
        difference_large = round(np.trapz(data['Diff Large'], x=mdates.date2num(data['ds'])), 2)
        
        fig = plt.figure(figsize=(10,6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        ax.plot(data['ds'], data['Fine Pred'], label='Fine prediction', c='black')
        ax.plot(data['ds'], data['Large Pred'], label = 'Large prediction', c='r')
        ax.plot(data['ds'], data['Real Price'], label = 'Real Price', c='green')

        ax.fill_between(data['ds'], data['Fine Pred'], data['Real Price'], label='Diff Real VS Fine P', alpha=0.2)
        ax.fill_between(data['ds'], data['Large Pred'], data['Real Price'], label='Diff Real VS Large P', alpha = 0.2)

        ax.set_title(f"Diff Fine: {difference_fine}, Diff Large: {difference_large}\n" )
        ax.set_ylabel('Price')
        ax.set_xlabel(f"Prediction since {data['ds'].min()} until {data['ds'].max()}")
        ax.legend()
        ax.grid(True)

        canvas.draw()

        width, height = map(int, fig.get_size_inches()*fig.get_dpi())
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(int(height), int(width), 4)
        image = image[:, :, :3]

        plt.close(fig)
        
        return image






    
    def ManyPlots(self, since_date, days_fine_pred, days_pred, crypto, time ):
        
        date = pd.to_datetime(since_date)
        today = pd.Timestamp.now()
        diff_days = (today - date).days
        test_days = {}

        # diff_days = 3 #Test for dicress the time, only for testing 

        for i in range(diff_days):
            date = date + pd.Timedelta(days=1)
            test_days[f'day_{i+1}'] = ModelIndicators().development_XGBoost_final(end_time=date, days_fine_pred=days_fine_pred, days_pred=days_pred, crypto=crypto, time=time)
            test_days[f'image_{i+1}'] = ModelIndicators().CreateImages(test_days[f'day_{i+1}'])
            img = Image.fromarray(test_days[f'image_{i+1}'])
            img.save(f'./plots/image_{i+1}.png')

        list_img = {name:img for name, img in test_days.items() if 'image' in name}

        num_images = len(list_img)
        cols = 3
        rows = math.ceil(num_images / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        # Si hay solo una fila o una columna, convertir a arreglo 2D para evitar errores
        axes = np.array(axes).reshape(rows, cols)

        for idx, (key, image) in enumerate(list_img.items()):
            row = idx // cols
            col = idx % cols
            axes[row, col].imshow(image.astype(np.uint8))
            axes[row, col].axis('off')
            axes[row, col].set_title(key)

        # Ocultar los ejes sobrantes si el número de imágenes no llena toda la grilla
        for idx in range(num_images, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        fig.savefig("full_grid_plot.png")


        plt.tight_layout()
        plt.show()




# end_time = '2025-04-12 23:00:00'
# days_fine_pred = 0.01
# days_pred = 0.05
# crypto='ETHUSDT'
# time='min'



# # test_1 = indicators().predictions(days_fine_pred=days_fine_pred, days_pred=days_pred, crypto=crypto, time=time)

#test_1 = ModelIndicators().development_model(end_time = end_time, days_fine_pred=days_fine_pred, days_pred = days_pred, crypto = crypto, time = time)
#test_1 = ModelIndicators().development_XGBoost_final(end_time = end_time, days_fine_pred=days_fine_pred, days_pred = days_pred, crypto = crypto, time = time)

#image = ModelIndicators().CreateImages(data=test_1)

#test_1 = ModelIndicators().ManyPlots(since_date=end_time, crypto=crypto, time = time, days_fine_pred=days_fine_pred, days_pred = days_pred)


# test = ModelIndicators().ManyPlots(since_date=end_time, days_fine_pred=days_fine_pred, days_pred = days_pred, crypto=crypto, time=time)