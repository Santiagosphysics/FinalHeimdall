
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
from IPython.display import display
from PIL import Image as PILImage

class ModelIndicators:
    def predictions(self, FirstTime, SecondTime, ThirdTime, FourthTime, FifthTime, crypto, time):
        start_time = datetime.datetime.now()

        first_time = start_time - pd.Timedelta(minutes  = FirstTime*24*60)
        second_time = start_time - pd.Timedelta(minutes = SecondTime*24*60)
        third_time = start_time - pd.Timedelta(minutes  = ThirdTime*24*60 )
        fourth_time = start_time - pd.Timedelta(minutes = FourthTime*24*60)
        fifth_time = start_time - pd.Timedelta(minutes   = FifthTime*24*60)

        first_data  = get_data_crypto().download_data_volume(start_time= first_time, end_time=start_time, crypto=crypto, time=time)
        second_data = get_data_crypto().download_data_volume(start_time=second_time, end_time=start_time, crypto=crypto, time=time)
        third_data  = get_data_crypto().download_data_volume(start_time= third_time, end_time=start_time, crypto=crypto, time=time)
        fourth_data = get_data_crypto().download_data_volume(start_time=fourth_time, end_time=start_time, crypto=crypto, time=time)
        fifth_data  = get_data_crypto().download_data_volume(start_time= fifth_time, end_time=start_time, crypto=crypto, time=time)

        df1, first_pred = XGBoost().XGBoost_final(data=first_data, time=time, crypto=crypto)
        df2, second_pred= XGBoost().XGBoost_final(data=second_data, time=time, crypto=crypto)
        df3, third_pred = XGBoost().XGBoost_final(data=third_data, time=time, crypto=crypto)
        df4, fourth_pred =XGBoost().XGBoost_final(data=fourth_data, time=time, crypto=crypto)
        df5, fifth_pred = XGBoost().XGBoost_final(data=fifth_data, time=time, crypto=crypto)

        first_pred  = first_pred.rename( columns={'Pred Price' :f'Train {FirstTime} days'})
        second_pred = second_pred.rename(columns= {'Pred Price':f'Train {SecondTime} days'})
        third_pred  = third_pred.rename(columns= {'Pred Price' :f'Train {ThirdTime} days'})
        fourth_pred = fourth_pred.rename(columns= {'Pred Price':f'Train {FourthTime} days'})
        fifth_pred  = fifth_pred.rename(columns= {'Pred Price' :f'Train {FifthTime} days'})

        first_pred = first_pred[['ds', f'Train {FirstTime} days']]
        second_pred = second_pred[['ds', f'Train {SecondTime} days']] 
        third_pred  = third_pred[['ds', f'Train {ThirdTime} days']] 
        fourth_pred = fourth_pred[['ds', f'Train {FourthTime} days']] 
        fifth_pred  = fifth_pred[['ds', f'Train {FifthTime} days']] 

        data = pd.merge(left=first_pred,right=second_pred, on='ds', how='left')
        data = pd.merge(left=data, right=third_pred, on='ds', how='left')
        data = pd.merge(left=data, right=fourth_pred, on='ds', how='left')
        data = pd.merge(left=data, right=fifth_pred, on='ds', how='left')


        data = data.dropna().reset_index(drop=True)

        return data

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

        return data
    
    def XGBoostFinalFivePreds(self, end_time, FirstTime, SecondTime, ThirdTime, FourthTime, FifthTime, crypto, time):

        end_time = pd.to_datetime(end_time)

        first_time = end_time - pd.Timedelta(minutes  = FirstTime*24*60)
        second_time = end_time - pd.Timedelta(minutes = SecondTime*24*60)
        third_time = end_time - pd.Timedelta(minutes  = ThirdTime*24*60 )
        fourth_time = end_time - pd.Timedelta(minutes = FourthTime*24*60)
        fifth_time = end_time - pd.Timedelta(minutes   = FifthTime*24*60)

        future_time = end_time + pd.Timedelta(minutes = 24*60)

        first_data  = get_data_crypto().download_data_volume(start_time= first_time, end_time=end_time, crypto=crypto, time=time)
        second_data = get_data_crypto().download_data_volume(start_time=second_time, end_time=end_time, crypto=crypto, time=time)
        third_data  = get_data_crypto().download_data_volume(start_time= third_time, end_time=end_time, crypto=crypto, time=time)
        fourth_data = get_data_crypto().download_data_volume(start_time=fourth_time, end_time=end_time, crypto=crypto, time=time)
        fifth_data  = get_data_crypto().download_data_volume(start_time= fifth_time, end_time=end_time, crypto=crypto, time=time)


        real_data = get_data_crypto().download_data(start_time=end_time, end_time= future_time, crypto=crypto, time=time)

        real_data = real_data.rename(columns={'close_time':'ds','close':'Real Price'})

        df1, first_pred =  XGBoost().XGBoost_final(data=first_data, time=time, crypto=crypto)
        df2, second_pred= XGBoost().XGBoost_final(data=second_data, time=time, crypto=crypto)
        df3, third_pred = XGBoost().XGBoost_final(data=third_data, time=time, crypto=crypto)
        df4, fourth_pred =XGBoost().XGBoost_final(data=fourth_data, time=time, crypto=crypto)
        df5, fifth_pred = XGBoost().XGBoost_final(data=fifth_data, time=time, crypto=crypto)

        first_pred  = first_pred.rename( columns={'Pred Price' :f'Train {FirstTime} days'})
        second_pred = second_pred.rename(columns= {'Pred Price':f'Train {SecondTime} days'})
        third_pred  = third_pred.rename(columns= {'Pred Price' :f'Train {ThirdTime} days'})
        fourth_pred = fourth_pred.rename(columns= {'Pred Price':f'Train {FourthTime} days'})
        fifth_pred  = fifth_pred.rename(columns= {'Pred Price' :f'Train {FifthTime} days'})

        first_pred = first_pred[['ds', f'Train {FirstTime} days']]
        second_pred = second_pred[['ds', f'Train {SecondTime} days']] 
        third_pred  = third_pred[['ds', f'Train {ThirdTime} days']] 
        fourth_pred = fourth_pred[['ds', f'Train {FourthTime} days']] 
        fifth_pred  = fifth_pred[['ds', f'Train {FifthTime} days']] 

        data = pd.merge(left=real_data, right = second_pred, on='ds',how='left')
        data = pd.merge(left=data, right=first_pred, on='ds', how='left')
        data = pd.merge(left=data, right=third_pred, on='ds', how='left')
        data = pd.merge(left=data, right=fourth_pred, on='ds', how='left')
        data = pd.merge(left=data, right=fifth_pred, on='ds', how='left')


        data['Diff Fine'] = np.absolute(data['Real Price'] - data[f'Train {SecondTime} days'])
        data['Diff Large'] = np.absolute(data['Real Price'] - data[f'Train {FourthTime} days'])

        data = data.dropna().reset_index(drop=True)

        return data
    

    
    def CreateImages(self, data, ShowImage):
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
        if ShowImage == 'on':
            plt.show()
        elif ShowImage == 'off':
            print('Img')
        else:
            raise ValueError("Please write 'on' if you want to show the real price else write 'off'")

        canvas.draw()

        width, height = map(int, fig.get_size_inches()*fig.get_dpi())
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(int(height), int(width), 4)
        image = image[:, :, :3]

        plt.close(fig)
        
        return image
    
    def CreateImagesFivePreds(self, data, RealPrice, ShowImage):
        names_columns = [col for col in data.columns if 'Train' in col]
        
        fig = plt.figure(figsize=(15,6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        ax.plot(data['ds'], data[names_columns[0]], label = names_columns[0], c='black')
        ax.plot(data['ds'], data[names_columns[1]], label = names_columns[1], c='orange')
        ax.plot(data['ds'], data[names_columns[2]], label = names_columns[2], c='blue')
        ax.plot(data['ds'], data[names_columns[3]], label = names_columns[3], c='gray')
        ax.plot(data['ds'], data[names_columns[4]], label = names_columns[4], c='brown')

        RealPrice = RealPrice.lower()

        if RealPrice == 'on':
            ax.plot(data['ds'], data['Real Price'], label = 'Real Price', c='green')
        elif RealPrice == 'off':
            print('With out real price')
        else:
            raise ValueError("Please write 'on' if you want to show the real price else write 'off'")


        ax.set_title(f"Many predictions" )
        ax.set_ylabel('Price')
        ax.set_xlabel(f"Prediction since {data['ds'].min()} until {data['ds'].max()}")
        ax.legend()
        ax.grid(True)
        if ShowImage == 'on':
            plt.show()
        elif ShowImage == 'off':
            print('Img')
        else:
            raise ValueError("Please write 'on' if you want to show the real price else write 'off'")

        canvas.draw()

        width, height = map(int, fig.get_size_inches()*fig.get_dpi())
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(int(height), int(width), 4)
        image = image[:, :, :3]

        plt.close(fig)
        
        return image



    
    def ManyPlots(self, SinceDate, RealPrice, FirstTime, SecondTime, ThirdTime, FourthTime, FifthTime, crypto, time ):
        
        date = pd.to_datetime(SinceDate)
        today = pd.Timestamp.now()
        diff_days = (today - date).days
        test_days = {}

        # diff_days = 3 #Test for dicress the time, only for testing 

        for i in range(diff_days):
            date = date + pd.Timedelta(days=1)
            test_days[f'day_{i+1}'] = ModelIndicators().XGBoostFinalFivePreds(end_time=date, FirstTime=FirstTime, SecondTime=SecondTime, ThirdTime=ThirdTime, FourthTime=FourthTime, FifthTime=FifthTime, crypto=crypto, time=time)
            test_days[f'image_{i+1}'] = ModelIndicators().CreateImagesFivePreds(test_days[f'day_{i+1}'], ShowImage='off', RealPrice=RealPrice)
            img = Image.fromarray(test_days[f'image_{i+1}'])
            img.save(f'./plots/image_{i+1}.png')

        list_img = {name:img for name, img in test_days.items() if 'image' in name}

        num_images = len(list_img)
        cols = 3
        rows = math.ceil(num_images / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        axes = np.array(axes).reshape(rows, cols)

        for idx, (key, image) in enumerate(list_img.items()):
            row = idx // cols
            col = idx % cols
            axes[row, col].imshow(image.astype(np.uint8))
            axes[row, col].axis('off')
            axes[row, col].set_title(key)

        for idx in range(num_images, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()
        fig.savefig("full_grid_plot.png")

        display(PILImage.open("full_grid_plot.png"))
        print('Its Working :)')

        return test_days
