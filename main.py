from indicators import ModelIndicators
from utils import get_data_crypto

import pandas as pd 
 
end_time = '2025-05-26 06:00:00'
start_time = '2025-05-01 06:00:00'
 
crypto='BNBUSDT'
time='min'

FirstTime = 5
SecondTime= 10
ThirdTime = 15
FourthTime= 20
FifthTime=  25

RealPrice = 'on'
ShowImage = 'on'

metrics, data = ModelIndicators().XGBoostTimeReduce(end_time, FirstTime, SecondTime, ThirdTime, FourthTime, FifthTime, crypto, time)
plot_1 = ModelIndicators().CreateImagesFivePreds(data=data, RealPrice=RealPrice, ShowImage=ShowImage, crypto=crypto)
print(metrics)

# df_1 = pd.read_excel('./datas/five_days.xlsx')
# df_2 = pd.read_excel('./datas/ten_days.xlsx')
# df_3 = pd.read_excel('./datas/fifteen_days.xlsx')
# df_4 = pd.read_excel('./datas/twenty_days.xlsx')
# df_5 = pd.read_excel('./datas/twentysix_days.xlsx')

