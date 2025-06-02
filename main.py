from indicators import ModelIndicators
 
end_time = '2024-11-17 00:00:00'
 
crypto='ETHUSDT'
time='min'

FirstTime= 10
SecondTime= 13
ThirdTime = 17
FourthTime=22
FifthTime= 25


RealPrice = 'on'
ShowImage = 'on'

metrics, data = ModelIndicators().XGBoostTimeReduce(end_time, FirstTime, SecondTime, ThirdTime, FourthTime, FifthTime, crypto, time)
plot_1 = ModelIndicators().CreateImagesFivePreds(data=data, RealPrice=RealPrice, ShowImage=ShowImage, crypto=crypto)
print(metrics)