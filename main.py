from indicators import ModelIndicators
 
end_time = '2025-02-12 00:00:00'
 
crypto='ETHUSDT'
time='min'

FirstTime= 5
SecondTime=10
ThirdTime= 15
FourthTime=20
FifthTime= 25

RealPrice = 'off'
ShowImage = 'on'

data = ModelIndicators().XGBoostTimeReduce(end_time, FirstTime, SecondTime, ThirdTime, FourthTime, FifthTime, crypto, time)
plot_1 = ModelIndicators().CreateImagesFivePreds(data=data, RealPrice=RealPrice, ShowImage=ShowImage, crypto=crypto)