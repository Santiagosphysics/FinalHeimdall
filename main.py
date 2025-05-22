from indicators import ModelIndicators
 
SinceDate = '2025-05-20 23:00:00'
 
crypto='ETHUSDT'
time='min'

FirstTime= 0.5
SecondTime=0.10
ThirdTime= 0.15
FourthTime=0.20
FifthTime= 0.25

RealPrice = 'off'

test = ModelIndicators().ManyPlots(SinceDate=SinceDate, FirstTime=FirstTime, SecondTime=SecondTime, 
                                   ThirdTime=ThirdTime, FourthTime=FourthTime, FifthTime=FifthTime, 
                                   crypto=crypto, time=time, RealPrice=RealPrice)
#test = ModelIndicators().predictions(FirstTime, SecondTime, ThirdTime, FourthTime, FifthTime, crypto, time)
#test = ModelIndicators().XGBoostFinalFivePreds(end_time, FirstTime, SecondTime, ThirdTime, FourthTime, FifthTime, crypto, time)
#plot_1 = ModelIndicators().CreateImagesFivePreds(data=test, RealPrice='off', ShowImage='on')