from indicators import ModelIndicators
 
crypto='ETHUSDT'
time='min'

FirstTime=  0.2
SecondTime= 0.4
ThirdTime=  0.6
FourthTime= 0.8
FifthTime=  1


RealPrice = 'off'
ShowImage = 'on'

test = ModelIndicators().predictions(FirstTime, SecondTime, ThirdTime, FourthTime, FifthTime, crypto, time)
plot_2 = ModelIndicators().CreateImagesFivePreds(data=test, RealPrice=RealPrice, ShowImage=ShowImage)