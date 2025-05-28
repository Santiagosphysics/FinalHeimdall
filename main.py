from utils import meassures

time_in= '03:03:59.999'
time_out = '05:00:59.999'

end_time = '2025-02-21 09:01:59.999'

day = 'second'
crypto='ETHUSDT'

data = meassures().ImportData(data_path='./data.csv')
test = meassures().testing(test=data, time_in=time_in, time_out=time_out, end_time=end_time, day=day, crypto=crypto)

print(test)