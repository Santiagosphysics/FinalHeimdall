from utils import meassures

data = meassures().ImportData(data_path='./data.csv')

end_time = '2025-02-21 00:00:00'

time_in= '10:03:59.999'
time_out = '16:00:59.999'

test = meassures().testing(test=data, time_in=time_in, time_out=time_out, end_time=end_time)