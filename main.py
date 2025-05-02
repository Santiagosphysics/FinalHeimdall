from utils import models, get_data_crypto

# data = get_data_crypto().download_data_cloud(start_time='2025-04-25', end_time='2025-04-30 13:00:00', crypto="BNBUSDT", time='S')
# prophet_model = models.prophet_model(data=data, time='S')


data = get_data_crypto().download_data_cloud(start_time='2025-04-25', end_time='2025-04-30 13:00:00', crypto="binancecoin", time='S')
prophet_model = models.prophet_model(data=data, time='S')