from utils import models, get_data_crypto, meassures

# data = get_data_crypto().download_data_cloud(start_time='2025-04-25', end_time='2025-04-30 13:00:00', crypto="BNBUSDT", time='S')
# prophet_model = models.prophet_model(data=data, time='S')

time = 'min'
start_time = '2025-04-25'
end_time = '2025-05-3 02:16:00'
crypto = "ETHUSDT"

data = get_data_crypto().download_data(start_time=start_time, end_time=end_time, crypto=crypto, time=time)
df_prophet, predict_prophet = models().XGBoost_model(data=data, time=time)
test_1 = meassures().data_predict(model_pred=predict_prophet, crypto=crypto, time=time)