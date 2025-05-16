from indicators import ModelIndicators
 
 
end_time = '2025-04-11 23:00:00'
 
days_fine_pred = 1
days_pred = 2
crypto='ETHUSDT'
time='min'
 
 
test_1 = ModelIndicators().development_XGBoost_final(end_time = end_time, days_fine_pred=days_fine_pred, days_pred = days_pred, crypto = crypto, time = time)