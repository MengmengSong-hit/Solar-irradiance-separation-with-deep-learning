from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import numpy as np
def cal(predict_data,label_data):
    mae=round(mean_absolute_error(label_data,predict_data),4)
    mse=round(mean_squared_error(label_data,predict_data),4)
    rmse=round(sqrt(mean_squared_error(label_data,predict_data)),4)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:",rmse)
    result=[mae,mse,rmse]
    return result

def point_forecast(predict_data,label_data):
    result=cal(predict_data,label_data)
    return result




        