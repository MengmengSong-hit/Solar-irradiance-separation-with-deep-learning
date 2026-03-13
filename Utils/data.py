import numpy as np
from pandas import DataFrame
import pandas as pd
from Utils.metric import *
def loadtxtmethod(filename):
    datalist = []
    columnlist = []
    with open(filename,'r') as of:
        # 获取第一行表头数据    
        firstline = of.readline()    
        #删除字符串头尾的特定字符    
        firstline = firstline.strip('\n')    
        #将字符串按照空格进行分割    
        columnlist = firstline.split()
        #print(columnlist)
        for line in of:        
        # 清除前后回车符，按照空格进行分割
            line = line.strip('\n').split()      
            linelist = []        
            for str in line:                       
                linelist = linelist + (str.strip(',').split(','))               
            datalist.append(linelist)
    df = DataFrame(datalist, columns=columnlist)
    return df

def get_train_and_valid_data(norm_data, train_num, predict_day, label_in_feature_index, time_step):
    feature_data = norm_data[:train_num]
    label_data = norm_data[:train_num, label_in_feature_index]  # 将延后几天的数据作为label
    train_x = [ feature_data[i:i + time_step] for i in range(train_num - (time_step + predict_day - 1))]
    train_y = [label_data[i + time_step + predict_day - 1] for i in range(train_num - (time_step + predict_day - 1))]
    train_x, train_y = np.array(train_x), np.array(train_y)
    #train_y = train_y.reshape([train_y.shape[0], 1])
    #train_x= train_x.squeeze()
    return train_x, train_y


def get_test_data(norm_data, data, train_num, time_step, label_in_feature_index, predict_day):
    feature_data = norm_data[train_num:]
    test_x = [feature_data[i:i + time_step] for i in range(feature_data.shape[0] - (time_step + predict_day - 1))]
    test_x = np.array(test_x)
    label_data = data[train_num + time_step + predict_day - 1:, label_in_feature_index]
    print(f"test_x{test_x.shape}")
    print(f"label_data{label_data.shape}")
    return test_x, label_data
