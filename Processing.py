# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd

def load_data(dataset_path):
    inputData = pd.read_csv(dataset_path)
    inputData.iloc[:, -1] = inputData.iloc[:, -1].astype('float')
    return inputData

def find_Null_Samples(input_data):
    input_data = pd.DataFrame(input_data)
    # inputdata=inputdata.fillna(method='ffill')
    input_data = input_data.apply(lambda x: x.fillna(x.mean()), axis=0)
    input_data = input_data.values
    return input_data

def generate_cost_matrix(y_train):
    y0_num = len([i for i in y_train if i == 0])
    y1_num = len([i for i in y_train if i == 1])

    FNcost = y0_num / y1_num
    FPcost = 1
    TNcost = 0
    TPcost = 0

    cost_mat = []
    for i in range(len(y_train)):
        cost_mat.append([FPcost, FNcost, TPcost, TNcost])

    cost_mat = np.array(cost_mat)

    return cost_mat
