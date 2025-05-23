#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
# from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics.pairwise import paired_distances
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
'''
这个脚本的主要功能是使用简单的自编码器（autoencoder）在 CERT 数据集上进行异常检测。具体步骤如下：
数据加载:
从 week-r5.2-percentile30.pkl 文件中加载数据。
移除一些不需要的列（如用户、时间等信息列），只保留特征列。
数据分割:
将数据集分为前半部分（用于训练）和后半部分（用于测试）。
随机选择 200 个用户用于训练。
数据标准化:
使用 StandardScaler 对训练和测试数据进行标准化。
模型训练:
使用 MLPRegressor 作为自编码器，训练模型以重构输入数据。
异常检测:
计算测试数据的重构误差。
使用 ROC AUC 评估模型性能。
在不同预算下计算检测率（Detection Rate）。
'''

print('This script runs a sample anomaly detection (using simple autoencoder) '
      'on CERT dataset. By default it takes CERT r5.2 data extracted with percentile '
      'representation, generated using temporal_data_representation script. '
      'It then trains on data of 200 random users in first half of the dataset, '
      'and output AUC score and detection rate at different budgets (instance-based)')

print('For more details, see this paper: Anomaly Detection for Insider Threats Using'
      ' Unsupervised Ensembles. Le, D. C.; and Zincir-Heywood, A. N. IEEE Transactions'
      ' on Network and Service Management, 18(2): 1152–1164. June 2021.')

data = pd.read_pickle('week-r5.2-percentile30.pkl')
removed_cols = ['user','day','week','starttime','endtime','sessionid','insider']
x_cols = [i for i in data.columns if i not in removed_cols]

run = 1
np.random.seed(run)

data1stHalf = data[data.week <= max(data.week)/2]
dataTest = data[data.week > max(data.week)/2]

nUsers = np.random.permutation(list(set(data1stHalf.user)))
trainUsers = nUsers[:200]


xTrain = data1stHalf[data1stHalf.user.isin(trainUsers)][x_cols].values
yTrain = data1stHalf[data1stHalf.user.isin(trainUsers)]['insider'].values
yTrainBin = yTrain > 0

xTest = data[x_cols].values
yTest = data['insider'].values
yTestBin = yTest > 0

scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

ae = MLPRegressor(hidden_layer_sizes=(int(data.shape[1]/4), int(data.shape[1]/8), 
                                      int(data.shape[1]/4)), max_iter=25, random_state=10)

ae.fit(xTrain, xTrain)

reconstructionError = paired_distances(xTest, ae.predict(xTest))

print('AUC score: ', roc_auc_score(yTestBin, reconstructionError))

print('Detection rate at different budgets:')
for ib in [0.001, 0.01, 0.05, 0.1, 0.2]:
    threshold = np.percentile(reconstructionError, 100-100*ib)
    flagged = np.where(reconstructionError>threshold)[0]
    dr = sum(yTestBin[flagged]>0)/sum(yTestBin>0)
    print(f'{100*ib}%, DR = {100*dr:.2f}%')