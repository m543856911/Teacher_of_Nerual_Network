# -*- coding: utf-8 -*-
"""
@Author: QuYue
@Time  : 2018/12/22 23:34
@File  : main2.py
"""
# %% Import Packages
import pandas as pd
import numpy as np
import torch
import math
from tqdm import tqdm
import Mr_Net as mn


# %% Parameter
class Parameter():
    def __init__(self):
        self.PATH = '../Data/'  # path 路径
        self.FILE = 'datingTestSet2.txt'  # file name 文件名
        self.EPOCH = 100  # epoch 迭代次数
        self.TRAINTEST = 4 / 1  # train/test 训练集与测试集的比例

    def stats(self, shape):  # statistics 统计
        self.total_amount = shape[0]
        self.train_amount = math.ceil(shape[0] * self.TRAINTEST / (self.TRAINTEST + 1))
        self.test_amount = shape[0] - self.train_amount
        self.feature_amount = shape[1] - 1


# %% Hyper Parameter
Parm = Parameter()
# %% Read Data
data = pd.read_csv(Parm.PATH + Parm.FILE, sep='\t', names=[
    '飞行公里', '冰淇淋消费', '游戏', '程度'])

Parm.stats(data.shape)
# %% Normalization
feature = data.iloc[:, :Parm.feature_amount]  # feature
label = data.iloc[:, -1]  # label
# feature norm (z-score)
feature -= feature.mean()
feature /= feature.std()
# label norm (from 0 to n)
label -= label.min()
# %% Split Data
# train_set
train_feature = np.array(feature.iloc[:Parm.train_amount])
train_label = np.array(label.iloc[:Parm.train_amount])
# test_set
test_feature = np.array(feature.iloc[Parm.train_amount:])
test_label = np.array(label.iloc[Parm.train_amount:])

all_feature = np.array(feature)
all_label = np.array(label)
# %% To Tensor
train_feature = torch.Tensor(train_feature)
train_label = torch.Tensor(train_label).type(torch.LongTensor)

test_feature = torch.Tensor(test_feature)
test_label = torch.Tensor(test_label).type(torch.LongTensor)

all_feature = torch.Tensor(all_feature)
all_label = torch.Tensor(all_label).type(torch.LongTensor)
# %% Create LR
LR1 = mn.LR_Classifier(Parm.feature_amount, 3)
# torch.save(LR1.state_dict(), Parm.PATH + 'LR1.pth')
LR2 = mn.LR_Classifier(Parm.feature_amount, 3)
LR2.load_state_dict(torch.load(Parm.PATH + 'LR1.pth'))
# %% Training
result = []
pbar = tqdm(total=Parm.EPOCH)
print('start')
for epoch in range(Parm.EPOCH):
    result += mn.MGD(LR2, train_feature, train_label, test_feature, test_label, batch_size=10, show=False)
    pbar.update(1)
pbar.close()
# %% plot
mn.plot_result(result)
