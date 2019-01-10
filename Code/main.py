# -*- coding: utf-8 -*-
"""
@Author: QuYue
@Time  : 2018/12/21 16:11
@File  : main.py
"""
# %% Import Packages
import pandas as pd
import numpy as np
import math
import torch

# %% Parameter
class Parameter():
    def __init__(self):
        self.PATH = '../Data/'  # path 路径
        self.FILE = 'datingTestSet2.txt'  # file name 文件名
        self.EPOCH = 10  # epoch 迭代次数
        self.TRAINTEST = 4/1  # train/test 训练集与测试集的比例

    def stats(self, shape):  # statistics 统计
        self.total_amount = shape[0]
        self.train_amount = math.ceil(shape[0]*self.TRAINTEST/(self.TRAINTEST+1))
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
test_label = np.array(label.iloc[:Parm.train_amount])
# %% To Tensor
train_feature = torch.Tensor(train_feature)
train_label_numpy = train_label.copy()
train_label = torch.Tensor(train_label).type(torch.LongTensor)

test_feature = torch.Tensor(test_feature)
test_label = torch.Tensor(test_label).type(torch.LongTensor)
# %%
class LR_Classifier(torch.nn.Module):
    def __init__(self):
        super(LR_Classifier, self).__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        y = self.sigmoid(x)
        return y

# %% Create Classifier
LR1 = LR_Classifier()
torch.save(LR1.state_dict(), Parm.PATH + 'LR1.pth')
LR2 = LR_Classifier()
LR2.load_state_dict(torch.load(Parm.PATH + 'LR1.pth'))
LR3 = LR_Classifier()
LR3.load_state_dict(torch.load(Parm.PATH + 'LR1.pth'))
LR4 = LR_Classifier()
LR4.load_state_dict(torch.load(Parm.PATH + 'LR1.pth'))
# %% Save
Acc = np.zeros((Parm.EPOCH, 3))
# %% Network1
optimizer1 = torch.optim.SGD(LR1.parameters(), lr=0.01)
criterion1 = torch.nn.CrossEntropyLoss()
for epoch in range(Parm.EPOCH):
    y1 = LR1(train_feature)

    loss1 = criterion1(y1, train_label)
    optimizer1.zero_grad()
    loss1.backward()
    optimizer1.step()

    pre_y1 = torch.max(y1, 1)[1].data.numpy().squeeze()
    Acc[epoch, 0] = (pre_y1 == train_label_numpy).sum()
    print(Acc[epoch, 0] / len(pre_y1))
print(1111111111111111111111)
# %% Network2
optimizer2 = torch.optim.SGD(LR2.parameters(), lr=0.01)
criterion2 = torch.nn.CrossEntropyLoss()
H = np.zeros(Parm.EPOCH)
for epoch in range(Parm.EPOCH):
    y2 = LR2(train_feature)

    loss2 = criterion2(y2, train_label)
    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()
    H[epoch] = loss2.data.numpy()

    pre_y2 = torch.max(y2, 1)[1].data.numpy().squeeze()
    Acc[epoch, 1] = (pre_y2 == train_label_numpy).sum()
    print(Acc[epoch, 1] / len(pre_y2))
print(1111111111111111111111)
# %% Network3
optimizer3 = torch.optim.SGD(LR3.parameters(), lr=0.01)
criterion3 = torch.nn.CrossEntropyLoss()

for epoch in range(Parm.EPOCH):
    Loss = torch.tensor([0.0], requires_grad=True)
    pre_y3 = np.zeros(train_label_numpy.shape)
    for i in range(len(train_feature)):
        y3 = LR3(train_feature[i:i + 1])
        pre_y3[i] = torch.max(y3, -1)[1].data.numpy()

        loss3 = criterion3(y3, train_label[i:i + 1])
        Loss = Loss + loss3

    optimizer3.zero_grad()
    Loss = Loss / len(train_feature)
    Loss.backward()
    optimizer3.step()

    Acc[epoch, 2] = (pre_y3 == train_label_numpy).sum()
    print(Acc[epoch, 2] / len(pre_y3))
#%% Network4