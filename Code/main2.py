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
import matplotlib.pyplot as plt
import Mr_Net as mn


# %% Parameter
class Parameter():
    def __init__(self):
        self.PATH = '../Data/'
        self.FILE = 'datingTestSet2.txt'
        self.EPOCH = 10


# %% Hyper Parameter
Parm = Parameter()
# %% Read Data
data = pd.read_csv(Parm.PATH + Parm.FILE, sep='\t', names=[
    '飞行公里', '冰淇淋消费', '游戏', '程度'])
# %% Normalization
feature = data.iloc[:, :3]  # feature
label = data.iloc[:, 3]  # label
# feature norm (z-score)
feature -= feature.mean()
feature /= feature.std()
# label norm (from 0 to n)
label -= label.min()
# %% Split Data
train_feature = np.array(feature.iloc[:800])
train_label = np.array(label.iloc[:800])

test_feature = np.array(feature.iloc[800:])
test_label = np.array(label.iloc[:800])
# %% To Tensor
train_feature = torch.Tensor(train_feature)
train_label_numpy = train_label.copy()
train_label = torch.Tensor(train_label).type(torch.LongTensor)

test_feature = torch.Tensor(test_feature)
test_label = torch.Tensor(test_label).type(torch.LongTensor)
#%% Create LR
LR1 = mn.LR_Classifier(3, 3)
torch.save(LR1.state_dict(), Parm.PATH + 'LR1.pth')
LR2 = mn.LR_Classifier(3, 3)
LR2.load_state_dict(torch.load(Parm.PATH + 'LR1.pth'))
#%%
result = mn.MGD(LR2, train_feature, train_label, show=False)

#%% plot
Acc = []
class_amount = len(result[0])-1

for i in result:
    Acc.append(i['Acc_rate'])
plt.plot(list(range(len(Acc))), Acc)
plt.ylim(0,1)

plt.show()







