# -*- coding: utf-8 -*-
"""
@Author: QuYue
@Time  : 2019/1/9 22:44
@File  : dataprocess.py

Package: Mr Net.dataprocess
A package which can provide some processing tasks for data.
"""
#%% Import Packages
import pandas as pd
import numpy as np
import torch

#%% Normalization
# z_score
def z_score(feature):
    feature -= feature.mean(axis=0)
    feature /= feature.std(axis=0)
    return feature

# Label from 0 to n
def zero2n(label):
    label -= label.min(axis=0)
    return label

#%% Split Data
def split(feature, label, train_amount, shuffle=False):
    N = len(feature)
    index = np.arange(N)
    if shuffle:
        np.random.shuffle(index)
    train_feature = feature.iloc[index[:train_amount]]
    train_label = label.iloc[index[:train_amount]]
    test_feature = feature.iloc[index[train_amount:]]
    test_label = label.iloc[index[train_amount:]]
    return train_feature, train_label, test_feature, test_label

#%% Change to Tensor
def toTensor(data, islabel=False):
    data =np.array(data)
    if islabel:
        data = torch.Tensor(data).type(torch.LongTensor)
    else:
        data = torch.Tensor(data)
    return data



