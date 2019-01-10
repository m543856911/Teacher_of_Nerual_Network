# -*- coding: utf-8 -*-
"""
@Author: QuYue
@Time  : 2019/1/9 22:40
@File  : network.py

Package: Mr Net.network
A package which provides a range of networks based on Pytorch.
"""
# %%　Import Packages
import numpy as np
import torch
# %% Logistic Regression
class LR_Classifier(torch.nn.Module):
    def __init__(self, feature_dim, label_dim):
        super(LR_Classifier, self).__init__()
        self.linear = torch.nn.Linear(feature_dim, label_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        y = self.sigmoid(x)
        return y


