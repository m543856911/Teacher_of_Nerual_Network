# -*- coding: utf-8 -*-
"""
@Author: QuYue
@Time  : 2018/12/22 22:22
@File  : __init__.py

Package: Mr Net
A package which can help the neural networks to learn much better.
"""
# %%　Import Packages
import numpy as np
import torch
import matplotlib.pyplot as plt


# %% Help
def help():
    print('Package: Mr Net\nA package which can help the neural networks to learn much better.')


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


# %% Mini-Batch Gradient descent
class Result():
    def __init__(self, correct, total):
        self.correct = correct
        self.total = total
        self.accuracy = correct / total


def compare_label(y, label):
    class_amount = y.shape[1]
    pre_y = torch.max(y, 1)[1].data.numpy()
    label = label.numpy()

    result = dict()
    for i in range(class_amount):
        result[i] = Result(int((pre_y[label == i] == i).sum()), int((label == i).sum()))
    result['Acc'] = (pre_y == label).sum()
    result['Acc_rate'] = result['Acc']/len(pre_y)
    return result


def MGD(network, feature, label, batch_size=1, lr=0.01, show=True):
    optimizer = torch.optim.SGD(network.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    N = len(feature)  # data amount
    Acc = np.array(range(0, N, batch_size))
    result_all = []
    for i in range(0, N, batch_size):
        y = network(feature[i: i + batch_size])

        loss = criterion(y, label[i: i + batch_size])
        loss.backward()
        optimizer.step()

        result = compare_label(network(feature), label)
        Acc[i] = result['Acc']
        result_all.append(result)

        if show:
            print('Accuracy is %s' % Acc[i])

    return result_all

