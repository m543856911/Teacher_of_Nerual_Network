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
class Mr_Result():
    def __init__(self):
        self.train = None
        self.test = None


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
    result['Total'] = Result(int((pre_y == label).sum()), len(label))
    return result


def MGD(network, feature, label, test_feature=None, test_label=None, batch_size=1, lr=0.01, show=True):
    N = len(feature)  # data amount
    result = []

    test_mode = False
    if (type(test_feature) != type(None)) and \
            (type(test_label) != type(None)):
        test_mode = True

    optimizer = torch.optim.SGD(network.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(0, N, batch_size):
        y = network(feature[i: i + batch_size])

        loss = criterion(y, label[i: i + batch_size])
        loss.backward()
        optimizer.step()

        result_step = Mr_Result()

        result_step.train = compare_label(network(feature), label)
        if test_mode:
            result_step.test = compare_label(network(test_feature), test_label)
        result.append(result_step)

        if show:
            if test_mode:
                print('Train Accuracy is %s, Test Accuracy is %s' % (
                result_step.train['Total'].accuracy, result_step.test['Total'].accuracy))
            else:
                print('Accuracy is %s' % result_step.train['Total'].accuracy)

    return result


# %% Plot
def plot_result(result):
    names = list(result[0].train.keys())
    times = len(result)
    test_mode = False
    if result[0].test != None:
        test_mode = True

    lines = dict([(i, []) for i in names])
    test_lines = dict([(i, []) for i in names])
    for i in result:
        for j in names:
            lines[j].append(i.train[j].accuracy)
            if test_mode:
                test_lines[j].append(i.test[j].accuracy)

    plt.figure()
    if test_mode:
        plt.subplot(2, 1, 1)
    plots = []
    for i in names:
        if i == 'Total':
            plots.append(plt.plot(list(range(times)), lines[i], linewidth=3)[0])
        else:
            plots.append(plt.plot(list(range(times)), lines[i])[0])
    plt.xlim(0, len(result))
    plt.ylim(0, 1)
    plt.title('Train Accuracy - %s' % result[-1].train['Total'].accuracy)
    plt.legend(plots, names)
    plt.grid(True)
    if test_mode:
        plt.subplot(2, 1, 2)
        plots = []
        for i in names:
            if i == 'Total':
                plots.append(plt.plot(list(range(times)), test_lines[i], linewidth=3)[0])
            else:
                plots.append(plt.plot(list(range(times)), test_lines[i])[0])
        plt.xlim(0, len(result))
        plt.ylim(0, 1)
        plt.title('Test Accuracy - %s' % result[-1].test['Total'].accuracy)
        plt.legend(plots, names)
        plt.grid(True)
    plt.show()
