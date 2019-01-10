# -*- coding: utf-8 -*-
"""
@Author: QuYue
@Time  : 2018/12/24 15:18
@File  : lesson.py

Package: Mr Net.lesson
A package which can help the neural networks to learn much better.
"""
#%% Import Packages
import numpy as np
import pandas as pd
import torch

#%% Lesson 1
# ---------- #
# I assume if the feature' s value which be recomputed after normalization is more close to the zero, the information included by this feature is lower.
# So the simple questions are the samples which the sum of values are low.
# infor_level = sum(abs(values))
# ---------- #
def Lesson1(feature, label, repeat = 1):
    # 信息量
    def infor_level(record):
        return record.abs().sum() - np.abs(record['label'])
    # 重复次数
    def repeat_record(record):
        index = []
        for i in range(len(record)):

            index.extend([i]*repeat)
        new_record = record.iloc[index]
        return new_record

    N = len(feature)  # sample amount

    data0 = pd.DataFrame(feature.copy())
    data0['label'] = label

    data0['infor_level'] = data0.apply(infor_level, axis=1)
    data0 = data0.sort_values(by='infor_level',ascending=True)

    new_data = data0.apply(repeat_record)
    feature = new_data.iloc[:, :-2]
    label = new_data['label']

    return feature, label
#%% Lesson 2

#%% Lesson RL_batch_input


