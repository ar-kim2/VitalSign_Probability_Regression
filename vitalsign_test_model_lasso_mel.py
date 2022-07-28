import numpy as np
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset2 import ViatalSignDataset_regression_mel_thickness_lasso
from sklearn import linear_model


'''
Mel과 Thicknss에 대해서 Lass Model로 Regression
'''


if __name__ == '__main__':
    cl = "mel"

    criterion = nn.MSELoss()

    train_dataset = ViatalSignDataset_regression_mel_thickness_lasso(name='train')
    test_dataset = ViatalSignDataset_regression_mel_thickness_lasso(name='test')


    # clf = linear_model.Lasso(alpha=1.0)
    clf = linear_model.Ridge(alpha=1.0)

    if cl == 'mel':
        clf.fit(train_dataset.ref_list, train_dataset.m_label)
    else:
        clf.fit(train_dataset.ref_list, train_dataset.th_label)

    pred_value = clf.predict(test_dataset.ref_list)


    if cl == 'mel':
        loss_mean = np.sqrt(np.average(((pred_value - test_dataset.m_label)**2)))
        loss_list = ((pred_value - test_dataset.m_label) ** 2) ** (1 / 2)
    else:
        loss_mean = np.sqrt(np.average(((pred_value - test_dataset.th_label) ** 2)))
        loss_list = ((pred_value - test_dataset.th_label) ** 2) ** (1 / 2)

    print("RMSE : ", loss_mean)

    deviation_list = []

    for di in range(len(loss_list)):
        deviation_list.append(((loss_list[di]-loss_mean)**2).item())

    sto_var = np.mean(deviation_list)
    print("Variance mean : ", sto_var)
    print("Standard variation mean : ", np.sqrt(sto_var))
