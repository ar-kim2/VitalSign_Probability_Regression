import numpy as np
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset2 import ViatalSignDataset_triplet2
from dataset2 import ViatalSignDataset_class
from dataset2 import ViatalSignDataset_regression

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

'''
Sto, Thb 추정을 위한 Fully Connected 기반 Regression Model 학습
'''

class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()

        self.input_dim = 25 #28 #49

        self.layer11 = nn.Linear(self.input_dim, 128)
        self.layer12 = nn.Linear(128, 128)
        self.layer13 = nn.Linear(128, 128)
        self.layer14 = nn.Linear(128, 64)
        self.layer15 = nn.Linear(64, 1)

    def forward(self, x):
        x1 = F.leaky_relu(self.layer11(x))
        x1 = F.leaky_relu(self.layer12(x1))
        x1 = F.leaky_relu(self.layer13(x1))
        x1 = F.leaky_relu(self.layer14(x1))
        x1 = self.layer15(x1)

        return x1


if __name__ == '__main__':
    use_gpu = True
    class_mode = ""

    save_dir = "vitalsign_sto_thb_0112_prob_01_input14_m1_epoch5000_addinput3_fc"

    path = os.path.dirname(__file__)

    feature_path = os.path.join(path, './result/{}/feature_weight_data'.format(save_dir))
    feature_path2 = os.path.join(path, './result/{}/feature_weight_data2'.format(save_dir))
    feature_path3 = os.path.join(path, './result/{}/feature_weight_data3'.format(save_dir))
    feature_path10000 = os.path.join(path, './result/{}/feature_weight_data_10000'.format(save_dir))

    classify_path = os.path.join(path, './result/{}/classification_weight_data'.format(save_dir))
    classify_path2 = os.path.join(path, './result/{}/classification_weight_data2'.format(save_dir))
    # classify_path3 = os.path.join(path, './result/{}/classification_weight_data3'.format(save_dir))

    regression_path_sto = os.path.join(path, './result/{}/regression_sto_weight_data'.format(save_dir))
    regression_path_sto2 = os.path.join(path, './result/{}/regression_sto_weight_data2'.format(save_dir))
    # regression_path_sto3 = os.path.join(path, './result/{}/regression_sto_weight_data3'.format(save_dir))
    # regression_path_sto4 = os.path.join(path, './result/{}/regression_sto_weight_data4'.format(save_dir))

    regression_path_thb = os.path.join(path, './result/{}/regression_thb_weight_data'.format(save_dir))
    regression_path_thb2 = os.path.join(path, './result/{}/regression_thb_weight_data2'.format(save_dir))
    # regression_path_thb3 = os.path.join(path, './result/{}/regression_thb_weight_data3'.format(save_dir))
    # regression_path_thb4 = os.path.join(path, './result/{}/regression_thb_weight_data4'.format(save_dir))

    if os.path.isdir("result/{}".format(save_dir)) == False:
        os.mkdir("result/{}".format(save_dir))

    print("**************************************************")
    print("****** 3.  Train Regression Model (Sto) ************")
    print("**************************************************")
    if use_gpu == True:
        Reg_model = Regression().to('cuda')
    else:
        Reg_model = Regression()

    data_loader = DataLoader(ViatalSignDataset_regression(mode='train', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1), batch_size=1000, shuffle=True)
    test_data_len = len(ViatalSignDataset_regression(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1))
    test_loader = DataLoader(ViatalSignDataset_regression(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1), batch_size=test_data_len, shuffle=False)

    optimizer = optim.Adam(Reg_model.parameters(), lr=0.001)

    criterion = nn.MSELoss()

    best_loss = 9.0
    best_test_loss = 9.0
    epochs =  1000

    for epoch in range(epochs):
        if epoch == 500:
            optimizer = optim.Adam(Reg_model.parameters(), lr=0.0005)
        elif epoch == 800:
            optimizer = optim.Adam(Reg_model.parameters(), lr=0.0001)

        running_loss = []
        # for anchor, m_label, tb_label, st_label, th_label in data_loader:
        for anchor, m_label, tb_label, st_label, th_label, _, _, _, _ in data_loader:
            Reg_model.train()
            pred_value = Reg_model(anchor)

            pred_value = pred_value.squeeze()

            loss = criterion(pred_value, st_label)
            loss = torch.sqrt(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_gpu == True:
                running_loss.append(loss.detach().cpu().numpy())
            else:
                running_loss.append(loss.detach().numpy())

        if epoch % 10 == 0:
            with torch.no_grad():
                mean_loss = np.mean(running_loss)

                for anchor_t, m_label_t, tb_label_t, st_label_t, th_label_t, _, _, _, _ in test_loader:
                    Reg_model.eval()
                    pred_value = Reg_model(anchor_t)
                    pred_value = pred_value.squeeze()
                    test_loss = criterion(pred_value, st_label_t)
                    test_loss = torch.sqrt(test_loss)

                if mean_loss < best_loss :
                    best_loss = mean_loss
                    print("Epoch: {}/{} - Loss: {:.4f},  Test Loss: {:.4f} (Save Model)".format(epoch + 1, epochs, mean_loss, test_loss))
                    torch.save(Reg_model.state_dict(), regression_path_sto)
                else:
                    print("Epoch: {}/{} - Loss: {:.4f},  Test Loss: {:.4f} ".format(epoch + 1, epochs, mean_loss, test_loss))

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    torch.save(Reg_model.state_dict(), regression_path_sto2)


    print("**************************************************")
    print("****** 3.  Train Regression Model (Thb) ************")
    print("**************************************************")

    if use_gpu == True:
        Reg_model2 = Regression().to('cuda')
    else:
        Reg_model2 = Regression()

    data_loader = DataLoader(ViatalSignDataset_regression(mode='train', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1),
                             batch_size=1000, shuffle=True)
    test_data_len = len(ViatalSignDataset_regression(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1))
    test_loader = DataLoader(ViatalSignDataset_regression(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1),
                             batch_size=test_data_len, shuffle=False)

    optimizer = optim.Adam(Reg_model2.parameters(), lr=0.001)

    criterion = nn.MSELoss()

    best_loss = 9.0
    best_test_loss = 9.0
    epochs = 1000

    for epoch in range(epochs):
        if epoch == 500:
            optimizer = optim.Adam(Reg_model2.parameters(), lr=0.0005)
        elif epoch == 800:
            optimizer = optim.Adam(Reg_model2.parameters(), lr=0.0001)

        running_loss = []
        # for anchor, m_label, tb_label, st_label, th_label in data_loader:
        for anchor, m_label, tb_label, st_label, th_label, _, _, _, _ in data_loader:
            Reg_model2.train()
            pred_value = Reg_model2(anchor)

            pred_value = pred_value.squeeze()

            loss = criterion(pred_value, tb_label)
            loss = torch.sqrt(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_gpu == True:
                running_loss.append(loss.detach().cpu().numpy())
            else:
                running_loss.append(loss.detach().numpy())

        if epoch % 10 == 0:
            with torch.no_grad():
                mean_loss = np.mean(running_loss)

                # for anchor_t, m_label_t, tb_label_t, st_label_t, th_label_t in test_loader:
                for anchor_t, m_label_t, tb_label_t, st_label_t, th_label_t, _, _, _, _ in test_loader:
                    Reg_model2.eval()
                    pred_value = Reg_model2(anchor_t)
                    pred_value = pred_value.squeeze()
                    test_loss = criterion(pred_value, tb_label_t)
                    test_loss = torch.sqrt(test_loss)

                if mean_loss < best_loss:
                    best_loss = mean_loss
                    print("Epoch: {}/{} - Loss: {:.4f},  Test Loss: {:.4f} (Save Model)".format(epoch + 1, epochs,
                                                                                                mean_loss, test_loss))
                    torch.save(Reg_model2.state_dict(), regression_path_thb)
                else:
                    print("Epoch: {}/{} - Loss: {:.4f},  Test Loss: {:.4f} ".format(epoch + 1, epochs, mean_loss,
                                                                                    test_loss))

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    torch.save(Reg_model2.state_dict(), regression_path_thb2)













