import numpy as np
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset2 import ViatalSignDataset_regression_mel_thickness

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

'''
멜라닌 추정을 위한 Fully Connected 기반 Regression Model 학습
'''

def cross_entropy_loss(y, t):
    return -np.sum(t * np.log(y))




class Regression(nn.Module):
    def __init__(self, cl_mode):
        super(Regression, self).__init__()

        self.input_dim = 8

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
    class_mode = "mel"

    save_dir = "vitalsign_mel_0112_prob_input14_m1_epoch1000_gt_fc"

    path = os.path.dirname(__file__)
    feature_path = os.path.join(path, './result/{}/feature_weight_data'.format(save_dir))
    feature_path2 = os.path.join(path, './result/{}/feature_weight_data2'.format(save_dir))
    feature_path3 = os.path.join(path, './result/{}/feature_weight_data3'.format(save_dir))
    feature_path4 = os.path.join(path, './result/{}/feature_weight_data_10000'.format(save_dir))

    classify_path = os.path.join(path, './result/{}/classification_weight_data'.format(save_dir))
    classify_path2 = os.path.join(path, './result/{}/classification_weight_data2'.format(save_dir))

    regression_path = os.path.join(path, './result/{}/regression_weight_data'.format(save_dir))
    regression_path2 = os.path.join(path, './result/{}/regression_weight_data2'.format(save_dir))

    if os.path.isdir("result/{}".format(save_dir)) == False:
        os.mkdir("result/{}".format(save_dir))
    # else:
    #     print("save file exist ")
    #     exit()


    print("**************************************************")
    print("****** 3.  Train Regression Model ************")
    print("**************************************************")

    if use_gpu == True:
        Reg_model = Regression(cl_mode='mel').to('cuda')
    else:
        Reg_model = Regression(cl_mode='mel')

    data_loader = DataLoader(ViatalSignDataset_regression_mel_thickness(mode='train', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1), batch_size=3000, shuffle=True)
    test_data_len = len(ViatalSignDataset_regression_mel_thickness(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1))
    test_loader = DataLoader(ViatalSignDataset_regression_mel_thickness(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1), batch_size=test_data_len, shuffle=False)

    optimizer = optim.Adam(Reg_model.parameters(), lr=0.001)

    criterion = nn.MSELoss()

    best_loss = 10066896896.0 #9.0
    best_test_loss =  10066896896.0 #9.0
    epochs = 1000

    for epoch in range(epochs):
        if epoch == 500:
            optimizer = optim.Adam(Reg_model.parameters(), lr=0.0005)
        elif epoch == 800:
            optimizer = optim.Adam(Reg_model.parameters(), lr=0.0001)

        running_loss = []
        # for anchor, m_label, tb_label, st_label, th_label in data_loader:
        for anchor, m_label, tb_label, st_label, th_label, _, _, _, _ in data_loader:
            gt_label = m_label

            Reg_model.train()
            pred_value = Reg_model(anchor)

            pred_value = pred_value.squeeze()

            loss = criterion(pred_value, gt_label)
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
                    if class_mode == 'mel':
                        gt_label_t = m_label_t
                    elif class_mode == 'thb':
                        gt_label_t = tb_label_t
                    elif class_mode == 'sto':
                        gt_label_t = st_label_t
                    elif class_mode == 'thickness':
                        gt_label_t = th_label_t
                    else:
                        gt_label_t = st_label_t


                    Reg_model.eval()
                    pred_value = Reg_model(anchor_t)
                    pred_value = pred_value.squeeze()
                    test_loss = criterion(pred_value, gt_label_t)
                    test_loss = torch.sqrt(test_loss)

                if mean_loss < best_loss :
                    best_loss = mean_loss
                    print("Epoch: {}/{} - Loss: {:.4f},  Test Loss: {:.4f} (Save Model)".format(epoch + 1, epochs, mean_loss, test_loss))
                    torch.save(Reg_model.state_dict(), regression_path)
                else:
                    print("Epoch: {}/{} - Loss: {:.4f},  Test Loss: {:.4f} ".format(epoch + 1, epochs, mean_loss, test_loss))

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    torch.save(Reg_model.state_dict(), regression_path2)





