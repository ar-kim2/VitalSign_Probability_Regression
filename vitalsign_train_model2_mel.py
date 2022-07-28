import numpy as np
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset2 import ViatalSignDataset_triplet_mel_thickness_v2
from dataset2 import ViatalSignDataset_class_mel_thickness
from dataset2 import ViatalSignDataset_regression_mel_thickness

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


import time

'''
멜라닌 추정을 위한 확률기반 Regression Model(Adapted Triplet Loss 방식) 학습
'''


def cross_entropy_loss(y, t):
    return -np.sum(t * np.log(y))

class VitalSign_Feature_mel_thickness(nn.Module):
    def __init__(self):
        super(VitalSign_Feature_mel_thickness, self).__init__()

        self.input_dim = 14

        self.common1 = nn.Linear(self.input_dim, 128)
        self.common2 = nn.Linear(128, 128)
        self.common3 = nn.Linear(128, 128)
        self.common4 = nn.Linear(128, 128)
        self.common5 = nn.Linear(128, 128)

    def forward(self, x):
        x = F.leaky_relu(self.common1(x))
        x = F.leaky_relu(self.common2(x))
        x = F.leaky_relu(self.common3(x))
        x = F.leaky_relu(self.common4(x))
        x = F.leaky_relu(self.common5(x))

        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.input_dim = 128

        self.layer11 = nn.Linear(self.input_dim, 128)
        self.layer12 = nn.Linear(128, 128)
        self.layer13 = nn.Linear(128, 128)
        self.layer14 = nn.Linear(128, 128)
        self.layer15 = nn.Linear(128, 8)

    def forward(self, x):
        x1 = F.leaky_relu(self.layer11(x))
        x1 = F.leaky_relu(self.layer12(x1))
        x1 = F.leaky_relu(self.layer13(x1))
        x1 = F.leaky_relu(self.layer14(x1))
        x1 = self.layer15(x1)

        return x1


class Regression(nn.Module):
    def __init__(self):
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

    save_dir = "vitalsign_mel_0418_prob_02_input14_m1_epoch20000_gt_test"

    path = os.path.dirname(__file__)
    feature_path = os.path.join(path, './result/{}/feature_weight_data'.format(save_dir))
    feature_path2 = os.path.join(path, './result/{}/feature_weight_data2'.format(save_dir))
    feature_path3 = os.path.join(path, './result/{}/feature_weight_data3'.format(save_dir))

    classify_path = os.path.join(path, './result/{}/classification_weight_data'.format(save_dir))
    classify_path2 = os.path.join(path, './result/{}/classification_weight_data2'.format(save_dir))

    regression_path = os.path.join(path, './result/{}/regression_weight_data'.format(save_dir))
    regression_path2 = os.path.join(path, './result/{}/regression_weight_data2'.format(save_dir))

    if os.path.isdir("result/{}".format(save_dir)) == False:
        os.mkdir("result/{}".format(save_dir))
    # else:
    #     print("save file exist ")
    #     exit()

    ############# 1. Train Feature Model ###################
    print("**************************************************")
    print("************ 1.  Train Feature Model *************")
    print("**************************************************")

    if use_gpu == True:
        feature_model = VitalSign_Feature_mel_thickness().to('cuda')
    else:
        feature_model = VitalSign_Feature_mel_thickness()

    data_loader = DataLoader(ViatalSignDataset_triplet_mel_thickness_v2(mode='train', cl=class_mode, model_mel=-1, model_thick=-1), batch_size=2000, shuffle=True)

    test_data_len = len(ViatalSignDataset_triplet_mel_thickness_v2(mode='test', cl=class_mode, model_mel=-1, model_thick=-1))
    test_data_loader = DataLoader(ViatalSignDataset_triplet_mel_thickness_v2(mode='test', cl=class_mode, model_mel=-1, model_thick=-1), batch_size=test_data_len, shuffle=False)

    optimizer = optim.Adam(feature_model.parameters(), lr=0.001)
    criterion = nn.TripletMarginLoss(margin=1, p=2)
    test_criterion = nn.TripletMarginLoss(margin=1, p=2)

    best_loss = 5.7
    best_test_loss = 5.7
    epochs = 20000

    train_loss_save = []
    test_loss_save = []
    test_loss_save2 = []

    for epoch in range(epochs):
        if epoch == 2000:
            optimizer = optim.Adam(feature_model.parameters(), lr=0.0005)
        if epoch == 4000:
            optimizer = optim.Adam(feature_model.parameters(), lr=0.0003)
        if epoch == 6000:
            optimizer = optim.Adam(feature_model.parameters(), lr=0.0001)
        if epoch == 10000:
            optimizer = optim.Adam(feature_model.parameters(), lr=0.00005)
        if epoch == 15000:
            optimizer = optim.Adam(feature_model.parameters(), lr=0.00001)

        running_loss = []

        for anchor, pos, neg, _, _, _, _, _ in data_loader:
            feature_model.train()
            anc_out = feature_model(anchor)

            pos_out = feature_model(pos)
            neg_out = feature_model(neg)
            loss = criterion(anc_out, pos_out, neg_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.detach().cpu().numpy())

        if epoch % 10 == 0:
            with torch.no_grad():
                mean_loss = np.mean(running_loss)

                feature_model.eval()

                for anchor_t, pos_t, neg_t, _, _, _, _, _ in test_data_loader:
                    anc_out_t = feature_model(anchor_t)

                    pos_out_t = feature_model(pos_t)
                    neg_out_t = feature_model(neg_t)


                    test_criterion2 = nn.TripletMarginLoss(margin=1.0, p=2)
                    test_loss = test_criterion(anc_out_t, pos_out_t, neg_out_t)
                    test_loss2 = test_criterion2(anc_out_t, pos_out_t, neg_out_t)

                if mean_loss < best_loss:
                    best_loss = mean_loss
                    t_stamp = time.time()
                    t_stamp = time.ctime(t_stamp)

                    print("{} | Epoch: {}/{} - Loss: {:.4f} , Test Loss: {:.4f}, Test Loss2: {:.4f} (Save Model)".format(t_stamp, epoch + 1, epochs,
                                                                                                mean_loss, test_loss, test_loss2))
                    # Training data Loss가 줄어들 때, Model Save
                    torch.save(feature_model.state_dict(), feature_path)
                    train_loss_save.append(mean_loss)
                    test_loss_save.append(test_loss)
                    test_loss_save2.append(test_loss2)
                else:
                    print("{} | Epoch: {}/{} - Loss: {:.4f} , Test Loss: {:.4f}, Test Loss2: {:.4f}".format(t_stamp, epoch + 1, epochs, mean_loss,
                                                                                   test_loss, test_loss2))
                    train_loss_save.append(mean_loss)
                    test_loss_save.append(test_loss)
                    test_loss_save2.append(test_loss2)
                    # 그 외의 경우 Model Save
                    torch.save(feature_model.state_dict(), feature_path3)

                if test_loss < best_test_loss:
                    best_test_loss = test_loss

                    # Test data Loss가 줄어들 때, Model Save
                    torch.save(feature_model.state_dict(), feature_path2)
                    print("[Save Feature Netwrok 2]")

    #
    ############# 2. Train Classification Model ###################
    print("**************************************************")
    print("****** 2.  Train Classification Model ************")
    print("**************************************************")

    feature_model.load_state_dict(torch.load(feature_path))
    feature_model.eval()

    temper_value = 1

    if use_gpu == True:
        classifier_model = Classifier(cl_mode='mel').to('cuda')
    else:
        classifier_model = Classifier(cl_mode='mel')

    data_loader = DataLoader(ViatalSignDataset_class_mel_thickness(mode='train', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1), batch_size=3000, shuffle=True)

    test_data_len = len(ViatalSignDataset_class_mel_thickness(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1))
    test_loader = DataLoader(ViatalSignDataset_class_mel_thickness(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1), batch_size=test_data_len,
                             shuffle=False)

    optimizer = optim.Adam(classifier_model.parameters(), lr=0.001)

    criterion = nn.MSELoss()

    best_loss = 9.0
    best_acc = 9.0
    epochs = 3000

    for epoch in range(epochs):
        if epoch == 1000:
            optimizer = optim.Adam(classifier_model.parameters(), lr=0.0005)
        elif epoch == 2000:
            optimizer = optim.Adam(classifier_model.parameters(), lr=0.0001)

        running_loss = []
        for anchor, m_label, m_label2, m_label3, tb_label, st_label, th_label, _, th_label3, total_label in data_loader:
            if class_mode == 'mel':
                gt_label = m_label
            elif class_mode == 'thb':
                gt_label = tb_label
            elif class_mode == 'sto':
                gt_label = st_label
            elif class_mode == 'thickness':
                gt_label = th_label
            else:
                gt_label = total_label

            x_data = feature_model(anchor)

            classifier_model.train()
            pred_prob_ = classifier_model(x_data)

            pred_prob = F.softmax(pred_prob_, dim=1)

            # Cross Entropy Loss
            loss = torch.mean(-(torch.sum(m_label3 * torch.log(pred_prob+0.00000000001), dim=1)))

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

                classifier_model.eval()

                for anchor, m_label, m_label2, m_label3, tb_label, st_label, th_label, _, th_label3, total_label in test_loader:
                    x_data = feature_model(anchor)
                    classifier_model.eval()

                    pred_prob_1 = classifier_model(x_data)
                    pred_prob_2 = pred_prob_1 * temper_value

                    # pred_prob = pred_prob_2

                    pred_prob = F.softmax(pred_prob_2, dim=1)
                    test_loss = torch.mean(-(torch.sum(m_label3 * torch.log(pred_prob+0.00000000001), dim=1)))

                    if torch.isnan(test_loss):
                        pred_prob_1_np = pred_prob_1.detach().cpu().numpy()
                        pred_prob_2_np = pred_prob_2.detach().cpu().numpy()
                        pred_prob_3_np = pred_prob.detach().cpu().numpy()

                    top_p, top_class = pred_prob.topk(1, dim=1)

                    total_cnt = 0
                    equal_cnt = 0

                    for i in range(top_class.shape[0]):
                        if top_class[i][0].item() == m_label[i].item():
                            equal_cnt += 1
                        total_cnt += 1

                    acc = equal_cnt / total_cnt

                if mean_loss < best_loss:
                    best_loss = mean_loss
                    print("Epoch: {}/{} - Loss: {:.4f} , Test loss : {:.4f},  Test Acc : {:.4f} (Save Model)".format(epoch + 1, epochs,
                                                                                                mean_loss, test_loss, acc))
                    torch.save(classifier_model.state_dict(), classify_path)
                else:
                    print("Epoch: {}/{} - Loss: {:.4f} , Test loss : {:.4f},  Test Acc : {:.4f} ".format(epoch + 1, epochs, mean_loss, test_loss, acc))

                if test_loss < best_acc:
                    torch.save(classifier_model.state_dict(), classify_path2)
                    best_acc = test_loss


    print("**************************************************")
    print("****** 3.  Train Regression Model ************")
    print("**************************************************")
    classifier_model.load_state_dict(torch.load(classify_path2))
    classifier_model.eval()

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

            x_data = feature_model(anchor)
            pred_prob = classifier_model(x_data)
            pred_prob = pred_prob*temper_value
            pred_prob = F.softmax(pred_prob, dim=1)

            Reg_model.train()
            pred_value = Reg_model(pred_prob)

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

                    x_data = feature_model(anchor_t)
                    pred_prob = classifier_model(x_data)
                    pred_prob = pred_prob * temper_value
                    pred_prob = F.softmax(pred_prob, dim=1)

                    Reg_model.eval()
                    pred_value = Reg_model(pred_prob)
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





