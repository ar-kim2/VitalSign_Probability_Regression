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

class VitalSign_Feature(nn.Module):
    def __init__(self):
        super(VitalSign_Feature, self).__init__()

        self.input_dim = 25 #43 #43

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
        self.layer15 = nn.Linear(128, 49)
        # self.layer15 = nn.Linear(128, 28)
        # self.layer15 = nn.Linear(128, 77)
        #self.layer15 = nn.Linear(128, 42)

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

        self.input_dim = 49 #28 #49

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

    save_dir = "vitalsign_sto_thb_0104_prob_02_025_input14_m1_epoch5000_addinput3_gt"

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


    ############# 1. Train Feature Model ###################
    print("**************************************************")
    print("************ 1.  Train Feature Model *************")
    print("**************************************************")

    if use_gpu == True:
        feature_model = VitalSign_Feature().to('cuda')
    else:
        feature_model = VitalSign_Feature()

    # feature_model.load_state_dict(torch.load(feature_path3))
    #
    # data_loader = DataLoader(ViatalSignDataset_triplet2(mode='train', cl=class_mode, model_mel=-1, model_thick=-1), batch_size=1000, shuffle=True)
    #
    # test_data_len = len(ViatalSignDataset_triplet2(mode='test', cl=class_mode, model_mel=-1, model_thick=-1))
    # test_data_loader = DataLoader(ViatalSignDataset_triplet2(mode='test', cl=class_mode, model_mel=-1, model_thick=-1), batch_size=test_data_len, shuffle=False)
    #
    # optimizer = optim.Adam(feature_model.parameters(), lr=0.001)
    # # optimizer = optim.Adam(feature_model.parameters(), lr=0.01)
    # criterion = nn.TripletMarginLoss(margin=1, p=2)
    # test_criterion = nn.TripletMarginLoss(margin=1, p=2)
    #
    # best_loss = 5.7
    # best_test_loss = 5.7
    # # epochs = 1000
    # epochs = 5000
    #
    # for epoch in range(epochs):
    #     if epoch == 2000:
    #         optimizer = optim.Adam(feature_model.parameters(), lr=0.0005)
    #     if epoch == 4000:
    #         optimizer = optim.Adam(feature_model.parameters(), lr=0.0003)
    #     if epoch == 6000:
    #         optimizer = optim.Adam(feature_model.parameters(), lr=0.0001)
    #     if epoch == 10000:
    #         optimizer = optim.Adam(feature_model.parameters(), lr=0.00005)
    #     if epoch == 15000:
    #         optimizer = optim.Adam(feature_model.parameters(), lr=0.00001)
    #
    #     running_loss = []
    #     for anchor, pos, neg, _, _, _, _, _ in data_loader:
    #         # if use_gpu == True:
    #         #     anchor, pos, neg = anchor.to('cuda'), pos.to('cuda'), neg.to('cuda')
    #
    #         feature_model.train()
    #         anc_out = feature_model(anchor)
    #
    #         pos_out = feature_model(pos)
    #         neg_out = feature_model(neg)
    #         loss = criterion(anc_out, pos_out, neg_out)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss.append(loss.detach().cpu().numpy())
    #
    #     if epoch % 10 == 0:
    #         with torch.no_grad():
    #             mean_loss = np.mean(running_loss)
    #
    #             feature_model.eval()
    #
    #             for anchor_t, pos_t, neg_t, _, _, _, _, _ in test_data_loader:
    #                 # if use_gpu == True:
    #                 #    anchor_t, pos_t, neg_t = anchor_t.to('cuda'), pos_t.to('cuda'), neg_t.to('cuda')
    #
    #                 anc_out_t = feature_model(anchor_t)
    #
    #                 pos_out_t = feature_model(pos_t)
    #                 neg_out_t = feature_model(neg_t)
    #
    #                 test_criterion2 = nn.TripletMarginLoss(margin=1.0, p=3)
    #                 test_loss = test_criterion(anc_out_t, pos_out_t, neg_out_t)
    #                 test_loss2 = test_criterion2(anc_out_t, pos_out_t, neg_out_t)
    #
    #             if mean_loss < best_loss:
    #                 best_loss = mean_loss
    #                 print("Epoch: {}/{} - Loss: {:.4f} , Test Loss: {:.4f} (Save Model)".format(epoch + 1, epochs,
    #                                                                                             mean_loss, test_loss))
    #                 torch.save(feature_model.state_dict(), feature_path)
    #
    #             else:
    #                 print("Epoch: {}/{} - Loss: {:.4f} , Test Loss: {:.4f}".format(epoch + 1, epochs, mean_loss,
    #                                                                                test_loss))
    #                 torch.save(feature_model.state_dict(), feature_path3)
    #
    #             if test_loss < best_test_loss:
    #                 best_test_loss = test_loss
    #                 torch.save(feature_model.state_dict(), feature_path2)
    #
    #     if epoch == 10000:
    #         torch.save(feature_model.state_dict(), feature_path10000)


    ############# 2. Train Classification Model ###################
    print("**************************************************")
    print("****** 2.  Train Classification Model ************")
    print("**************************************************")

    feature_model.load_state_dict(torch.load(feature_path))
    feature_model.eval()

    temper_value = 1

    if use_gpu == True:
        classifier_model = Classifier().to('cuda')
    else:
        classifier_model = Classifier()

    data_loader = DataLoader(ViatalSignDataset_class(mode='train', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1), batch_size=3000, shuffle=True)

    test_data_len = len(ViatalSignDataset_class(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1))
    test_loader = DataLoader(ViatalSignDataset_class(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1), batch_size=test_data_len,
                             shuffle=False)

    optimizer = optim.Adam(classifier_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_loss = 9.0
    best_acc = 0.1
    best_test_loss = 9.0 #0.1
    epochs = 3000

    for epoch in range(epochs):
        if epoch == 1000:
            optimizer = optim.Adam(classifier_model.parameters(), lr=0.0005)
        elif epoch == 2000:
            optimizer = optim.Adam(classifier_model.parameters(), lr=0.0001)

        running_loss = []
        for anchor, m_label, tb_label, tb_label3, st_label, st_label3, th_label, total_label, total_label3 in data_loader:
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
            pred_prob = classifier_model(x_data)
            pred_prob = F.softmax(pred_prob, dim=1)
            loss = torch.mean(-(torch.sum(total_label3 * torch.log(pred_prob+0.000000000001), dim=1)))  # Cross Entropy Loss
            #pred_prob = pred_prob * temper_value

            #loss = criterion(pred_prob, gt_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # running_loss = torch.cat([running_loss, torch.unsqueeze(loss, dim=0)])

            if use_gpu == True:
                running_loss.append(loss.detach().cpu().numpy())
            else:
                running_loss.append(loss.detach().numpy())

        if epoch % 10 == 0:
            with torch.no_grad():
                mean_loss = np.mean(running_loss)

                classifier_model.eval()

                for anchor, m_label, tb_label, tb_label3, st_label, st_label3, th_label, total_label, total_label3 in test_loader:
                    x_data = feature_model(anchor)
                    classifier_model.eval()

                    pred_prob = classifier_model(x_data)
                    pred_prob = pred_prob * temper_value

                    pred_prob = F.softmax(pred_prob, dim=1)
                    test_loss = torch.mean(-(torch.sum(total_label3 * torch.log(pred_prob+0.000000000001), dim=1)))  # Cross Entropy Loss

                    top_p, top_class = pred_prob.topk(1, dim=1)

                    total_cnt = 0
                    equal_cnt = 0

                    for i in range(top_class.shape[0]):
                        if top_class[i][0].item() == total_label[i].item():
                            equal_cnt += 1
                        total_cnt += 1

                    acc = equal_cnt / total_cnt

                if mean_loss < best_loss:
                    best_loss = mean_loss
                    print("Epoch: {}/{} - Loss: {:.4f} , Test loss : {:.4f}, Test Acc : {:.4f} (Save Model)".format(epoch + 1, epochs,
                                                                                                mean_loss, test_loss, acc))
                    torch.save(classifier_model.state_dict(), classify_path)
                else:
                    print("Epoch: {}/{} - Loss: {:.4f} , Test loss : {:.4f}, Test Acc : {:.4f} ".format(epoch + 1, epochs, mean_loss, test_loss, acc))

                if test_loss < best_test_loss:
                    torch.save(classifier_model.state_dict(), classify_path2)
                    best_test_loss = test_loss

                # if acc > best_acc:
                #     torch.save(classifier_model.state_dict(), classify_path3)
                #     best_acc = acc


    print("**************************************************")
    print("****** 3.  Train Regression Model (Sto) ************")
    print("**************************************************")
    classifier_model.load_state_dict(torch.load(classify_path))
    classifier_model.eval()

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
            x_data = feature_model(anchor)
            pred_prob = classifier_model(x_data)
            pred_prob = F.softmax(pred_prob, dim=1)

            Reg_model.train()
            pred_value = Reg_model(pred_prob)

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
                    x_data = feature_model(anchor_t)
                    pred_prob = classifier_model(x_data)
                    # pred_prob = pred_prob * temper_value
                    pred_prob = F.softmax(pred_prob, dim=1)

                    Reg_model.eval()
                    pred_value = Reg_model(pred_prob)
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
    classifier_model.load_state_dict(torch.load(classify_path))
    classifier_model.eval()

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
            x_data = feature_model(anchor)
            pred_prob = classifier_model(x_data)
            # pred_prob = pred_prob * temper_value
            pred_prob = F.softmax(pred_prob, dim=1)

            Reg_model2.train()
            pred_value = Reg_model2(pred_prob)

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
                    x_data = feature_model(anchor_t)
                    pred_prob = classifier_model(x_data)
                    # pred_prob = pred_prob * temper_value
                    pred_prob = F.softmax(pred_prob, dim=1)

                    Reg_model2.eval()
                    pred_value = Reg_model2(pred_prob)
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













