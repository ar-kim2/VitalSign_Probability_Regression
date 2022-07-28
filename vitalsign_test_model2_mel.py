import numpy as np
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torch.utils.data import DataLoader

from dataset2 import ViatalSignDataset_triplet_mel_thickness
from dataset2 import ViatalSignDataset_triplet_mel_thickness_v2
from dataset2 import ViatalSignDataset_class_mel_thickness
from dataset2 import ViatalSignDataset_regression_mel_thickness

from sklearn.metrics import confusion_matrix
from util.confusion_matrix_plot import plot_confusion_matrix

from sklearn.manifold import TSNE

from online_triplet_loss.losses import batch_hard_triplet_loss
from online_triplet_loss.losses import batch_all_triplet_loss


'''
Melanin Model 검증
'''

class VitalSign_Feature_mel_thickness(nn.Module):
    def __init__(self):
        super(VitalSign_Feature_mel_thickness, self).__init__()

        self.input_dim = 14

        self.common1 = nn.Linear(self.input_dim, 128)
        self.common2 = nn.Linear(128, 128)
        self.common3 = nn.Linear(128, 128)
        self.common4 = nn.Linear(128, 128)
        self.common5 = nn.Linear(128, 128)

        # self.common1 = nn.Linear(self.input_dim, 28)
        # self.common2 = nn.Linear(28, 28)
        # self.common3 = nn.Linear(28, 28)
        # self.common4 = nn.Linear(28, 28)
        # self.common5 = nn.Linear(28, 28)

        # self.common1 = nn.Linear(self.input_dim, 28)
        # self.common2 = nn.Linear(28, 28)
        # self.common3 = nn.Linear(28, 56)
        # self.common4 = nn.Linear(56, 56)
        # self.common5 = nn.Linear(56, 56)

    def forward(self, x):
        x = F.leaky_relu(self.common1(x))
        x = F.leaky_relu(self.common2(x))
        x = F.leaky_relu(self.common3(x))
        x = F.leaky_relu(self.common4(x))
        x = F.leaky_relu(self.common5(x))

        return x



class Classifier(nn.Module):
    def __init__(self, cl_mode):
        super(Classifier, self).__init__()

        self.input_dim = 128 #64 #128

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

    mel_class = [0, 1, 2, 3, 4, 5, 6, 7]

    # save_dir_name = "vitalsign_mel_1004_prob_005_input_64"
    save_dir_name = "vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3"

    if os.path.isdir("result/{}".format(save_dir_name)) == False:
        os.mkdir("result/{}".format(save_dir_name))

    path = os.path.dirname(__file__)
    feature_path = os.path.join(path, './result/{}/feature_weight_data'.format(save_dir_name))
    feature_path2 = os.path.join(path, './result/{}/feature_weight_data2'.format(save_dir_name))
    # feature_path3 = os.path.join(path, './result/{}/feature_weight_data3'.format(save_dir_name))

    classify_path = os.path.join(path, './result/{}/classification_weight_data'.format(save_dir_name))
    classify_path2 = os.path.join(path, './result/{}/classification_weight_data2'.format(save_dir_name))

    regression_path = os.path.join(path, './result/{}/regression_weight_data'.format(save_dir_name))
    regression_path2 = os.path.join(path, './result/{}/regression_weight_data2'.format(save_dir_name))

    if use_gpu == True:
        feature_model = VitalSign_Feature_mel_thickness().to('cuda')
        classifier_model = Classifier(cl_mode=class_mode).to('cuda')
        Reg_model = Regression(cl_mode=class_mode).to('cuda')
    else:
        feature_model = VitalSign_Feature_mel_thickness()
        classifier_model = Classifier(cl_mode=class_mode)
        Reg_model = Regression(cl_mode=class_mode)

    feature_model.load_state_dict(torch.load(feature_path))
    try:
        classifier_model.load_state_dict(torch.load(classify_path2))
    except:
        classifier_model.load_state_dict(torch.load(classify_path))
    Reg_model.load_state_dict(torch.load(regression_path2))

    feature_model.eval()
    classifier_model.eval()
    Reg_model.eval()

    ############# 1. Train Feature Model ###################
    print("**************************************************")
    print("************ 1.  Train Feature Model *************")
    print("**************************************************")
    test_data_len = len(
        ViatalSignDataset_triplet_mel_thickness_v2(mode='test', cl=class_mode, model_mel=-1, model_thick=-1))
    test_data_loader = DataLoader(
        ViatalSignDataset_triplet_mel_thickness_v2(mode='test', cl=class_mode, model_mel=-1, model_thick=-1),
        batch_size=test_data_len, shuffle=False)

    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    out_list = []
    label = []


    with torch.no_grad():
        feature_model.eval()

        # for anchor_t, pos_t, neg_t, mel_pos_t, mel_neg_t, thb_pos_t, thb_neg_t, sto_pos_t, sto_neg_t, thickness_pos_t, thickness_neg_t, _, _, _, _, _ in test_data_loader:
        for anchor_t, pos_t, neg_t, m_label, _, _, _, _ in test_data_loader:
            if use_gpu == True:
               anchor_t = anchor_t.to('cuda')

            anc_out_t = feature_model(anchor_t)

            pos_out_t = feature_model(pos_t)
            neg_out_t = feature_model(neg_t)

            # test_loss = batch_hard_triplet_loss(m_label, anc_out_t, margin=1)

            test_criterion = nn.TripletMarginLoss(margin=1.0, p=2)
            test_loss2 = test_criterion(anc_out_t, pos_out_t, neg_out_t)

            if len(out_list) == 0:
                out_list = anc_out_t.detach().cpu().numpy()
                label = m_label.detach().cpu().numpy()
            else:
                out_list = np.concatenate([out_list, anc_out_t.detach().cpu().numpy()])
                label = np.concatenate([label, m_label.numpy()])

        print("CHeck test_loss : ", test_loss2.detach().cpu().numpy())


    model = TSNE(learning_rate=100, random_state=1)
    transformed = model.fit_transform(out_list)

    x_trans1 = []
    x_trans2 = []
    x_trans3 = []
    x_trans4 = []
    x_trans5 = []
    x_trans6 = []
    x_trans7 = []
    x_trans8 = []

    y_trans1 = []
    y_trans2 = []
    y_trans3 = []
    y_trans4 = []
    y_trans5 = []
    y_trans6 = []
    y_trans7 = []
    y_trans8 = []

    for i in range(len(transformed)):
        if label[i] == 0:
            x_trans1.append(transformed[i, 0])
            y_trans1.append(transformed[i, 1])
        elif label[i] == 1:
            x_trans2.append(transformed[i, 0])
            y_trans2.append(transformed[i, 1])
        elif label[i] == 2:
            x_trans3.append(transformed[i, 0])
            y_trans3.append(transformed[i, 1])
        elif label[i] == 3:
            x_trans4.append(transformed[i, 0])
            y_trans4.append(transformed[i, 1])
        elif label[i] == 4:
            x_trans5.append(transformed[i, 0])
            y_trans5.append(transformed[i, 1])
        elif label[i] == 5:
            x_trans6.append(transformed[i, 0])
            y_trans6.append(transformed[i, 1])
        elif label[i] == 6:
            x_trans7.append(transformed[i, 0])
            y_trans7.append(transformed[i, 1])
        elif label[i] == 7:
            x_trans8.append(transformed[i, 0])
            y_trans8.append(transformed[i, 1])

    plt.figure()
    plt.scatter(x_trans1, y_trans1, color='r', s=5, label='0 < mel <= 2')
    plt.scatter(x_trans2, y_trans2, color='orange', s=5, label='2 < mel <= 4')
    plt.scatter(x_trans3, y_trans3, color='gold', s=5, label='4 < mel <= 6')
    plt.scatter(x_trans4, y_trans4, color='g', s=5, label='6 < mel <= 8')
    plt.scatter(x_trans5, y_trans5, color='cyan', s=5, label='8 < mel <= 10')
    plt.scatter(x_trans6, y_trans6, color='blue', s=5, label='10 < mel <= 12')
    plt.scatter(x_trans7, y_trans7, color='purple', s=5, label='12 < mel <= 14')
    plt.scatter(x_trans8, y_trans8, color='gray', s=5, label='14 < mel')
    plt.legend()
    plt.show()


    ################### Test Classification Model ######################
    print("**************************************************")
    print("************ 2.  Test Classification Model *************")
    print("**************************************************")
    classifier_data_len = len(ViatalSignDataset_class_mel_thickness(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1))
    classifier_loader = DataLoader(ViatalSignDataset_class_mel_thickness(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1), batch_size=classifier_data_len,
                             shuffle=False)

    temper_value = 1

    total_cnt = 0
    equal_cnt = 0

    predict_result = []
    gt_list = []

    criterion = nn.MSELoss()

    for anchor, m_label, m_label2, m_label3, tb_label, st_label, th_label, _, th_label3, total_label in classifier_loader:
        x_data = feature_model(anchor)
        pred_prob = classifier_model(x_data)
        pred_prob = pred_prob * temper_value
        pred_prob = F.softmax(pred_prob, dim=1)
        #
        # check_idx = 1500
        #
        # plt.figure()
        # for cidx in range(25):
        #     check_idx = cidx * 51
        #     print("CHECK pred_prob : ", pred_prob[check_idx].detach().cpu().numpy())
        #     print("CHECK gt value : ", m_label2[check_idx].detach().cpu().numpy())
        #     print("CHECK gt prob : ", m_label3[check_idx].detach().cpu().numpy())
        #
        #     gt_value = m_label2[check_idx].detach().cpu().numpy()
        #     gt_value = np.round(gt_value, 4)
        #
        #     plt.subplot(5, 5, cidx+1)
        #     plt.title("Mel GT Value : {}".format(gt_value))
        #     plt.plot(pred_prob[check_idx].detach().cpu().numpy(), label='Pred Prob')
        #     plt.plot(m_label3[check_idx].detach().cpu().numpy(), label='GT Prob')
        #     plt.legend()
        # plt.show()

        test_loss = torch.mean(-(torch.sum(m_label3 * torch.log(pred_prob+0.000000000000001), dim=1)))

        rmse_loss = criterion(pred_prob, m_label3)

        top_p, top_class = pred_prob.topk(1, dim=1)

        for data_idx in range(top_class.shape[0]):
            if top_class[data_idx][0].item() == m_label[data_idx].item():
                equal_cnt += 1
            predict_result.append(top_class[data_idx][0].item())
            gt_list.append(m_label[data_idx].item())
            total_cnt += 1

    acc = equal_cnt / total_cnt
    print("Classifier Accuracy : ", acc)
    print("Classifier CE Loss  : ", test_loss.item())
    print("Classifier RMSE Loss  : ", rmse_loss.item())

    cm = confusion_matrix(predict_result, gt_list)
    plot_confusion_matrix(cm=cm,
                          normalize=False,
                          target_names = ['0', '1', '2', '3', '4', '5', '6', '7'])

    print("**************************************************")
    print("****** 3.  Test Regression Model ************")
    print("**************************************************")
    criterion = nn.MSELoss()

    reg_data_len = len(ViatalSignDataset_regression_mel_thickness(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1))
    reg_loader = DataLoader(ViatalSignDataset_regression_mel_thickness(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1), batch_size=reg_data_len, shuffle=False)

    for anchor, m_value, tb_value, st_value, th_value, m_label, tb_label, st_label, th_label in reg_loader:
        x_data = feature_model(anchor)
        pred_prob = classifier_model(x_data)
        pred_prob = F.softmax(pred_prob, dim=1)
        pred_value = Reg_model(pred_prob)

    # prev_value_result_torch = np.array(pred_value.detach().cpu(), dtype=np.float32)
    # prev_value_result_torch = torch.FloatTensor(prev_value_result_torch)
    #
    # gt_value_list_torch = np.array(m_value, dtype=np.float32)
    # gt_value_list_torch = torch.FloatTensor(gt_value_list_torch)
    #
    # test_loss = criterion(prev_value_result_torch, gt_value_list_torch)

    pred_value = torch.squeeze(pred_value)

    test_loss = criterion(pred_value, m_value)
    test_loss = torch.sqrt(test_loss)

    print("Regression Test loss : ", test_loss.item())

    # Calculate Variance
    # loss_list = ((prev_value_result_torch - gt_value_list_torch)**2)**(1/2)
    loss_list = ((pred_value - m_value)**2)**(1/2)
    loss_mean = test_loss.item()

    deviation_list = []

    for di in range(len(loss_list)):
        deviation_list.append(((loss_list[di]-loss_mean)**2).item())

    sto_var = np.mean(deviation_list)
    print("Variance mean : ", sto_var)
    print("Standard variation mean : ", np.sqrt(sto_var))


    label_sort = []
    pred_sort = []

    prev_value_result = []
    gt_value_list = []

    for i in range(len(pred_value)):
        prev_value_result.append(pred_value[i].item())
        gt_value_list.append(m_value[i].item())

    for i in range(len(gt_value_list)):
        min_idx = np.argmin(gt_value_list)
        label_sort.append(gt_value_list[min_idx])
        pred_sort.append(prev_value_result[min_idx])
        gt_value_list = np.delete(gt_value_list, min_idx)
        prev_value_result = np.delete(prev_value_result, min_idx)

    x_axis = []

    for i in range(np.shape(label_sort)[0]):
        x_axis.append(i)

    plt.figure()
    plt.scatter(x_axis, pred_sort, label="pred_value", s=3)
    plt.scatter(x_axis, label_sort, label='gt', s=3)
    plt.legend()
    plt.show()
