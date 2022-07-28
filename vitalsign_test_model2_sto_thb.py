import numpy as np
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset2 import ViatalSignDataset_class
from dataset2 import ViatalSignDataset_regression

from sklearn.metrics import confusion_matrix
from util.confusion_matrix_plot import plot_confusion_matrix

'''
Sto, Thb Model 검증
'''

class VitalSign_Feature(nn.Module):
    def __init__(self):
        super(VitalSign_Feature, self).__init__()

        self.input_dim = 25 #43 #75

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
    def __init__(self, cl_mode):
        super(Classifier, self).__init__()

        self.input_dim = 128

        self.layer11 = nn.Linear(self.input_dim, 128)
        self.layer12 = nn.Linear(128, 128)
        self.layer13 = nn.Linear(128, 128)
        self.layer14 = nn.Linear(128, 128)

        self.layer15 = nn.Linear(128, 49)
        #self.layer15 = nn.Linear(128, 42)

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


        self.input_dim = 49
        #self.input_dim = 42

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

    save_dir_name = "vitalsign_sto_thb_0113_prob_02_0125_input14_m1_epoch5000_addinput3"
    # save_dir_name = "vitalsign_sto_thb_0928_integ_prob_changefesature"


    path = os.path.dirname(__file__)
    feature_path = os.path.join(path, './result/{}/feature_weight_data'.format(save_dir_name))
    feature_path2 = os.path.join(path, './result/{}/feature_weight_data2'.format(save_dir_name))

    classify_path = os.path.join(path, './result/{}/classification_weight_data'.format(save_dir_name))
    classify_path2 = os.path.join(path, './result/{}/classification_weight_data2'.format(save_dir_name))

    regression_path = os.path.join(path, './result/{}/regression_sto_weight_data'.format(save_dir_name))
    regression_path2 = os.path.join(path, './result/{}/regression_sto_weight_data2'.format(save_dir_name))

    regression_path_thb = os.path.join(path, './result/{}/regression_thb_weight_data'.format(save_dir_name))
    regression_path2_thb = os.path.join(path, './result/{}/regression_thb_weight_data2'.format(save_dir_name))


    if use_gpu == True:
        feature_model = VitalSign_Feature().to('cuda')
        classifier_model = Classifier(cl_mode=class_mode).to('cuda')
        Reg_model = Regression(cl_mode='sto').to('cuda')
        Reg_model_thb = Regression(cl_mode='thb').to('cuda')
    else:
        feature_model = VitalSign_Feature()
        classifier_model = Classifier(cl_mode=class_mode)
        Reg_model = Regression(cl_mode='sto')
        Reg_model_thb = Regression(cl_mode='thb')

    feature_model.load_state_dict(torch.load(feature_path))

    classifier_model.load_state_dict(torch.load(classify_path2))
    Reg_model.load_state_dict(torch.load(regression_path2))
    Reg_model_thb.load_state_dict(torch.load(regression_path2_thb))
    # try:
    #     classifier_model.load_state_dict(torch.load(classify_path2))
    # except:
    #     classifier_model.load_state_dict(torch.load(classify_path))
    # Reg_model.load_state_dict(torch.load(regression_path2))
    # Reg_model_thb.load_state_dict(torch.load(regression_path2_thb))

    feature_model.eval()
    classifier_model.eval()
    Reg_model.eval()

    ################### Test Classification Model ######################
    classifier_data_len = len(ViatalSignDataset_class(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1))
    classifier_loader = DataLoader(ViatalSignDataset_class(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1), batch_size=classifier_data_len,
                             shuffle=False)

    temper_value = 1

    total_cnt = 0
    equal_cnt = 0

    predict_result = []
    gt_list = []

    for anchor, m_label, tb_label,  tb_label3, st_label, st_label3, th_label, total_label, total_label3 in classifier_loader:
        x_data = feature_model(anchor)
        pred_prob = classifier_model(x_data)
        pred_prob = pred_prob * temper_value
        pred_prob = F.softmax(pred_prob, dim=1)

        test_loss = torch.mean(-(torch.sum(total_label3 * torch.log(pred_prob+0.000000000001), dim=1)))

        top_p, top_class = pred_prob.topk(1, dim=1)

        for data_idx in range(top_class.shape[0]):
            if top_class[data_idx][0].item() == total_label[data_idx].item():
                equal_cnt += 1
            predict_result.append(top_class[data_idx][0].item())
            gt_list.append(total_label[data_idx].item())
            total_cnt += 1

    acc = equal_cnt / total_cnt
    print("Classifier Accuracy : ", acc)
    print("Classifier Loss  : ", test_loss.item())

    cm = confusion_matrix(predict_result, gt_list)
    plot_confusion_matrix(cm=cm,
                          normalize=False,
                          target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
                                        '15',
                                        '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28',
                                        '29', '30', '31',
                                        '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44',
                                        '45', '46', '47', '48'])
                          # target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
                          #               '15',
                          #               '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28',
                          #               '29', '30', '31',
                          #               '32', '33', '34', '35', '36', '37', '38', '39', '40', '41'])

    print("**************************************************")
    print("****** 3.  Test Regression Model (Sto) ************")
    print("**************************************************")
    criterion = nn.MSELoss()

    reg_data_len = len(ViatalSignDataset_regression(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1))
    reg_loader = DataLoader(ViatalSignDataset_regression(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1), batch_size=reg_data_len, shuffle=False)

    for anchor, m_label, tb_label, st_label, th_label, _, _, _, _ in reg_loader:
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

    test_loss = criterion(pred_value, st_label)
    test_loss = torch.sqrt(test_loss)

    print("Regression Test loss : ", test_loss.item())

    # Calculate Variance
    # loss_list = ((prev_value_result_torch - gt_value_list_torch)**2)**(1/2)
    loss_list = ((pred_value - st_label)**2)**(1/2)
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
        gt_value_list.append(st_label[i].item())

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

    print("**************************************************")
    print("****** 3.  Test Regression Model (Thb) ************")
    print("**************************************************")
    criterion = nn.MSELoss()

    reg_data_len = len(
        ViatalSignDataset_regression(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1))
    reg_loader = DataLoader(
        ViatalSignDataset_regression(mode='val', cl=class_mode, use_gpu=use_gpu, model_mel=-1, model_thick=-1),
        batch_size=reg_data_len, shuffle=False)

    for anchor, m_label, tb_label, st_label, th_label, _, _, _, _ in reg_loader:
        x_data = feature_model(anchor)
        pred_prob = classifier_model(x_data)
        pred_prob = F.softmax(pred_prob, dim=1)
        pred_value = Reg_model_thb(pred_prob)

    # prev_value_result_torch = np.array(pred_value.detach().cpu(), dtype=np.float32)
    # prev_value_result_torch = torch.FloatTensor(prev_value_result_torch)
    #
    # gt_value_list_torch = np.array(m_value, dtype=np.float32)
    # gt_value_list_torch = torch.FloatTensor(gt_value_list_torch)
    #
    # test_loss = criterion(prev_value_result_torch, gt_value_list_torch)

    pred_value = torch.squeeze(pred_value)

    test_loss = criterion(pred_value, tb_label)
    test_loss = torch.sqrt(test_loss)

    print("Regression Test loss : ", test_loss.item())

    # Calculate Variance
    # loss_list = ((prev_value_result_torch - gt_value_list_torch)**2)**(1/2)
    loss_list = ((pred_value - tb_label) ** 2) ** (1 / 2)
    loss_mean = test_loss.item()

    deviation_list = []

    for di in range(len(loss_list)):
        deviation_list.append(((loss_list[di] - loss_mean) ** 2).item())

    sto_var = np.mean(deviation_list)
    print("Variance mean : ", sto_var)
    print("Standard variation mean : ", np.sqrt(sto_var))

    label_sort = []
    pred_sort = []

    prev_value_result = []
    gt_value_list = []

    for i in range(len(pred_value)):
        prev_value_result.append(pred_value[i].item())
        gt_value_list.append(tb_label[i].item())

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
