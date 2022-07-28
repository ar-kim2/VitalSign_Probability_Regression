import numpy as np
import os

import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import random
import torch.nn as nn

#from vitalsign_feature_model import VitalSign_Feature_mel_thickness
#from vitalsign_classfication_model import Classifier

import torch.nn.functional as F


# [0902] classification class thb + sto combination
# [0902] input data absorbance
# [0908] input data absorbance + mel one-hot + thickness one-hot
# [0908] classification class thb
# [0908] classification class sto
# [0909] input data absorbance + mel one-hot + thickness one-hot + thb one-hot


class VitalSign_Feature_mel_thickness(nn.Module):
    def __init__(self):
        super(VitalSign_Feature_mel_thickness, self).__init__()

        self.input_dim = 14 #32 #75 #82 #64 #75 #77 # 66 #64

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

        self.input_dim = 128 #64 #128

        self.layer11 = nn.Linear(self.input_dim, 128)
        self.layer12 = nn.Linear(128, 128)
        self.layer13 = nn.Linear(128, 128)
        self.layer14 = nn.Linear(128, 128)

        if cl_mode == 'mel':
            self.layer15 = nn.Linear(128, 8)
        elif cl_mode == 'thb':
            self.layer15 = nn.Linear(128, 7)
        elif cl_mode == 'sto':
            self.layer15 = nn.Linear(128, 7)
        elif cl_mode == 'thickness':
            self.layer15 = nn.Linear(128, 3)
        else:
            self.layer15 = nn.Linear(128, 49)
            #self.layer15 = nn.Linear(128, 7)

    def forward(self, x):
        x1 = F.leaky_relu(self.layer11(x))
        x1 = F.leaky_relu(self.layer12(x1))
        x1 = F.leaky_relu(self.layer13(x1))
        x1 = F.leaky_relu(self.layer14(x1))
        x1 = self.layer15(x1)

        return x1


def calculate_k(x, y, z):
    # x : 851.35  measure idx : 24
    # y : 490.83  measure idx : 0
    # z : 668.79  measure idx : 10

    a = -54.540945783151464
    b = 21.095479254243322
    c = 78.56709080853545
    d =  -4.968144415839242

    t = -(a*x+b*y+c*z+d)/(a+b+c)
    meaure_r_p = x+t
    meaure_g_p = y+t
    meaure_b_p = z+t

    k = x- meaure_r_p

    distance = (((x- meaure_r_p) ** 2) + ((y - meaure_g_p) ** 2) +((z- meaure_b_p) ** 2))/3
    distance = distance**0.5

    # print("Origin point : ", x, ", ",y, ", ",z)
    # print("Covert point : ", meaure_r_p, ", ",meaure_g_p, ", ",meaure_b_p)
    # print("CHECK distance : ", distance, " k : ", k)

    return distance, k

class ViatalSignDataset_triplet(data.Dataset):
    def __init__(self, mode='train', cl='', model_mel= 0, model_thick=0):
        self.mode = mode
        self.cl = cl

        if mode == "train":
            reflect_list, m_label, tb_label, st_label, th_label = self.read_vitalsign_dataset(name='train', model_mel=model_mel, model_thick=model_thick)
        else:
            reflect_list, m_label, tb_label, st_label, th_label = self.read_vitalsign_dataset(name='test', model_mel=model_mel, model_thick=model_thick)

        self.ref_list, self.m_label, self.tb_label, self.st_label, self.th_label = reflect_list, m_label, tb_label, st_label, th_label

        mel_feature_model = VitalSign_Feature_mel_thickness().to('cuda')
        thickness_feature_model = VitalSign_Feature_mel_thickness().to('cuda')

        mel_classifier_model = Classifier(cl_mode="mel").to('cuda')
        thickness_classifier_model = Classifier(cl_mode="thickness").to('cuda')

        path = os.path.dirname(__file__)

        mel_feature_path = os.path.join(path, './result/vitalsign_mel_0104_prob_01_input14_m1_epoch5000_addinput3/feature_weight_data')
        thickness_feature_path = os.path.join(path, './result/vitalsign_thickness_0104_prob_01_input14_m1_epoch5000_addinput3/feature_weight_data')

        mel_classify_path = os.path.join(path, './result/vitalsign_mel_0104_prob_01_input14_m1_epoch5000_addinput3/classification_weight_data2')
        thickness_classify_path = os.path.join(path, './result/vitalsign_thickness_0104_prob_01_input14_m1_epoch5000_addinput3/classification_weight_data2')

        mel_feature_model.load_state_dict(torch.load(mel_feature_path))
        mel_classifier_model.load_state_dict(torch.load(mel_classify_path))

        thickness_feature_model.load_state_dict(torch.load(thickness_feature_path))
        thickness_classifier_model.load_state_dict(torch.load(thickness_classify_path))

        mel_x = mel_feature_model(torch.FloatTensor(self.ref_list).to('cuda'))
        mel_out = mel_classifier_model(mel_x)
        mel_prob = F.softmax(mel_out, dim=1)
        mel_prob = mel_prob.detach().cpu().numpy()

        thickness_x = thickness_feature_model(torch.FloatTensor(self.ref_list).to('cuda'))
        thickenss_out = thickness_classifier_model(thickness_x)
        thickness_prob = F.softmax(thickenss_out, dim=1)
        thickness_prob = thickness_prob.detach().cpu().numpy()

        self.ref_list = np.concatenate((self.ref_list, mel_prob, thickness_prob), axis=1)

        # [0902] classification class thb + sto combination
        self.total_label = (self.tb_label * 7) + self.st_label

        self.positive_list = []
        self.negative_list = []

        for i in range(len(m_label)):
            totall = self.total_label[i]

            negative_idx = np.where(self.total_label != totall)
            positive_idx = np.where(self.total_label == totall)

            self.positive_list.append(positive_idx[0])
            self.negative_list.append(negative_idx[0])

        self.random_idx = 0

        self.ref_list = torch.FloatTensor(self.ref_list).to('cuda')
        self.m_label = torch.LongTensor(m_label).to('cuda')
        self.tb_label = torch.LongTensor(tb_label).to('cuda')
        self.st_label = torch.LongTensor(st_label).to('cuda')
        self.th_label = torch.LongTensor(th_label).to('cuda')
        self.total_label = torch.LongTensor(self.total_label).to('cuda')


    def __getitem__(self, index):
        anchor = self.ref_list[index]
        ml = self.m_label[index]
        tbl = self.tb_label[index]
        stl = self.st_label[index]
        thl = self.th_label[index]
        totall = self.total_label[index]

        p_idx = self.positive_list[index][torch.randint(len(self.positive_list[index]), (1,))]
        positive = self.ref_list[p_idx]

        n_idx = self.negative_list[index][torch.randint(len(self.negative_list[index]), (1,))]
        negative = self.ref_list[n_idx]

        # Combination으로 세밀하게 나누는 경우
        # positive_idx = np.where(self.total_label == totall)
        # negative_idx = np.where(self.total_label != totall)

        # mel_positive_idx = np.where(self.m_label == ml)
        # mel_negative_idx = np.where(self.m_label != ml)
        #
        # thb_positive_idx = np.where(self.tb_label == tbl)
        # thb_negative_idx = np.where(self.tb_label != tbl)
        #
        # sto_positive_idx = np.where(self.st_label == stl)
        # sto_negative_idx = np.where(self.st_label != stl)
        #
        # thickness_positive_idx = np.where(self.th_label == thl)
        # thickness_negative_idx = np.where(self.th_label != thl)

        # positive = self.ref_list[random.choice(positive_idx[0])]
        # negative = self.ref_list[random.choice(negative_idx[0])]

        # mel_positive = self.ref_list[random.choice(mel_positive_idx[0])]
        # mel_negative = self.ref_list[random.choice(mel_negative_idx[0])]
        #
        # thb_positive = self.ref_list[random.choice(thb_positive_idx[0])]
        # thb_negative = self.ref_list[random.choice(thb_negative_idx[0])]
        #
        # sto_positive = self.ref_list[random.choice(sto_positive_idx[0])]
        # sto_negative = self.ref_list[random.choice(sto_negative_idx[0])]
        #
        # thickness_positive = self.ref_list[random.choice(thickness_positive_idx[0])]
        # thickness_negative = self.ref_list[random.choice(thickness_negative_idx[0])]

        # ml = torch.LongTensor([ml])
        # tbl = torch.LongTensor([tbl])
        # stl = torch.LongTensor([stl])
        # thl = torch.LongTensor([thl])
        # totall = torch.LongTensor([totall])

        #return anchor, positive, negative, mel_positive, mel_negative, thb_positive, thb_negative, sto_positive, sto_negative, thickness_positive, thickness_negative, ml, tbl, stl, thl, totall
        return anchor, positive, negative, ml, tbl, stl, thl, totall


    def __len__(self):
        return len(self.ref_list)

    def read_vitalsign_dataset(self, name, model_mel, model_thick):
        '''
        Read Data
        '''

        path = os.path.dirname(__file__)

        # fileNameList = ['input_data9.npy', 'input_data10.npy', 'input_data11.npy', 'input_data13.npy',\
        #                 'input_data_random9.npy', 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        #                 'input_data_random21.npy', 'input_data_random22.npy','input_data_random23.npy', 'input_data_random24.npy']
        #
        fileNameList = ['input_data9.npy', 'input_data10.npy', \
                        'input_data11.npy', 'input_data13.npy', 'input_data14.npy', 'input_data15.npy',\
                        'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy', \
                        'input_data_random18.npy', 'input_data_random19.npy', 'input_data_random20.npy', 'input_data_random21.npy',\
                        'input_data_random22.npy', 'input_data_random23.npy', 'input_data_random24.npy']

        # fileNameList = ['input_data9.npy', 'input_data14.npy', 'input_data15.npy',\
        #                 'input_data_random9.npy', 'input_data_random21.npy', 'input_data_random22.npy','input_data_random23.npy', 'input_data_random24.npy', 'input_data13.npy', 'input_data_random20.npy']

        test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']

        # fileNameList = ['input_data6.npy', 'input_data7.npy', 'input_data8.npy', 'input_data9.npy', 'input_data10.npy', \
        #                 'input_data11.npy', 'input_data12.npy', 'input_data13.npy', 'input_data14.npy', 'input_data15.npy',\
        #                 'input_data_random6.npy', 'input_data_random7.npy', 'input_data_random8.npy', 'input_data_random9.npy', \
        #                 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        #                 'input_data_random14.npy', 'input_data_random15.npy', 'input_data_random16.npy','input_data_random17.npy', \
        #                 'input_data_random18.npy', 'input_data_random19.npy', 'input_data_random20.npy', 'input_data_random21.npy', \
        #                 'input_data_random22.npy', 'input_data_random23.npy', 'input_data_random24.npy']
        #
        # test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']
        #
        # fileNameList = ['input_data_4000_1.npy', 'input_data_4000_2.npy','input_data_4000_3.npy','input_data_4000_4.npy',\
        #                 'random_data_4000_1.npy', 'random_data_4000_2.npy', 'random_data_4000_3.npy',\
        #                 'random_data_4000_4.npy', 'random_data_4000_5.npy', 'random_data_4000_6.npy']
        #
        # test_fileNameList = ['input_data_4000_1.npy', 'input_data_4000_2.npy','input_data_4000_3.npy','input_data_4000_4.npy',\
        #                 'random_data_4000_1.npy', 'random_data_4000_2.npy', 'random_data_4000_3.npy',\
        #                 'random_data_4000_4.npy', 'random_data_4000_5.npy', 'random_data_4000_6.npy']

        if name == 'train':
            fileNameList = fileNameList
        else:
            fileNameList = test_fileNameList

        train_data = []

        for fn in fileNameList:
            temp_data = np.load(path + '/input_data/{}'.format(fn), allow_pickle=True)

            if len(train_data) == 0 :
                train_data = temp_data
            else:
                train_data = np.concatenate([train_data, temp_data], axis=1)

        reflect_list = np.transpose(train_data[:, :, 5])

        out2 = train_data[0, :, 0:4]

        m_label = []
        tb_label = []
        st_label = []
        th_label = []

        m_label2 = []
        tb_label2 = []
        st_label2 = []
        th_label2 = []

        for d_idx in range(len(out2)):
            if out2[d_idx][0] > 0 and out2[d_idx][0] <= 0.02:
                m_i2 = [1, 0, 0, 0, 0, 0, 0, 0]
                m_i = 0
            elif out2[d_idx][0] > 0.02 and out2[d_idx][0] <= 0.04:
                m_i2 = [0, 1, 0, 0, 0, 0, 0, 0]
                m_i = 1
            elif out2[d_idx][0] > 0.04 and out2[d_idx][0] <= 0.06:
                m_i2 = [0, 0, 1, 0, 0, 0, 0, 0]
                m_i = 2
            elif out2[d_idx][0] > 0.06 and out2[d_idx][0] <= 0.08:
                m_i2 = [0, 0, 0, 1, 0, 0, 0, 0]
                m_i = 3
            elif out2[d_idx][0] > 0.08 and out2[d_idx][0] <= 0.1:
                m_i2 = [0, 0, 0, 0, 1, 0, 0, 0]
                m_i = 4
            elif out2[d_idx][0] > 0.1 and out2[d_idx][0] <= 0.12:
                m_i2 = [0, 0, 0, 0, 0, 1, 0, 0]
                m_i = 5
            elif out2[d_idx][0] > 0.12 and out2[d_idx][0] <= 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 1, 0]
                m_i = 6
            elif out2[d_idx][0] > 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 0, 1]
                m_i = 7

            m_label.append(m_i)
            m_label2.append(m_i2)

            if out2[d_idx][1] > 0 and out2[d_idx][1] <= 0.01:
                tb_i2 = [1, 0, 0, 0, 0, 0, 0]
                tb_i = 0
            elif out2[d_idx][1] > 0.01 and out2[d_idx][1] <= 0.02:
                tb_i2 = [0, 1, 0, 0, 0, 0, 0]
                tb_i = 1
            elif out2[d_idx][1] > 0.02 and out2[d_idx][1] <= 0.03:
                tb_i2 = [0, 0, 1, 0, 0, 0, 0]
                tb_i = 2
            elif out2[d_idx][1] > 0.03 and out2[d_idx][1] <= 0.04:
                tb_i2 = [0, 0, 0, 1, 0, 0, 0]
                tb_i = 3
            elif out2[d_idx][1] > 0.04 and out2[d_idx][1] <= 0.05:
                tb_i2 = [0, 0, 0, 0, 1, 0, 0]
                tb_i = 4
            elif out2[d_idx][1] > 0.05 and out2[d_idx][1] <= 0.06:
                tb_i2 = [0, 0, 0, 0, 0, 1, 0]
                tb_i = 5
            elif out2[d_idx][1] > 0.06 and out2[d_idx][1] <= 0.07:
                tb_i2 = [0, 0, 0, 0, 0, 0, 1]
                tb_i = 6

            tb_label.append(tb_i)
            tb_label2.append(tb_i2)

            if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
                st_i2 = [1, 0, 0, 0, 0, 0, 0]
                st_i = 0
            elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.75:
                st_i2 = [0, 1, 0, 0, 0, 0, 0]
                st_i = 1
            elif out2[d_idx][2] > 0.75 and out2[d_idx][2] <= 0.8:
                st_i2 = [0, 0, 1, 0, 0, 0, 0]
                st_i = 2
            elif out2[d_idx][2] > 0.8 and out2[d_idx][2] <= 0.85:
                st_i2 = [0, 0, 0, 1, 0, 0, 0]
                st_i = 3
            elif out2[d_idx][2] > 0.85 and out2[d_idx][2] <= 0.9:
                st_i2 = [0, 0, 0, 0, 1, 0, 0]
                st_i = 4
            elif out2[d_idx][2] > 0.9 and out2[d_idx][2] <= 0.95:
                st_i2 = [0, 0, 0, 0, 0, 1, 0]
                st_i = 5
            elif out2[d_idx][2] > 0.95 and out2[d_idx][2] <= 1.0:
                st_i2 = [0, 0, 0, 0, 0, 0, 1]
                st_i = 6

            # if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
            #     st_i2 = [1, 0, 0, 0]
            #     st_i = 0
            # elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.8:
            #     st_i2 = [0, 1, 0, 0]
            #     st_i = 1
            # elif out2[d_idx][2] > 0.8 and out2[d_idx][2] <= 0.9:
            #     st_i2 = [0, 0, 1, 0]
            #     st_i = 2
            # elif out2[d_idx][2] > 0.9 and out2[d_idx][2] <= 1.0:
            #     st_i2 = [0, 0, 0, 1]
            #     st_i = 3

            # if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
            #     st_i2 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 0
            # elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.73:
            #     st_i2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 1
            # elif out2[d_idx][2] > 0.73 and out2[d_idx][2] <= 0.76:
            #     st_i2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 2
            # elif out2[d_idx][2] > 0.76 and out2[d_idx][2] <= 0.79:
            #     st_i2 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 3
            # elif out2[d_idx][2] > 0.79 and out2[d_idx][2] <= 0.82:
            #     st_i2 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            #     st_i = 4
            # elif out2[d_idx][2] > 0.82 and out2[d_idx][2] <= 0.85:
            #     st_i2 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            #     st_i = 5
            # elif out2[d_idx][2] > 0.85 and out2[d_idx][2] <= 0.88:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            #     st_i = 6
            # elif out2[d_idx][2] > 0.88 and out2[d_idx][2] <= 0.91:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            #     st_i = 7
            # elif out2[d_idx][2] > 0.91 and out2[d_idx][2] <= 0.94:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
            #     st_i = 8
            # elif out2[d_idx][2] > 0.94 and out2[d_idx][2] <= 0.97:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            #     st_i = 9
            # elif out2[d_idx][2] > 0.97 and out2[d_idx][2] <= 1.0:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            #     st_i = 10

            st_label.append(st_i)
            st_label2.append(st_i2)

            if out2[d_idx][3] <= 0.035:
                th_i2 = [1, 0, 0]
                th_i = 0
            elif out2[d_idx][3] > 0.035 and out2[d_idx][3] <= 0.055:
                th_i2 = [0, 1, 0]
                th_i = 1
            elif out2[d_idx][3] > 0.055 and out2[d_idx][3] <= 0.075:
                th_i2 = [0, 0, 1]
                th_i = 2

            th_label.append(th_i)
            th_label2.append(th_i2)

        reflect_list_ex = []
        m_label_ex = []
        tb_label_ex = []
        st_label_ex = []
        th_label_ex = []

        for i in range(len(reflect_list)):
            if model_mel == -1 and model_thick == -1:
                temp_ref = []
                # for ri in range(54):
                #     temp_ref.append(reflect_list[i][ri])

                # temp_ref.append(reflect_list[i][5])

                # temp_ref.append(reflect_list[i][8])
                temp_ref.append(reflect_list[i][9])

                # temp_ref.append(reflect_list[i][11])
                temp_ref.append(reflect_list[i][12])

                # temp_ref.append(reflect_list[i][15])
                temp_ref.append(reflect_list[i][16])

                # temp_ref.append(reflect_list[i][18])
                temp_ref.append(reflect_list[i][19])

                # temp_ref.append(reflect_list[i][21])
                # temp_ref.append(reflect_list[i][22])  # temp_ref.append(reflect_list[i][25])

                # temp_ref.append(reflect_list[i][24])
                temp_ref.append(reflect_list[i][25])  # temp_ref.append(reflect_list[i][25])

                # temp_ref.append(reflect_list[i][26])
                temp_ref.append(reflect_list[i][27])

                # temp_ref.append(reflect_list[i][28])
                temp_ref.append(reflect_list[i][29])

                # temp_ref.append(reflect_list[i][30])
                temp_ref.append(reflect_list[i][31])

                # temp_ref.append(reflect_list[i][33])
                temp_ref.append(reflect_list[i][34])
                temp_ref.append(reflect_list[i][35])
                # temp_ref.append(reflect_list[i][37])
                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.6)
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.83)

                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][39])

                # temp_ref.append(reflect_list[i][41])
                # temp_ref.append(reflect_list[i][42])  # temp_ref.append(reflect_list[i][43])

                # temp_ref.append(reflect_list[i][43])
                temp_ref.append(reflect_list[i][44])
                temp_ref.append((reflect_list[i][46] + reflect_list[i][47]) / 2)

                # temp_ref.append(reflect_list[i][45])
                # temp_ref.append(reflect_list[i][46])

                # temp_ref.append(reflect_list[i][47])
                temp_ref.append(reflect_list[i][48])
                temp_ref.append(reflect_list[i][49])

                # temp_ref.append(reflect_list[i][50])
                # temp_ref.append(reflect_list[i][51])
                # temp_ref.append((reflect_list[i][52] + reflect_list[i][53])/2)
                # temp_ref.append(reflect_list[i][54])
                # temp_ref.append((reflect_list[i][55] + reflect_list[i][56]) / 2)
                # temp_ref.append((reflect_list[i][56] + reflect_list[i][57]) / 2)
                # temp_ref.append(reflect_list[i][57])
                # temp_ref.append(reflect_list[i][59])
                # temp_ref.append(reflect_list[i][60])
                # temp_ref.append(reflect_list[i][61])
                # temp_ref.append(reflect_list[i][62])
                # temp_ref.append(reflect_list[i][63])

                reflect_list_ex.append(temp_ref)
                #reflect_list_ex.append(reflect_list_concat)
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])
            elif m_label[i] == model_mel and th_label[i] == model_thick:
                reflect_list_ex.append(reflect_list[i])
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])

        # m_label = np.array(m_label, dtype=np.float32)
        # tb_label = np.array(tb_label, dtype=np.float32)
        # st_label = np.array(st_label, dtype=np.float32)
        # th_label = np.array(th_label, dtype=np.float32)

        m_label = np.array(m_label_ex, dtype=np.float32)
        tb_label = np.array(tb_label_ex, dtype=np.float32)
        st_label = np.array(st_label_ex, dtype=np.float32)
        th_label = np.array(th_label_ex, dtype=np.float32)

        #reflect_list = np.concatenate([reflect_list, m_label2, th_label2], axis=1)
        #reflect_list = np.concatenate((reflect_list, m_label[:, np.newaxis], th_label[:, np.newaxis]), axis=1)
        #reflect_list = np.array(reflect_list, dtype=np.float32)
        reflect_list = np.array(reflect_list_ex, dtype=np.float32)

        return reflect_list, m_label, tb_label, st_label, th_label



class ViatalSignDataset_triplet2(data.Dataset):
    def __init__(self, mode='train', cl='', model_mel= 0, model_thick=0):
        self.mode = mode
        self.cl = cl

        if mode == "train":
            reflect_list, m_label, tb_label, st_label, th_label, m_label3, th_label3 = self.read_vitalsign_dataset(name='train', model_mel=model_mel, model_thick=model_thick)
        else:
            reflect_list, m_label, tb_label, st_label, th_label, m_label3, th_label3 = self.read_vitalsign_dataset(name='test', model_mel=model_mel, model_thick=model_thick)

        self.ref_list, self.m_label, self.tb_label, self.st_label, self.th_label = reflect_list, m_label, tb_label, st_label, th_label

        mel_feature_model = VitalSign_Feature_mel_thickness().to('cuda')
        thickness_feature_model = VitalSign_Feature_mel_thickness().to('cuda')

        mel_classifier_model = Classifier(cl_mode="mel").to('cuda')
        thickness_classifier_model = Classifier(cl_mode="thickness").to('cuda')

        path = os.path.dirname(__file__)

        # mel_feature_path = os.path.join(path, './result/vitalsign_mel_1230_prob_005_input14_m05_epoch20000_alldata/feature_weight_data')
        # thickness_feature_path = os.path.join(path, './result/vitalsign_thickness_1230_prob_005_input14_m05_epoch20000_alldata/feature_weight_data')
        #
        # mel_classify_path = os.path.join(path, './result/vitalsign_mel_1230_prob_005_input14_m05_epoch20000_alldata/classification_weight_data2')
        # thickness_classify_path = os.path.join(path, './result/vitalsign_thickness_1230_prob_005_input14_m05_epoch20000_alldata/classification_weight_data2')
        #
        # mel_feature_model.load_state_dict(torch.load(mel_feature_path))
        # mel_classifier_model.load_state_dict(torch.load(mel_classify_path))
        #
        # thickness_feature_model.load_state_dict(torch.load(thickness_feature_path))
        # thickness_classifier_model.load_state_dict(torch.load(thickness_classify_path))
        #
        # mel_x = mel_feature_model(torch.FloatTensor(self.ref_list).to('cuda'))
        # mel_out = mel_classifier_model(mel_x)
        # mel_prob = F.softmax(mel_out, dim=1)
        # mel_prob = mel_prob.detach().cpu().numpy()
        #
        # thickness_x = thickness_feature_model(torch.FloatTensor(self.ref_list).to('cuda'))
        # thickenss_out = thickness_classifier_model(thickness_x)
        # thickness_prob = F.softmax(thickenss_out, dim=1)
        # thickness_prob = thickness_prob.detach().cpu().numpy()

        # self.ref_list = np.concatenate((self.ref_list, mel_prob, thickness_prob), axis=1)
        self.ref_list = np.concatenate((self.ref_list, m_label3, th_label3), axis=1)

        # [0902] classification class thb + sto combination
        self.total_label = (self.tb_label * 7) + self.st_label

        self.positive_list = []
        self.negative_list = []

        for i in range(len(m_label)):
            totall = self.total_label[i]

            negative_idx = np.where(self.total_label != totall)
            positive_idx = np.where(self.total_label == totall)

            self.positive_list.append(positive_idx[0])
            self.negative_list.append(negative_idx[0])

        self.random_idx = 0

        self.ref_list = torch.FloatTensor(self.ref_list).to('cuda')
        self.m_label = torch.LongTensor(m_label).to('cuda')
        self.tb_label = torch.LongTensor(tb_label).to('cuda')
        self.st_label = torch.LongTensor(st_label).to('cuda')
        self.th_label = torch.LongTensor(th_label).to('cuda')
        self.total_label = torch.LongTensor(self.total_label).to('cuda')


    def __getitem__(self, index):
        anchor = self.ref_list[index]
        ml = self.m_label[index]
        tbl = self.tb_label[index]
        stl = self.st_label[index]
        thl = self.th_label[index]
        totall = self.total_label[index]

        p_idx = self.positive_list[index][torch.randint(len(self.positive_list[index]), (1,))]
        positive = self.ref_list[p_idx]

        n_idx = self.negative_list[index][torch.randint(len(self.negative_list[index]), (1,))]
        negative = self.ref_list[n_idx]

        # Combination으로 세밀하게 나누는 경우
        # positive_idx = np.where(self.total_label == totall)
        # negative_idx = np.where(self.total_label != totall)

        # mel_positive_idx = np.where(self.m_label == ml)
        # mel_negative_idx = np.where(self.m_label != ml)
        #
        # thb_positive_idx = np.where(self.tb_label == tbl)
        # thb_negative_idx = np.where(self.tb_label != tbl)
        #
        # sto_positive_idx = np.where(self.st_label == stl)
        # sto_negative_idx = np.where(self.st_label != stl)
        #
        # thickness_positive_idx = np.where(self.th_label == thl)
        # thickness_negative_idx = np.where(self.th_label != thl)

        # positive = self.ref_list[random.choice(positive_idx[0])]
        # negative = self.ref_list[random.choice(negative_idx[0])]

        # mel_positive = self.ref_list[random.choice(mel_positive_idx[0])]
        # mel_negative = self.ref_list[random.choice(mel_negative_idx[0])]
        #
        # thb_positive = self.ref_list[random.choice(thb_positive_idx[0])]
        # thb_negative = self.ref_list[random.choice(thb_negative_idx[0])]
        #
        # sto_positive = self.ref_list[random.choice(sto_positive_idx[0])]
        # sto_negative = self.ref_list[random.choice(sto_negative_idx[0])]
        #
        # thickness_positive = self.ref_list[random.choice(thickness_positive_idx[0])]
        # thickness_negative = self.ref_list[random.choice(thickness_negative_idx[0])]

        # ml = torch.LongTensor([ml])
        # tbl = torch.LongTensor([tbl])
        # stl = torch.LongTensor([stl])
        # thl = torch.LongTensor([thl])
        # totall = torch.LongTensor([totall])

        #return anchor, positive, negative, mel_positive, mel_negative, thb_positive, thb_negative, sto_positive, sto_negative, thickness_positive, thickness_negative, ml, tbl, stl, thl, totall
        return anchor, positive, negative, ml, tbl, stl, thl, totall


    def __len__(self):
        return len(self.ref_list)

    def read_vitalsign_dataset(self, name, model_mel, model_thick):
        '''
        Read Data
        '''

        path = os.path.dirname(__file__)

        # fileNameList = ['input_data9.npy', 'input_data10.npy', 'input_data11.npy', 'input_data13.npy',\
        #                 'input_data_random9.npy', 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        #                 'input_data_random21.npy', 'input_data_random22.npy','input_data_random23.npy', 'input_data_random24.npy']
        #
        # fileNameList = ['input_data9.npy', 'input_data10.npy', \
        #                 'input_data11.npy', 'input_data13.npy', 'input_data14.npy', 'input_data15.npy',\
        #                 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy', \
        #                 'input_data_random18.npy', 'input_data_random19.npy', 'input_data_random20.npy', 'input_data_random21.npy',\
        #                 'input_data_random22.npy', 'input_data_random23.npy', 'input_data_random24.npy']

        fileNameList = ['input_data9.npy', 'input_data14.npy', 'input_data15.npy',\
                        'input_data_random9.npy', 'input_data_random23.npy', 'input_data_random21.npy','input_data_random22.npy', 'input_data_random24.npy',\
                        'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy','input_data_random20_3.npy'] #\
                        # 'input_data_random16.npy', 'input_data_random17.npy', 'input_data_random18.npy', 'input_data_random19.npy',]


        test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']

        # fileNameList = ['input_data6.npy', 'input_data7.npy', 'input_data8.npy', 'input_data9.npy', 'input_data10.npy', \
        #                 'input_data11.npy', 'input_data12.npy', 'input_data13.npy', 'input_data14.npy', 'input_data15.npy',\
        #                 'input_data_random6.npy', 'input_data_random7.npy', 'input_data_random8.npy', 'input_data_random9.npy', \
        #                 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        #                 'input_data_random14.npy', 'input_data_random15.npy', 'input_data_random16.npy','input_data_random17.npy', \
        #                 'input_data_random18.npy', 'input_data_random19.npy', 'input_data_random20.npy', 'input_data_random21.npy', \
        #                 'input_data_random22.npy', 'input_data_random23.npy', 'input_data_random24.npy']
        #
        # test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']
        #
        # fileNameList = ['input_data_4000_1.npy', 'input_data_4000_2.npy','input_data_4000_3.npy','input_data_4000_4.npy',\
        #                 'random_data_4000_1.npy', 'random_data_4000_2.npy', 'random_data_4000_3.npy',\
        #                 'random_data_4000_4.npy', 'random_data_4000_5.npy', 'random_data_4000_6.npy']
        #
        # test_fileNameList = ['input_data_4000_1.npy', 'input_data_4000_2.npy','input_data_4000_3.npy','input_data_4000_4.npy',\
        #                 'random_data_4000_1.npy', 'random_data_4000_2.npy', 'random_data_4000_3.npy',\
        #                 'random_data_4000_4.npy', 'random_data_4000_5.npy', 'random_data_4000_6.npy']

        if name == 'train':
            fileNameList = fileNameList
        else:
            fileNameList = test_fileNameList

        train_data = []

        for fn in fileNameList:
            temp_data = np.load(path + '/input_data/{}'.format(fn), allow_pickle=True)

            if len(train_data) == 0 :
                train_data = temp_data
            else:
                train_data = np.concatenate([train_data, temp_data], axis=1)

        reflect_list = np.transpose(train_data[:, :, 5])

        out2 = train_data[0, :, 0:4]

        m_label = []
        tb_label = []
        st_label = []
        th_label = []

        m_label2 = []
        tb_label2 = []
        st_label2 = []
        th_label2 = []

        m_label3 = []
        tb_label3 = []
        st_label3 = []
        th_label3 = []

        for d_idx in range(len(out2)):
            if out2[d_idx][0] > 0 and out2[d_idx][0] <= 0.02:
                m_i2 = [1, 0, 0, 0, 0, 0, 0, 0]
                m_i = 0
            elif out2[d_idx][0] > 0.02 and out2[d_idx][0] <= 0.04:
                m_i2 = [0, 1, 0, 0, 0, 0, 0, 0]
                m_i = 1
            elif out2[d_idx][0] > 0.04 and out2[d_idx][0] <= 0.06:
                m_i2 = [0, 0, 1, 0, 0, 0, 0, 0]
                m_i = 2
            elif out2[d_idx][0] > 0.06 and out2[d_idx][0] <= 0.08:
                m_i2 = [0, 0, 0, 1, 0, 0, 0, 0]
                m_i = 3
            elif out2[d_idx][0] > 0.08 and out2[d_idx][0] <= 0.1:
                m_i2 = [0, 0, 0, 0, 1, 0, 0, 0]
                m_i = 4
            elif out2[d_idx][0] > 0.1 and out2[d_idx][0] <= 0.12:
                m_i2 = [0, 0, 0, 0, 0, 1, 0, 0]
                m_i = 5
            elif out2[d_idx][0] > 0.12 and out2[d_idx][0] <= 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 1, 0]
                m_i = 6
            elif out2[d_idx][0] > 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 0, 1]
                m_i = 7

            g_p = []
            g_sig = 0.005

            g_p.append(np.exp(-(out2[d_idx][0] - 0.01) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.03) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.05) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.07) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.09) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.11) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.13) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.15) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))

            m_i3 = []
            for gp_i in range(len(g_p)):
                m_i3.append(g_p[gp_i]/np.sum(g_p))

            m_label.append(m_i)
            m_label2.append(m_i2)
            m_label3.append(m_i3)


            if out2[d_idx][1] > 0 and out2[d_idx][1] <= 0.01:
                tb_i2 = [1, 0, 0, 0, 0, 0, 0]
                tb_i = 0
            elif out2[d_idx][1] > 0.01 and out2[d_idx][1] <= 0.02:
                tb_i2 = [0, 1, 0, 0, 0, 0, 0]
                tb_i = 1
            elif out2[d_idx][1] > 0.02 and out2[d_idx][1] <= 0.03:
                tb_i2 = [0, 0, 1, 0, 0, 0, 0]
                tb_i = 2
            elif out2[d_idx][1] > 0.03 and out2[d_idx][1] <= 0.04:
                tb_i2 = [0, 0, 0, 1, 0, 0, 0]
                tb_i = 3
            elif out2[d_idx][1] > 0.04 and out2[d_idx][1] <= 0.05:
                tb_i2 = [0, 0, 0, 0, 1, 0, 0]
                tb_i = 4
            elif out2[d_idx][1] > 0.05 and out2[d_idx][1] <= 0.06:
                tb_i2 = [0, 0, 0, 0, 0, 1, 0]
                tb_i = 5
            elif out2[d_idx][1] > 0.06 and out2[d_idx][1] <= 0.07:
                tb_i2 = [0, 0, 0, 0, 0, 0, 1]
                tb_i = 6

            tb_label.append(tb_i)
            tb_label2.append(tb_i2)

            if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
                st_i2 = [1, 0, 0, 0, 0, 0, 0]
                st_i = 0
            elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.75:
                st_i2 = [0, 1, 0, 0, 0, 0, 0]
                st_i = 1
            elif out2[d_idx][2] > 0.75 and out2[d_idx][2] <= 0.8:
                st_i2 = [0, 0, 1, 0, 0, 0, 0]
                st_i = 2
            elif out2[d_idx][2] > 0.8 and out2[d_idx][2] <= 0.85:
                st_i2 = [0, 0, 0, 1, 0, 0, 0]
                st_i = 3
            elif out2[d_idx][2] > 0.85 and out2[d_idx][2] <= 0.9:
                st_i2 = [0, 0, 0, 0, 1, 0, 0]
                st_i = 4
            elif out2[d_idx][2] > 0.9 and out2[d_idx][2] <= 0.95:
                st_i2 = [0, 0, 0, 0, 0, 1, 0]
                st_i = 5
            elif out2[d_idx][2] > 0.95 and out2[d_idx][2] <= 1.0:
                st_i2 = [0, 0, 0, 0, 0, 0, 1]
                st_i = 6

            # if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
            #     st_i2 = [1, 0, 0, 0]
            #     st_i = 0
            # elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.8:
            #     st_i2 = [0, 1, 0, 0]
            #     st_i = 1
            # elif out2[d_idx][2] > 0.8 and out2[d_idx][2] <= 0.9:
            #     st_i2 = [0, 0, 1, 0]
            #     st_i = 2
            # elif out2[d_idx][2] > 0.9 and out2[d_idx][2] <= 1.0:
            #     st_i2 = [0, 0, 0, 1]
            #     st_i = 3

            # if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
            #     st_i2 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 0
            # elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.73:
            #     st_i2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 1
            # elif out2[d_idx][2] > 0.73 and out2[d_idx][2] <= 0.76:
            #     st_i2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 2
            # elif out2[d_idx][2] > 0.76 and out2[d_idx][2] <= 0.79:
            #     st_i2 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 3
            # elif out2[d_idx][2] > 0.79 and out2[d_idx][2] <= 0.82:
            #     st_i2 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            #     st_i = 4
            # elif out2[d_idx][2] > 0.82 and out2[d_idx][2] <= 0.85:
            #     st_i2 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            #     st_i = 5
            # elif out2[d_idx][2] > 0.85 and out2[d_idx][2] <= 0.88:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            #     st_i = 6
            # elif out2[d_idx][2] > 0.88 and out2[d_idx][2] <= 0.91:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            #     st_i = 7
            # elif out2[d_idx][2] > 0.91 and out2[d_idx][2] <= 0.94:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
            #     st_i = 8
            # elif out2[d_idx][2] > 0.94 and out2[d_idx][2] <= 0.97:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            #     st_i = 9
            # elif out2[d_idx][2] > 0.97 and out2[d_idx][2] <= 1.0:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            #     st_i = 10

            st_label.append(st_i)
            st_label2.append(st_i2)

            if out2[d_idx][3] <= 0.035:
                th_i2 = [1, 0, 0]
                th_i = 0
            elif out2[d_idx][3] > 0.035 and out2[d_idx][3] <= 0.055:
                th_i2 = [0, 1, 0]
                th_i = 1
            elif out2[d_idx][3] > 0.055 and out2[d_idx][3] <= 0.075:
                th_i2 = [0, 0, 1]
                th_i = 2

            g_p = []
            g_sig = 0.005

            g_p.append(np.exp(-(out2[d_idx][3] - 0.025) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][3] - 0.045) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][3] - 0.065) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))

            th_i3 = []
            for gp_i in range(len(g_p)):
                th_i3.append(g_p[gp_i] / np.sum(g_p))

            th_label.append(th_i)
            th_label2.append(th_i2)
            th_label3.append(th_i3)

        reflect_list_ex = []
        m_label_ex = []
        tb_label_ex = []
        st_label_ex = []
        th_label_ex = []

        for i in range(len(reflect_list)):
            if model_mel == -1 and model_thick == -1:
                temp_ref = []
                # for ri in range(54):
                #     temp_ref.append(reflect_list[i][ri])

                # temp_ref.append(reflect_list[i][5])

                # temp_ref.append(reflect_list[i][8])
                temp_ref.append(reflect_list[i][9])

                # temp_ref.append(reflect_list[i][11])
                temp_ref.append(reflect_list[i][12])

                # temp_ref.append(reflect_list[i][15])
                temp_ref.append(reflect_list[i][16])

                # temp_ref.append(reflect_list[i][18])
                temp_ref.append(reflect_list[i][19])

                # temp_ref.append(reflect_list[i][21])
                # temp_ref.append(reflect_list[i][22])  # temp_ref.append(reflect_list[i][25])

                # temp_ref.append(reflect_list[i][24])
                temp_ref.append(reflect_list[i][25])  # temp_ref.append(reflect_list[i][25])

                # temp_ref.append(reflect_list[i][26])
                temp_ref.append(reflect_list[i][27])

                # temp_ref.append(reflect_list[i][28])
                temp_ref.append(reflect_list[i][29])

                # temp_ref.append(reflect_list[i][30])
                temp_ref.append(reflect_list[i][31])

                # temp_ref.append(reflect_list[i][33])
                temp_ref.append(reflect_list[i][34])
                temp_ref.append(reflect_list[i][35])
                # temp_ref.append(reflect_list[i][37])
                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.6)
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.83)

                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][39])

                # temp_ref.append(reflect_list[i][41])
                # temp_ref.append(reflect_list[i][42])  # temp_ref.append(reflect_list[i][43])

                # temp_ref.append(reflect_list[i][43])
                temp_ref.append(reflect_list[i][44])
                temp_ref.append((reflect_list[i][46] + reflect_list[i][47]) / 2)

                # temp_ref.append(reflect_list[i][45])
                # temp_ref.append(reflect_list[i][46])

                # temp_ref.append(reflect_list[i][47])
                temp_ref.append(reflect_list[i][48])
                temp_ref.append(reflect_list[i][49])

                # temp_ref.append(reflect_list[i][50])
                # temp_ref.append(reflect_list[i][51])
                # temp_ref.append((reflect_list[i][52] + reflect_list[i][53])/2)
                # temp_ref.append(reflect_list[i][54])
                # temp_ref.append((reflect_list[i][55] + reflect_list[i][56]) / 2)
                # temp_ref.append((reflect_list[i][56] + reflect_list[i][57]) / 2)
                # temp_ref.append(reflect_list[i][57])
                # temp_ref.append(reflect_list[i][59])
                # temp_ref.append(reflect_list[i][60])
                # temp_ref.append(reflect_list[i][61])
                # temp_ref.append(reflect_list[i][62])
                # temp_ref.append(reflect_list[i][63])

                reflect_list_ex.append(temp_ref)
                #reflect_list_ex.append(reflect_list_concat)
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])
            elif m_label[i] == model_mel and th_label[i] == model_thick:
                reflect_list_ex.append(reflect_list[i])
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])

        # m_label = np.array(m_label, dtype=np.float32)
        # tb_label = np.array(tb_label, dtype=np.float32)
        # st_label = np.array(st_label, dtype=np.float32)
        # th_label = np.array(th_label, dtype=np.float32)

        m_label = np.array(m_label_ex, dtype=np.float32)
        tb_label = np.array(tb_label_ex, dtype=np.float32)
        st_label = np.array(st_label_ex, dtype=np.float32)
        th_label = np.array(th_label_ex, dtype=np.float32)

        m_label3 = np.array(m_label3, dtype=np.float32)
        th_label3 = np.array(th_label3, dtype=np.float32)

        #reflect_list = np.concatenate([reflect_list, m_label2, th_label2], axis=1)
        #reflect_list = np.concatenate((reflect_list, m_label[:, np.newaxis], th_label[:, np.newaxis]), axis=1)
        #reflect_list = np.array(reflect_list, dtype=np.float32)
        reflect_list = np.array(reflect_list_ex, dtype=np.float32)

        return reflect_list, m_label, tb_label, st_label, th_label, m_label3, th_label3


class ViatalSignDataset_triplet_mel_thickness(data.Dataset):
    def __init__(self, mode='train', cl='', model_mel= 0, model_thick=0):
        self.mode = mode
        self.cl = cl

        if mode == "train":
            reflect_list, m_label, tb_label, st_label, th_label = self.read_vitalsign_dataset(name='train', model_mel=model_mel, model_thick=model_thick)
        else:
            reflect_list, m_label, tb_label, st_label, th_label = self.read_vitalsign_dataset(name='test', model_mel=model_mel, model_thick=model_thick)

        self.ref_list, self.m_label, self.tb_label, self.st_label, self.th_label = reflect_list, m_label, tb_label, st_label, th_label

        if cl == 'mel':
            self.total_label = self.m_label
        elif cl == 'thickness':
            self.total_label = self.th_label

    def __getitem__(self, index):
        anchor = self.ref_list[index]
        ml = self.m_label[index]
        tbl = self.tb_label[index]
        stl = self.st_label[index]
        thl = self.th_label[index]
        totall = self.total_label[index]

        # Combination으로 세밀하게 나누는 경우
        positive_idx = np.where(self.total_label == totall)
        negative_idx = np.where(self.total_label != totall)

        positive = self.ref_list[random.choice(positive_idx[0])]
        negative = self.ref_list[random.choice(negative_idx[0])]

        ml = torch.LongTensor([ml])
        tbl = torch.LongTensor([tbl])
        stl = torch.LongTensor([stl])
        thl = torch.LongTensor([thl])
        totall = torch.LongTensor([totall])

        #return anchor, positive, negative, mel_positive, mel_negative, thb_positive, thb_negative, sto_positive, sto_negative, thickness_positive, thickness_negative, ml, tbl, stl, thl, totall
        return anchor, positive, negative, ml, tbl, stl, thl, totall


    def __len__(self):
        return len(self.ref_list)

    def read_vitalsign_dataset(self, name, model_mel, model_thick):
        '''
        Read Data
        '''

        path = os.path.dirname(__file__)

        # fileNameList = ['input_data9.npy', 'input_data10.npy', 'input_data11.npy', 'input_data13.npy',\
        #                 'input_data_random9.npy', 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        #                 'input_data_random21.npy', 'input_data_random22.npy','input_data_random23.npy', 'input_data_random24.npy']
        #
        # test_fileNameList = ['input_data9.npy', 'input_data10.npy', 'input_data11.npy', 'input_data13.npy',\
        #                 'input_data_random9.npy', 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        #                 'input_data_random21.npy', 'input_data_random22.npy','input_data_random23.npy', 'input_data_random24.npy']

        #
        # test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']

        # fileNameList = ['input_data6.npy', 'input_data7.npy', 'input_data8.npy', 'input_data9.npy', 'input_data10.npy', \
        #                 'input_data11.npy', 'input_data12.npy', 'input_data13.npy', 'input_data14.npy', 'input_data15.npy',\
        #                 'input_data_random6.npy', 'input_data_random7.npy', 'input_data_random8.npy', 'input_data_random9.npy', \
        #                 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        #                 'input_data_random14.npy', 'input_data_random15.npy', 'input_data_random16.npy','input_data_random17.npy', \
        #                 'input_data_random18.npy', 'input_data_random19.npy', 'input_data_random20.npy', 'input_data_random21.npy',\
        #                 'input_data_random22.npy', 'input_data_random23.npy', 'input_data_random24.npy']
        #
        # test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']

        fileNameList = ['input_data9.npy', 'input_data14.npy', 'input_data15.npy',\
                        'input_data_random9.npy', 'input_data_random23.npy', 'input_data_random21.npy',\
                        'input_data_random22.npy', 'input_data_random24.npy']
                        # 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
                        # 'input_data_random14.npy', 'input_data_random15.npy']

        test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']

        # fileNameList = ['input_data9.npy', 'input_data14.npy', 'input_data_random24.npy']
        #
        # test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']

        if name == 'train':
            fileNameList = fileNameList
        else:
            fileNameList = test_fileNameList

        train_data = []

        for fn in fileNameList:
            temp_data = np.load(path + '/input_data/{}'.format(fn), allow_pickle=True)

            if len(train_data) == 0 :
                train_data = temp_data
            else:
                train_data = np.concatenate([train_data, temp_data], axis=1)

        reflect_list = np.transpose(train_data[:, :, 5])

        out2 = train_data[0, :, 0:4]

        m_label = []
        tb_label = []
        st_label = []
        th_label = []

        m_label2 = []
        tb_label2 = []
        st_label2 = []
        th_label2 = []

        for d_idx in range(len(out2)):
            if out2[d_idx][0] > 0 and out2[d_idx][0] <= 0.02:
                m_i2 = [1, 0, 0, 0, 0, 0, 0, 0]
                m_i = 0
            elif out2[d_idx][0] > 0.02 and out2[d_idx][0] <= 0.04:
                m_i2 = [0, 1, 0, 0, 0, 0, 0, 0]
                m_i = 1
            elif out2[d_idx][0] > 0.04 and out2[d_idx][0] <= 0.06:
                m_i2 = [0, 0, 1, 0, 0, 0, 0, 0]
                m_i = 2
            elif out2[d_idx][0] > 0.06 and out2[d_idx][0] <= 0.08:
                m_i2 = [0, 0, 0, 1, 0, 0, 0, 0]
                m_i = 3
            elif out2[d_idx][0] > 0.08 and out2[d_idx][0] <= 0.1:
                m_i2 = [0, 0, 0, 0, 1, 0, 0, 0]
                m_i = 4
            elif out2[d_idx][0] > 0.1 and out2[d_idx][0] <= 0.12:
                m_i2 = [0, 0, 0, 0, 0, 1, 0, 0]
                m_i = 5
            elif out2[d_idx][0] > 0.12 and out2[d_idx][0] <= 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 1, 0]
                m_i = 6
            elif out2[d_idx][0] > 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 0, 1]
                m_i = 7

            m_label.append(m_i)
            m_label2.append(m_i2)

            if out2[d_idx][1] > 0 and out2[d_idx][1] <= 0.01:
                tb_i2 = [1, 0, 0, 0, 0, 0, 0]
                tb_i = 0
            elif out2[d_idx][1] > 0.01 and out2[d_idx][1] <= 0.02:
                tb_i2 = [0, 1, 0, 0, 0, 0, 0]
                tb_i = 1
            elif out2[d_idx][1] > 0.02 and out2[d_idx][1] <= 0.03:
                tb_i2 = [0, 0, 1, 0, 0, 0, 0]
                tb_i = 2
            elif out2[d_idx][1] > 0.03 and out2[d_idx][1] <= 0.04:
                tb_i2 = [0, 0, 0, 1, 0, 0, 0]
                tb_i = 3
            elif out2[d_idx][1] > 0.04 and out2[d_idx][1] <= 0.05:
                tb_i2 = [0, 0, 0, 0, 1, 0, 0]
                tb_i = 4
            elif out2[d_idx][1] > 0.05 and out2[d_idx][1] <= 0.06:
                tb_i2 = [0, 0, 0, 0, 0, 1, 0]
                tb_i = 5
            elif out2[d_idx][1] > 0.06 and out2[d_idx][1] <= 0.07:
                tb_i2 = [0, 0, 0, 0, 0, 0, 1]
                tb_i = 6

            tb_label.append(tb_i)
            tb_label2.append(tb_i2)

            if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
                st_i2 = [1, 0, 0, 0, 0, 0, 0]
                st_i = 0
            elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.75:
                st_i2 = [0, 1, 0, 0, 0, 0, 0]
                st_i = 1
            elif out2[d_idx][2] > 0.75 and out2[d_idx][2] <= 0.8:
                st_i2 = [0, 0, 1, 0, 0, 0, 0]
                st_i = 2
            elif out2[d_idx][2] > 0.8 and out2[d_idx][2] <= 0.85:
                st_i2 = [0, 0, 0, 1, 0, 0, 0]
                st_i = 3
            elif out2[d_idx][2] > 0.85 and out2[d_idx][2] <= 0.9:
                st_i2 = [0, 0, 0, 0, 1, 0, 0]
                st_i = 4
            elif out2[d_idx][2] > 0.9 and out2[d_idx][2] <= 0.95:
                st_i2 = [0, 0, 0, 0, 0, 1, 0]
                st_i = 5
            elif out2[d_idx][2] > 0.95 and out2[d_idx][2] <= 1.0:
                st_i2 = [0, 0, 0, 0, 0, 0, 1]
                st_i = 6

            st_label.append(st_i)
            st_label2.append(st_i2)

            if out2[d_idx][3] <= 0.035:
                th_i2 = [1, 0, 0]
                th_i = 0
            elif out2[d_idx][3] > 0.035 and out2[d_idx][3] <= 0.055:
                th_i2 = [0, 1, 0]
                th_i = 1
            elif out2[d_idx][3] > 0.055 and out2[d_idx][3] <= 0.075:
                th_i2 = [0, 0, 1]
                th_i = 2

            th_label.append(th_i)
            th_label2.append(th_i2)

        reflect_list_ex = []
        m_label_ex = []
        tb_label_ex = []
        st_label_ex = []
        th_label_ex = []

        for i in range(len(reflect_list)):
            if model_mel == -1 and model_thick == -1:
                temp_ref = []
                # for ri in range(64):
                #     temp_ref.append(reflect_list[i][ri])

                # temp_ref.append(reflect_list[i][5])

                # temp_ref.append(reflect_list[i][8])
                temp_ref.append(reflect_list[i][9])

                # temp_ref.append(reflect_list[i][11])
                temp_ref.append(reflect_list[i][12])

                # temp_ref.append(reflect_list[i][15])
                temp_ref.append(reflect_list[i][16])

                # temp_ref.append(reflect_list[i][18])
                temp_ref.append(reflect_list[i][19])

                # temp_ref.append(reflect_list[i][21])
                # temp_ref.append(reflect_list[i][22])  #temp_ref.append(reflect_list[i][25])
                #
                # temp_ref.append(reflect_list[i][24])
                temp_ref.append(reflect_list[i][25])  # temp_ref.append(reflect_list[i][25])

                # temp_ref.append(reflect_list[i][26])
                temp_ref.append(reflect_list[i][27])

                # temp_ref.append(reflect_list[i][28])
                temp_ref.append(reflect_list[i][29])

                # temp_ref.append(reflect_list[i][30])
                temp_ref.append(reflect_list[i][31])

                # temp_ref.append(reflect_list[i][33])
                temp_ref.append(reflect_list[i][34])
                temp_ref.append(reflect_list[i][35])
                # temp_ref.append(reflect_list[i][37])
                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.6)
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.83)

                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][39])

                # temp_ref.append(reflect_list[i][41])
                # temp_ref.append(reflect_list[i][42])  # temp_ref.append(reflect_list[i][43])

                # temp_ref.append(reflect_list[i][43])
                temp_ref.append(reflect_list[i][44])
                # temp_ref.append((reflect_list[i][46] + reflect_list[i][47]) / 2)

                # temp_ref.append(reflect_list[i][45])
                # temp_ref.append(reflect_list[i][46])

                temp_ref.append(reflect_list[i][47])
                temp_ref.append(reflect_list[i][48])
                temp_ref.append(reflect_list[i][49])

                # temp_ref.append(reflect_list[i][50])
                # temp_ref.append(reflect_list[i][51])
                # temp_ref.append((reflect_list[i][52] + reflect_list[i][53])/2)
                # temp_ref.append(reflect_list[i][54])
                # temp_ref.append((reflect_list[i][55] + reflect_list[i][56]) / 2)
                # temp_ref.append((reflect_list[i][56] + reflect_list[i][57]) / 2)
                # temp_ref.append(reflect_list[i][57])
                # temp_ref.append(reflect_list[i][59])
                # temp_ref.append(reflect_list[i][60])
                # temp_ref.append(reflect_list[i][61])
                # temp_ref.append(reflect_list[i][62])
                # temp_ref.append(reflect_list[i][63])

                reflect_list_ex.append(temp_ref)

                #reflect_list_ex.append(reflect_list_concat)
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])
            elif m_label[i] == model_mel and th_label[i] == model_thick:
                reflect_list_ex.append(reflect_list[i])
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])

        # m_label = np.array(m_label, dtype=np.float32)
        # tb_label = np.array(tb_label, dtype=np.float32)
        # st_label = np.array(st_label, dtype=np.float32)
        # th_label = np.array(th_label, dtype=np.float32)

        m_label = np.array(m_label_ex, dtype=np.float32)
        tb_label = np.array(tb_label_ex, dtype=np.float32)
        st_label = np.array(st_label_ex, dtype=np.float32)
        th_label = np.array(th_label_ex, dtype=np.float32)

        #reflect_list = np.concatenate([reflect_list, m_label2, th_label2], axis=1)
        #reflect_list = np.concatenate((reflect_list, m_label[:, np.newaxis], th_label[:, np.newaxis]), axis=1)
        #reflect_list = np.array(reflect_list, dtype=np.float32)
        reflect_list = np.array(reflect_list_ex, dtype=np.float32)

        return reflect_list, m_label, tb_label, st_label, th_label

class ViatalSignDataset_triplet_mel_thickness_v2(data.Dataset):
    def __init__(self, mode='train', cl='', model_mel= 0, model_thick=0):
        self.mode = mode
        self.cl = cl

        if mode == "train":
            reflect_list, m_label, tb_label, st_label, th_label = self.read_vitalsign_dataset(name='train', model_mel=model_mel, model_thick=model_thick)
        else:
            reflect_list, m_label, tb_label, st_label, th_label = self.read_vitalsign_dataset(name='test', model_mel=model_mel, model_thick=model_thick)

        self.ref_list, self.m_label, self.tb_label, self.st_label, self.th_label = reflect_list, m_label, tb_label, st_label, th_label

        if cl == 'mel':
            self.total_label = self.m_label
        elif cl == 'thickness':
            self.total_label = self.th_label

        self.positive_list = []
        self.negative_list = []

        for i in range(len(m_label)):
            totall = self.total_label[i]

            negative_idx = np.where(self.total_label != totall)
            positive_idx = np.where(self.total_label == totall)

            self.positive_list.append(positive_idx[0])
            self.negative_list.append(negative_idx[0])

        self.random_idx = 0

        self.ref_list = torch.FloatTensor(reflect_list).to('cuda')
        self.m_label = torch.LongTensor(m_label).to('cuda')
        self.tb_label = torch.LongTensor(tb_label).to('cuda')
        self.st_label = torch.LongTensor(st_label).to('cuda')
        self.th_label = torch.LongTensor(th_label).to('cuda')

    def __getitem__(self, index):
        anchor = self.ref_list[index]
        ml = self.m_label[index]
        tbl = self.tb_label[index]
        stl = self.st_label[index]
        thl = self.th_label[index]
        totall = self.total_label[index]

        # print("index : ", index)

        p_idx = self.positive_list[index][torch.randint(len(self.positive_list[index]), (1,))]

        # positive_list = torch.Tensor(self.positive_list[index])
        # p_idx = positive_list.multinomial(num_samples=1)
        # positive = self.ref_list[self.positive_list[index][p_idx]]
        positive = self.ref_list[p_idx]

        # print("CHECK p idx L ", p_idx )

        n_idx = self.negative_list[index][torch.randint(len(self.negative_list[index]), (1,))]
        # negative_list = torch.Tensor(self.negative_list[index])
        # n_idx = negative_list.multinomial(num_samples=1)
        # negative = self.ref_list[self.negative_list[index][n_idx]]
        negative = self.ref_list[n_idx]

        # if index == 0:
        #     print("n_iex : ", n_idx)
        #     print("mel : ", ml)
        #     print("neg mel : ", self.m_label[n_idx])

        # if self.random_idx >= len(self.positive_list[index]) or self.random_idx >= len(self.negative_list[index]):
        #     self.random_idx = 0
        #
        # positive = self.ref_list[self.positive_list[index][self.random_idx]]
        # negative = self.ref_list[self.negative_list[index][self.random_idx]]
        #
        # self.random_idx = self.random_idx +1

        # if self.random_idx >= self.min_value:
        #     self.random_idx = 0

        # # Combination으로 세밀하게 나누는 경우
        # positive_idx = np.where(self.total_label == totall)
        # negative_idx = np.where(self.total_label != totall)
        #
        # positive = self.ref_list[random.choice(positive_idx[0])]
        # negative = self.ref_list[random.choice(negative_idx[0])]

        # ml = torch.LongTensor([ml])
        # tbl = torch.LongTensor([tbl])
        # stl = torch.LongTensor([stl])
        # thl = torch.LongTensor([thl])
        # totall = torch.LongTensor([totall])

        #return anchor, positive, negative, mel_positive, mel_negative, thb_positive, thb_negative, sto_positive, sto_negative, thickness_positive, thickness_negative, ml, tbl, stl, thl, totall
        return anchor, positive, negative, ml, tbl, stl, thl, totall


    def __len__(self):
        return len(self.ref_list)

    def read_vitalsign_dataset(self, name, model_mel, model_thick):
        '''
        Read Data
        '''

        path = os.path.dirname(__file__)

        fileNameList = ['input_data9.npy', 'input_data14.npy', 'input_data15.npy',\
                        'input_data_random9.npy', 'input_data_random23.npy', 'input_data_random21.npy','input_data_random22.npy', 'input_data_random24.npy',\
                        'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy','input_data_random20_3.npy']

        test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']

        if name == 'train':
            fileNameList = fileNameList
        else:
            fileNameList = test_fileNameList

        train_data = []

        for fn in fileNameList:
            temp_data = np.load(path + '/input_data/{}'.format(fn), allow_pickle=True)

            if len(train_data) == 0 :
                train_data = temp_data
            else:
                train_data = np.concatenate([train_data, temp_data], axis=1)

        # Read Absorbance, data index 5는 Absorbance data임.
        reflect_list = np.transpose(train_data[:, :, 5])

        out2 = train_data[0, :, 0:4]

        m_label = []
        tb_label = []
        st_label = []
        th_label = []

        m_label2 = []
        tb_label2 = []
        st_label2 = []
        th_label2 = []

        for d_idx in range(len(out2)):
            if out2[d_idx][0] > 0 and out2[d_idx][0] <= 0.02:
                m_i2 = [1, 0, 0, 0, 0, 0, 0, 0]
                m_i = 0
            elif out2[d_idx][0] > 0.02 and out2[d_idx][0] <= 0.04:
                m_i2 = [0, 1, 0, 0, 0, 0, 0, 0]
                m_i = 1
            elif out2[d_idx][0] > 0.04 and out2[d_idx][0] <= 0.06:
                m_i2 = [0, 0, 1, 0, 0, 0, 0, 0]
                m_i = 2
            elif out2[d_idx][0] > 0.06 and out2[d_idx][0] <= 0.08:
                m_i2 = [0, 0, 0, 1, 0, 0, 0, 0]
                m_i = 3
            elif out2[d_idx][0] > 0.08 and out2[d_idx][0] <= 0.1:
                m_i2 = [0, 0, 0, 0, 1, 0, 0, 0]
                m_i = 4
            elif out2[d_idx][0] > 0.1 and out2[d_idx][0] <= 0.12:
                m_i2 = [0, 0, 0, 0, 0, 1, 0, 0]
                m_i = 5
            elif out2[d_idx][0] > 0.12 and out2[d_idx][0] <= 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 1, 0]
                m_i = 6
            elif out2[d_idx][0] > 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 0, 1]
                m_i = 7

            m_label.append(m_i)
            m_label2.append(m_i2)

            if out2[d_idx][1] > 0 and out2[d_idx][1] <= 0.01:
                tb_i2 = [1, 0, 0, 0, 0, 0, 0]
                tb_i = 0
            elif out2[d_idx][1] > 0.01 and out2[d_idx][1] <= 0.02:
                tb_i2 = [0, 1, 0, 0, 0, 0, 0]
                tb_i = 1
            elif out2[d_idx][1] > 0.02 and out2[d_idx][1] <= 0.03:
                tb_i2 = [0, 0, 1, 0, 0, 0, 0]
                tb_i = 2
            elif out2[d_idx][1] > 0.03 and out2[d_idx][1] <= 0.04:
                tb_i2 = [0, 0, 0, 1, 0, 0, 0]
                tb_i = 3
            elif out2[d_idx][1] > 0.04 and out2[d_idx][1] <= 0.05:
                tb_i2 = [0, 0, 0, 0, 1, 0, 0]
                tb_i = 4
            elif out2[d_idx][1] > 0.05 and out2[d_idx][1] <= 0.06:
                tb_i2 = [0, 0, 0, 0, 0, 1, 0]
                tb_i = 5
            elif out2[d_idx][1] > 0.06 and out2[d_idx][1] <= 0.07:
                tb_i2 = [0, 0, 0, 0, 0, 0, 1]
                tb_i = 6

            tb_label.append(tb_i)
            tb_label2.append(tb_i2)

            if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
                st_i2 = [1, 0, 0, 0, 0, 0, 0]
                st_i = 0
            elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.75:
                st_i2 = [0, 1, 0, 0, 0, 0, 0]
                st_i = 1
            elif out2[d_idx][2] > 0.75 and out2[d_idx][2] <= 0.8:
                st_i2 = [0, 0, 1, 0, 0, 0, 0]
                st_i = 2
            elif out2[d_idx][2] > 0.8 and out2[d_idx][2] <= 0.85:
                st_i2 = [0, 0, 0, 1, 0, 0, 0]
                st_i = 3
            elif out2[d_idx][2] > 0.85 and out2[d_idx][2] <= 0.9:
                st_i2 = [0, 0, 0, 0, 1, 0, 0]
                st_i = 4
            elif out2[d_idx][2] > 0.9 and out2[d_idx][2] <= 0.95:
                st_i2 = [0, 0, 0, 0, 0, 1, 0]
                st_i = 5
            elif out2[d_idx][2] > 0.95 and out2[d_idx][2] <= 1.0:
                st_i2 = [0, 0, 0, 0, 0, 0, 1]
                st_i = 6

            st_label.append(st_i)
            st_label2.append(st_i2)

            if out2[d_idx][3] <= 0.035:
                th_i2 = [1, 0, 0]
                th_i = 0
            elif out2[d_idx][3] > 0.035 and out2[d_idx][3] <= 0.055:
                th_i2 = [0, 1, 0]
                th_i = 1
            elif out2[d_idx][3] > 0.055 and out2[d_idx][3] <= 0.075:
                th_i2 = [0, 0, 1]
                th_i = 2

            th_label.append(th_i)
            th_label2.append(th_i2)

        reflect_list_ex = []
        m_label_ex = []
        tb_label_ex = []
        st_label_ex = []
        th_label_ex = []

        for i in range(len(reflect_list)):
            if model_mel == -1 and model_thick == -1:
                temp_ref = []
                # for ri in range(64):
                #     temp_ref.append(reflect_list[i][ri])
                #
                # temp_ref.append(reflect_list[i][5])

                # temp_ref.append(reflect_list[i][8])
                temp_ref.append(reflect_list[i][9])

                # temp_ref.append(reflect_list[i][11])
                temp_ref.append(reflect_list[i][12])

                # temp_ref.append(reflect_list[i][15])
                temp_ref.append(reflect_list[i][16])

                # temp_ref.append(reflect_list[i][18])
                temp_ref.append(reflect_list[i][19])

                # temp_ref.append(reflect_list[i][21])
                # temp_ref.append(reflect_list[i][22])  #temp_ref.append(reflect_list[i][25])
                #
                # temp_ref.append(reflect_list[i][24])
                temp_ref.append(reflect_list[i][25])  # temp_ref.append(reflect_list[i][25])

                # temp_ref.append(reflect_list[i][26])
                temp_ref.append(reflect_list[i][27])

                # temp_ref.append(reflect_list[i][28])
                temp_ref.append(reflect_list[i][29])

                # temp_ref.append(reflect_list[i][30])
                temp_ref.append(reflect_list[i][31])

                # temp_ref.append(reflect_list[i][33])
                temp_ref.append(reflect_list[i][34])
                temp_ref.append(reflect_list[i][35])
                # temp_ref.append(reflect_list[i][37])
                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.6)
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.83)

                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][39])

                # temp_ref.append(reflect_list[i][41])
                # temp_ref.append(reflect_list[i][42])  # temp_ref.append(reflect_list[i][43])

                # temp_ref.append(reflect_list[i][43])
                temp_ref.append(reflect_list[i][44])
                temp_ref.append((reflect_list[i][46] + reflect_list[i][47]) / 2)

                # temp_ref.append(reflect_list[i][45])
                # temp_ref.append(reflect_list[i][46])

                # temp_ref.append(reflect_list[i][47])
                temp_ref.append(reflect_list[i][48])
                temp_ref.append(reflect_list[i][49])

                # temp_ref.append(reflect_list[i][50])
                # temp_ref.append(reflect_list[i][51])
                # temp_ref.append((reflect_list[i][52] + reflect_list[i][53])/2)
                # temp_ref.append(reflect_list[i][54])
                # temp_ref.append((reflect_list[i][55] + reflect_list[i][56]) / 2)
                # temp_ref.append((reflect_list[i][56] + reflect_list[i][57]) / 2)
                # temp_ref.append(reflect_list[i][57])
                # temp_ref.append(reflect_list[i][59])
                # temp_ref.append(reflect_list[i][60])
                # temp_ref.append(reflect_list[i][61])
                # temp_ref.append(reflect_list[i][62])
                # temp_ref.append(reflect_list[i][63])

                reflect_list_ex.append(temp_ref)

                #reflect_list_ex.append(reflect_list_concat)
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])
            elif m_label[i] == model_mel and th_label[i] == model_thick:
                reflect_list_ex.append(reflect_list[i])
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])

        # m_label = np.array(m_label, dtype=np.float32)
        # tb_label = np.array(tb_label, dtype=np.float32)
        # st_label = np.array(st_label, dtype=np.float32)
        # th_label = np.array(th_label, dtype=np.float32)

        m_label = np.array(m_label_ex, dtype=np.float32)
        tb_label = np.array(tb_label_ex, dtype=np.float32)
        st_label = np.array(st_label_ex, dtype=np.float32)
        th_label = np.array(th_label_ex, dtype=np.float32)

        #reflect_list = np.concatenate([reflect_list, m_label2, th_label2], axis=1)
        #reflect_list = np.concatenate((reflect_list, m_label[:, np.newaxis], th_label[:, np.newaxis]), axis=1)
        #reflect_list = np.array(reflect_list, dtype=np.float32)
        reflect_list = np.array(reflect_list_ex, dtype=np.float32)

        return reflect_list, m_label, tb_label, st_label, th_label


class ViatalSignDataset_class(data.Dataset):
    def __init__(self, mode='train', cl='', use_gpu = False, model_mel= -1, model_thick=-1):
        self.mode = mode
        self.cl = cl
        self.use_gpu = use_gpu

        if mode == "train":
            reflect_list, m_label, tb_label, tb_label3, st_label, st_label3, th_label, comb_label, m_label3, th_label3 = self.read_vitalsign_dataset(name='train', model_mel=model_mel, model_thick=model_thick)
        else:
            reflect_list, m_label, tb_label, tb_label3, st_label, st_label3, th_label, comb_label, m_label3, th_label3 = self.read_vitalsign_dataset(name='test', model_mel=model_mel, model_thick=model_thick)

        mel_feature_model = VitalSign_Feature_mel_thickness().to('cuda')
        thickness_feature_model = VitalSign_Feature_mel_thickness().to('cuda')

        mel_classifier_model = Classifier(cl_mode="mel").to('cuda')
        thickness_classifier_model = Classifier(cl_mode="thickness").to('cuda')

        path = os.path.dirname(__file__)

        mel_feature_path = os.path.join(path, './result/vitalsign_mel_0104_prob_02_input14_m1_epoch5000_addinput3/feature_weight_data')
        thickness_feature_path = os.path.join(path, './result/vitalsign_thickness_0104_prob_02_input14_m1_epoch5000_addinput3/feature_weight_data')

        mel_classify_path = os.path.join(path, './result/vitalsign_mel_0104_prob_02_input14_m1_epoch5000_addinput3/classification_weight_data2')
        thickness_classify_path = os.path.join(path, './result/vitalsign_thickness_0104_prob_02_input14_m1_epoch5000_addinput3/classification_weight_data2')

        mel_feature_model.load_state_dict(torch.load(mel_feature_path))
        mel_classifier_model.load_state_dict(torch.load(mel_classify_path))

        thickness_feature_model.load_state_dict(torch.load(thickness_feature_path))
        thickness_classifier_model.load_state_dict(torch.load(thickness_classify_path))

        mel_x = mel_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        mel_out = mel_classifier_model(mel_x)
        mel_prob = F.softmax(mel_out, dim=1)
        mel_prob = mel_prob.detach().cpu().numpy()

        thickness_x = thickness_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        thickenss_out = thickness_classifier_model(thickness_x)
        thickness_prob = F.softmax(thickenss_out, dim=1)
        thickness_prob = thickness_prob.detach().cpu().numpy()

        reflect_list = np.concatenate((reflect_list, mel_prob, thickness_prob), axis=1)
        # reflect_list = np.concatenate((reflect_list, m_label3, th_label3), axis=1)

        # [0902] classification class thb + sto combination
        total_label = (tb_label * 7) + st_label
        # total_label = (tb_label * 11) + st_label
        # [0908] classification class thb + sto combination
        #total_label = tb_label
        # [0908] classification class sto
        #total_label = st_label
        #total_label = m_label
        #total_label = th_label

        if self.use_gpu == True:
            self.ref_list = torch.FloatTensor(reflect_list).to('cuda')
            self.m_label = torch.LongTensor(m_label).to('cuda')
            self.tb_label = torch.LongTensor(tb_label).to('cuda')
            self.tb_label3 = torch.FloatTensor(tb_label3).to('cuda')
            self.st_label = torch.LongTensor(st_label).to('cuda')
            self.st_label3 = torch.FloatTensor(st_label3).to('cuda')
            self.th_label = torch.LongTensor(th_label).to('cuda')
            self.total_label = torch.LongTensor(total_label).to('cuda')
            self.comb_label = torch.FloatTensor(comb_label).to('cuda')
        else:
            self.ref_list = torch.FloatTensor(reflect_list)
            self.m_label = torch.LongTensor(m_label)
            self.tb_label = torch.LongTensor(tb_label)
            self.tb_label3 = torch.FloatTensor(tb_label3)
            self.st_label = torch.LongTensor(st_label)
            self.st_label3 = torch.FloatTensor(st_label3)
            self.th_label = torch.LongTensor(th_label)
            self.total_label = torch.LongTensor(total_label)
            self.comb_label = torch.FloatTensor(comb_label)

    def __getitem__(self, index):
        anchor = self.ref_list[index]
        ml = self.m_label[index]
        tbl = self.tb_label[index]
        tbl3 = self.tb_label3[index]
        stl = self.st_label[index]
        stl3 = self.st_label3[index]
        thl = self.th_label[index]
        totall = self.total_label[index]
        totall3 = self.comb_label[index]

        return anchor, ml, tbl, tbl3, stl, stl3, thl, totall, totall3

    def __len__(self):
        return len(self.ref_list)

    def read_vitalsign_dataset(self, name, model_mel, model_thick):
        '''
        Read Data
        '''
        path = os.path.dirname(__file__)

        # fileNameList = ['input_data9.npy', 'input_data10.npy', 'input_data11.npy', 'input_data13.npy',\
        #                 'input_data_random9.npy', 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        #                 'input_data_random21.npy', 'input_data_random22.npy','input_data_random23.npy', 'input_data_random24.npy']
        #
        # # test_fileNameList = ['input_data9.npy', 'input_data10.npy', 'input_data11.npy', 'input_data13.npy',\
        # #                 'input_data_random9.npy', 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        # #                 'input_data_random21.npy', 'input_data_random22.npy','input_data_random23.npy', 'input_data_random24.npy']
        #
        # test_fileNameList = ['input_data9.npy', 'input_data_random10.npy']

        # fileNameList = ['input_data9.npy', 'input_data10.npy', \
        #                 'input_data11.npy', 'input_data13.npy', 'input_data14.npy', 'input_data15.npy',\
        #                 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy', \
        #                 'input_data_random18.npy', 'input_data_random19.npy', 'input_data_random20.npy', 'input_data_random21.npy',\
        #                 'input_data_random22.npy', 'input_data_random23.npy', 'input_data_random24.npy']


        fileNameList = ['input_data9.npy', 'input_data14.npy', 'input_data15.npy',\
                        'input_data_random9.npy', 'input_data_random23.npy', 'input_data_random21.npy','input_data_random22.npy', 'input_data_random24.npy',\
                        'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy','input_data_random20_3.npy'] #\
                        # 'input_data_random16.npy', 'input_data_random17.npy', 'input_data_random18.npy', 'input_data_random19.npy',]


        test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']


        # fileNameList = ['input_data6.npy', 'input_data7.npy', 'input_data8.npy', 'input_data9.npy', 'input_data10.npy', \
        #                 'input_data11.npy', 'input_data12.npy', 'input_data13.npy', 'input_data14.npy', 'input_data15.npy',\
        #                 'input_data_random6.npy', 'input_data_random7.npy', 'input_data_random8.npy', 'input_data_random9.npy', \
        #                 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        #                 'input_data_random14.npy', 'input_data_random15.npy', 'input_data_random16.npy','input_data_random17.npy', \
        #                 'input_data_random18.npy', 'input_data_random19.npy', 'input_data_random20.npy', 'input_data_random21.npy', \
        #                 'input_data_random22.npy', 'input_data_random23.npy', 'input_data_random24.npy']
        #
        # test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']
        #
        # fileNameList = ['input_data_4000_1.npy', 'input_data_4000_2.npy','input_data_4000_3.npy','input_data_4000_4.npy',\
        #                 'random_data_4000_1.npy', 'random_data_4000_2.npy', 'random_data_4000_3.npy',\
        #                 'random_data_4000_4.npy', 'random_data_4000_5.npy', 'random_data_4000_6.npy']
        #
        # test_fileNameList = ['input_data_4000_1.npy', 'input_data_4000_2.npy','input_data_4000_3.npy','input_data_4000_4.npy',\
        #                 'random_data_4000_1.npy', 'random_data_4000_2.npy', 'random_data_4000_3.npy',\
        #                 'random_data_4000_4.npy', 'random_data_4000_5.npy', 'random_data_4000_6.npy']

        if name == 'train':
            fileNameList = fileNameList
        else:
            fileNameList = test_fileNameList

        train_data = []

        for fn in fileNameList:
            temp_data = np.load(path + '/input_data/{}'.format(fn), allow_pickle=True)

            if len(train_data) == 0 :
                train_data = temp_data
            else:
                train_data = np.concatenate([train_data, temp_data], axis=1)

        reflect_list = np.transpose(train_data[:, :, 5])

        out2 = train_data[0, :, 0:4]

        m_label = []
        tb_label = []
        st_label = []
        th_label = []

        m_label2 = []
        tb_label2 = []
        st_label2 = []
        th_label2 = []

        m_label3 = []
        tb_label3 = []
        st_label3 = []
        th_label3 = []
        combination_label3 = []

        for d_idx in range(len(out2)):
            if out2[d_idx][0] > 0 and out2[d_idx][0] <= 0.02:
                m_i2 = [1, 0, 0, 0, 0, 0, 0, 0]
                m_i = 0
            elif out2[d_idx][0] > 0.02 and out2[d_idx][0] <= 0.04:
                m_i2 = [0, 1, 0, 0, 0, 0, 0, 0]
                m_i = 1
            elif out2[d_idx][0] > 0.04 and out2[d_idx][0] <= 0.06:
                m_i2 = [0, 0, 1, 0, 0, 0, 0, 0]
                m_i = 2
            elif out2[d_idx][0] > 0.06 and out2[d_idx][0] <= 0.08:
                m_i2 = [0, 0, 0, 1, 0, 0, 0, 0]
                m_i = 3
            elif out2[d_idx][0] > 0.08 and out2[d_idx][0] <= 0.1:
                m_i2 = [0, 0, 0, 0, 1, 0, 0, 0]
                m_i = 4
            elif out2[d_idx][0] > 0.1 and out2[d_idx][0] <= 0.12:
                m_i2 = [0, 0, 0, 0, 0, 1, 0, 0]
                m_i = 5
            elif out2[d_idx][0] > 0.12 and out2[d_idx][0] <= 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 1, 0]
                m_i = 6
            elif out2[d_idx][0] > 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 0, 1]
                m_i = 7

            g_p = []
            g_sig = 0.02

            g_p.append(np.exp(-(out2[d_idx][0] - 0.01) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.03) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.05) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.07) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.09) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.11) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.13) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.15) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))

            m_i3 = []
            for gp_i in range(len(g_p)):
                m_i3.append(g_p[gp_i] / np.sum(g_p))

            m_label.append(m_i)
            m_label2.append(m_i2)
            m_label3.append(m_i3)

            if out2[d_idx][1] > 0 and out2[d_idx][1] <= 0.01:
                tb_i2 = [1, 0, 0, 0, 0, 0, 0]
                tb_i = 0
            elif out2[d_idx][1] > 0.01 and out2[d_idx][1] <= 0.02:
                tb_i2 = [0, 1, 0, 0, 0, 0, 0]
                tb_i = 1
            elif out2[d_idx][1] > 0.02 and out2[d_idx][1] <= 0.03:
                tb_i2 = [0, 0, 1, 0, 0, 0, 0]
                tb_i = 2
            elif out2[d_idx][1] > 0.03 and out2[d_idx][1] <= 0.04:
                tb_i2 = [0, 0, 0, 1, 0, 0, 0]
                tb_i = 3
            elif out2[d_idx][1] > 0.04 and out2[d_idx][1] <= 0.05:
                tb_i2 = [0, 0, 0, 0, 1, 0, 0]
                tb_i = 4
            elif out2[d_idx][1] > 0.05 and out2[d_idx][1] <= 0.06:
                tb_i2 = [0, 0, 0, 0, 0, 1, 0]
                tb_i = 5
            elif out2[d_idx][1] > 0.06 and out2[d_idx][1] <= 0.07:
                tb_i2 = [0, 0, 0, 0, 0, 0, 1]
                tb_i = 6

            g_p = []
            # g_sig = 0.005
            g_sig = 0.0025
            # g_sig = 0.01

            g_p.append(np.exp(-(out2[d_idx][1] - 0.005) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][1] - 0.015) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][1] - 0.025) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][1] - 0.035) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][1] - 0.045) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][1] - 0.055) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][1] - 0.065) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))

            tb_i3 = []
            for gp_i in range(len(g_p)):
                tb_i3.append(g_p[gp_i]/np.sum(g_p))

            tb_label.append(tb_i)
            tb_label2.append(tb_i2)
            tb_label3.append(tb_i3)

            if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
                st_i2 = [1, 0, 0, 0, 0, 0, 0]
                st_i = 0
            elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.75:
                st_i2 = [0, 1, 0, 0, 0, 0, 0]
                st_i = 1
            elif out2[d_idx][2] > 0.75 and out2[d_idx][2] <= 0.8:
                st_i2 = [0, 0, 1, 0, 0, 0, 0]
                st_i = 2
            elif out2[d_idx][2] > 0.8 and out2[d_idx][2] <= 0.85:
                st_i2 = [0, 0, 0, 1, 0, 0, 0]
                st_i = 3
            elif out2[d_idx][2] > 0.85 and out2[d_idx][2] <= 0.9:
                st_i2 = [0, 0, 0, 0, 1, 0, 0]
                st_i = 4
            elif out2[d_idx][2] > 0.9 and out2[d_idx][2] <= 0.95:
                st_i2 = [0, 0, 0, 0, 0, 1, 0]
                st_i = 5
            elif out2[d_idx][2] > 0.95 and out2[d_idx][2] <= 1.0:
                st_i2 = [0, 0, 0, 0, 0, 0, 1]
                st_i = 6

            # if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
            #     st_i2 = [1, 0, 0, 0]
            #     st_i = 0
            # elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.8:
            #     st_i2 = [0, 1, 0, 0]
            #     st_i = 1
            # elif out2[d_idx][2] > 0.8 and out2[d_idx][2] <= 0.9:
            #     st_i2 = [0, 0, 1, 0]
            #     st_i = 2
            # elif out2[d_idx][2] > 0.9 and out2[d_idx][2] <= 1.0:
            #     st_i2 = [0, 0, 0, 1]
            #     st_i = 3

            # if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
            #     st_i2 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 0
            # elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.73:
            #     st_i2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 1
            # elif out2[d_idx][2] > 0.73 and out2[d_idx][2] <= 0.76:
            #     st_i2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 2
            # elif out2[d_idx][2] > 0.76 and out2[d_idx][2] <= 0.79:
            #     st_i2 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 3
            # elif out2[d_idx][2] > 0.79 and out2[d_idx][2] <= 0.82:
            #     st_i2 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            #     st_i = 4
            # elif out2[d_idx][2] > 0.82 and out2[d_idx][2] <= 0.85:
            #     st_i2 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            #     st_i = 5
            # elif out2[d_idx][2] > 0.85 and out2[d_idx][2] <= 0.88:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            #     st_i = 6
            # elif out2[d_idx][2] > 0.88 and out2[d_idx][2] <= 0.91:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            #     st_i = 7
            # elif out2[d_idx][2] > 0.91 and out2[d_idx][2] <= 0.94:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
            #     st_i = 8
            # elif out2[d_idx][2] > 0.94 and out2[d_idx][2] <= 0.97:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            #     st_i = 9
            # elif out2[d_idx][2] > 0.97 and out2[d_idx][2] <= 1.0:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            #     st_i = 10

            g_p = []
            # g_sig = 0.025
            g_sig = 0.0125
            # g_sig = 0.05

            g_p.append(np.exp(-(out2[d_idx][2] - 0.675) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][2] - 0.725) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][2] - 0.775) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][2] - 0.825) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][2] - 0.875) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][2] - 0.925) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][2] - 0.975) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))

            # g_p.append(np.exp(-(out2[d_idx][2] - 0.65) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            # g_p.append(np.exp(-(out2[d_idx][2] - 0.75) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            # g_p.append(np.exp(-(out2[d_idx][2] - 0.85) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            # g_p.append(np.exp(-(out2[d_idx][2] - 0.95) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))

            # g_p.append(np.exp(-(out2[d_idx][2] - 0.685) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            # g_p.append(np.exp(-(out2[d_idx][2] - 0.715) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            # g_p.append(np.exp(-(out2[d_idx][2] - 0.745) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            # g_p.append(np.exp(-(out2[d_idx][2] - 0.775) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            # g_p.append(np.exp(-(out2[d_idx][2] - 0.805) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            # g_p.append(np.exp(-(out2[d_idx][2] - 0.835) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            # g_p.append(np.exp(-(out2[d_idx][2] - 0.865) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            # g_p.append(np.exp(-(out2[d_idx][2] - 0.895) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            # g_p.append(np.exp(-(out2[d_idx][2] - 0.925) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            # g_p.append(np.exp(-(out2[d_idx][2] - 0.955) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            # g_p.append(np.exp(-(out2[d_idx][2] - 0.985) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))


            st_i3 = []
            for gp_i in range(len(g_p)):
                st_i3.append(g_p[gp_i]/np.sum(g_p))

            st_label.append(st_i)
            st_label2.append(st_i2)
            st_label3.append(st_i3)

            if out2[d_idx][3] <= 0.035:
                th_i2 = [1, 0, 0]
                th_i = 0
            elif out2[d_idx][3] > 0.035 and out2[d_idx][3] <= 0.055:
                th_i2 = [0, 1, 0]
                th_i = 1
            elif out2[d_idx][3] > 0.055 and out2[d_idx][3] <= 0.075:
                th_i2 = [0, 0, 1]
                th_i = 2

            g_p = []
            g_sig = 0.02

            g_p.append(np.exp(-(out2[d_idx][3] - 0.025) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][3] - 0.045) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][3] - 0.065) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))

            th_i3 = []
            for gp_i in range(len(g_p)):
                th_i3.append(g_p[gp_i] / np.sum(g_p))

            th_label.append(th_i)
            th_label2.append(th_i2)
            th_label3.append(th_i3)


            # STO THB Combination Probability

            g_p = []
            # g_sig = np.array([[0.005**2, 0], [0, 0.025**2]])
            g_sig = np.array([[0.0025**2, 0], [0, 0.0125**2]])
            # g_sig = np.array([[0.01**2, 0], [0, 0.05**2]])
            # g_sig = np.array([[0.02**2, 0], [0, 0.1**2]])
            g_x = np.array([out2[d_idx][1], out2[d_idx][2]])

            g_thb_u = [0.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065]
            g_sto_u = [0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975]
            # g_sto_u = [0.65, 0.75, 0.85, 0.95]
            # g_sto_u = [0.685, 0.715, 0.745, 0.775, 0.805, 0.835, 0.865, 0.895, 0.925, 0.955, 0.985]

            #g_sto_u = [0.725, 0.775, 0.825, 0.875, 0.925, 0.975]

            for g_t in range(len(g_thb_u)):
                for g_s in range(len(g_sto_u)):
                    g_u = np.array([g_thb_u[g_t], g_sto_u[g_s]])
                    g_p.append(np.exp(-(1/2)*np.dot(np.dot((g_x-g_u).T, np.linalg.inv(g_sig)), (g_x-g_u))) / ((((2*np.pi)**2) * np.linalg.det(g_sig))**(1/2)))

            st_tb_i3 = []
            for gp_i in range(len(g_p)):
                st_tb_i3.append(g_p[gp_i] / np.sum(g_p))

            combination_label3.append(st_tb_i3)

        reflect_list_ex = []
        m_label_ex = []
        tb_label_ex = []
        st_label_ex = []
        th_label_ex = []

        for i in range(len(reflect_list)):
            if model_mel == -1 and model_thick == -1:
                temp_ref = []
                # for ri in range(64):
                #     temp_ref.append(reflect_list[i][ri])

                # temp_ref.append(reflect_list[i][5])

                # temp_ref.append(reflect_list[i][8])
                temp_ref.append(reflect_list[i][9])

                # temp_ref.append(reflect_list[i][11])
                temp_ref.append(reflect_list[i][12])

                # temp_ref.append(reflect_list[i][15])
                temp_ref.append(reflect_list[i][16])

                # temp_ref.append(reflect_list[i][18])
                temp_ref.append(reflect_list[i][19])

                # temp_ref.append(reflect_list[i][21])
                # temp_ref.append(reflect_list[i][22])  # temp_ref.append(reflect_list[i][25])

                # temp_ref.append(reflect_list[i][24])
                temp_ref.append(reflect_list[i][25])  # temp_ref.append(reflect_list[i][25])

                # temp_ref.append(reflect_list[i][26])
                temp_ref.append(reflect_list[i][27])

                # temp_ref.append(reflect_list[i][28])
                temp_ref.append(reflect_list[i][29])

                # temp_ref.append(reflect_list[i][30])
                temp_ref.append(reflect_list[i][31])

                # temp_ref.append(reflect_list[i][33])
                temp_ref.append(reflect_list[i][34])
                temp_ref.append(reflect_list[i][35])
                # temp_ref.append(reflect_list[i][37])
                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.6)
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.83)

                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][39])

                # temp_ref.append(reflect_list[i][41])
                # temp_ref.append(reflect_list[i][42])  # temp_ref.append(reflect_list[i][43])

                # temp_ref.append(reflect_list[i][43])
                temp_ref.append(reflect_list[i][44])
                temp_ref.append((reflect_list[i][46] + reflect_list[i][47]) / 2)

                # temp_ref.append(reflect_list[i][45])
                # temp_ref.append(reflect_list[i][46])

                # temp_ref.append(reflect_list[i][47])
                temp_ref.append(reflect_list[i][48])
                temp_ref.append(reflect_list[i][49])

                # temp_ref.append(reflect_list[i][50])
                # temp_ref.append(reflect_list[i][51])
                # temp_ref.append((reflect_list[i][52] + reflect_list[i][53])/2)
                # temp_ref.append(reflect_list[i][54])
                # temp_ref.append((reflect_list[i][55] + reflect_list[i][56]) / 2)
                # temp_ref.append((reflect_list[i][56] + reflect_list[i][57]) / 2)
                # temp_ref.append(reflect_list[i][57])
                # temp_ref.append(reflect_list[i][59])
                # temp_ref.append(reflect_list[i][60])
                # temp_ref.append(reflect_list[i][61])
                # temp_ref.append(reflect_list[i][62])
                # temp_ref.append(reflect_list[i][63])

                reflect_list_ex.append(temp_ref)
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])
            elif m_label[i] == model_mel and th_label[i] == model_thick:
                reflect_list_ex.append(reflect_list[i])
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])

        # m_label = np.array(m_label, dtype=np.float32)
        # tb_label = np.array(tb_label, dtype=np.float32)
        # st_label = np.array(st_label, dtype=np.float32)
        # th_label = np.array(th_label, dtype=np.float32)

        m_label = np.array(m_label_ex, dtype=np.float32)
        tb_label = np.array(tb_label_ex, dtype=np.float32)
        tb_label3 = np.array(tb_label3, dtype=np.float32)
        st_label = np.array(st_label_ex, dtype=np.float32)
        st_label3 = np.array(st_label3, dtype=np.float32)
        th_label = np.array(th_label_ex, dtype=np.float32)
        combination_label3 = np.array(combination_label3, dtype=np.float32)

        m_label3 = np.array(m_label3, dtype=np.float32)
        th_label3 = np.array(th_label3, dtype=np.float32)



        #reflect_list = np.concatenate([reflect_list, m_label2, th_label2], axis=1)
        #reflect_list = np.concatenate((reflect_list, m_label[:, np.newaxis], th_label[:, np.newaxis]), axis=1)
        #reflect_list = np.array(reflect_list, dtype=np.float32)
        reflect_list = np.array(reflect_list_ex, dtype=np.float32)

        return reflect_list, m_label, tb_label, tb_label3, st_label,  st_label3, th_label, combination_label3, m_label3, th_label3


class ViatalSignDataset_class_mel_thickness(data.Dataset):
    def __init__(self, mode='train', cl='', use_gpu = False, model_mel= -1, model_thick=-1):
        self.mode = mode
        self.cl = cl
        self.use_gpu = use_gpu

        if mode == "train":
            reflect_list, m_label, m_label2, m_label3, tb_label, st_label, th_label, th_label2, th_label3 = self.read_vitalsign_dataset(name='train', model_mel=model_mel, model_thick=model_thick)
        else:
            reflect_list, m_label, m_label2, m_label3, tb_label, st_label, th_label, th_label2, th_label3 = self.read_vitalsign_dataset(name='test', model_mel=model_mel, model_thick=model_thick)

        total_label = m_label

        if self.use_gpu == True:
            self.ref_list = torch.FloatTensor(reflect_list).to('cuda')
            self.m_label = torch.LongTensor(m_label).to('cuda')
            # self.m_label2 = torch.LongTensor(m_label2).to('cuda')
            self.m_label2 = torch.FloatTensor(m_label2).to('cuda')
            self.m_label3 = torch.FloatTensor(m_label3).to('cuda')
            self.tb_label = torch.LongTensor(tb_label).to('cuda')
            self.st_label = torch.LongTensor(st_label).to('cuda')
            self.th_label = torch.LongTensor(th_label).to('cuda')
            self.th_label2 = torch.FloatTensor(th_label2).to('cuda')
            self.th_label3 = torch.FloatTensor(th_label3).to('cuda')
            self.total_label = torch.LongTensor(total_label).to('cuda')
        else:
            self.ref_list = torch.FloatTensor(reflect_list)
            self.m_label = torch.LongTensor(m_label)
            self.m_label2 = torch.LongTensor(m_label2)
            self.m_label3 = torch.LongTensor(m_label3)
            self.tb_label = torch.LongTensor(tb_label)
            self.st_label = torch.LongTensor(st_label)
            self.th_label = torch.LongTensor(th_label)
            self.th_label2 = torch.FloatTensor(th_label2)
            self.th_label3 = torch.LongTensor(th_label3)
            self.total_label = torch.LongTensor(total_label)

    def __getitem__(self, index):
        anchor = self.ref_list[index]
        ml = self.m_label[index]
        ml2 = self.m_label2[index]
        ml3 = self.m_label3[index]
        tbl = self.tb_label[index]
        stl = self.st_label[index]
        thl = self.th_label[index]
        thl2 = self.th_label2[index]
        thl3 = self.th_label3[index]
        totall = self.total_label[index]

        return anchor, ml, ml2, ml3, tbl, stl, thl, thl2, thl3, totall

    def __len__(self):
        return len(self.ref_list)

    def read_vitalsign_dataset(self, name, model_mel, model_thick):
        '''
        Read Data
        '''
        path = os.path.dirname(__file__)

        # fileNameList = ['input_data9.npy', 'input_data10.npy', 'input_data11.npy', 'input_data13.npy',\
        #                 'input_data_random9.npy', 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        #                 'input_data_random21.npy', 'input_data_random22.npy','input_data_random23.npy', 'input_data_random24.npy']
        #
        # # test_fileNameList = ['input_data9.npy', 'input_data10.npy', 'input_data11.npy', 'input_data13.npy',\
        # #                 'input_data_random9.npy', 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        # #                 'input_data_random21.npy', 'input_data_random22.npy','input_data_random23.npy', 'input_data_random24.npy']
        # test_fileNameList = ['input_data8.npy', 'input_data_random8.npy']

        # fileNameList = ['input_data9.npy', 'input_data10.npy', \
        #                 'input_data11.npy', 'input_data13.npy', 'input_data14.npy', 'input_data15.npy',\
        #                 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy', \
        #                 'input_data_random18.npy', 'input_data_random19.npy', 'input_data_random20.npy', 'input_data_random21.npy',\
        #                 'input_data_random22.npy', 'input_data_random23.npy', 'input_data_random24.npy']
        #
        # test_fileNameList = ['input_data12.npy', 'input_data_random17.npy']

        #
        # test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']

        fileNameList = ['input_data9.npy', 'input_data14.npy', 'input_data15.npy',\
                        'input_data_random9.npy', 'input_data_random23.npy', 'input_data_random21.npy','input_data_random22.npy', 'input_data_random24.npy',\
                        'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy','input_data_random20_3.npy'] #\
                        # 'input_data_random16.npy', 'input_data_random17.npy', 'input_data_random18.npy', 'input_data_random19.npy',]

        test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']

        #
        # fileNameList = ['input_data_4000_1.npy', 'input_data_4000_2.npy','input_data_4000_3.npy','input_data_4000_4.npy',\
        #                 'random_data_4000_1.npy', 'random_data_4000_2.npy', 'random_data_4000_3.npy', \
        #                      'random_data_4000_4.npy', 'random_data_4000_5.npy', 'random_data_4000_6.npy' ]
        #
        # test_fileNameList = ['input_data_4000_1.npy', 'input_data_4000_2.npy','input_data_4000_3.npy','input_data_4000_4.npy',\
        #                 'random_data_4000_1.npy', 'random_data_4000_2.npy', 'random_data_4000_3.npy', \
        #                      'random_data_4000_4.npy', 'random_data_4000_5.npy', 'random_data_4000_6.npy' ]

        if name == 'train':
            fileNameList = fileNameList
        else:
            fileNameList = test_fileNameList

        train_data = []

        for fn in fileNameList:
            temp_data = np.load(path + '/input_data/{}'.format(fn), allow_pickle=True)

            if len(train_data) == 0 :
                train_data = temp_data
            else:
                train_data = np.concatenate([train_data, temp_data], axis=1)

        reflect_list = np.transpose(train_data[:, :, 5])

        out2 = train_data[0, :, 0:4]

        m_label = []
        tb_label = []
        st_label = []
        th_label = []

        m_label2 = []
        tb_label2 = []
        st_label2 = []
        th_label2 = []

        m_label3 = []
        tb_label3 = []
        st_label3 = []
        th_label3 = []

        for d_idx in range(len(out2)):
            if out2[d_idx][0] > 0 and out2[d_idx][0] <= 0.02:
                m_i2 = [1, 0, 0, 0, 0, 0, 0, 0]
                m_i = 0
            elif out2[d_idx][0] > 0.02 and out2[d_idx][0] <= 0.04:
                m_i2 = [0, 1, 0, 0, 0, 0, 0, 0]
                m_i = 1
            elif out2[d_idx][0] > 0.04 and out2[d_idx][0] <= 0.06:
                m_i2 = [0, 0, 1, 0, 0, 0, 0, 0]
                m_i = 2
            elif out2[d_idx][0] > 0.06 and out2[d_idx][0] <= 0.08:
                m_i2 = [0, 0, 0, 1, 0, 0, 0, 0]
                m_i = 3
            elif out2[d_idx][0] > 0.08 and out2[d_idx][0] <= 0.1:
                m_i2 = [0, 0, 0, 0, 1, 0, 0, 0]
                m_i = 4
            elif out2[d_idx][0] > 0.1 and out2[d_idx][0] <= 0.12:
                m_i2 = [0, 0, 0, 0, 0, 1, 0, 0]
                m_i = 5
            elif out2[d_idx][0] > 0.12 and out2[d_idx][0] <= 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 1, 0]
                m_i = 6
            elif out2[d_idx][0] > 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 0, 1]
                m_i = 7

            g_p = []
            g_sig = 0.01
            # g_sig = 0.02

            g_p.append(np.exp(-(out2[d_idx][0] - 0.01) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.03) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.05) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.07) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.09) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.11) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.13) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.15) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))

            m_i3 = []
            for gp_i in range(len(g_p)):
                m_i3.append(g_p[gp_i]/np.sum(g_p))

            m_label.append(m_i)
            # m_label2.append(m_i2)
            m_label2.append(out2[d_idx][0])
            m_label3.append(m_i3)

            if out2[d_idx][1] > 0 and out2[d_idx][1] <= 0.01:
                tb_i2 = [1, 0, 0, 0, 0, 0, 0]
                tb_i = 0
            elif out2[d_idx][1] > 0.01 and out2[d_idx][1] <= 0.02:
                tb_i2 = [0, 1, 0, 0, 0, 0, 0]
                tb_i = 1
            elif out2[d_idx][1] > 0.02 and out2[d_idx][1] <= 0.03:
                tb_i2 = [0, 0, 1, 0, 0, 0, 0]
                tb_i = 2
            elif out2[d_idx][1] > 0.03 and out2[d_idx][1] <= 0.04:
                tb_i2 = [0, 0, 0, 1, 0, 0, 0]
                tb_i = 3
            elif out2[d_idx][1] > 0.04 and out2[d_idx][1] <= 0.05:
                tb_i2 = [0, 0, 0, 0, 1, 0, 0]
                tb_i = 4
            elif out2[d_idx][1] > 0.05 and out2[d_idx][1] <= 0.06:
                tb_i2 = [0, 0, 0, 0, 0, 1, 0]
                tb_i = 5
            elif out2[d_idx][1] > 0.06 and out2[d_idx][1] <= 0.07:
                tb_i2 = [0, 0, 0, 0, 0, 0, 1]
                tb_i = 6

            tb_label.append(tb_i)
            tb_label2.append(tb_i2)

            if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
                st_i2 = [1, 0, 0, 0, 0, 0, 0]
                st_i = 0
            elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.75:
                st_i2 = [0, 1, 0, 0, 0, 0, 0]
                st_i = 1
            elif out2[d_idx][2] > 0.75 and out2[d_idx][2] <= 0.8:
                st_i2 = [0, 0, 1, 0, 0, 0, 0]
                st_i = 2
            elif out2[d_idx][2] > 0.8 and out2[d_idx][2] <= 0.85:
                st_i2 = [0, 0, 0, 1, 0, 0, 0]
                st_i = 3
            elif out2[d_idx][2] > 0.85 and out2[d_idx][2] <= 0.9:
                st_i2 = [0, 0, 0, 0, 1, 0, 0]
                st_i = 4
            elif out2[d_idx][2] > 0.9 and out2[d_idx][2] <= 0.95:
                st_i2 = [0, 0, 0, 0, 0, 1, 0]
                st_i = 5
            elif out2[d_idx][2] > 0.95 and out2[d_idx][2] <= 1.0:
                st_i2 = [0, 0, 0, 0, 0, 0, 1]
                st_i = 6

            st_label.append(st_i)
            st_label2.append(st_i2)

            if out2[d_idx][3] <= 0.035:
                th_i2 = [1, 0, 0]
                th_i = 0
            elif out2[d_idx][3] > 0.035 and out2[d_idx][3] <= 0.055:
                th_i2 = [0, 1, 0]
                th_i = 1
            elif out2[d_idx][3] > 0.055 and out2[d_idx][3] <= 0.075:
                th_i2 = [0, 0, 1]
                th_i = 2
            # if out2[d_idx][3] <= 0.025:
            #     th_i2 = [1, 0, 0, 0, 0, 0]
            #     th_i = 0
            # elif out2[d_idx][3] > 0.025 and out2[d_idx][3] <= 0.035:
            #     th_i2 = [0, 1, 0, 0, 0, 0]
            #     th_i = 1
            # elif out2[d_idx][3] > 0.035 and out2[d_idx][3] <= 0.045:
            #     th_i2 = [0, 0, 1, 0, 0, 0]
            #     th_i = 2
            # elif out2[d_idx][3] > 0.045 and out2[d_idx][3] <= 0.055:
            #     th_i2 = [0, 0, 0, 1, 0, 0]
            #     th_i = 3
            # elif out2[d_idx][3] > 0.055 and out2[d_idx][3] <= 0.065:
            #     th_i2 = [0, 0, 0, 0, 1, 0]
            #     th_i = 4
            # elif out2[d_idx][3] > 0.065:
            #     th_i2 = [0, 0, 0, 0, 0, 1]
            #     th_i = 5

            g_p = []
            g_sig = 0.01
            # g_sig = 0.02

            g_p.append(np.exp(-(out2[d_idx][3] - 0.025) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][3] - 0.045) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(out2[d_idx][3] - 0.065) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))

            th_i3 = []
            for gp_i in range(len(g_p)):
                th_i3.append(g_p[gp_i] / np.sum(g_p))

            th_label.append(th_i)
            # th_label2.append(th_i2)
            th_label2.append(out2[d_idx][3])
            th_label3.append(th_i3)

        reflect_list_ex = []
        m_label_ex = []
        tb_label_ex = []
        st_label_ex = []
        th_label_ex = []


        for i in range(len(reflect_list)):
            if model_mel == -1 and model_thick == -1:
                temp_ref = []
                # for ri in range(64):
                #     temp_ref.append(reflect_list[i][ri])

                # temp_ref.append(reflect_list[i][5])

                # temp_ref.append(reflect_list[i][8])
                temp_ref.append(reflect_list[i][9])

                # temp_ref.append(reflect_list[i][11])
                temp_ref.append(reflect_list[i][12])

                # temp_ref.append(reflect_list[i][15])
                temp_ref.append(reflect_list[i][16])

                # temp_ref.append(reflect_list[i][18])
                temp_ref.append(reflect_list[i][19])

                # temp_ref.append(reflect_list[i][21])
                # temp_ref.append(reflect_list[i][22])  #temp_ref.append(reflect_list[i][25])
                #
                # temp_ref.append(reflect_list[i][24])
                temp_ref.append(reflect_list[i][25])  #temp_ref.append(reflect_list[i][25])

                # temp_ref.append(reflect_list[i][26])
                temp_ref.append(reflect_list[i][27])

                # temp_ref.append(reflect_list[i][28])
                temp_ref.append(reflect_list[i][29])

                # temp_ref.append(reflect_list[i][30])
                temp_ref.append(reflect_list[i][31])

                # temp_ref.append(reflect_list[i][33])
                temp_ref.append(reflect_list[i][34])
                temp_ref.append(reflect_list[i][35])
                # temp_ref.append(reflect_list[i][37])
                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.6)
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.83)

                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][39])

                # temp_ref.append(reflect_list[i][41])
                # temp_ref.append(reflect_list[i][42])  # temp_ref.append(reflect_list[i][43])

                # temp_ref.append(reflect_list[i][43])
                temp_ref.append(reflect_list[i][44])
                temp_ref.append((reflect_list[i][46] + reflect_list[i][47]) / 2)

                # temp_ref.append(reflect_list[i][45])
                # temp_ref.append(reflect_list[i][46])

                # temp_ref.append(reflect_list[i][47])
                temp_ref.append(reflect_list[i][48])
                temp_ref.append(reflect_list[i][49])

                # temp_ref.append(reflect_list[i][50])
                # temp_ref.append(reflect_list[i][51])
                # temp_ref.append((reflect_list[i][52] + reflect_list[i][53])/2)
                # temp_ref.append(reflect_list[i][54])
                # temp_ref.append((reflect_list[i][55] + reflect_list[i][56]) / 2)
                # temp_ref.append((reflect_list[i][56] + reflect_list[i][57]) / 2)
                # temp_ref.append(reflect_list[i][57])
                # temp_ref.append(reflect_list[i][59])
                # temp_ref.append(reflect_list[i][60])
                # temp_ref.append(reflect_list[i][61])
                # temp_ref.append(reflect_list[i][62])
                # temp_ref.append(reflect_list[i][63])

                # [0909] input data absorbance + mel one-hot + thickness one-hot + thb one-hot
                #reflect_list_concat = list(reflect_list[i]) + m_label2[i] + tb_label2[i] + th_label2[i]
                # [0908] input data absorbance + mel one-hot + thickness one-hot
                #reflect_list_concat = list(reflect_list[i]) + m_label2[i] + th_label2[i]
                # [0902] input data absourbance
                reflect_list_ex.append(temp_ref)
                #reflect_list_ex.append(reflect_list_concat)
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])
            elif m_label[i] == model_mel and th_label[i] == model_thick:
                reflect_list_ex.append(reflect_list[i])
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])

        # m_label = np.array(m_label, dtype=np.float32)
        # tb_label = np.array(tb_label, dtype=np.float32)
        # st_label = np.array(st_label, dtype=np.float32)
        # th_label = np.array(th_label, dtype=np.float32)

        m_label = np.array(m_label_ex, dtype=np.float32)
        m_label2 = np.array(m_label2, dtype=np.float32)
        m_label3 = np.array(m_label3, dtype=np.float32)
        tb_label = np.array(tb_label_ex, dtype=np.float32)
        st_label = np.array(st_label_ex, dtype=np.float32)
        th_label = np.array(th_label_ex, dtype=np.float32)
        th_label2 = np.array(th_label2, dtype=np.float32)
        th_label3 = np.array(th_label3, dtype=np.float32)


        #reflect_list = np.concatenate([reflect_list, m_label2, th_label2], axis=1)
        #reflect_list = np.concatenate((reflect_list, m_label[:, np.newaxis], th_label[:, np.newaxis]), axis=1)
        #reflect_list = np.array(reflect_list, dtype=np.float32)
        reflect_list = np.array(reflect_list_ex, dtype=np.float32)

        return reflect_list, m_label, m_label2, m_label3, tb_label, st_label, th_label, th_label2, th_label3

class ViatalSignDataset_regression(data.Dataset):
    def __init__(self, mode='train', cl='', use_gpu = False, model_mel= -1, model_thick=-1):
        self.mode = mode
        self.cl = cl
        self.use_gpu = use_gpu

        if mode == "train":
            reflect_list, m_label, tb_label, st_label, th_label, m_label1, tb_label1, st_label1, th_label1, m_label3, th_label3 = self.read_vitalsign_dataset_regression(name='train', model_mel=model_mel, model_thick=model_thick)
        else:
            reflect_list, m_label, tb_label, st_label, th_label, m_label1, tb_label1, st_label1, th_label1, m_label3, th_label3 = self.read_vitalsign_dataset_regression(name='test', model_mel=model_mel, model_thick=model_thick)

        mel_feature_model = VitalSign_Feature_mel_thickness().to('cuda')
        thickness_feature_model = VitalSign_Feature_mel_thickness().to('cuda')

        mel_classifier_model = Classifier(cl_mode="mel").to('cuda')
        thickness_classifier_model = Classifier(cl_mode="thickness").to('cuda')

        path = os.path.dirname(__file__)

        mel_feature_path = os.path.join(path, './result/vitalsign_mel_0104_prob_02_input14_m1_epoch5000_addinput3/feature_weight_data')
        thickness_feature_path = os.path.join(path, './result/vitalsign_thickness_0104_prob_02_input14_m1_epoch5000_addinput3/feature_weight_data')

        mel_classify_path = os.path.join(path, './result/vitalsign_mel_0104_prob_02_input14_m1_epoch5000_addinput3/classification_weight_data2')
        thickness_classify_path = os.path.join(path, './result/vitalsign_thickness_0104_prob_02_input14_m1_epoch5000_addinput3/classification_weight_data2')

        mel_feature_model.load_state_dict(torch.load(mel_feature_path))
        mel_classifier_model.load_state_dict(torch.load(mel_classify_path))

        thickness_feature_model.load_state_dict(torch.load(thickness_feature_path))
        thickness_classifier_model.load_state_dict(torch.load(thickness_classify_path))

        mel_x = mel_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        mel_out = mel_classifier_model(mel_x)
        mel_prob = F.softmax(mel_out, dim=1)
        mel_prob = mel_prob.detach().cpu().numpy()

        thickness_x = thickness_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        thickenss_out = thickness_classifier_model(thickness_x)
        thickness_prob = F.softmax(thickenss_out, dim=1)
        thickness_prob = thickness_prob.detach().cpu().numpy()

        reflect_list = np.concatenate((reflect_list, mel_prob, thickness_prob), axis=1)
        # reflect_list = np.concatenate((reflect_list, m_label3, th_label3), axis=1)

        if self.use_gpu == True:
            self.ref_list = torch.FloatTensor(reflect_list).to('cuda')
            self.m_label = torch.FloatTensor(m_label).to('cuda')
            self.tb_label = torch.FloatTensor(tb_label).to('cuda')
            self.st_label = torch.FloatTensor(st_label).to('cuda')
            self.th_label = torch.FloatTensor(th_label).to('cuda')
            # self.m_label1 = torch.FloatTensor(m_label1).to('cuda')
            # self.tb_label1 = torch.FloatTensor(tb_label1).to('cuda')
            # self.st_label1 = torch.FloatTensor(st_label1).to('cuda')
            # self.th_label1 = torch.FloatTensor(th_label1).to('cuda')
        else:
            self.ref_list = torch.FloatTensor(reflect_list)
            self.m_label = torch.FloatTensor(m_label)
            self.tb_label = torch.FloatTensor(tb_label)
            self.st_label = torch.FloatTensor(st_label)
            self.th_label = torch.FloatTensor(th_label)
            # self.m_label1 = torch.FloatTensor(m_label1)
            # self.tb_label1 = torch.FloatTensor(tb_label1)
            # self.st_label1 = torch.FloatTensor(st_label1)
            # self.th_label1 = torch.FloatTensor(th_label1)

        self.m_label1 = m_label1
        self.tb_label1 = tb_label1
        self.st_label1 = st_label1
        self.th_label1 = th_label1


    def __getitem__(self, index):
        anchor = self.ref_list[index]
        ml = self.m_label[index]
        tbl = self.tb_label[index]
        stl = self.st_label[index]
        thl = self.th_label[index]

        ml1 = self.m_label1[index]
        tbl1 = self.tb_label1[index]
        stl1 = self.st_label1[index]
        thl1 = self.th_label1[index]

        return anchor, ml, tbl, stl, thl, ml1, tbl1, stl1, thl1

    def __len__(self):
        return len(self.ref_list)

    def read_vitalsign_dataset_regression(self, name, model_mel, model_thick):
        '''
        Read Data
        '''
        path = os.path.dirname(__file__)
        #
        # fileNameList = ['input_data9.npy', 'input_data10.npy', 'input_data11.npy', 'input_data13.npy',\
        #                 'input_data_random9.npy', 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        #                 'input_data_random21.npy', 'input_data_random22.npy','input_data_random23.npy', 'input_data_random24.npy']
        #
        # test_fileNameList = ['input_data8.npy', 'input_data_random13.npy']

        # fileNameList = ['input_data9.npy', 'input_data14.npy', 'input_data15.npy',\
                        # 'input_data_random9.npy', 'input_data_random21.npy', 'input_data_random22.npy','input_data_random23.npy', 'input_data_random24.npy']


        # fileNameList = ['input_data9.npy', 'input_data10.npy', \
        #                 'input_data11.npy', 'input_data13.npy', 'input_data14.npy', 'input_data15.npy',\
        #                 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy', \
        #                 'input_data_random18.npy', 'input_data_random19.npy', 'input_data_random20.npy', 'input_data_random21.npy',\
        #                 'input_data_random22.npy', 'input_data_random23.npy', 'input_data_random24.npy']

        fileNameList = ['input_data9.npy', 'input_data14.npy', 'input_data15.npy',\
                        'input_data_random9.npy', 'input_data_random23.npy', 'input_data_random21.npy','input_data_random22.npy', 'input_data_random24.npy',\
                        'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy', 'input_data_random20_3.npy'] #\
                        # 'input_data_random16.npy', 'input_data_random17.npy', 'input_data_random18.npy', 'input_data_random19.npy',]


        test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']

        # test_fileNameList = ['input_data9.npy', 'input_data10.npy', 'input_data11.npy', 'input_data13.npy',\
        #                 'input_data_random9.npy', 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        #                 'input_data_random21.npy', 'input_data_random22.npy','input_data_random23.npy', 'input_data_random24.npy']

        # fileNameList = ['input_data6.npy', 'input_data7.npy', 'input_data8.npy', 'input_data9.npy', 'input_data10.npy', \
        #                 'input_data11.npy', 'input_data12.npy', 'input_data13.npy', 'input_data14.npy', 'input_data15.npy',\
        #                 'input_data_random6.npy', 'input_data_random7.npy', 'input_data_random8.npy', 'input_data_random9.npy', \
        #                 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        #                 'input_data_random14.npy', 'input_data_random15.npy', 'input_data_random16.npy','input_data_random17.npy', \
        #                 'input_data_random18.npy', 'input_data_random19.npy', 'input_data_random20.npy', 'input_data_random21.npy', \
        #                 'input_data_random22.npy', 'input_data_random23.npy', 'input_data_random24.npy']
        #
        # test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']

        # fileNameList = ['input_data_4000_1.npy', 'input_data_4000_2.npy','input_data_4000_3.npy','input_data_4000_4.npy',\
        #                 'random_data_4000_1.npy', 'random_data_4000_2.npy', 'random_data_4000_3.npy',\
        #                 'random_data_4000_4.npy', 'random_data_4000_5.npy', 'random_data_4000_6.npy']
        #
        # test_fileNameList = ['input_data_4000_1.npy', 'input_data_4000_2.npy','input_data_4000_3.npy','input_data_4000_4.npy',\
        #                 'random_data_4000_1.npy', 'random_data_4000_2.npy', 'random_data_4000_3.npy',\
        #                 'random_data_4000_4.npy', 'random_data_4000_5.npy', 'random_data_4000_6.npy']

        if name == 'train':
            fileNameList = fileNameList
        else:
            fileNameList = test_fileNameList

        train_data = []

        for fn in fileNameList:
            temp_data = np.load(path + '/input_data/{}'.format(fn), allow_pickle=True)

            if len(train_data) == 0 :
                train_data = temp_data
            else:
                train_data = np.concatenate([train_data, temp_data], axis=1)

        reflect_list = np.transpose(train_data[:, :, 5])

        out2 = train_data[0, :, 0:4]

        m_label = []
        tb_label = []
        st_label = []
        th_label = []

        for d_idx in range(len(out2)):
            m_label.append(out2[d_idx][0])
            tb_label.append(out2[d_idx][1])
            st_label.append(out2[d_idx][2])
            th_label.append(out2[d_idx][3])

        m_label1 = []
        tb_label1 = []
        st_label1 = []
        th_label1 = []

        m_label2 = []
        tb_label2 = []
        st_label2 = []
        th_label2 = []

        m_label3 = []
        tb_label3 = []
        st_label3 = []
        th_label3 = []

        for d_idx in range(len(out2)):
            if out2[d_idx][0] > 0 and out2[d_idx][0] <= 0.02:
                m_i2 = [1, 0, 0, 0, 0, 0, 0, 0]
                m_i = 0
            elif out2[d_idx][0] > 0.02 and out2[d_idx][0] <= 0.04:
                m_i2 = [0, 1, 0, 0, 0, 0, 0, 0]
                m_i = 1
            elif out2[d_idx][0] > 0.04 and out2[d_idx][0] <= 0.06:
                m_i2 = [0, 0, 1, 0, 0, 0, 0, 0]
                m_i = 2
            elif out2[d_idx][0] > 0.06 and out2[d_idx][0] <= 0.08:
                m_i2 = [0, 0, 0, 1, 0, 0, 0, 0]
                m_i = 3
            elif out2[d_idx][0] > 0.08 and out2[d_idx][0] <= 0.1:
                m_i2 = [0, 0, 0, 0, 1, 0, 0, 0]
                m_i = 4
            elif out2[d_idx][0] > 0.1 and out2[d_idx][0] <= 0.12:
                m_i2 = [0, 0, 0, 0, 0, 1, 0, 0]
                m_i = 5
            elif out2[d_idx][0] > 0.12 and out2[d_idx][0] <= 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 1, 0]
                m_i = 6
            elif out2[d_idx][0] > 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 0, 1]
                m_i = 7

            g_p = []
            g_sig = 0.02

            g_p.append(np.exp(-(out2[d_idx][0] - 0.01) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.03) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.05) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.07) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.09) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.11) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.13) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.15) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))

            m_i3 = []
            for gp_i in range(len(g_p)):
                m_i3.append(g_p[gp_i]/np.sum(g_p))

            m_label1.append(m_i)
            m_label2.append(m_i2)
            m_label3.append(m_i3)

            if out2[d_idx][1] > 0 and out2[d_idx][1] <= 0.01:
                tb_i2 = [1, 0, 0, 0, 0, 0, 0]
                tb_i = 0
            elif out2[d_idx][1] > 0.01 and out2[d_idx][1] <= 0.02:
                tb_i2 = [0, 1, 0, 0, 0, 0, 0]
                tb_i = 1
            elif out2[d_idx][1] > 0.02 and out2[d_idx][1] <= 0.03:
                tb_i2 = [0, 0, 1, 0, 0, 0, 0]
                tb_i = 2
            elif out2[d_idx][1] > 0.03 and out2[d_idx][1] <= 0.04:
                tb_i2 = [0, 0, 0, 1, 0, 0, 0]
                tb_i = 3
            elif out2[d_idx][1] > 0.04 and out2[d_idx][1] <= 0.05:
                tb_i2 = [0, 0, 0, 0, 1, 0, 0]
                tb_i = 4
            elif out2[d_idx][1] > 0.05 and out2[d_idx][1] <= 0.06:
                tb_i2 = [0, 0, 0, 0, 0, 1, 0]
                tb_i = 5
            elif out2[d_idx][1] > 0.06 and out2[d_idx][1] <= 0.07:
                tb_i2 = [0, 0, 0, 0, 0, 0, 1]
                tb_i = 6

            tb_label1.append(tb_i)
            tb_label2.append(tb_i2)

            if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
                st_i2 = [1, 0, 0, 0, 0, 0, 0]
                st_i = 0
            elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.75:
                st_i2 = [0, 1, 0, 0, 0, 0, 0]
                st_i = 1
            elif out2[d_idx][2] > 0.75 and out2[d_idx][2] <= 0.8:
                st_i2 = [0, 0, 1, 0, 0, 0, 0]
                st_i = 2
            elif out2[d_idx][2] > 0.8 and out2[d_idx][2] <= 0.85:
                st_i2 = [0, 0, 0, 1, 0, 0, 0]
                st_i = 3
            elif out2[d_idx][2] > 0.85 and out2[d_idx][2] <= 0.9:
                st_i2 = [0, 0, 0, 0, 1, 0, 0]
                st_i = 4
            elif out2[d_idx][2] > 0.9 and out2[d_idx][2] <= 0.95:
                st_i2 = [0, 0, 0, 0, 0, 1, 0]
                st_i = 5
            elif out2[d_idx][2] > 0.95 and out2[d_idx][2] <= 1.0:
                st_i2 = [0, 0, 0, 0, 0, 0, 1]
                st_i = 6

            # if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
            #     st_i2 = [1, 0, 0, 0]
            #     st_i = 0
            # elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.8:
            #     st_i2 = [0, 1, 0, 0, 0, 0, 0]
            #     st_i = 1
            # elif out2[d_idx][2] > 0.8 and out2[d_idx][2] <= 0.9:
            #     st_i2 = [0, 0, 1, 0, 0, 0, 0]
            #     st_i = 2
            # elif out2[d_idx][2] > 0.9 and out2[d_idx][2] <= 1.0:
            #     st_i2 = [0, 0, 0, 1, 0, 0, 0]
            #     st_i = 3

            # if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
            #     st_i2 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 0
            # elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.73:
            #     st_i2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 1
            # elif out2[d_idx][2] > 0.73 and out2[d_idx][2] <= 0.76:
            #     st_i2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 2
            # elif out2[d_idx][2] > 0.76 and out2[d_idx][2] <= 0.79:
            #     st_i2 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            #     st_i = 3
            # elif out2[d_idx][2] > 0.79 and out2[d_idx][2] <= 0.82:
            #     st_i2 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            #     st_i = 4
            # elif out2[d_idx][2] > 0.82 and out2[d_idx][2] <= 0.85:
            #     st_i2 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            #     st_i = 5
            # elif out2[d_idx][2] > 0.85 and out2[d_idx][2] <= 0.88:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            #     st_i = 6
            # elif out2[d_idx][2] > 0.88 and out2[d_idx][2] <= 0.91:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            #     st_i = 7
            # elif out2[d_idx][2] > 0.91 and out2[d_idx][2] <= 0.94:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
            #     st_i = 8
            # elif out2[d_idx][2] > 0.94 and out2[d_idx][2] <= 0.97:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            #     st_i = 9
            # elif out2[d_idx][2] > 0.97 and out2[d_idx][2] <= 1.0:
            #     st_i2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            #     st_i = 10

            st_label1.append(st_i)
            st_label2.append(st_i2)

            if out2[d_idx][3] <= 0.035:
                th_i2 = [1, 0, 0]
                th_i = 0
            elif out2[d_idx][3] > 0.035 and out2[d_idx][3] <= 0.055:
                th_i2 = [0, 1, 0]
                th_i = 1
            elif out2[d_idx][3] > 0.055 and out2[d_idx][3] <= 0.075:
                th_i2 = [0, 0, 1]
                th_i = 2
            # if out2[d_idx][3] <= 0.025:
            #     th_i2 = [1, 0, 0, 0, 0, 0]
            #     th_i = 0
            # elif out2[d_idx][3] > 0.025 and out2[d_idx][3] <= 0.035:
            #     th_i2 = [0, 1, 0, 0, 0, 0]
            #     th_i = 1
            # elif out2[d_idx][3] > 0.035 and out2[d_idx][3] <= 0.045:
            #     th_i2 = [0, 0, 1, 0, 0, 0]
            #     th_i = 2
            # elif out2[d_idx][3] > 0.045 and out2[d_idx][3] <= 0.055:
            #     th_i2 = [0, 0, 0, 1, 0, 0]
            #     th_i = 3
            # elif out2[d_idx][3] > 0.055 and out2[d_idx][3] <= 0.065:
            #     th_i2 = [0, 0, 0, 0, 1, 0]
            #     th_i = 4
            # elif out2[d_idx][3] > 0.065:
            #     th_i2 = [0, 0, 0, 0, 0, 1]
            #     th_i = 5

            g_p = []
            g_sig = 0.02

            g_p.append(np.exp(-(out2[d_idx][3] - 0.025) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][3] - 0.045) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][3] - 0.065) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))

            th_i3 = []
            for gp_i in range(len(g_p)):
                th_i3.append(g_p[gp_i] / np.sum(g_p))

            th_label1.append(th_i)
            th_label2.append(th_i2)
            th_label3.append(th_i3)

        reflect_list_ex = []
        m_label_ex = []
        tb_label_ex = []
        st_label_ex = []
        th_label_ex = []

        for i in range(len(reflect_list)):
            if model_mel == -1 and model_thick == -1:
                temp_ref = []
                # for ri in range(64):
                #     temp_ref.append(reflect_list[i][ri])

                # temp_ref.append(reflect_list[i][5])

                # temp_ref.append(reflect_list[i][8])
                temp_ref.append(reflect_list[i][9])

                # temp_ref.append(reflect_list[i][11])
                temp_ref.append(reflect_list[i][12])

                # temp_ref.append(reflect_list[i][15])
                temp_ref.append(reflect_list[i][16])

                # temp_ref.append(reflect_list[i][18])
                temp_ref.append(reflect_list[i][19])

                # temp_ref.append(reflect_list[i][21])
                # temp_ref.append(reflect_list[i][22])  # temp_ref.append(reflect_list[i][25])

                # temp_ref.append(reflect_list[i][24])
                temp_ref.append(reflect_list[i][25])  # temp_ref.append(reflect_list[i][25])

                # temp_ref.append(reflect_list[i][26])
                temp_ref.append(reflect_list[i][27])

                # temp_ref.append(reflect_list[i][28])
                temp_ref.append(reflect_list[i][29])

                # temp_ref.append(reflect_list[i][30])
                temp_ref.append(reflect_list[i][31])

                # temp_ref.append(reflect_list[i][33])
                temp_ref.append(reflect_list[i][34])
                temp_ref.append(reflect_list[i][35])
                # temp_ref.append(reflect_list[i][37])
                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.6)
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.83)

                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][39])

                # temp_ref.append(reflect_list[i][41])
                # temp_ref.append(reflect_list[i][42])  # temp_ref.append(reflect_list[i][43])

                # temp_ref.append(reflect_list[i][43])
                temp_ref.append(reflect_list[i][44])
                temp_ref.append((reflect_list[i][46] + reflect_list[i][47]) / 2)

                # temp_ref.append(reflect_list[i][45])
                # temp_ref.append(reflect_list[i][46])

                # temp_ref.append(reflect_list[i][47])
                temp_ref.append(reflect_list[i][48])
                temp_ref.append(reflect_list[i][49])

                # temp_ref.append(reflect_list[i][50])
                # temp_ref.append(reflect_list[i][51])
                # temp_ref.append((reflect_list[i][52] + reflect_list[i][53])/2)
                # temp_ref.append(reflect_list[i][54])
                # temp_ref.append((reflect_list[i][55] + reflect_list[i][56]) / 2)
                # temp_ref.append((reflect_list[i][56] + reflect_list[i][57]) / 2)
                # temp_ref.append(reflect_list[i][57])
                # temp_ref.append(reflect_list[i][59])
                # temp_ref.append(reflect_list[i][60])
                # temp_ref.append(reflect_list[i][61])
                # temp_ref.append(reflect_list[i][62])
                # temp_ref.append(reflect_list[i][63])


                reflect_list_ex.append(temp_ref)
                #reflect_list_ex.append(reflect_list_concat)
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])
            elif m_label1[i] == model_mel and th_label1[i] == model_thick:
                reflect_list_ex.append(reflect_list[i])
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])

        # m_label = np.array(m_label, dtype=np.float32)
        # tb_label = np.array(tb_label, dtype=np.float32)
        # st_label = np.array(st_label, dtype=np.float32)
        # th_label = np.array(th_label, dtype=np.float32)

        m_label = np.array(m_label_ex, dtype=np.float32)
        tb_label = np.array(tb_label_ex, dtype=np.float32)
        st_label = np.array(st_label_ex, dtype=np.float32)
        th_label = np.array(th_label_ex, dtype=np.float32)

        # reflect_list = np.concatenate([reflect_list, m_label2, th_label2], axis=1)
        # reflect_list = np.concatenate((reflect_list, m_label[:, np.newaxis], th_label[:, np.newaxis]), axis=1)
        reflect_list = np.array(reflect_list_ex, dtype=np.float32)

        m_label3 = np.array(m_label3, dtype=np.float32)
        th_label3 = np.array(th_label3, dtype=np.float32)

        # m_label1 = np.array(m_label1, dtype=np.float32)
        # tb_label1 = np.array(tb_label1, dtype=np.float32)
        # st_label1 = np.array(st_label1, dtype=np.float32)
        # th_label1 = np.array(th_label1, dtype=np.float32)

        return reflect_list, m_label, tb_label, st_label, th_label, m_label1, tb_label1, st_label1, th_label1, m_label3, th_label3



class ViatalSignDataset_regression_mel_thickness(data.Dataset):
    def __init__(self, mode='train', cl='', use_gpu = False, model_mel= -1, model_thick=-1):
        self.mode = mode
        self.cl = cl
        self.use_gpu = use_gpu

        if mode == "train":
            reflect_list, m_label, tb_label, st_label, th_label, m_label1, tb_label1, st_label1, th_label1 = self.read_vitalsign_dataset_regression(name='train', model_mel=model_mel, model_thick=model_thick)
        else:
            reflect_list, m_label, tb_label, st_label, th_label, m_label1, tb_label1, st_label1, th_label1 = self.read_vitalsign_dataset_regression(name='test', model_mel=model_mel, model_thick=model_thick)

        if self.use_gpu == True:
            self.ref_list = torch.FloatTensor(reflect_list).to('cuda')
            self.m_label = torch.FloatTensor(m_label).to('cuda')
            self.tb_label = torch.FloatTensor(tb_label).to('cuda')
            self.st_label = torch.FloatTensor(st_label).to('cuda')
            self.th_label = torch.FloatTensor(th_label).to('cuda')
            # self.m_label1 = torch.FloatTensor(m_label1).to('cuda')
            # self.tb_label1 = torch.FloatTensor(tb_label1).to('cuda')
            # self.st_label1 = torch.FloatTensor(st_label1).to('cuda')
            # self.th_label1 = torch.FloatTensor(th_label1).to('cuda')
        else:
            self.ref_list = torch.FloatTensor(reflect_list)
            self.m_label = torch.FloatTensor(m_label)
            self.tb_label = torch.FloatTensor(tb_label)
            self.st_label = torch.FloatTensor(st_label)
            self.th_label = torch.FloatTensor(th_label)
            # self.m_label1 = torch.FloatTensor(m_label1)
            # self.tb_label1 = torch.FloatTensor(tb_label1)
            # self.st_label1 = torch.FloatTensor(st_label1)
            # self.th_label1 = torch.FloatTensor(th_label1)

        self.m_label1 = m_label1
        self.tb_label1 = tb_label1
        self.st_label1 = st_label1
        self.th_label1 = th_label1


    def __getitem__(self, index):
        anchor = self.ref_list[index]
        ml = self.m_label[index]
        tbl = self.tb_label[index]
        stl = self.st_label[index]
        thl = self.th_label[index]

        ml1 = self.m_label1[index]
        tbl1 = self.tb_label1[index]
        stl1 = self.st_label1[index]
        thl1 = self.th_label1[index]

        return anchor, ml, tbl, stl, thl, ml1, tbl1, stl1, thl1

    def __len__(self):
        return len(self.ref_list)

    def read_vitalsign_dataset_regression(self, name, model_mel, model_thick):
        '''
        Read Data
        '''
        path = os.path.dirname(__file__)

        # fileNameList = ['input_data9.npy', 'input_data10.npy', 'input_data11.npy', 'input_data13.npy',\
        #                 'input_data_random9.npy', 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        #                 'input_data_random21.npy', 'input_data_random22.npy','input_data_random23.npy', 'input_data_random24.npy']
        #
        # # test_fileNameList = ['input_data9.npy', 'input_data10.npy', 'input_data11.npy', 'input_data13.npy',\
        # #                 'input_data_random9.npy', 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy',\
        # #                 'input_data_random21.npy', 'input_data_random22.npy','input_data_random23.npy', 'input_data_random24.npy']
        #
        # test_fileNameList = ['input_data8.npy', 'input_data_random8.npy']

        # fileNameList = ['input_data9.npy', 'input_data10.npy', \
        #                 'input_data11.npy', 'input_data13.npy', 'input_data14.npy', 'input_data15.npy',\
        #                 'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy', \
        #                 'input_data_random18.npy', 'input_data_random19.npy', 'input_data_random20.npy', 'input_data_random21.npy',\
        #                 'input_data_random22.npy', 'input_data_random23.npy', 'input_data_random24.npy']
        #
        # test_fileNameList = ['input_data12.npy', 'input_data_random17.npy']
        #
        # test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']

        fileNameList = ['input_data9.npy', 'input_data14.npy', 'input_data15.npy',\
                        'input_data_random9.npy', 'input_data_random23.npy', 'input_data_random21.npy','input_data_random22.npy', 'input_data_random24.npy',\
                        'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy'] #,'input_data_random20_3.npy'] #\
                        # 'input_data_random16.npy', 'input_data_random17.npy', 'input_data_random18.npy', 'input_data_random19.npy',]

        test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']

        # fileNameList = ['input_data_4000_1.npy', 'input_data_4000_2.npy','input_data_4000_3.npy','input_data_4000_4.npy',\
        #                 'random_data_4000_1.npy', 'random_data_4000_2.npy', 'random_data_4000_3.npy', \
        #                      'random_data_4000_4.npy', 'random_data_4000_5.npy', 'random_data_4000_6.npy' ]
        #
        # test_fileNameList = ['input_data_4000_1.npy', 'input_data_4000_2.npy','input_data_4000_3.npy','input_data_4000_4.npy',\
        #                 'random_data_4000_1.npy', 'random_data_4000_2.npy', 'random_data_4000_3.npy', \
        #                      'random_data_4000_4.npy', 'random_data_4000_5.npy', 'random_data_4000_6.npy' ]

        if name == 'train':
            fileNameList = fileNameList
        else:
            fileNameList = test_fileNameList

        train_data = []

        for fn in fileNameList:
            temp_data = np.load(path + '/input_data/{}'.format(fn), allow_pickle=True)

            if len(train_data) == 0 :
                train_data = temp_data
            else:
                train_data = np.concatenate([train_data, temp_data], axis=1)

        reflect_list = np.transpose(train_data[:, :, 5])

        out2 = train_data[0, :, 0:4]

        m_label = []
        tb_label = []
        st_label = []
        th_label = []

        for d_idx in range(len(out2)):
            m_label.append(out2[d_idx][0])
            tb_label.append(out2[d_idx][1])
            st_label.append(out2[d_idx][2])
            th_label.append(out2[d_idx][3])

        m_label1 = []
        tb_label1 = []
        st_label1 = []
        th_label1 = []

        m_label2 = []
        tb_label2 = []
        st_label2 = []
        th_label2 = []

        m_label3 = []
        th_label3 = []

        for d_idx in range(len(out2)):
            if out2[d_idx][0] > 0 and out2[d_idx][0] <= 0.02:
                m_i2 = [1, 0, 0, 0, 0, 0, 0, 0]
                m_i = 0
            elif out2[d_idx][0] > 0.02 and out2[d_idx][0] <= 0.04:
                m_i2 = [0, 1, 0, 0, 0, 0, 0, 0]
                m_i = 1
            elif out2[d_idx][0] > 0.04 and out2[d_idx][0] <= 0.06:
                m_i2 = [0, 0, 1, 0, 0, 0, 0, 0]
                m_i = 2
            elif out2[d_idx][0] > 0.06 and out2[d_idx][0] <= 0.08:
                m_i2 = [0, 0, 0, 1, 0, 0, 0, 0]
                m_i = 3
            elif out2[d_idx][0] > 0.08 and out2[d_idx][0] <= 0.1:
                m_i2 = [0, 0, 0, 0, 1, 0, 0, 0]
                m_i = 4
            elif out2[d_idx][0] > 0.1 and out2[d_idx][0] <= 0.12:
                m_i2 = [0, 0, 0, 0, 0, 1, 0, 0]
                m_i = 5
            elif out2[d_idx][0] > 0.12 and out2[d_idx][0] <= 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 1, 0]
                m_i = 6
            elif out2[d_idx][0] > 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 0, 1]
                m_i = 7

            g_p = []
            g_sig = 0.01
            # g_sig = 0.02

            g_p.append(np.exp(-(out2[d_idx][0] - 0.01) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.03) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.05) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.07) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.09) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.11) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.13) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.15) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))

            m_i3 = []
            for gp_i in range(len(g_p)):
                m_i3.append(g_p[gp_i] / np.sum(g_p))

            m_label1.append(m_i)
            m_label2.append(m_i2)
            m_label3.append(m_i3)

            if out2[d_idx][1] > 0 and out2[d_idx][1] <= 0.01:
                tb_i2 = [1, 0, 0, 0, 0, 0, 0]
                tb_i = 0
            elif out2[d_idx][1] > 0.01 and out2[d_idx][1] <= 0.02:
                tb_i2 = [0, 1, 0, 0, 0, 0, 0]
                tb_i = 1
            elif out2[d_idx][1] > 0.02 and out2[d_idx][1] <= 0.03:
                tb_i2 = [0, 0, 1, 0, 0, 0, 0]
                tb_i = 2
            elif out2[d_idx][1] > 0.03 and out2[d_idx][1] <= 0.04:
                tb_i2 = [0, 0, 0, 1, 0, 0, 0]
                tb_i = 3
            elif out2[d_idx][1] > 0.04 and out2[d_idx][1] <= 0.05:
                tb_i2 = [0, 0, 0, 0, 1, 0, 0]
                tb_i = 4
            elif out2[d_idx][1] > 0.05 and out2[d_idx][1] <= 0.06:
                tb_i2 = [0, 0, 0, 0, 0, 1, 0]
                tb_i = 5
            elif out2[d_idx][1] > 0.06 and out2[d_idx][1] <= 0.07:
                tb_i2 = [0, 0, 0, 0, 0, 0, 1]
                tb_i = 6

            tb_label1.append(tb_i)
            tb_label2.append(tb_i2)

            if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
                st_i2 = [1, 0, 0, 0, 0, 0, 0]
                st_i = 0
            elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.75:
                st_i2 = [0, 1, 0, 0, 0, 0, 0]
                st_i = 1
            elif out2[d_idx][2] > 0.75 and out2[d_idx][2] <= 0.8:
                st_i2 = [0, 0, 1, 0, 0, 0, 0]
                st_i = 2
            elif out2[d_idx][2] > 0.8 and out2[d_idx][2] <= 0.85:
                st_i2 = [0, 0, 0, 1, 0, 0, 0]
                st_i = 3
            elif out2[d_idx][2] > 0.85 and out2[d_idx][2] <= 0.9:
                st_i2 = [0, 0, 0, 0, 1, 0, 0]
                st_i = 4
            elif out2[d_idx][2] > 0.9 and out2[d_idx][2] <= 0.95:
                st_i2 = [0, 0, 0, 0, 0, 1, 0]
                st_i = 5
            elif out2[d_idx][2] > 0.95 and out2[d_idx][2] <= 1.0:
                st_i2 = [0, 0, 0, 0, 0, 0, 1]
                st_i = 6

            st_label1.append(st_i)
            st_label2.append(st_i2)

            if out2[d_idx][3] <= 0.035:
                th_i2 = [1, 0, 0]
                th_i = 0
            elif out2[d_idx][3] > 0.035 and out2[d_idx][3] <= 0.055:
                th_i2 = [0, 1, 0]
                th_i = 1
            elif out2[d_idx][3] > 0.055 and out2[d_idx][3] <= 0.075:
                th_i2 = [0, 0, 1]
                th_i = 2

            g_p = []
            g_sig = 0.01
            # g_sig = 0.02

            g_p.append(np.exp(-(out2[d_idx][3] - 0.025) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][3] - 0.045) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][3] - 0.065) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))

            th_i3 = []
            for gp_i in range(len(g_p)):
                th_i3.append(g_p[gp_i] / np.sum(g_p))

            th_label1.append(th_i)
            th_label2.append(th_i2)
            th_label3.append(th_i3)

        reflect_list_ex = []
        m_label_ex = []
        tb_label_ex = []
        st_label_ex = []
        th_label_ex = []

        for i in range(len(reflect_list)):
            if model_mel == -1 and model_thick == -1:
                temp_ref = []
                # for ri in range(64):
                #     temp_ref.append(reflect_list[i][ri])

                # temp_ref.append(reflect_list[i][5])

                # temp_ref.append(reflect_list[i][8])
                temp_ref.append(reflect_list[i][9])

                # temp_ref.append(reflect_list[i][11])
                temp_ref.append(reflect_list[i][12])

                # temp_ref.append(reflect_list[i][15])
                temp_ref.append(reflect_list[i][16])

                # temp_ref.append(reflect_list[i][18])
                temp_ref.append(reflect_list[i][19])

                # temp_ref.append(reflect_list[i][21])
                # temp_ref.append(reflect_list[i][22])  #temp_ref.append(reflect_list[i][25])
                #
                # temp_ref.append(reflect_list[i][24])
                temp_ref.append(reflect_list[i][25])  # temp_ref.append(reflect_list[i][25])

                # temp_ref.append(reflect_list[i][26])
                temp_ref.append(reflect_list[i][27])

                # temp_ref.append(reflect_list[i][28])
                temp_ref.append(reflect_list[i][29])

                # temp_ref.append(reflect_list[i][30])
                temp_ref.append(reflect_list[i][31])

                # temp_ref.append(reflect_list[i][33])
                temp_ref.append(reflect_list[i][34])
                temp_ref.append(reflect_list[i][35])
                # temp_ref.append(reflect_list[i][37])
                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.6)
                # temp_ref.append(reflect_list[i][38]+(reflect_list[i][39]-reflect_list[i][38])*0.83)

                # temp_ref.append(reflect_list[i][38])
                # temp_ref.append(reflect_list[i][39])

                # temp_ref.append(reflect_list[i][41])
                # temp_ref.append(reflect_list[i][42])  # temp_ref.append(reflect_list[i][43])

                # temp_ref.append(reflect_list[i][43])
                temp_ref.append(reflect_list[i][44])
                temp_ref.append((reflect_list[i][46] + reflect_list[i][47]) / 2)

                # temp_ref.append(reflect_list[i][45])
                # temp_ref.append(reflect_list[i][46])

                # temp_ref.append(reflect_list[i][47])
                temp_ref.append(reflect_list[i][48])
                temp_ref.append(reflect_list[i][49])

                # temp_ref.append(reflect_list[i][50])
                # temp_ref.append(reflect_list[i][51])
                # temp_ref.append((reflect_list[i][52] + reflect_list[i][53])/2)
                # temp_ref.append(reflect_list[i][54])
                # temp_ref.append((reflect_list[i][55] + reflect_list[i][56]) / 2)
                # temp_ref.append((reflect_list[i][56] + reflect_list[i][57]) / 2)
                # temp_ref.append(reflect_list[i][57])
                # temp_ref.append(reflect_list[i][59])
                # temp_ref.append(reflect_list[i][60])
                # temp_ref.append(reflect_list[i][61])
                # temp_ref.append(reflect_list[i][62])
                # temp_ref.append(reflect_list[i][63])


                reflect_list_ex.append(temp_ref)
                #reflect_list_ex.append(reflect_list_concat)
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])
            elif m_label1[i] == model_mel and th_label1[i] == model_thick:
                reflect_list_ex.append(reflect_list[i])
                m_label_ex.append(m_label[i])
                tb_label_ex.append(tb_label[i])
                st_label_ex.append(st_label[i])
                th_label_ex.append(th_label[i])

        # m_label = np.array(m_label, dtype=np.float32)
        # tb_label = np.array(tb_label, dtype=np.float32)
        # st_label = np.array(st_label, dtype=np.float32)
        # th_label = np.array(th_label, dtype=np.float32)

        m_label = np.array(m_label_ex, dtype=np.float32)
        tb_label = np.array(tb_label_ex, dtype=np.float32)
        st_label = np.array(st_label_ex, dtype=np.float32)
        th_label = np.array(th_label_ex, dtype=np.float32)

        m_label3 = np.array(m_label3, dtype=np.float32)
        th_label3 = np.array(th_label3, dtype=np.float32)

        # reflect_list = np.concatenate([reflect_list, m_label2, th_label2], axis=1)
        # reflect_list = np.concatenate((reflect_list, m_label[:, np.newaxis], th_label[:, np.newaxis]), axis=1)
        reflect_list = np.array(reflect_list_ex, dtype=np.float32)

        # m_label1 = np.array(m_label1, dtype=np.float32)
        # tb_label1 = np.array(tb_label1, dtype=np.float32)
        # st_label1 = np.array(st_label1, dtype=np.float32)
        # th_label1 = np.array(th_label1, dtype=np.float32)

        return reflect_list, m_label, tb_label, st_label, th_label, m_label1, tb_label1, st_label1, th_label1



class ViatalSignDataset_fc_regression(data.Dataset):
    def __init__(self, mode='train', cl='', use_gpu = False):
        self.mode = mode
        self.cl = cl
        self.use_gpu = use_gpu

        if mode == "train":
            reflect_list, m_label, tb_label, st_label, th_label = self.read_vitalsign_dataset_regression(name='train')
        else:
            reflect_list, m_label, tb_label, st_label, th_label = self.read_vitalsign_dataset_regression(name='test')

        if self.use_gpu == True:
            self.ref_list = torch.FloatTensor(reflect_list).to('cuda')
            self.m_label = torch.FloatTensor(m_label).to('cuda')
            self.tb_label = torch.FloatTensor(tb_label).to('cuda')
            self.st_label = torch.FloatTensor(st_label).to('cuda')
            self.th_label = torch.FloatTensor(th_label).to('cuda')
        else:
            self.ref_list = torch.FloatTensor(reflect_list)
            self.m_label = torch.FloatTensor(m_label)
            self.tb_label = torch.FloatTensor(tb_label)
            self.st_label = torch.FloatTensor(st_label)
            self.th_label = torch.FloatTensor(th_label)

    def __getitem__(self, index):
        anchor = self.ref_list[index]
        ml = self.m_label[index]
        tbl = self.tb_label[index]
        stl = self.st_label[index]
        thl = self.th_label[index]

        return anchor, ml, tbl, stl, thl

    def __len__(self):
        return len(self.ref_list)

    def read_vitalsign_dataset_regression(self, name):
        '''
        Read Data
        '''
        path = os.path.dirname(__file__)

        fileNameList = ['input_data1.npy', 'input_data2.npy', 'input_data3.npy', 'input_data_random1.npy',
                        'input_data_random2.npy', 'input_data_random3.npy', 'input_data_random4.npy']

        test_fileNameList = ['input_data4.npy', 'input_data_random5.npy']


        if name == 'train':
            fileNameList = fileNameList
        else:
            fileNameList = test_fileNameList

        train_data = []

        for fn in fileNameList:
            temp_data = np.load(path + '/../input_data/{}'.format(fn), allow_pickle=True)

            if len(train_data) == 0 :
                train_data = temp_data
            else:
                train_data = np.concatenate([train_data, temp_data], axis=1)

        reflect_list = np.transpose(train_data[:, :, 5])

        out2 = train_data[0, :, 0:4]

        m_label = []
        tb_label = []
        st_label = []
        th_label = []

        for d_idx in range(len(out2)):
            if out2[d_idx][0] > 0 and out2[d_idx][0] <= 0.02:
                m_i = 0
            elif out2[d_idx][0] > 0.02 and out2[d_idx][0] <= 0.04:
                m_i = 1
            elif out2[d_idx][0] > 0.04 and out2[d_idx][0] <= 0.06:
                m_i = 2
            elif out2[d_idx][0] > 0.06 and out2[d_idx][0] <= 0.08:
                m_i = 3
            elif out2[d_idx][0] > 0.08 and out2[d_idx][0] <= 0.1:
                m_i = 4
            elif out2[d_idx][0] > 0.1 and out2[d_idx][0] <= 0.12:
                m_i = 5
            elif out2[d_idx][0] > 0.12 and out2[d_idx][0] <= 0.14:
                m_i = 6
            elif out2[d_idx][0] > 0.14:
                m_i = 7

            m_label.append(m_i)

            if out2[d_idx][1] > 0 and out2[d_idx][1] <= 0.01:
                tb_i = 0
            elif out2[d_idx][1] > 0.01 and out2[d_idx][1] <= 0.02:
                tb_i = 1
            elif out2[d_idx][1] > 0.02 and out2[d_idx][1] <= 0.03:
                tb_i = 2
            elif out2[d_idx][1] > 0.03 and out2[d_idx][1] <= 0.04:
                tb_i = 3
            elif out2[d_idx][1] > 0.04 and out2[d_idx][1] <= 0.05:
                tb_i = 4
            elif out2[d_idx][1] > 0.05 and out2[d_idx][1] <= 0.06:
                tb_i = 5
            elif out2[d_idx][1] > 0.06 and out2[d_idx][1] <= 0.07:
                tb_i = 6

            tb_label.append(tb_i)
            st_label.append(out2[d_idx][2])
            if out2[d_idx][3] <= 0.03:
                th_i = 0
            elif out2[d_idx][3] > 0.03 and out2[d_idx][3] <= 0.04:
                th_i = 1
            elif out2[d_idx][3] > 0.04 and out2[d_idx][3] <= 0.05:
                th_i = 2
            elif out2[d_idx][3] > 0.05 and out2[d_idx][3] <= 0.06:
                th_i = 3
            elif out2[d_idx][3] > 0.06:
                th_i = 4

            th_label.append(th_i)

        reflect_list = np.concatenate((reflect_list, np.array(m_label)[:, np.newaxis], np.array(tb_label)[:, np.newaxis], np.array(th_label)[:, np.newaxis]), axis=1)

        reflect_list = np.array(reflect_list, dtype=np.float32)
        m_label = np.array(m_label, dtype=np.float32)
        tb_label = np.array(tb_label, dtype=np.float32)
        st_label = np.array(st_label, dtype=np.float32)
        th_label = np.array(th_label, dtype=np.float32)


        return reflect_list, m_label, tb_label, st_label, th_label



class ViatalSignDataset_regression_mel_thickness_lasso():
    def __init__(self, name = 'train'):
        reflect_list, m_label, tb_label, st_label, th_label = self.read_vitalsign_dataset_regression(name=name)

        self.ref_list = reflect_list
        self.m_label = m_label
        self.th_label = th_label

    def read_vitalsign_dataset_regression(self, name):
        '''
        Read Data
        '''
        path = os.path.dirname(__file__)

        fileNameList = ['input_data9.npy', 'input_data14.npy', 'input_data15.npy',\
                        'input_data_random9.npy', 'input_data_random23.npy', 'input_data_random21.npy','input_data_random22.npy', 'input_data_random24.npy',\
                        'input_data_random10.npy', 'input_data_random11.npy', 'input_data_random12.npy']

        test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']


        if name == 'train':
            fileNameList = fileNameList
        else:
            fileNameList = test_fileNameList

        train_data = []

        for fn in fileNameList:
            temp_data = np.load(path + '/input_data/{}'.format(fn), allow_pickle=True)

            if len(train_data) == 0 :
                train_data = temp_data
            else:
                train_data = np.concatenate([train_data, temp_data], axis=1)

        reflect_list = np.transpose(train_data[:, :, 5])

        out2 = train_data[0, :, 0:4]

        m_label = []
        tb_label = []
        st_label = []
        th_label = []

        for d_idx in range(len(out2)):
            m_label.append(out2[d_idx][0])
            tb_label.append(out2[d_idx][1])
            st_label.append(out2[d_idx][2])
            th_label.append(out2[d_idx][3])

        m_label1 = []
        tb_label1 = []
        st_label1 = []
        th_label1 = []

        m_label2 = []
        tb_label2 = []
        st_label2 = []
        th_label2 = []

        m_label3 = []
        th_label3 = []

        for d_idx in range(len(out2)):
            if out2[d_idx][0] > 0 and out2[d_idx][0] <= 0.02:
                m_i2 = [1, 0, 0, 0, 0, 0, 0, 0]
                m_i = 0
            elif out2[d_idx][0] > 0.02 and out2[d_idx][0] <= 0.04:
                m_i2 = [0, 1, 0, 0, 0, 0, 0, 0]
                m_i = 1
            elif out2[d_idx][0] > 0.04 and out2[d_idx][0] <= 0.06:
                m_i2 = [0, 0, 1, 0, 0, 0, 0, 0]
                m_i = 2
            elif out2[d_idx][0] > 0.06 and out2[d_idx][0] <= 0.08:
                m_i2 = [0, 0, 0, 1, 0, 0, 0, 0]
                m_i = 3
            elif out2[d_idx][0] > 0.08 and out2[d_idx][0] <= 0.1:
                m_i2 = [0, 0, 0, 0, 1, 0, 0, 0]
                m_i = 4
            elif out2[d_idx][0] > 0.1 and out2[d_idx][0] <= 0.12:
                m_i2 = [0, 0, 0, 0, 0, 1, 0, 0]
                m_i = 5
            elif out2[d_idx][0] > 0.12 and out2[d_idx][0] <= 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 1, 0]
                m_i = 6
            elif out2[d_idx][0] > 0.14:
                m_i2 = [0, 0, 0, 0, 0, 0, 0, 1]
                m_i = 7

            g_p = []
            g_sig = 0.01
            # g_sig = 0.02

            g_p.append(np.exp(-(out2[d_idx][0] - 0.01) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.03) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.05) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.07) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.09) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.11) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.13) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][0] - 0.15) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))

            m_i3 = []
            for gp_i in range(len(g_p)):
                m_i3.append(g_p[gp_i] / np.sum(g_p))

            m_label1.append(m_i)
            m_label2.append(m_i2)
            m_label3.append(m_i3)

            if out2[d_idx][1] > 0 and out2[d_idx][1] <= 0.01:
                tb_i2 = [1, 0, 0, 0, 0, 0, 0]
                tb_i = 0
            elif out2[d_idx][1] > 0.01 and out2[d_idx][1] <= 0.02:
                tb_i2 = [0, 1, 0, 0, 0, 0, 0]
                tb_i = 1
            elif out2[d_idx][1] > 0.02 and out2[d_idx][1] <= 0.03:
                tb_i2 = [0, 0, 1, 0, 0, 0, 0]
                tb_i = 2
            elif out2[d_idx][1] > 0.03 and out2[d_idx][1] <= 0.04:
                tb_i2 = [0, 0, 0, 1, 0, 0, 0]
                tb_i = 3
            elif out2[d_idx][1] > 0.04 and out2[d_idx][1] <= 0.05:
                tb_i2 = [0, 0, 0, 0, 1, 0, 0]
                tb_i = 4
            elif out2[d_idx][1] > 0.05 and out2[d_idx][1] <= 0.06:
                tb_i2 = [0, 0, 0, 0, 0, 1, 0]
                tb_i = 5
            elif out2[d_idx][1] > 0.06 and out2[d_idx][1] <= 0.07:
                tb_i2 = [0, 0, 0, 0, 0, 0, 1]
                tb_i = 6

            tb_label1.append(tb_i)
            tb_label2.append(tb_i2)

            if out2[d_idx][2] >= 0 and out2[d_idx][2] <= 0.7:
                st_i2 = [1, 0, 0, 0, 0, 0, 0]
                st_i = 0
            elif out2[d_idx][2] > 0.7 and out2[d_idx][2] <= 0.75:
                st_i2 = [0, 1, 0, 0, 0, 0, 0]
                st_i = 1
            elif out2[d_idx][2] > 0.75 and out2[d_idx][2] <= 0.8:
                st_i2 = [0, 0, 1, 0, 0, 0, 0]
                st_i = 2
            elif out2[d_idx][2] > 0.8 and out2[d_idx][2] <= 0.85:
                st_i2 = [0, 0, 0, 1, 0, 0, 0]
                st_i = 3
            elif out2[d_idx][2] > 0.85 and out2[d_idx][2] <= 0.9:
                st_i2 = [0, 0, 0, 0, 1, 0, 0]
                st_i = 4
            elif out2[d_idx][2] > 0.9 and out2[d_idx][2] <= 0.95:
                st_i2 = [0, 0, 0, 0, 0, 1, 0]
                st_i = 5
            elif out2[d_idx][2] > 0.95 and out2[d_idx][2] <= 1.0:
                st_i2 = [0, 0, 0, 0, 0, 0, 1]
                st_i = 6

            st_label1.append(st_i)
            st_label2.append(st_i2)

            if out2[d_idx][3] <= 0.035:
                th_i2 = [1, 0, 0]
                th_i = 0
            elif out2[d_idx][3] > 0.035 and out2[d_idx][3] <= 0.055:
                th_i2 = [0, 1, 0]
                th_i = 1
            elif out2[d_idx][3] > 0.055 and out2[d_idx][3] <= 0.075:
                th_i2 = [0, 0, 1]
                th_i = 2

            g_p = []
            g_sig = 0.01
            # g_sig = 0.02

            g_p.append(np.exp(-(out2[d_idx][3] - 0.025) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][3] - 0.045) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))
            g_p.append(np.exp(-(out2[d_idx][3] - 0.065) ** 2 / (2 * g_sig ** 2)) / (np.sqrt(2 * np.pi * g_sig ** 2)))

            th_i3 = []
            for gp_i in range(len(g_p)):
                th_i3.append(g_p[gp_i] / np.sum(g_p))

            th_label1.append(th_i)
            th_label2.append(th_i2)
            th_label3.append(th_i3)

        reflect_list_ex = []
        m_label_ex = []
        tb_label_ex = []
        st_label_ex = []
        th_label_ex = []

        for i in range(len(reflect_list)):
            temp_ref = []

            temp_ref.append(reflect_list[i][9])
            temp_ref.append(reflect_list[i][12])
            temp_ref.append(reflect_list[i][16])
            temp_ref.append(reflect_list[i][19])
            temp_ref.append(reflect_list[i][25])  # temp_ref.append(reflect_list[i][25])
            temp_ref.append(reflect_list[i][27])
            temp_ref.append(reflect_list[i][29])
            temp_ref.append(reflect_list[i][31])
            temp_ref.append(reflect_list[i][34])
            temp_ref.append(reflect_list[i][35])
            temp_ref.append(reflect_list[i][44])
            temp_ref.append((reflect_list[i][46] + reflect_list[i][47]) / 2)
            temp_ref.append(reflect_list[i][48])
            temp_ref.append(reflect_list[i][49])

            # shading, k = calculate_k(-(np.log(reflect_list[i][9])),
            #                          -(np.log(reflect_list[i][29])),
            #                          -(np.log(reflect_list[i][49])))
            #
            # temp_ref.append(-(np.log(reflect_list[i][9])) - k)
            # temp_ref.append(-(np.log(reflect_list[i][12])) - k)
            # temp_ref.append(-(np.log(reflect_list[i][16])) - k)
            # temp_ref.append(-(np.log(reflect_list[i][19])) - k)
            # temp_ref.append(-(np.log(reflect_list[i][25])) - k)
            # temp_ref.append(-(np.log(reflect_list[i][27])) - k)
            # temp_ref.append(-(np.log(reflect_list[i][29])) - k)
            # temp_ref.append(-(np.log(reflect_list[i][31])) - k)
            # temp_ref.append(-(np.log(reflect_list[i][34])) - k)
            # temp_ref.append(-(np.log(reflect_list[i][35])) - k)
            # temp_ref.append(-(np.log(reflect_list[i][44])) - k)
            # temp_ref.append(-(np.log((reflect_list[i][46] + reflect_list[i][47]) / 2)) - k)
            # temp_ref.append(-(np.log(reflect_list[i][48])) - k)
            # temp_ref.append(-(np.log(reflect_list[i][49])) - k)

            reflect_list_ex.append(temp_ref)

            m_label_ex.append(m_label[i])
            tb_label_ex.append(tb_label[i])
            st_label_ex.append(st_label[i])
            th_label_ex.append(th_label[i])

        m_label = np.array(m_label_ex, dtype=np.float32)
        tb_label = np.array(tb_label_ex, dtype=np.float32)
        st_label = np.array(st_label_ex, dtype=np.float32)
        th_label = np.array(th_label_ex, dtype=np.float32)

        reflect_list = np.array(reflect_list_ex, dtype=np.float32)

        return reflect_list, m_label, tb_label, st_label, th_label




