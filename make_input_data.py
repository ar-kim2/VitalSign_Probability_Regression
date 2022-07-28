import numpy as np
import os

path = os.path.dirname(__file__)

# test_fileNameList = ['input_data13.npy', 'input_data_random20.npy']

temp_data = np.load(path + '/input_data/input_data_random20.npy', allow_pickle=True)

# print("check temp data shape : ", np.shape(temp_data))

for i in range(64):
    for j in range(6000):
        random_noise = np.random.rand() * 0.03
        temp_data[i][j][5] = temp_data[i][j][5] + random_noise

np.save(path + '/input_data/input_data_random20_3.npy', temp_data)

