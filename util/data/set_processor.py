import scipy.io as sio
import numpy as np
import os

SET_SPLIT = ['train', 'test']
#SET_DIM = {'cifar10': 4096}
#SET_DIM = {'cifar10': 3072}
SET_DIM = {'cifar10': 2048, 'nus-wide': 2048, 'ms-coco': 2048}
SET_LABEL = {'cifar10': 10, 'nus-wide': 21, 'ms-coco': 79}
SET_SIZE = {'cifar10': [50000, 10000], 'nus-wide': [50000,10000], 'ms-coco': [40000,8000]}


def cifar_processor(root_folder):
    class_num = 10

    def reader(file_name, part=SET_SPLIT[0]):
        data_mat = sio.loadmat(file_name)
        feat = data_mat[part + '_data']
        label = np.squeeze(data_mat[part + '_label'])
        fid = np.arange(0, feat.shape[0]) # convert to the one-hot vector
        label = np.eye(class_num)[label]

        return {'feat': feat, 'label': label, 'fid': fid}

    train_name = os.path.join(root_folder, 'cifar10_fc7_train.mat')
    train_dict = reader(train_name)
    test_name = os.path.join(root_folder, 'cifar10_fc7_test.mat')
    test_dict = reader(test_name, part=SET_SPLIT[1])

    return train_dict, test_dict


SET_PROCESSOR = {'cifar10': cifar_processor}
