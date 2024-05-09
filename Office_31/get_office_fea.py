import torchvision.transforms as transforms
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def re_label(data_, target_=False):
    new_dt = []
    new_lb = []
    if target_:
        for i in range(len(data_['labels'][0])):
            if data_['labels'][0][i] < 10:
                new_dt.append(data_['fea'][i])
                new_lb.append(data_['labels'][0][i])
            elif data_['labels'][0][i] >= 20:
                new_dt.append(data_['fea'][i])
                new_lb.append(10)
    else:
        for i in range(len(data_['labels'][0])):
            if data_['labels'][0][i] < 10:
                new_dt.append(data_['fea'][i])
                new_lb.append(data_['labels'][0][i])
    print("length of data after relabeling:{}".format(len(new_dt), len(new_lb)))
    return np.array(new_dt), np.array(new_lb)


class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.data = sio.loadmat(data_dir)

    def __len__(self):
        return len(self.data['labels'][0])

    def __getitem__(self, index):
        return self.data['fea'][index], self.data['labels'][0][index]


class MyDatasetRelabel(Dataset):
    def __init__(self, data_dir, target_=False):
        self.data = sio.loadmat(data_dir)
        self.data = re_label(self.data, target_=target_)

    def __len__(self):
        return len(self.data[1])

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index]


def get_dataset(root, source, target):
    source_data = MyDatasetRelabel(root + source)
    target_data = MyDatasetRelabel(root + target, target_=True)
    return source_data, target_data


if __name__ == "__main__":
    data_dir = '/data/bzhang3/dataset/fea_office31_vgg16/VGG16_Office31_webcam.mat'
    dataset1 = MyDatasetRelabel(data_dir, target_=True)
    # dt_load = DataLoader(dataset1, 16, shuffle=True, num_workers=1)
    # for th, (dt, label) in enumerate(dt_load):
    #     print(th, label)
    # print(dt_load)
