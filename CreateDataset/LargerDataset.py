import os.path as osp

import torch
# 这里就不能用InMemoryDataset了
from torch_geometric.data import Dataset

class MyDataset(Dataset):
    # 默认预处理函数的参数都是None
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['file_1', 'file_2']

    @property
    def processed_file_names(self):
        # 一次无法加载所有数据，所以对数据进行了分解
        return ['data1.pt', 'data2.pt', 'data3.pt']

    def download(self):
        # Download to raw_dir
        pass

    def process(self):
        i = 0
        # 遍历每一个文件路径
        for raw_path in self.raw_paths:
            data = Data(...)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data{}.pt',format(idx)))
        return data