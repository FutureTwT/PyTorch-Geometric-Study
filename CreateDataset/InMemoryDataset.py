import torch
from torch_geometric.data import InMemoryDataset

class MyDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        # 数据的下载和处理过程在父类中调用实现
        super(MyDataset, self).__init__(root, transform, pre_transform)
        # 加载数据
        self.data, self.slices = torch.load(self.processed_paths[0])

    # 将函数修饰为类属性
    @property
    def raw_file_names(self):
        return ['file_1', 'file_2']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # download to self.raw_dir
        pass

    def process(self):
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # 这里的save方式以及路径需要对应构造函数中的load操作
        torch.save((data, slices), self.processed_paths[0])