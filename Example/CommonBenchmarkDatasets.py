from torch_geometric.datasets import TUDataset
import torch

# 加载数据集，下载转换过程
dataset = TUDataset(root='data/', name='ENZYMES')
print(dataset)
# ENZYMES(600)
print(type(dataset))
# <class 'torch_geometric.datasets.tu_dataset.TUDataset'>
print(len(dataset))
# 600
print(dataset.num_node_features)
# 3

# dataset是一个可迭代对象，并且每一个元素都是一个Data实例，但是y是一个单独的元素，所以说这个数据集是Graph-level的
data = dataset[0]
print(data)
# Data(edge_index=[2, 168], x=[37, 3], y=[1])

# 数据集切分
dataset_train = dataset[:500]
dataset_test = dataset[500:]
print(dataset_train, dataset_test)
# ENZYMES(500) ENZYMES(100)
dataset_sample1 = dataset[torch.tensor([i for i in range(500)], dtype=torch.long)]
print(dataset_sample1)
# ENZYMES(500)
dataset_sample2 = dataset[torch.tensor([True, False])]
print(dataset_sample2)
# ENZYMES(1)
print(dataset[0])
# Data(edge_index=[2, 168], x=[37, 3], y=[1])
print(dataset[1])
# Data(edge_index=[2, 102], x=[23, 3], y=[1])
print(dataset_sample2[0])
# Data(edge_index=[2, 168], x=[37, 3], y=[1])

dataset = dataset.shuffle()
# 等价于
dataset = dataset[torch.randperm(len(dataset))]
