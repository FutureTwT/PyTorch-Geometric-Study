# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/agnn.py
# 代码注释

import os.path as osp

import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import AGNNConv

dataset = 'Cora'

# osp.realpath(__file__) 输出当前代码文件的绝对路径
# .. 访问到当前代码目录的上一层目录
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)

# path为root路径，dataset为name路径，path+dataset才是数据的路径（processed所在的路径）
dataset = Planetoid(path, dataset, T.NormalizeFeatures())

# dataset为可迭代对象，每一个元素都是一个Data实例
# 所有的Cora数据内容都在第一个data中存放，因为只有一个图
data = dataset[0]
# print(len(dataset)) # 1
print(type(data))

# 定义网络结构
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 初始化一些函数接口
        self.lin1 = torch.nn.Linear(dataset.num_features, 16)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        # 因为使用的是节点分类数据集，所以需要映射一个全类别概率向量
        self.lin2 = torch.nn.Linear(16, dataset.num_classes)

    # 官方代码中没有传递data参数，而是直接在forward中调用data，将data作为全局变量的方式进行使用
    def forward(self, data:torch_geometric.data.data.Data):
        # dropout默认的training状态是false，而且此处仅仅是调用了一个外部函数F.dropout，即使内部training状态发生改变，
        # 也不会影响dropout，所以需要将模型的training状态传递给dropout
        x = F.dropout(data.x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, data.edge_index)
        x = self.prop2(x, data.edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        #
        return F.log_softmax(x, dim=1)

# 选择GPU/CPU设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Net()相当于创建了一个Net实例化对象，然后调用父类Module的函数
model, data = Net().to(device), data.to(device)
# 构造优化器，指定优化目标（也就是模型的参数），学习率和衰减参数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 定义训练函数
def train():
    # 1.启动训练
    model.train()
    # 2.优化器梯度初始化
    optimizer.zero_grad()
    # 3.反向传播过程
    # 4.调用优化器
    optimizer.step()


# 定义测试函数