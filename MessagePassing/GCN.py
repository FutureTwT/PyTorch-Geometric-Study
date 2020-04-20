import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # X: [N, in_channels]
        # edge_index: [2, E]

        # 1.在邻接矩阵中增加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 2.对节点特征进行一个非线性转换
        # x的维度会由[N, in_channels]转换为[N, out_channels]
        x = self.lin(x)

        # 3.计算标准化系数
        # edge_index的第一个向量作为行坐标，第二个向量作为列坐标
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-1/2)
        # norm的第一个元素就是edge_index中的第一列（第一条边）上的标准化系数
        # tensor的乘法为对应元素乘法，tensor1[tensor2]后的维度与tensor2一致
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 4-6步的开始标志，内部实现了message-AGGREGATE-update
        return self.propagate(edge_index, size=(x.size(0), x.size(1)), x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j的维度为[E, out_channels]
        print(x_j)
        # 4.进行传递消息的构造，将标准化系数乘以邻域节点的特征信息得到传递信息
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out的维度为[N, out_channels]

        # 6.更新新的节点嵌入，这里没有做任何多余的映射过程
        return aggr_out

# 实例化对象
conv = GCNConv(3, 3)
# 构建数据
edge_index = torch.tensor([
    [0, 1, 1, 2],
    [1, 0, 2, 1]
], dtype=torch.long)
x = torch.tensor([
    [0, 0, 0],
    [1, 1, 1],
    [2, 2, 2]
], dtype=torch.float)

# 默认为调用对象的forward函数
x = conv(x, edge_index)





