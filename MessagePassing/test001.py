import torch
from torch_geometric.utils import add_self_loops, degree

# 节点编号是从1开始的，但是在后面增加自环的时候，是从0开始的
'''
edge_index = torch.tensor([
    [1, 2, 3],
    [2, 3, 2]
], dtype=torch.long)
'''
edge_index = torch.tensor([
    [0, 1, 2],
    [1, 2, 1]
], dtype=torch.long)

# 如果节点编号从1开始，那么不设置节点个数的话，则会增加[0,1,2,3]的自环；节点个数设置为3，增加[0,1,2]的自环
edge_index, _ = add_self_loops(edge_index, num_nodes=3)
print(edge_index)
# tensor([[0, 1, 2, 0, 1, 2],
#         [1, 2, 1, 0, 1, 2]])

# 分别取出所有边的第一个索引和第二个索引
row, col = edge_index
print(row)
# tensor([0, 1, 2, 0, 1, 2])

# 节点2出现了四次，所以0和1节点的度为0，而节点2的度为4。从节点编号开始，统计每一个编号的出现次数，如果没有出现的不会被空过去，而是记录为0
# row = torch.tensor([2, 2, 2, 2], dtype=torch.long) # degree: tensor([0., 0., 4.])

deg = degree(row)
# print('deg: ', deg)

row, col = edge_index
norm = deg[row] * deg[col]
print(norm)

# 转换为n行1列
norm_ = norm.view(-1, 1)
# print(norm_.shape)

# 特征矩阵，每一行代表一个元素的特征向量
x_j = torch.tensor([
    [1,2,3],
    [2,3,4],
    [1,2,3],
    [2,3,4],
    [1,2,3],
    [2,3,4]
])
# (n, 1) * (n * m)广播式乘法
print(norm_ * x_j)