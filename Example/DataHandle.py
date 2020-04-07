import torch
from torch_geometric.data import Data

edge_index = torch.tensor([
    [0, 1, 1, 2],
    [1, 0, 2, 1]],dtype=torch.long)
# 注意x是二维的，不是一维的，每一行代表一个节点的特征向量，此处特征维度为1
x = torch.tensor([[-1],
                  [0],
                  [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(data)

# 通过节点对的方式给出
edge_index = torch.tensor([
    [0, 1], [1, 0], [1, 2], [2, 1]
], dtype=torch.long)
data = Data(x=x, edge_index=edge_index.t().contiguous())
print(data)