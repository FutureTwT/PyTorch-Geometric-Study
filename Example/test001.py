### Q1: X维度和Y的维度不统一
import torch
from torch_geometric.data import Data

# 构建边
edge_index = torch.tensor([
    [3, 1, 1, 2],
    [1, 3, 2, 1]], dtype=torch.long)
# 构建X
x = torch.tensor([[-1],
                  [0],
                  [1],[2]], dtype=torch.float)
y = torch.tensor([[1], [2]], dtype=torch.float)
data = Data(x=x, y=y, edge_index=edge_index)

print(data)

### Q2: 手动加载数据集