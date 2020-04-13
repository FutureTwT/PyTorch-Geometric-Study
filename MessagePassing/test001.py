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

row, col = edge_index
print(row)
# tensor([0, 1, 2, 0, 1, 2])

# row = torch.tensor([2, 2, 2, 2], dtype=torch.long) # degree: tensor([0., 0., 4.])
# 从节点编号开始，统计每一个编号的出现次数，如果没有出现的不会被空过去，而是记录为0
deg = degree(row)
print(deg)


