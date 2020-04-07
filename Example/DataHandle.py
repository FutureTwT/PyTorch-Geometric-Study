import torch
from torch_geometric.data import Data

edge_index = torch.tensor([
    [3, 1, 1, 2],
    [1, 3, 2, 1]],dtype=torch.long)
# 注意x是二维的，不是一维的，每一行代表一个节点的特征向量，此处特征维度为1
x = torch.tensor([[-1],
                  [0],
                  [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(data)
'''
# 通过节点对的方式给出
edge_index = torch.tensor([
    [0, 1], [1, 0], [1, 2], [2, 1]
], dtype=torch.long)
data = Data(x=x, edge_index=edge_index.t().contiguous())
print(data)
'''
# 输出data的属性关键字，只有传递参数的才会被输出
print(data.keys)
# ['x', 'edge_index']

# 按照关键字进行输出，注意是字符串
print(data['x'])
# tensor([[-1.],
#         [ 0.],
#         [ 1.]])
print(data['edge_index'])
# tensor([[0, 1, 1, 2],
#         [1, 0, 2, 1]])

print('edge_attr: ', data['edge_attr'])
# edge_attr:  None

# 遍历所有关键字及其对应的数值
for key, item in data:
    print(key, '---', item)

# 可以直接检索key，也可以检索data内函数
if 'edge_attr' not in data.keys:
    print('Not in')
    # Not in

if 'x' in data:
    print('In')
    # In

# print(type(data.keys))
# <class 'list'>

print(data.num_nodes)
# 3

# 这里的边数为4
print(data.num_edges)
# 4

print(data.num_edge_features)
# 0

print(data.num_node_features)
# 1

print(data.contains_isolated_nodes())
# False

print(data.contains_self_loops())
# False

print(data.is_undirected())
# True
