import torch
from torch_geometric.data import Data
from itertools import product

edge_index = torch.tensor([
    [3, 1, 1, 2],
    [1, 3, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1],
                  [0],
                  [1]], dtype=torch.float)

d = Data(x=x, edge_index=edge_index)
# print(type(d)) # # <class 'torch_geometric.data.data.Data'>

data_list = [0]
data_list[0] = d

keys = data_list[0].keys
# data->Data()
data = data_list[0].__class__()
# print(data_list[0].keys) # ['x', 'edge_index']
# print(type(data)) # <class 'torch_geometric.data.data.Data'>

for key in keys:
    data[key] = []
print(data) # Data(edge_index=[0], x=[0])

slices = {key: [0] for key in keys}
# print(slices) # {'x': [0], 'edge_index': [0]}

for item, key in product(data_list, keys):
    print(item, key)
    data[key].append(item[key])
    print(item[key])
    if torch.is_tensor(item[key]):
        s = slices[key][-1] + item[key].size(
            item.__cat_dim__(key, item[key]))
    else:
        s = slices[key][-1] + 1
    slices[key].append(s)

print(data)

if hasattr(data_list[0], '__num_nodes__'):
    data.__num_nodes__ = []
    for item in data_list:
        data.__num_nodes__.append(item.num_nodes)

for key in keys:
    item = data_list[0][key]
    if torch.is_tensor(item):
        data[key] = torch.cat(data[key],
                              dim=data.__cat_dim__(key, item))
    elif isinstance(item, int) or isinstance(item, float):
        data[key] = torch.tensor(data[key])

    slices[key] = torch.tensor(slices[key], dtype=torch.long)

