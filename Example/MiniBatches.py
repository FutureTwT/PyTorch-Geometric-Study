from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean

dataset = TUDataset(root='data/', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in loader:
    print(data)
    # Batch(batch=[1005], edge_index=[2, 3948], x=[1005, 21], y=[32])
    x = scatter_mean(data.x, data.batch, dim=0)
    print(x.size())
    # torch.Size([32, 21])