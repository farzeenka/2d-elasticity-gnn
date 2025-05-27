import os
import numpy as np
import torch
import torch_geometric.data.data
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse

# Allow unpickling of DataEdgeAttr
import torch.serialization
torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])

class FEMDataset(InMemoryDataset):
    def __init__(self, root, nx=20, ny=20, transform=None, pre_transform=None):
        self.nx, self.ny = nx, ny
        super().__init__(root, transform, pre_transform)
        # Load the processed data with full unpickling
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return sorted(f for f in os.listdir(self.raw_dir) if f.endswith('.npz'))

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass  # no external download

    def process(self):
        data_list = []
        for raw_path in self.raw_paths:
            npz = np.load(raw_path)
            coords  = npz["coords"]          # (N,2)
            disp    = npz["disp"]            # (N,2)
            stress  = npz["stress"].reshape(len(coords), -1)  # (N,4)
            bc_mask = npz["bc_mask"].reshape(-1,1)            # (N,1)

            # Node features: x, y, bc_flag
            x = torch.from_numpy(np.hstack([coords, bc_mask])).float()

            # Targets: displacement + flattened stress
            y_disp   = torch.from_numpy(disp).float()         # (N,2)
            y_stress = torch.from_numpy(stress).float()       # (N,4)
            y = torch.cat([y_disp, y_stress], dim=1)          # (N,6)

            # Build 4-neighbor mesh connectivity
            N = len(coords)
            idx = np.arange(N).reshape((self.ny+1, self.nx+1))
            adj = np.zeros((N, N), dtype=int)
            for i in range(self.ny+1):
                for j in range(self.nx+1):
                    u = idx[i,j]
                    for di, dj in [(1,0),(0,1),(-1,0),(0,-1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni <= self.ny and 0 <= nj <= self.nx:
                            v = idx[ni,nj]
                            adj[u,v] = 1
            edge_index = dense_to_sparse(torch.from_numpy(adj))[0]

            data_list.append(Data(x=x, edge_index=edge_index, y=y))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
