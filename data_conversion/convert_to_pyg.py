# convert_to_pyg.py

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

def load_sample(npz_path):
    data = np.load(npz_path)
    coords  = data['coords']    # (N,2)
    disp    = data['disp']      # (N,2)
    stress  = data['stress']    # (N,2,2)
    bc_mask = data['bc_mask']   # (N,)
    return coords, disp, stress, bc_mask

def build_edge_index(nx, ny):
    """
    Reconstruct the unit-square triangulation connectivity.
    The mesh was created via UnitSquareMesh(nx, ny), 
    which divides the unit square into nx*ny rectangles, each split into 2 triangles.
    We'll build the adjacency matrix of the resulting graph.
    """
    N = (nx + 1)*(ny + 1)
    # Compute grid indices
    idx = np.arange(N).reshape((ny+1, nx+1))
    # Build adjacency via 4-connectivity (you can also use 8-connectivity or actual triangle edges)
    adj = np.zeros((N, N), dtype=int)
    for i in range(ny+1):
        for j in range(nx+1):
            u = idx[i,j]
            for di, dj in [(1,0), (0,1), (-1,0), (0,-1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni <= ny and 0 <= nj <= nx:
                    v = idx[ni,nj]
                    adj[u,v] = 1
    # Convert to edge_index
    edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
    return edge_index

def make_pyg_data(npz_path, nx=20, ny=20):
    coords, disp, stress, bc_mask = load_sample(npz_path)
    N = coords.shape[0]

    # Node features: x, y, BC_flag
    x = torch.from_numpy(np.hstack([coords, bc_mask.reshape(-1,1)])).float()  # (N,3)

    # Targets: displacement + flattened stress
    y_disp   = torch.from_numpy(disp).float()                                # (N,2)
    y_stress = torch.from_numpy(stress.reshape(N, -1)).float()               # (N,4)
    y = torch.cat([y_disp, y_stress], dim=1)                                 # (N,6)

    # Edge index
    edge_index = build_edge_index(nx, ny)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

if __name__ == "__main__":
    data = make_pyg_data("sample0.npz", nx=20, ny=20)
    print(data)
    # e.g. Data(x=[441,3], edge_index=[2,?], y=[441,6])
