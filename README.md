# 2D Elasticity GNN

Predict nodal displacements and stresses in a 2D elastic plate using a Graph Neural Network.  
**Status:** In progress – prototype achieves ~10% MSE reduction in 50 epochs on initial dataset.

## Motivation

Finite‐element simulations (FEM) are widespread in engineering but can be computationally expensive when repeated under varying loads or geometries. Inspired by MeshGraphNets (Pfaff et al., ICLR 2021), which learn mesh‐based physical simulations with graph networks, this project aims to train a GNN to approximate FEM results for 2D linear elasticity. By learning nodal displacements and stress fields, a GNN surrogate can drastically reduce online computation once trained.

**Key reference:**  
> Pfaff, T., et al. “Learning Mesh‐Based Simulations with Graph Networks.” *ICLR 2021*.

## Features

- **Data Generation (FEniCS)**  
  - `data_generation/hello_elasticity.py`: solves a 2D linear‐elastic PDE on a unit‐square mesh.  
  - Exports `.npz` samples containing node coordinates, displacements, stress tensors, and boundary flags.

- **Data Conversion (PyTorch Geometric)**  
  - `data_conversion/convert_to_pyg.py` + `data_conversion/dataset.py`: load `.npz`, build `Data(x, edge_index, y)`, and collate into an `InMemoryDataset`.

- **Model & Training**  
  - `models/gnns.py`: two‐layer GraphSAGE + MLP.  
  - `models/train.py`: trains the GNN end‐to‐end on initial dataset; MSE ↓ ~10% in 50 epochs.  
  - Saved model checkpoint: `models/simple_gnn.pt`.

