# 2D Elasticity GNN

Predict nodal displacements and stresses in a 2D elastic plate using a Graph Neural Network.

> **Status:** In progress – prototype achieves ~10% MSE reduction in 50 epochs on initial dataset.

## Features

- **Data Generation**: FEniCS script (`data_generation/hello_elasticity.py`) solves linear-elastic PDE and exports `.npz` samples.  
- **Data Conversion**: PyTorch-Geometric script (`data_conversion/convert_to_pyg.py`) + `FEMDataset` turn `.npz` into `Data(x, edge_index, y)`.  
- **Model & Training**: `models/gnns.py` defines a two-layer GraphSAGE; `models/train.py` trains it end-to-end.
