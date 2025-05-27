# 2D Elasticity GNN

Predict nodal displacements and stresses in a 2D elastic plate using a Graph Neural Network.

## Features

- **Data Generation**: FEniCS script (`data_generation/hello_elasticity.py`) solves linear-elastic PDE and exports `.npz` samples.
- **Data Conversion**: PyTorch-Geometric script (`data_conversion/convert_to_pyg.py`) + `FEMDataset` turn `.npz` into `Data(x, edge_index, y)`.
- **Model & Training**: `models/gnns.py` defines a two-layer GraphSAGE; `models/train.py` trains it end-to-end.

## Quickstart

```bash
git clone https://github.com/farzeenka/2d-elasticity-gnn.git
cd 2d-elasticity-gnn

# 1) Create environment
conda env create -f environment.yml
conda activate elasticity-gnn

# 2) Generate one sample
python data_generation/hello_elasticity.py

# 3) Convert to graph
python data_conversion/convert_to_pyg.py

# 4) Train prototype
python -m models.train
