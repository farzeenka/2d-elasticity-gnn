import torch
from torch_geometric.loader import DataLoader
from data_conversion.dataset import FEMDataset
from models.gnns import SimpleGNN

def train():
    # Hyperparameters
    nx, ny      = 20, 20
    batch_size  = 4
    hidden_dim  = 64
    lr          = 1e-3
    epochs      = 50

    # 1) Load dataset and DataLoader
    dataset = FEMDataset(root="data", nx=nx, ny=ny)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2) Instantiate model, optimizer, loss
    in_feats  = 3  # x, y, bc_flag
    out_feats = 6  # disp_x, disp_y, stress_xx, stress_xy, stress_yx, stress_yy
    model = SimpleGNN(in_feats, hidden_dim, out_feats)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # 3) Training loop
    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for batch in loader:
            opt.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} — Loss: {avg_loss:.6f}")

    # 4) Save model
    torch.save(model.state_dict(), "models/simple_gnn.pt")
    print("Training complete. Model saved to models/simple_gnn.pt")

if __name__ == "__main__":
    train()
