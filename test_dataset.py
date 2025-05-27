from data_conversion.dataset import FEMDataset

if __name__ == "__main__":
    # Assumes your .npz files live in data/raw/
    ds = FEMDataset(root="data", nx=20, ny=20)
    print("Number of samples:", len(ds))
    print("First sample:", ds[0])
git init