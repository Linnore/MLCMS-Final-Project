import numpy as np
import torch

class crowdDataset(torch.utils.data.Dataset):
    def __init__(self, rawdataset, device="cpu"):

        numOfCols = rawdataset.shape[1]
        mask = np.repeat(False, numOfCols)
        mask[2] = True
        
        self.v_data = torch.tensor(rawdataset[:, mask], dtype=torch.float32).to(device)
        self.X_data = torch.tensor(rawdataset[:, ~mask].reshape(-1,1), dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return (self.X_data[idx], self.v_data[idx])