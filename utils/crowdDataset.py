import numpy as np
import torch

class crowdDataset(torch.utils.data.Dataset):
    """The customized dataset class for dataloader in torch.
    """
    def __init__(self, rawdataset, device="cpu"):
        """

        Args:
            rawdataset (ndarray): the dataset output by the vadereOutputLoader, where the velocity is the 3rd column.
            device (str, optional): the device torch will use. Defaults to "cpu".
        """

        numOfCols = rawdataset.shape[1]
        mask = np.repeat(False, numOfCols)
        mask[0] = True # velocity is in the column 0
        
        self.v_data = torch.tensor(rawdataset[:, mask].reshape(-1,1), dtype=torch.float32).to(device)
        self.X_data = torch.tensor(rawdataset[:, ~mask], dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return (self.X_data[idx, :], self.v_data[idx, :])