import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
class CustomDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor, transform: callable):
        super().__init__()
        assert data.size(0) == targets.size(0), "Size mismatch between tensors"
        self.data = data.unsqueeze(1).expand(-1, 3, -1, -1) # N C H W
        self.targets = targets.tolist()
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(T.ToPILImage()(self.data[index])), self.targets[index]

    def __len__(self):
        return len(self.targets)