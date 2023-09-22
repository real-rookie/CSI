from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms as T
import numpy as np

class MVTec_AD(Dataset):
    def __init__(self, img_dir, train, resize, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.train = train
        self.resize = resize
        self.transform = transform
        self.target_transform = target_transform
        self.targets_train = None
        self.targets_test = None
        self.mapping_train = [280,
            544,
            789,
            1019,
            1266,
            1475,
            1699,
            1918,
            2309,
            2529,
            2796,
            3116,
            3176,
            3389,
            3629,
            ]
        self.mapping_test = [28,
            49,
            81,
            114,
            133,
            153,
            211,
            234,
            274,
            296,
            322,
            363,
            375,
            435,
            467,
            ]

    def __len__(self):
        if self.train is True:
            return 3629
        else:
            return 467

    def __getitem__(self, idx):
        if self.train is True:
            folder = "train"
            mapping = self.mapping_train
        else:
            folder = "test"
            mapping = self.mapping_test

        cls = None
        for i, val in enumerate(mapping):
            if idx < val:
                cls = i
                break
        if cls != 0:
            image = read_image(f"{self.img_dir}/{folder}/{cls}/{cls}_{idx - mapping[cls-1]}.png")
        else:
            image = read_image(f"{self.img_dir}/{folder}/{cls}/{cls}_{idx}.png")
        image = T.Resize([self.resize, self.resize])(image)
        image = T.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            cls = self.target_transform(cls)
        return image, cls
    
    @property
    def targets(self):
        if self.train is True:
            if self.targets_train is not None:
                return self.targets_train
            targets = np.zeros(3629)
            for i in range(3629):
                _, target = self.__getitem__(i)
                targets[i] = target
            self.targets_train = targets
            return targets
        else:
            if self.targets_test is not None:
                return self.targets_test
            targets = np.zeros(467)
            for i in range(467):
                _, target = self.__getitem__(i)
                targets[i] = target
            self.targets_test = targets
            return targets