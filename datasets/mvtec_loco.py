from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms as T
import numpy as np

class mvtec_loco(Dataset):
    def __init__(self, img_dir, train, resize, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.train = train
        self.resize = resize
        self.transform = transform
        self.target_transform = target_transform
        self.targets_train = None
        self.targets_test = None
        self.mapping_train = [
            351,
            686,
            1058,
            1418,
            1778
            ]
        self.mapping_test = [
            102,
            196,
            334,
            456,
            575
            ]

    def __len__(self):
        if self.train is True:
            return 1778
        else:
            return 575

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
            targets = np.zeros(1778)
            for i in range(1778):
                _, target = self.__getitem__(i)
                targets[i] = target
            self.targets_train = targets
            return targets
        else:
            if self.targets_test is not None:
                return self.targets_test
            targets = np.zeros(575)
            for i in range(575):
                _, target = self.__getitem__(i)
                targets[i] = target
            self.targets_test = targets
            return targets