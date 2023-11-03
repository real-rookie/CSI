from abc import *
import torch.nn as nn

# of interest
class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, num_classes=10, simclr_dim=128):
        print("__init__(self, last_dim, num_classes=10, simclr_dim=128):")
        print(last_dim)
        print(num_classes)
        print(simclr_dim)
        super(BaseModel, self).__init__()
        self.last_dim = last_dim
        self.linear = nn.Linear(last_dim, num_classes)
        print("self.linear = nn.Linear(last_dim, num_classes)")
        print(last_dim)
        print(num_classes)
        self.simclr_layer = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, simclr_dim),
        )
        self.shift_cls_layer = nn.Linear(last_dim, 2)
        self.joint_distribution_layer = nn.Linear(last_dim, 4 * num_classes)

    @abstractmethod
    def penultimate(self, inputs, all_features=False):
        pass

    def forward(self, inputs, penultimate=False, simclr=False, shift=False, joint=False):
        print("base input shape")
        print(inputs.shape)
        _aux = {}
        _return_aux = False

        features = self.penultimate(inputs)
        print("features = self.penultimate(inputs)")
        print(features.shape)
        features = nn.Linear(features.shape[1], self.last_dim)(features)
        # (length, 512)

        output = self.linear(features)
        print("output = self.linear(features)")
        print(output.shape)
        # (length, number_classes)

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features
            print(_aux['penultimate'].shape)
            # (length, 512)

        if simclr:
            _return_aux = True
            _aux['simclr'] = self.simclr_layer(features)
            print(_aux['simclr'].shape)
            # (length, 128(simclr_dim))
            
        if shift:
            _return_aux = True
            _aux['shift'] = self.shift_cls_layer(features)
            print(_aux['shift'].shape)
            # (length, 4) it is 4 because of K

        if joint:
            _return_aux = True
            _aux['joint'] = self.joint_distribution_layer(features)

        if _return_aux:
            return output, _aux
        print(output.shape)
        return output
