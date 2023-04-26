import numpy as np
import torch
import torch.nn as nn

# Backbone
from classifier.backbone import Conv4, ResNet18

# Classifier
from classifier.backbone import distLinear, Linear_fw

class Conv4Cos(nn.Module):
    def __init__(self, input_shape, n_classes):
        super().__init__()
        assert len(input_shape) == 3
        self.backbone = Conv4(input_shape[0])
        # Probe the output shape of the backbone
        self.backbone.eval()
        with torch.no_grad():
            out = self.backbone(torch.zeros(1, *input_shape))
            linaer_in = np.prod(out.shape[1:])
        self.fc = distLinear(linaer_in, n_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
