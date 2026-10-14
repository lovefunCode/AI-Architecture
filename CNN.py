#  CNN is a neural network designed to extract local patterns from data, especially images

#  a typical CNN has 4 main operations:
#  Convolution(conv2d) : detect lcoal features
#  Activation(ReLU): introduce norlinearity
#  Pooling(MaxPooling): reduces spatial dimensions
#  Linear(Linear) : uses extracted features to make predictions

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Convolution layer
        self.conv1d = nn.Conv2d(
            in_channels = 1,
            out_channels = 16,
            kernel_size = 3,
            padding = 1
        )

        self.conv2d = nn.Conv2d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = 3,
            padding = 1
        )

        # 2. Pooling Layer
        self.pool = nn.MaxPooling2d(kernel_size=2, stride = 2)

        #  3 fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #  first convolution block
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        #  second convolution block
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten features maps --> vector
        x = torch.flatten(x, start_dim = 1)

        #  classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

        

