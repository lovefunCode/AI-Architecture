#  MLP: is a neural network composed of fully connected(linear) layers and usually nonlinear activation functions

# input x --> Linear layer z = wx + b  --> ReLU activation a = max(0, z)
# ouput a

import torch
import torch.nn as nn
# import torch.nn.functional as F

def relu(z):
    return torch.maximum(torch.zeros_like(z), z)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # combine all layer dimensions

        sizes = [input_size] + hidden_size + [output_size]

        print('sizes====', sizes)

        # self.layers = nn.ModuleList([
        #     nn.Linear(sizes[i], sizes[i+1])
        #     for i in range(len(sizes) - 1)
        # ])

        self.layers = nn.ModuleList([])
        for i in range(len(sizes) - 1):
            layer = nn.Linear(sizes[i], sizes[i+1])
            self.layers.append(layer)


        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = relu(self.fc1(x))
        # x = relu(self.fc2(x))
        # x = relu(self.fc3(x))

        for i, layer in enumerate(self.layers):
            x = layer(x)
            print('x---', x)
            # relu for hidden layers only
            if i < len(self.layers) - 1:
                x = relu(x)

        return x

model = MLP(input_size = 2, hidden_size = [16, 8], output_size = 2)

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

# Target class for each input
y = torch.tensor([0, 1, 1, 0])

# define loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.01
)


for epoch in range(1001):
    # forward pass
    logits = model(X)

    # calculate loss
    loss = criterion(logits, y)

    # clear old gradients
    optimizer.zero_grad()

    # compute new gradients
    loss.backward()

    # update parameters
    optimizer.step()

    # if epoch % 200 == 0:
        # print(f"Epoch {epoch}, Loss: {loss.item(): .4f}")
