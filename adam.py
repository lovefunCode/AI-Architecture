import torch
# from torch import nn
import torch.nn as nn

import matplotlib.pyplot as plt


def learning_rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup **(-1.5)))


class MyAdam:
    def __init__(self, params, model_size, warmup, beta1, beta2, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2

        self.params = list(params)

        self.model_size = model_size
        self.warmup = warmup
        self.eps = eps
        self.m = []
        self.v = []
        self.t = 0

        for i, p in enumerate(self.params):
            self.m.append(torch.zeros_like(p))
            self.v.append(torch.zeros_like(p))

        self.lr = learning_rate(self.t, self.model_size, 1, self.warmup)



    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


    def step(self):
        self.t += 1

        self.lr = learning_rate(self.t, self.model_size, 1, self.warmup)

        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * p.grad ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            with torch.no_grad():
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)



model = nn.Linear(1,1)
x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
])

y = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0],
])


optimizer = MyAdam(model.parameters(), model_size = 1, warmup = 10, beta1=0.9, beta2=0.98)

epochs = []
losses = []
learning_rates = []


for epoch in range(100):
    # forward
    prediction = model(x)

    loss = torch.mean((prediction-y) ** 2)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    # optimizer.zero_grad()

    epochs.append(epoch)
    losses.append(loss.item())
    learning_rates.append(optimizer.lr)

    if epoch % 10 == 0:
        print(
            f'epoch={epoch},  ' 
            f'loss={loss.item():.4f} '
            f'lr={optimizer.lr:.6f}'
        )
        

# -------------------------
# Plot
# -------------------------

fig, ax1 = plt.subplots()

# Loss
ax1.plot(epochs, losses)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")

# Learning rate
ax2 = ax1.twinx()
ax2.plot(epochs, learning_rates)
ax2.set_ylabel("Learning Rate")

plt.title("Training Loss and Learning Rate")

plt.show()


# python3 adam.py
