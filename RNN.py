import torch
import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape

        h = torch.zeros(
            batch_size, 
            self.hidden_size,
            device = x.device
        )

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            h = torch.tanh(
                self.input_to_hidden(x_t)
                + self.hidden_to_hidden(h)
            )

            y_t = self.hidden_to_output(h)

            outputs.append(y_t)

        outputs = torch.stack(outputs, dim = 1)

        return outputs, h
    


class RNNClassifier(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 16, num_classes = 2):
        super().__init__()

        self.rnn = MyRNN(
            input_size = input_size,
            hidden_size = hidden_size,
            output_size = 1
        )

        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        _, final_hidden = self.rnn(x)
        logits = self.classifier(final_hidden)
        return logits



#  generate  a toy dataset
torch.manual_seed(42)

num_samples = 1000
seq_len = 5

X = torch.rand(num_samples, seq_len, 1)

y = (X.mean(dim =1).squeeze(-1) > 0.5).long()

model = RNNClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()

    logits = model(X)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        predictions = logits.argmax(dim = 1)
        accuracy = (predictions == y).float().mean()

        print(
            f'Epoch {epoch +1},'
            f"Loss: {loss.item(): .4f},"
            f'Accuracy: {accuracy.item():.4f}' 
        )


# python3 RNN.py