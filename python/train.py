import torch, torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader

torch.manual_seed(0)

tf = tv.transforms.ToTensor()
train_ds = tv.datasets.MNIST('../data', train=True, download=True, transform=tf)
test_ds  = tv.datasets.MNIST('../data', train=False, download=True, transform=tf)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl  = DataLoader(test_ds, batch_size=1000)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for x, y in train_dl:
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            correct += (model(x).argmax(1) == y).sum().item()
    print(f"epoch {epoch+1}: test accuracy = {correct/len(test_ds)*100:.2f}%")

torch.save(model.state_dict(), 'float_model.pt')
