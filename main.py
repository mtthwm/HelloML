import torch as tch
import torch.nn as nn
import torchvision as tv

train_data = tv.datasets.EMNIST(
    root="./data",
    split="letters",
    train=True,
    download=True
)

model = nn.modules.Sequential([
    nn.Conv2d(1, 1, 3, 1, 1),
    nn.MaxPool2d(4, 1),
    nn.Flatten(),
    nn.Linear(256, 26),
    nn.ReLU()
])

model.compile()

model.train()

print(train_data[0])