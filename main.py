import torch as tch
import torchvision as tv

train_data = tv.datasets.EMNIST(
    root="./data",
    split="letters",
    train=True,
    download=True
)

print(train_data[0])