import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for image classification on MNIST dataset.
    Three layered cnn followed by a fully connected layer.

    Architecture:
    - Conv2d -> ReLU -> MaxPool2d
    - Conv2d -> ReLU -> MaxPool2d
    - Conv2d -> ReLU -> MaxPool2d
    - Dropout
    - Fully Connected Layer

    Input: 1x28x28 grayscale images
    Output: 10 class scores
    """

    def __init__(self) -> None:
        super(MyAwesomeModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128, 10)
        self.reLU = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reLU(self.conv1(x))
        x = self.maxpool(x)
        x = self.reLU(self.conv2(x))
        x = self.maxpool(x)
        x = self.reLU(self.conv3(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
