import torch
from torch import nn

class FoodClassModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        

