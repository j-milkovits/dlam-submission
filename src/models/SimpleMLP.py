import torch
from torch.nn import Linear, Module, ReLU, Sequential, Sigmoid


class SimpleMLP(Module):
    def __init__(self, input_size: int = 300) -> None:
        super(SimpleMLP, self).__init__()
        self.mlp_stack = Sequential(
            Linear(input_size, 600),
            ReLU(),
            Linear(600, 300),
            ReLU(),
            Linear(300, 300),
            ReLU(),
            Linear(300, 1),
            Sigmoid(),  # binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp_stack(x)
        return x
