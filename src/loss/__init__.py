import torch

class CrossEntropyLossWrapper(torch.nn.CrossEntropyLoss):
    def __init__(self, weight):
        super().__init__(weight=torch.tensor(weight, dtype=torch.float))

__all__ = [
    "CrossEntropyLoss"
]
