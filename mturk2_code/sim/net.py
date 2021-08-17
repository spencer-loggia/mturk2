try:
    import torch
    from torch import nn
    from torchvision.models import alexnet
except ModuleNotFoundError:
    print("Torch or Torchvision not found")
    exit(-3)


class ConvNet(nn.Module):
    """
    a wrapper around pretrained AlexNet
    """
    def __init__(self):
        super().__init__()
        self.alex = alexnet(pretrained=True)

