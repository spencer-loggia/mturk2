import torch

class Deform(torch.nn.Module):

    def __init__(self, channels=4, deform_basis=4):
        