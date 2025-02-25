import torch
import torch.nn as nn

class UniOCTSeg(nn.Module):
    def __init__(self, network=None, prompt_decoder=None):
        super(UniOCTSeg, self).__init__()

        self.network = network
        self.prompt_decoder = prompt_decoder

    def forward(self, x):
        masked_features, features = self.network(x)
        out = self.prompt_decoder(features, masked_features)
        return out