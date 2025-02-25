import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from UniOCTSeg import *
from Vision_network import vision_network
from Prompt_decoder import decoder


if '__main__' == __name__:

    data_pth = r''
    data = np.load(data_pth)

    network = vision_network(in_channels=1,
                             hidden_size=768,
                             weights='ViT-B-16')

    p_decoder = decoder(basic_prompt_num=9,
                        hidden_dim=64,
                        n_heads=8,
                        dim_feedforward=512,
                        dec_layers=10
                        )

    model = UniOCTSeg(network=vision_network,
                      prompt_decoder=p_decoder)

    img = data['img']
    device = torch.device('cpu')
    img = torch.tensor(img, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
    pred = model(img)
    output = np.argmax(F.sigmoid(pred).cpu().numpy(), axis=1)
    plt.imshow(output, 'gray')
    plt.show()

