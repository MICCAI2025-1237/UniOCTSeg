import os
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import random
from evaluation import SoftDiceLoss, BCELoss, PCL
from UniOCTSeg import *
from Vision_network import vision_network
from Prompt_decoder import decoder

parser = argparse.ArgumentParser()
parser.add_argument('--data_pth', default=r'', type=str,
                    help='Directory to save checkpoint.')
parser.add_argument('--save_pth', type=str, default='')
parser.add_argument('--exp', type=str, default='UniOCTSeg', help='experiment_name')
parser.add_argument('--max_iterations', type=int, default=80000,
                    help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--device', type=int, default=0, help='batch_size per gpu')

parser.add_argument('--num_workers', default=24, type=int, help='num of workers')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-4,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int,  default=42, help='random seed')
parser.add_argument('--resume', default='',
                    help='Resume from checkpoint or not.')
args = parser.parse_args()
device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")


class EMA():
    def __init__(self, decay=0.99):
        self.decay = decay
    def copy_para(self, ema_model, model):
        self.ema_dict = OrderedDict()
        for name, param in model.named_parameters():
                self.ema_dict[name]=param.data.clone()
        for name, buffer in model.named_buffers():
            self.ema_dict[name] = buffer.data.clone()
        ema_model.load_state_dict(self.ema_dict)
    def update(self, ema_model, model):
        model.eval()
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1 - self.decay) * param.data + self.decay * self.ema_dict[name]
                self.ema_dict[name]=new_average
        for name, buffer in model.named_buffers():
            new_average = (1 - self.decay) * buffer.data + self.decay * self.ema_dict[name]
            self.ema_dict[name] = new_average
        ema_model.load_state_dict(self.ema_dict)
        model.train()
    def resume(self, ema_model):
        self.ema_dict = OrderedDict()
        for name, param in ema_model.named_parameters():
            self.ema_dict[name] = param.data.clone()
        for name, buffer in ema_model.named_buffers():
            self.ema_dict[name] = buffer.data.clone()

def train(args):
    base_lr = args.base_lr
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    iter_num = 0

    network = vision_network(in_channels=1,
                             hidden_size=768,
                             weights='ViT-B-16')

    p_decoder = decoder(basic_prompt_num=9,
                        hidden_dim=64,
                        n_heads=8,
                        dim_feedforward=512,
                        dec_layers=10)

    model = UniOCTSeg(network=vision_network,
                      prompt_decoder=p_decoder)

    model = model.to(device)
    model.train()

    train_Labeled_dataset = Load_train_data(root=args.data_pth,
                                            flag='train')

    train_Labeled_dataloder = torch.utils.data.DataLoader(dataset=train_Labeled_dataset,
                                                          batch_size=24,
                                                          num_workers=24,
                                                          shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=base_lr,
                                 betas=(0.9, 0.95),
                                 weight_decay=0.0001)

    loss_dice = SoftDiceLoss()
    loss_ce = BCELoss()
    loss_consistency = PCL()

    if args.resume != '':
        latest_status = torch.load(args.resume,
                                   map_location=torch.device('cpu'))
        model.load_state_dict(latest_status['model'])
        optimizer.load_state_dict(latest_status['optimizer'])
        iter_num = latest_status['iter_num']
    max_epoch = max_iterations // len(train_Labeled_dataloder) + 1
    iterator = tqdm(range(max_epoch),
                    ncols=70)

    for epoch in iterator:
        for idx, (imgs, labels, dataset_names) in enumerate(train_Labeled_dataloder):

            imgs = torch.tensor(imgs, device=device, dtype=torch.float).unsqueeze(1)
            labels = torch.tensor(labels, device=device, dtype=torch.float)

            layer_preds = model(imgs)
            layer_preds_sigmoid = F.sigmoid(layer_preds)

            layer_loss1 = loss_dice(layer_preds_sigmoid, labels)
            layer_loss2 = loss_ce(layer_preds, labels)

            Loss = layer_loss1 + layer_loss2
            print(
                'iteration %d : loss : %.5f, loss_layer_dice: %.5f, loss_layer_ce: %.5f' %
                (iter_num, Loss.item(), layer_loss1.item(), layer_loss2.item()))

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.snapshot_path = args.result_path + args.exp
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    train(args)
