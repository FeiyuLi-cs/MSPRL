import json
import math
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DistributedSampler

from dataloader import HalftoneDataSet
from model import build_net


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True


# init
def init_dir(args):
    # print('>>>>>>>>>>>>>>>>>>>>>>>  init_dir')

    args.model_save_dir = os.path.join('./checkpoint/', args.model_name)
    if args.mode == 'train':
        args.log = os.path.join(args.log, args.model_name, args.mode)
        args.tb_path = os.path.join(args.tb_path, args.model_name, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    if args.mode == 'test':
        temp = os.path.normpath(args.test_halftone_path)
        temp = temp.split(os.sep)
        args.log = os.path.join(args.log, args.model_name, args.mode, temp[-2])
        args.test_output_path = os.path.join(args.test_output_path, args.model_name, temp[-2] + '/')
    if args.mode == 'valid':
        args.log = os.path.join(args.log, args.model_name, args.mode)
    args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    if not os.path.isdir(args.tb_path):
        os.makedirs(args.tb_path)
    if not os.path.isdir(args.test_output_path):
        os.makedirs(args.test_output_path)

    init_state(args)


# init log save info
def init_state(args):
    state = {
        'epoch': 0,
        'current_iter': 0,
        'learning_rate': 0,
        'train_avg_loss': 0,
        'avg_psnr': 0,
        'max_psnr': 0
    }
    args.state = state


# init loggers
def init_loggers(args):
    # print('>>>>>>>>>>>>>>>>>>>>>>>  init_log', args.log, args.resume)
    if args.resume:
        log = open(os.path.join(args.log, 'log.txt'), 'a')
    else:
        log = open(os.path.join(args.log, 'log.txt'), 'w')

        log.write('%s\n' % json.dumps(
            {
                'model_name': args.model_name, 'image_size': args.image_size,
                'batch_size': args.batch_size,
                'total_iters': args.total_iters, 'total_epochs': args.total_epochs,
                'per_epoch_num_iter': args.per_epoch_num_iter,
                'learning_rate': args.learning_rate, 'min_lr': args.min_lr,
                'cosine_annealing T_max': args.total_epochs, 'fft_loss_factor': args.fft_loss_factor,
                'pin_memory': args.pin_memory,
            }
        ))
        log.flush()
    return log


# init model
def init_model(args, device):
    net = build_net(args.model_name, args.image_channel).to(device)
    # print(net)
    l1_criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=args.total_epochs)
    return net, l1_criterion, optimizer, scheduler


def create_train_valid_data_loader(args):
    train_data = HalftoneDataSet.PairDataSet('train', halftone_path=args.train_halftone_path,
                                             target_path=args.train_target_path,
                                             image_size=args.image_size, image_channel=args.image_channel)
    valid_data = HalftoneDataSet.PairDataSet('valid', halftone_path=args.valid_halftone_path,
                                             target_path=args.valid_target_path,
                                             image_size=args.image_size, image_channel=args.image_channel)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=args.pin_memory)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=1, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=args.pin_memory)
    args.train_data_len = len(train_loader)
    args.valid_data_len = len(valid_loader)
    args.per_epoch_num_iter = args.train_data_len
    args.total_epochs = math.ceil(args.total_iters / args.per_epoch_num_iter)
    return train_loader, valid_loader


def create_valid_data_loader(args):
    valid_data = HalftoneDataSet.PairDataSet('valid', halftone_path=args.valid_halftone_path,
                                             target_path=args.valid_target_path,
                                             image_size=args.image_size, image_channel=args.image_channel)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=1, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=args.pin_memory)
    return valid_loader
