import json
import os
import random
import time
from datetime import timedelta
import torch
from config import get_config
from test import test
from train import train
from utils.init_util import init_dir, create_valid_data_loader
from valid import valid
from model import build_net


def main(args):
    print('----------------------------------------------------------------------')
    print(args.model_name)
    print(args.mode)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'valid':
        log = open(os.path.join(args.log, '{}.txt'.format(args.mode)), 'w')
        # loader valid data
        valid_loader = create_valid_data_loader(args)

        net = build_net(args.model_name, args.image_channel)
        if args.resume:
            if str(args.device) == 'cpu':
                resume = torch.load(args.resume, map_location='cpu')
            else:
                resume = torch.load(args.resume)
            state_dict = resume['model']
        else:
            if str(args.device) == 'cpu':
                state_dict = torch.load(args.test_model, map_location='cpu')
            else:
                state_dict = torch.load(args.test_model)
        net.load_state_dict(state_dict)
        net = net.to(args.device)
        valid(args, net, log, valid_loader)
    elif args.mode == 'test':
        temp = os.path.normpath(args.test_halftone_path)
        temp = temp.split(os.sep)
        log = open(os.path.join(args.log, '{}.txt'.format(temp[-2])), 'w')
        net = build_net(args.model_name, args.image_channel)
        if args.resume:
            if str(args.device) == 'cpu':
                resume = torch.load(args.resume, map_location='cpu')
            else:
                resume = torch.load(args.resume)
            state_dict = resume['model']
        else:
            if str(args.device) == 'cpu':
                state_dict = torch.load(args.test_model, map_location='cpu')
            else:
                state_dict = torch.load(args.test_model)
        net.load_state_dict(state_dict)
        net = net.to(args.device)
        test(args, net, log)


if __name__ == '__main__':
    args = get_config()
    # print(args)

    # Init dir
    init_dir(args)
    main(args)
