from __future__ import division

import datetime
import json
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from utils.init_util import *
import torch.distributed as dist


def train(args):
    set_random_seed(args.seed)

    device = args.device
    # log = log

    print_freq = args.print_freq
    model_name = args.model_name
    valid_freq = args.valid_freq
    save_freq = args.save_freq
    fft_loss_factor = args.fft_loss_factor

    train_loader, valid_loader = create_train_valid_data_loader(args)
    train_data_len = args.train_data_len
    valid_data_len = args.valid_data_len
    net, l1_criterion, optimizer, scheduler = init_model(args, device)

    log = init_loggers(args)

    writer = SummaryWriter(args.tb_path)
    state = args.state
    model_save_dir = args.model_save_dir

    if args.resume:
        resume = torch.load(args.resume)
        state['max_psnr'] = resume['max_psnr']
        start_epoch = resume['epoch']
        current_iter = resume['current_iter']
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        net.load_state_dict(resume['model'])
        state = resume['state']
        print('======> Resume from {}'.format(start_epoch))
        print('======> Max psnr : {}.'.format(state['max_psnr']))
        start_epoch += 1
        current_iter += 1
    else:
        start_epoch = 0
        current_iter = 0

    epoch = start_epoch
    total_iters = args.total_iters
    # Main loop
    while current_iter < total_iters:
        net.train()
        train_loss = 0
        state['epoch'] = epoch
        state['learning_rate'] = optimizer.state_dict()['param_groups'][0]['lr']
        print('======>', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        start_time = time.time()

        for batch_id, (predict, target) in enumerate(train_loader):
            if current_iter >= total_iters:
                break

            predict, target = predict.to(device), target.to(device)
            optimizer.zero_grad()

            # forward
            output_map = net(predict)
            # backward

            l1_loss = l1_criterion(output_map, target)
            output_map_fft = torch.fft.rfft2(output_map)
            target_fft = torch.fft.rfft2(target)
            fft_loss = fft_loss_factor * l1_criterion(output_map_fft, target_fft)
            loss = l1_loss + fft_loss
            # loss = l1_loss

            train_loss += loss.item()
            loss.backward()

            # optimizer
            optimizer.step()

            writer.add_scalar('L1 Loss', l1_loss, batch_id + epoch * train_data_len)
            writer.add_scalar('fft_loss', fft_loss, batch_id + epoch * train_data_len)
            writer.add_scalar('loss', loss, batch_id + epoch * train_data_len)

            if batch_id % print_freq == 0:
                print(
                    "======> Epoch[{}]({}/{}): Loss: {:.4f} lr:{}".format(
                        epoch, batch_id, train_data_len,
                        loss.item(),
                        optimizer.state_dict()['param_groups'][0]['lr']))
            current_iter += 1

        # scheduler
        scheduler.step()

        train_avg_loss = train_loss / train_data_len
        state['train_avg_loss'] = train_avg_loss
        print("======> Epoch[{}] Complete: Avg. Loss: {:.4f}".format(epoch, train_avg_loss))
        print("======> Train time", datetime.timedelta(seconds=int(time.time() - start_time)))

        if epoch % save_freq == 0:
            # save
            checkpoint = {
                'max_psnr': state['max_psnr'],
                'epoch': epoch,
                'current_iter': current_iter,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'state': state
            }
            save_model(checkpoint, model_save_dir, model_name, total_iters, mode='resume', epoch=epoch)

        # valid_freq
        if epoch % valid_freq == 0:
            # test
            valid(net, device, valid_loader, valid_data_len, state, l1_criterion, fft_loss_factor)
            if state['avg_psnr'] > state['max_psnr']:
                state['max_psnr'] = state['avg_psnr']
                save_model(net.state_dict(), model_save_dir, model_name, total_iters, mode='net', epoch=epoch)
            print("Max. PSNR: {:.4f} dB".format(state['max_psnr']))

        # write log
        state['current_iter'] = current_iter
        log.write('%s\n' % json.dumps(state))
        log.flush()

        # end one epoch
        epoch += 1

    log.write('done!!!!!\n')
    log.close()
    print('done!!!!!')


def valid(net, device, valid_loader, valid_data_len, state, l1_criterion, fft_loss_factor):
    epoch_psnr = 0
    net.eval()
    start_time = time.time()
    with torch.no_grad():
        for predict, target in valid_loader:
            predict, target = predict.to(device), target.to(device)

            # forward
            output_map = net(predict)

            img1 = transforms.ToPILImage()(output_map.squeeze(0))
            img1 = np.asanyarray(img1)

            img2 = transforms.ToPILImage()(target.squeeze(0))
            img2 = np.asanyarray(img2)

            psnr = compare_psnr(img1, img2)

            epoch_psnr += psnr

    avg_psnr = epoch_psnr / valid_data_len
    state['avg_psnr'] = avg_psnr
    print("======> Val time", datetime.timedelta(seconds=int(time.time() - start_time)), end='')
    print("  Val Avg. PSNR: {:.4f} dB  ".format(avg_psnr), end='')


def save_model(net, model_save_dir, model_name, total_iters, mode, epoch):
    if mode == 'resume':
        model_save_dir = '{}/{}_{}_epoch{}.pth'.format(model_save_dir, model_name, total_iters, epoch)
    else:
        model_save_dir = '{}/{}_{}_best.pth'.format(model_save_dir, model_name, total_iters)
    torch.save(net, model_save_dir)
