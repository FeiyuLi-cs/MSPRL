from __future__ import print_function

import json
import os
import time

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import cv2
import torch
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
from torchvision.utils import save_image


def valid(args, net, log, val_loader):
    device = args.device
    val_data_len = len(val_loader)
    net.eval()
    avg_psnr = 0
    avg_ssim = 0
    state = {
        'img_num': 0,
        'time': '',
        'psnr': 0.0,
        'ssim': 0.0,
    }
    print('======> valid start~')
    i = 0
    with torch.no_grad():
        for predict, target in val_loader:
            predict, target = predict.to(device), target.to(device)
            output_map = net(predict)

            img1 = transforms.ToPILImage()(output_map.squeeze(0))
            img1 = np.asanyarray(img1)

            img2 = transforms.ToPILImage()(target.squeeze(0))
            img2 = np.asanyarray(img2)

            psnr = compare_psnr(img1, img2)
            ssim = compare_ssim(img1, img2)

            state['img_num'] = i
            state['time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            state['psnr'] = psnr
            state['ssim'] = ssim
            avg_psnr += psnr
            avg_ssim += ssim
            i += 1
            log.write('%s\n' % json.dumps(state))
            log.flush()

    print('======> total img count ', i, val_data_len)
    log.write('%s\n' % json.dumps(
        {'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
         'avg_psnr': avg_psnr / val_data_len,
         'avg_ssim': avg_ssim / val_data_len}))
    log.flush()
    log.close()
    print('done!!!')
