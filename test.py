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
from torchvision.utils import save_image


def test(args, net, log):
    device = args.device
    test_halftone_path = args.test_halftone_path
    test_output_path = args.test_output_path
    test_target_path = args.test_target_path
    net.eval()
    with torch.no_grad():
        for filename in os.listdir(test_halftone_path):
            halftone_img = test_halftone_path + "/" + filename
            print('======> ', filename)
            halftone_img = ImageToTensor(halftone_img)
            halftone_img = halftone_img.unsqueeze(0)
            halftone_img = halftone_img.to(device)
            output = net(halftone_img)
            save_image(output, '%s%s' % (test_output_path, filename))
    print('-------------------psnr ssim-------------------')
    psnr_ssim(test_target_path, test_output_path, log)


def ImageToTensor(halftone_img):
    halftone_img = Image.open(halftone_img)
    t = transforms.ToTensor()
    halftone_img = t(halftone_img)
    return halftone_img


def psnr_ssim(test_target_path, test_output_path, log):
    state = {
        'IMG': '',
        'time': '',
        'psnr': 0.0,
        'ssim': 0.0,
    }

    avg_psnr = 0
    avg_ssim = 0
    count = 0

    for filename in os.listdir(test_target_path):
        print('======> psnr-ssim ', filename)

        img1_path = test_output_path + filename
        img2_path = test_target_path + filename

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        psnr = compare_psnr(img1, img2)
        ssim = compare_ssim(img1, img2)

        state['IMG'] = filename
        state['time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        state['psnr'] = psnr
        state['ssim'] = ssim
        avg_psnr += psnr
        avg_ssim += ssim
        count += 1
        log.write('%s\n' % json.dumps(state))
        log.flush()

    print('======> total img count ', count)
    log.write('%s\n' % json.dumps(
        {'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
         'avg_psnr': avg_psnr / count, 'avg_ssim': avg_ssim / count}))
    log.flush()
    log.close()
    print('done!!!')


if __name__ == '__main__':
    test_target_path = 'E://graduate/data/halftoneImg/halftoneImg/test/Kodak/target/'
    test_output_path = 'E://graduate/python/InverseHalftoning/output/results/Kodak/'
    log = open(os.path.join('./', 'log.txt'), 'w')
    psnr_ssim(test_target_path, test_output_path, log)
