from __future__ import print_function

import os
import torch
from PIL import Image
from torchvision.transforms import transforms
from torchvision.utils import save_image

from model import build_net


def test(demo_path, demo_output, net, device):
    net.eval()
    with torch.no_grad():
        for filename in os.listdir(demo_path):
            halftone_img = demo_path + "/" + filename
            print('======> ', filename)
            halftone_img = ImageToTensor(halftone_img)
            halftone_img = halftone_img.unsqueeze(0)
            halftone_img = halftone_img.to(device)
            output = net(halftone_img)
            output = torch.cat([halftone_img, output], dim=0)
            save_image(output, '%s%s' % (demo_output, filename))


def ImageToTensor(halftone_img):
    halftone_img = Image.open(halftone_img)
    halftone_img = transforms.ToTensor()(halftone_img)
    return halftone_img


if __name__ == '__main__':
    demo_path = './demo/halftone/'
    demo_output = './demo/output/'
    model_path = './checkpoint/MSPRL/MSPRL_300000.pth'
    net = build_net(model_name='MSPRL', image_channel=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if str(device) == 'cpu':
        state_dict = torch.load(model_path, map_location='cpu')
    else:
        state_dict = torch.load(model_path)
    net.load_state_dict(state_dict)
    net = net.to(device)
    test(demo_path, demo_output, net, device)
