import argparse

parser = argparse.ArgumentParser()

# common
parser.add_argument('--model_name', default='MSPRL7', type=str)
parser.add_argument('--mode', default='test', choices=['train', 'valid', 'test'], type=str)
parser.add_argument('--tb_path', type=str, default='./runs/', help='tensorboard path')
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--gpu', type=int, default=0, help='point single gpu')
parser.add_argument('--print_freq', type=int, default=300)
parser.add_argument('--valid_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=1)

# data load
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=14)
parser.add_argument('--pin_memory', type=bool, default=True)
parser.add_argument('--image_channel', type=int, default='1', help='1:gray')

parser.add_argument('--train_halftone_path', type=str, default='./dataset/train/input/*.*')
parser.add_argument('--train_target_path', type=str, default='./dataset/train/target/*.*')
parser.add_argument('--valid_halftone_path', type=str, default='./dataset/valid/input/*.*')
parser.add_argument('--valid_target_path', type=str, default='./dataset/valid/target/*.*')

# resume
parser.add_argument('--resume', type=str, default=None)

# i/o
parser.add_argument('--log', type=str, default='./logs/')

# Train
parser.add_argument('--total_iters', type=int, default=300000)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--min_lr', type=float, default=1e-6)
parser.add_argument('--fft_loss_factor', type=float, default=0.1)

# Test
parser.add_argument('--test_model', type=str, default='./checkpoint/MSPRL7/MSPRL7_300000_best.pth')
parser.add_argument('--test_halftone_path', type=str, default='./dataset/test/Class/data/')
parser.add_argument('--test_target_path', type=str, default='./dataset/test/Class/target/')
parser.add_argument('--test_output_path', type=str, default='./results/')


def get_config():
    return parser.parse_args()
