import argparse
from train import *


## Parser 생성하기
parser = argparse.ArgumentParser(description="MElGAN modeling",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--g_lr", default=2e-4, type=float, dest="g_lr")
parser.add_argument("--d_lr", default=2e-4, type=float, dest="d_lr")
parser.add_argument("--batch_size", default=16, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=300, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./dataset", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./ckpt", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")
parser.add_argument("--decay_epoch", default = 500, type=int,dest ="decay_epoch")
parser.add_argument("--test_path", default = './test', type=str,dest ="test_path")

parser.add_argument("--only_D", default=1 , type=int, dest="only_D")

args = parser.parse_args()

if __name__ == '__main__':
    if args.mode == 'train' or args.mode == 'test':
        train(args)