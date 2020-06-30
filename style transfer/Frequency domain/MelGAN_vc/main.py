import argparse
from train import *


## Parser 생성하기
parser = argparse.ArgumentParser(description="DCGAN modeling",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--g_lr", default=1e-5, type=float, dest="g_lr")
parser.add_argument("--d_lr", default=1e-5, type=float, dest="d_lr")
parser.add_argument("--batch_size", default=16, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=300, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets/BSR/BSDS500/data/images", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--gupt", default=5 , type=int, dest="gupt")

args = parser.parse_args()

if __name__ == '__main__':
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)