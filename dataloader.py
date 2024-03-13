import sys
sys.path.append("/pscratch/sd/y/yanggao/SuperBench/")

from src.data_loader_crop import getData
from utils import *
import argparse

parser = argparse.ArgumentParser(description='training parameters')
# arguments for data
parser.add_argument('--data_name', type=str, default='nskt_16k', help='dataset')
parser.add_argument('--data_path', type=str, default='/pscratch/sd/y/yanggao/SuperBench/superbench_v1/nskt16000_1024', help='the folder path of dataset')
parser.add_argument('--crop_size', type=int, default=512, help='crop size for high-resolution snapshots')
parser.add_argument('--n_patches', type=int, default=8, help='number of patches')    
parser.add_argument('--method', type=str, default="bicubic", help='downsample method')
parser.add_argument('--model_path', type=str, default='results/model_EDSR_sst4_0.0001_5544.pt', help='saved model')
parser.add_argument('--pretrained', default=False, type=lambda x: (str(x).lower() == 'true'), help='load the pretrained model')

# arguments for training
parser.add_argument('--model', type=str, default='subpixelCNN', help='model')
parser.add_argument('--epochs', type=int, default=300, help='max epochs')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-6, help='weight decay')
parser.add_argument('--seed', type=int, default=5544, help='random seed')
parser.add_argument('--step_size', type=int, default=100, help='step size for scheduler')
parser.add_argument('--gamma', type=float, default=0.97, help='coefficient for scheduler')
parser.add_argument('--noise_ratio', type=float, default=0.0, help='noise ratio')

# arguments for model
parser.add_argument('--upscale_factor', type=int, default=8, help='upscale factor')
parser.add_argument('--in_channels', type=int, default=2, help='num of input channels')
parser.add_argument('--hidden_channels', type=int, default=32, help='num of hidden channels')
parser.add_argument('--out_channels', type=int, default=2, help='num of output channels')
parser.add_argument('--n_res_blocks', type=int, default=18, help='num of resdiual blocks')
parser.add_argument('--loss_type', type=str, default='l1', help='L1 or L2 loss')
parser.add_argument('--optimizer_type', type=str, default='Adam', help='type of optimizer')
parser.add_argument('--scheduler_type', type=str, default='ExponentialLR', help='type of scheduler')

args = parser.parse_args()
print(args)


resol, n_fields, n_train_samples, mean, std = get_data_info(args.data_name)
train_loader, val1_loader, val2_loader, test1_loader, test2_loader  = getData(args, args.n_patches, std=std)


def get_train_loader():
    return train_loader

sys.path.pop()

if __name__ == "__main__":
    print('\nThe data resolution is: ', resol)
    print("mean is: ",mean)
    print("std is: ",std)

    print(type(train_loader))
    print(f"len = {len(train_loader)}")
    for i, (data, target) in enumerate(train_loader):
        print(f"Batch {i+1} size: ", data.size())
        break

# torch.save(dataset.file_paths, 'dataset_file_paths.pt')
# print(f"\ntrain_loader saved at {}")