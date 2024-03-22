import sys
sys.path.append("/pscratch/sd/y/yanggao/SuperBench/")

from src.data_loader_crop import getData
from utils import *
import argparse

# Instead of parsing arguments from the command line, create a class or a simple namespace
# that holds all your default values for the arguments.
class DefaultArgs:
    data_name = 'nskt_16k'
    data_path = '/pscratch/sd/y/yanggao/SuperBench/superbench_v1/nskt16000_1024'
    crop_size = 512
    n_patches = 8
    method = "bicubic"
    model_path = 'results/model_EDSR_sst4_0.0001_5544.pt'
    pretrained = False
    model = 'subpixelCNN'
    epochs = 300
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    lr = 0.0001
    wd = 1e-6
    seed = 5544
    step_size = 100
    gamma = 0.97
    noise_ratio = 0.0
    upscale_factor = 8
    in_channels = 2
    hidden_channels = 32
    out_channels = 2
    n_res_blocks = 18
    loss_type = 'l1'
    optimizer_type = 'Adam'
    scheduler_type = 'ExponentialLR'

# Now use this class instead of parsing args
args = DefaultArgs()

# Your script continues as before...
print(args)  # Example usage


resol, n_fields, n_train_samples, mean, std = get_data_info(args.data_name)
train_loader, val1_loader, val2_loader, test1_loader, test2_loader  = getData(args, args.n_patches, std=std)


def get_train_loader():
    return train_loader

# sys.path.pop("/pscratch/sd/y/yanggao/SuperBench/")
# print(sys.path)

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