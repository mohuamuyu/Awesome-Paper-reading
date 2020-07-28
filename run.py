
import os
import sys
import yaml
import json
import argparse
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", default="/home/work/user-job-dir/CenterPoint7/configs/centerpoint/nusc_centerpoint_voxelnet_01voxel.py", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    parser.add_argument("--validate", action="store_true", help="whether to evaluate the checkpoint during training")
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus to use " "(only applicable to non-distributed training)")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--autoscale-lr", action="store_true", help="automatically scale lr with the number of gpus")
    parser.add_argument('--data_url', type=str, default='s3://bucket-8280/chenxinghao/data/NUSCENES.tar', help='path to dataset')
    parser.add_argument('--train_url', type=str, default='s3://bucket-2707/zm/CenterPoint7/output/', help='train_dir')

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    ###########################################
    ########## install
    os.system('cd /home/work/user-job-dir/CenterPoint7')
    os.system('ls')
    os.system('pip install -r requirements.txt')

    ########## Advanced Installation

    # nuScenes dev-kit

    # Cuda Extensions
    try:
        os.system('export PATH=/usr/local/cuda-10.0/bin:$PATH')
        os.system('export CUDA_PATH=/usr/local/cuda-10.0')
        os.system('export CUDA_HOME=/usr/local/cuda-10.0')
        os.system('export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH')
    except:
        print("export PATH=/usr/local/cuda-10.0 wromg wrong wrong !!!!!!!!!!!!")
    os.system('cd /home/work/user-job-dir/CenterPoint7')
    os.system('bash setup.sh')

    # APEX
    os.system('cd /home/work/user-job-dir/CenterPoint7')
    os.system('cd apex')
    os.system('pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./')
    os.system('cd /home/work/user-job-dir/CenterPoint7')

    # spconv
    os.system('cd spconv')
    os.system('pip install cmake')
    os.system('python setup.py bdist_wheel')
    os.system('cd ./dist && pip install *')
    os.system('cd /home/work/user-job-dir/CenterPoint7')

    ########## copy dataset
    import moxing as mox
    mox.file.copy_parallel(args.data_url, '/cache/data/nuscenes-zm')
    os.system('cd /cache/data/nuscenes-zm')
    os.system('tar -xf NUSCENES.tar')

    os.system('cd /home/work/user-job-dir/CenterPoint7')
    os.system('mkdir data')
    os.system('cd data')
    os.system('ln -s /cache/data/nuscenes-zm')
    os.system('mv nuscenes-zm nuScenes')

    # nuScenes
    os.system('python tools/create_data.py nuscenes_data_prep --root_path=data/nuScenes --version="v1.0-trainval" --nsweeps=10')

    # train
    os.system('python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py')

    mox.file.copy_parallel(args.work_dir, 's3://bucket-2707/zm/CenterPoint7/output2')


if __name__ == "__main__":
    main()
