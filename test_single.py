#!/usr/bin/env python3

import torch
import torch.utils.data 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np 
import trimesh


import os
import json
import time
from tqdm import tqdm

# remember to add paths in model/__init__.py for new models
from model import *



def main():
    
    model = init_model(specs["Model"], specs, 1)
    
    ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
    resume = os.path.join(args.exp_dir, ckpt)
     
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)

    # 提取输入文件名（不含扩展名）
    input_filename = os.path.splitext(os.path.basename(args.file))[0]
    
    file_ext = args.file[-4:]
    if file_ext == ".csv":
        f = pd.read_csv(args.file, sep=',',header=None).values
        f = f[f[:,-1]==0][:,:3]
    elif file_ext == ".ply":
        f = trimesh.load(args.file).vertices
    else:
        print("add your extension type here! currently not supported...")
        exit()

    sampled_points = args.num_points
    
    # recenter and normalize
    f -= np.mean(f, axis=0)
    bbox_length = np.sqrt( np.sum((np.max(f, axis=0) - np.min(f, axis=0))**2) )
    f /= bbox_length

    f = torch.from_numpy(f)[torch.randperm(f.shape[0])[0:sampled_points]].float().unsqueeze(0)
    model.load_state_dict(checkpoint['state_dict'])
    
    # 添加清理显存
    torch.cuda.empty_cache()
    
    # 创建包含输入文件名的输出目录
    output_dir = os.path.join(args.outdir, input_filename)
    os.makedirs(output_dir, exist_ok=True)  # 提前创建目录
    
    # 保存输入采样点云（归一化后的点云）
    input_pc_numpy = f.squeeze(0).cpu().numpy()
    input_pc_file = os.path.join(output_dir, "input_sampled_points.xyz")
    np.savetxt(input_pc_file, input_pc_numpy, fmt='%.6f', delimiter=' ', 
               header=f'{input_pc_numpy.shape[0]} points (normalized and centered)', 
               comments='# ')
    print(f"已保存输入采样点云到: {input_pc_file}")
    
    print(f"\n{'='*60}")
    print(f"重建配置:")
    print(f"  输入文件: {args.file}")
    print(f"  采样点数: {sampled_points}")
    print(f"  网格分辨率: {args.resolution}")
    print(f"  批处理大小: {args.batch_size}")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*60}\n")
    
    model.reconstruct(
        model, 
        {'point_cloud': f, 'mesh_name': input_filename}, 
        eval_dir=output_dir, 
        testopt=args.test_optimize,
        sampled_points=sampled_points,
        resolution=args.resolution,
        batch_size=args.batch_size
    ) 


def init_model(model, specs, num_objects):
    if model == "DeepSDF":
        return DeepSDF(specs, num_objects).cuda()
    elif model == "NeuralPull":
        return NeuralPull(specs, num_objects).cuda()
    elif model == "ConvOccNet":
        return ConvOccNet(specs).cuda()
    elif model == "GenSDF":
        return GenSDF(specs, None).cuda()
    else:
        print("model not loaded...")

    
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e",
        default="config/gensdf/semi",
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r",
        default="last",
        help="continue from previous saved logs, integer value or 'last'",
    )

    arg_parser.add_argument(
        "--outdir", "-o",
        required=True,
        help="output directory of reconstruction",
    )

    arg_parser.add_argument(
        "--file", "-f",
        required=True,
        help="input point cloud filepath, in csv or ply format",
    )
    
    arg_parser.add_argument(
        "--resolution", "-res",
        type=int,
        default=192,
        help="grid resolution for marching cubes (default: 192, higher=better quality but more memory)",
    )
    
    arg_parser.add_argument(
        "--batch_size", "-bs",
        type=int,
        default=64000,
        help="batch size for SDF queries (default: 64000, lower if out of memory)",
    )
    
    arg_parser.add_argument(
        "--num_points", "-np",
        type=int,
        default=1000,
        help="number of points to sample from input (default: 1000)",
    )
    
    arg_parser.add_argument(
        "--test_optimize",
        action="store_true",
        default=True,
        help="perform test-time optimization (default: True)",
    )
    
    arg_parser.add_argument(
        "--no_test_optimize",
        action="store_false",
        dest="test_optimize",
        help="skip test-time optimization (faster but lower quality)",
    )

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"][0])

    main()
