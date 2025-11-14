#!/usr/bin/env python3
"""
Point Cloud Fusion Example

This script demonstrates the complete workflow for:
1. Normalizing point clouds to a unified space
2. Fusing two point clouds using GenSDF's encoder
3. Reconstructing the fused result as a 3D mesh

Usage:
    python fusion_example.py -m config/gensdf/semi -p1 data/part1.ply -p2 data/part2.ply -o output
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import trimesh
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GenSDF
from utils.pointcloud_normalize import normalize_pointcloud, save_transform_params, apply_transform, load_transform_params
from utils.sdf_fusion import SDFFusion, create_fusion_example


def load_pointcloud(path):
    """Load point cloud from various formats."""
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.csv':
        pc = pd.read_csv(path, sep=',', header=None).values
        # If CSV has SDF values, only take surface points (sdv=0)
        if pc.shape[1] == 4:
            pc = pc[pc[:, -1] == 0][:, :3]
        elif pc.shape[1] > 3:
            pc = pc[:, :3]
    elif ext == '.ply':
        pc = trimesh.load(path).vertices
    elif ext == '.xyz':
        pc = np.loadtxt(path)
        if pc.ndim == 1:
            pc = pc.reshape(-1, 3)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return pc


def save_pointcloud(pc, path):
    """Save point cloud to file."""
    ext = os.path.splitext(path)[1].lower()
    
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if ext in ['.xyz', '.txt']:
        np.savetxt(path, pc, fmt='%.6f')
    elif ext == '.ply':
        mesh = trimesh.Trimesh(vertices=pc, faces=[])
        mesh.export(path)
    else:
        raise ValueError(f"Unsupported output format: {ext}")


def main():
    parser = argparse.ArgumentParser(
        description='Fuse two point clouds using GenSDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic fusion with union
    python fusion_example.py -m config/gensdf/semi -p1 part1.ply -p2 part2.ply -o output
    
    # Fusion with smooth blending
    python fusion_example.py -m config/gensdf/semi -p1 part1.ply -p2 part2.ply -o output --method blend --alpha 15
    
    # Use reference transform parameters
    python fusion_example.py -m config/gensdf/semi -p1 part1.ply -p2 part2.ply -o output --ref-transform transform.json
        """
    )
    
    parser.add_argument(
        '--model_dir', '-m',
        required=True,
        help='Path to model directory containing specs.json and checkpoint'
    )
    
    parser.add_argument(
        '--pointcloud1', '-p1',
        required=True,
        help='Path to first point cloud file (.ply, .xyz, or .csv)'
    )
    
    parser.add_argument(
        '--pointcloud2', '-p2',
        required=True,
        help='Path to second point cloud file (.ply, .xyz, or .csv)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--method',
        choices=['union', 'blend'],
        default='union',
        help='Fusion method: union (min) or blend (smooth)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=10.0,
        help='Blending parameter for blend method (higher = sharper)'
    )
    
    parser.add_argument(
        '--resolution',
        type=int,
        default=128,
        help='Marching cubes resolution (default: 128)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64000,
        help='Batch size for SDF queries (default: 64000)'
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        default='last',
        help='Checkpoint to load (default: last)'
    )
    
    parser.add_argument(
        '--ref_transform',
        help='Use existing transformation parameters from JSON file'
    )
    
    parser.add_argument(
        '--save_transform',
        help='Save transformation parameters to JSON file'
    )
    
    parser.add_argument(
        '--no_normalize',
        action='store_true',
        help='Skip normalization (assume already normalized)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 70)
    print("GenSDF Point Cloud Fusion")
    print("=" * 70)
    
    # Load model
    print("\n1. Loading GenSDF model...")
    specs_path = os.path.join(args.model_dir, "specs.json")
    with open(specs_path, 'r') as f:
        specs = json.load(f)
    
    model = GenSDF(specs, None).cuda()
    
    ckpt = f"{args.checkpoint}.ckpt" if args.checkpoint == 'last' else f"epoch={args.checkpoint}.ckpt"
    checkpoint_path = os.path.join(args.model_dir, ckpt)
    
    print(f"   Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("   Model loaded successfully!")
    
    # Load point clouds
    print("\n2. Loading point clouds...")
    print(f"   Point cloud 1: {args.pointcloud1}")
    pc1 = load_pointcloud(args.pointcloud1)
    print(f"   Loaded {pc1.shape[0]} points")
    
    print(f"   Point cloud 2: {args.pointcloud2}")
    pc2 = load_pointcloud(args.pointcloud2)
    print(f"   Loaded {pc2.shape[0]} points")
    
    # Normalize point clouds
    if not args.no_normalize:
        print("\n3. Normalizing point clouds...")
        
        if args.ref_transform:
            # Use existing transformation
            print(f"   Loading reference transformation from: {args.ref_transform}")
            transform_params = load_transform_params(args.ref_transform)
            pc1_norm = apply_transform(pc1, transform_params)
            pc2_norm = apply_transform(pc2, transform_params)
            print(f"   Applied transformation: centroid={transform_params['centroid']}, scale={transform_params['scale']:.4f}")
        else:
            # Create new transformation from first point cloud
            print("   Creating normalization from first point cloud...")
            pc1_norm, transform_params = normalize_pointcloud(pc1, return_params=True)
            pc2_norm = apply_transform(pc2, transform_params)
            print(f"   Centroid: {transform_params['centroid']}")
            print(f"   Scale: {transform_params['scale']:.4f}")
            
            # Save transformation if requested
            if args.save_transform:
                save_transform_params(transform_params, args.save_transform)
            else:
                # Save to output directory by default
                default_transform_path = os.path.join(args.output, "transform_params.json")
                save_transform_params(transform_params, default_transform_path)
        
        # Save normalized point clouds
        pc1_norm_path = os.path.join(args.output, "pc1_normalized.xyz")
        pc2_norm_path = os.path.join(args.output, "pc2_normalized.xyz")
        save_pointcloud(pc1_norm, pc1_norm_path)
        save_pointcloud(pc2_norm, pc2_norm_path)
        print(f"   Saved normalized point clouds to {args.output}")
    else:
        print("\n3. Skipping normalization (using input as-is)")
        pc1_norm = pc1
        pc2_norm = pc2
    
    # Convert to torch tensors
    pc1_tensor = torch.from_numpy(pc1_norm).float()
    pc2_tensor = torch.from_numpy(pc2_norm).float()
    
    # Fuse point clouds
    print("\n4. Fusing point clouds...")
    print(f"   Method: {args.method}")
    if args.method == 'blend':
        print(f"   Alpha: {args.alpha}")
    
    fusion = SDFFusion(model)
    fused_sdf = fusion.fuse_pointclouds(
        pc1_tensor, 
        pc2_tensor, 
        method=args.method, 
        alpha=args.alpha
    )
    
    # Reconstruct mesh
    print("\n5. Reconstructing 3D mesh from fused SDF...")
    print(f"   Resolution: {args.resolution}")
    print(f"   Batch size: {args.batch_size}")
    
    output_mesh_path = os.path.join(args.output, f"fused_{args.method}.ply")
    fusion.reconstruct_from_sdf(
        fused_sdf, 
        output_mesh_path, 
        resolution=args.resolution,
        batch_size=args.batch_size
    )
    
    print("\n" + "=" * 70)
    print("Fusion Complete!")
    print("=" * 70)
    print(f"Output mesh: {output_mesh_path}")
    print(f"Output directory: {args.output}")
    print("\nYou can visualize the result with:")
    print(f"  meshlab {output_mesh_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
