#!/usr/bin/env python3
"""
Point Cloud Normalization Utilities

This module provides functions to normalize point clouds by centering to origin
and scaling the bounding box diagonal to [-1, 1] range, as well as saving/loading
transformation parameters to apply the same transformation to new point clouds.
"""

import numpy as np
import json
import os


def normalize_pointcloud(pointcloud, return_params=False):
    """
    Normalize a point cloud by centering to origin and scaling bounding box to [-1, 1].
    
    This follows the GenSDF preprocessing convention where:
    1. The centroid is moved to the origin
    2. The bounding box diagonal length is normalized to span [-1, 1]
    
    Args:
        pointcloud: numpy array of shape (N, 3) containing x, y, z coordinates
        return_params: if True, return both normalized pointcloud and transformation parameters
        
    Returns:
        If return_params=False:
            normalized_pointcloud: numpy array of shape (N, 3)
        If return_params=True:
            tuple of (normalized_pointcloud, transform_params)
            where transform_params is a dict with keys 'centroid' and 'scale'
    """
    if not isinstance(pointcloud, np.ndarray):
        pointcloud = np.array(pointcloud)
    
    if pointcloud.shape[1] != 3:
        raise ValueError(f"Expected pointcloud with 3 coordinates (x,y,z), got shape {pointcloud.shape}")
    
    # Step 1: Calculate centroid and recenter
    centroid = np.mean(pointcloud, axis=0)
    centered = pointcloud - centroid
    
    # Step 2: Calculate bounding box diagonal and normalize
    bbox_min = np.min(centered, axis=0)
    bbox_max = np.max(centered, axis=0)
    bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
    
    if bbox_diagonal > 0:
        scale = bbox_diagonal / 2.0  # Scale to make diagonal span 2 (from -1 to 1)
        normalized = centered / scale
    else:
        scale = 1.0
        normalized = centered
    
    if return_params:
        transform_params = {
            'centroid': centroid.tolist(),
            'scale': float(scale)
        }
        return normalized, transform_params
    
    return normalized


def save_transform_params(transform_params, filepath):
    """
    Save transformation parameters to a JSON file.
    
    Args:
        transform_params: dict with keys 'centroid' and 'scale'
        filepath: path to save the JSON file
    """
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(transform_params, f, indent=2)
    
    print(f"Transformation parameters saved to: {filepath}")


def load_transform_params(filepath):
    """
    Load transformation parameters from a JSON file.
    
    Args:
        filepath: path to the JSON file
        
    Returns:
        transform_params: dict with keys 'centroid' and 'scale'
    """
    with open(filepath, 'r') as f:
        transform_params = json.load(f)
    
    # Convert lists back to numpy arrays for centroid
    transform_params['centroid'] = np.array(transform_params['centroid'])
    
    return transform_params


def apply_transform(pointcloud, transform_params):
    """
    Apply saved transformation parameters to a new point cloud.
    
    This applies the same centering and scaling that was used on the reference point cloud.
    
    Args:
        pointcloud: numpy array of shape (N, 3) containing x, y, z coordinates
        transform_params: dict with keys 'centroid' and 'scale' from normalize_pointcloud
        
    Returns:
        transformed_pointcloud: numpy array of shape (N, 3)
    """
    if not isinstance(pointcloud, np.ndarray):
        pointcloud = np.array(pointcloud)
    
    if pointcloud.shape[1] != 3:
        raise ValueError(f"Expected pointcloud with 3 coordinates (x,y,z), got shape {pointcloud.shape}")
    
    centroid = np.array(transform_params['centroid'])
    scale = transform_params['scale']
    
    # Apply the same transformation: center and scale
    centered = pointcloud - centroid
    transformed = centered / scale
    
    return transformed


def inverse_transform(pointcloud, transform_params):
    """
    Apply inverse transformation to convert normalized coordinates back to original space.
    
    Args:
        pointcloud: numpy array of shape (N, 3) in normalized space
        transform_params: dict with keys 'centroid' and 'scale'
        
    Returns:
        original_pointcloud: numpy array of shape (N, 3) in original space
    """
    if not isinstance(pointcloud, np.ndarray):
        pointcloud = np.array(pointcloud)
    
    centroid = np.array(transform_params['centroid'])
    scale = transform_params['scale']
    
    # Reverse the transformation: scale and uncenter
    scaled = pointcloud * scale
    original = scaled + centroid
    
    return original


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Normalize point cloud and save transformation parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Normalize a point cloud and save parameters
    python pointcloud_normalize.py -i input.xyz -o normalized.xyz -p transform.json
    
    # Apply saved transformation to a new point cloud
    python pointcloud_normalize.py -i new_input.xyz -o new_normalized.xyz --apply transform.json
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input point cloud file (xyz format)')
    parser.add_argument('-o', '--output', required=True, help='Output point cloud file (xyz format)')
    parser.add_argument('-p', '--params', help='Path to save transformation parameters (JSON)')
    parser.add_argument('--apply', help='Apply transformation from this parameter file')
    
    args = parser.parse_args()
    
    # Load point cloud
    print(f"Loading point cloud from: {args.input}")
    pc = np.loadtxt(args.input)
    if pc.ndim == 1:
        pc = pc.reshape(-1, 3)
    
    if args.apply:
        # Apply existing transformation
        print(f"Loading transformation parameters from: {args.apply}")
        params = load_transform_params(args.apply)
        normalized_pc = apply_transform(pc, params)
        print(f"Applied transformation with centroid={params['centroid']}, scale={params['scale']}")
    else:
        # Create new transformation
        normalized_pc, params = normalize_pointcloud(pc, return_params=True)
        print(f"Normalized point cloud:")
        print(f"  Centroid: {params['centroid']}")
        print(f"  Scale: {params['scale']}")
        
        if args.params:
            save_transform_params(params, args.params)
    
    # Save normalized point cloud
    print(f"Saving normalized point cloud to: {args.output}")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savetxt(args.output, normalized_pc, fmt='%.6f')
    print("Done!")
