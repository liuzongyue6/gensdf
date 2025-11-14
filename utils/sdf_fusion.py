#!/usr/bin/env python3
"""
SDF Fusion Utilities

This module provides functions to fuse multiple SDF representations using GenSDF's
encoder-decoder architecture. It supports two fusion methods:
1. Union: Takes the minimum SDF value at each point
2. Blend: Smooth blending using exponential weighting
"""

import numpy as np
import torch
import os
from pathlib import Path


def sdf_union(x, sdf_a, sdf_b):
    """
    SDF union operation: takes the minimum of two SDF values.
    
    This creates a union of two shapes by taking the minimum signed distance
    at each query point.
    
    Args:
        x: query points, numpy array of shape (N, 3) or (B, N, 3)
        sdf_a: first SDF function or values
        sdf_b: second SDF function or values
        
    Returns:
        Fused SDF values
    """
    # Handle both callable SDF functions and precomputed values
    if callable(sdf_a):
        sdf_a = sdf_a(x)
    if callable(sdf_b):
        sdf_b = sdf_b(x)
    
    return np.minimum(sdf_a, sdf_b)


def sdf_blend(x, sdf_a, sdf_b, alpha=10):
    """
    SDF smooth blending operation using exponential weighting.
    
    This creates a smooth blend between two shapes. Higher alpha values
    make the blend sharper (more like union), while lower values create
    smoother transitions.
    
    Args:
        x: query points, numpy array of shape (N, 3) or (B, N, 3)
        sdf_a: first SDF function or values
        sdf_b: second SDF function or values
        alpha: blending parameter (default=10). Higher = sharper transition.
        
    Returns:
        Fused SDF values
    """
    # Handle both callable SDF functions and precomputed values
    if callable(sdf_a):
        sdf_a = sdf_a(x)
    if callable(sdf_b):
        sdf_b = sdf_b(x)
    
    # Smooth exponential blend
    f1 = np.exp(-alpha * sdf_a)
    f2 = np.exp(-alpha * sdf_b)
    result = -np.log(f1 + f2 + 1e-10) / alpha  # Add small epsilon to avoid log(0)
    
    return result


class SDFFusion:
    """
    Helper class for fusing point clouds using GenSDF encoder-decoder.
    
    This class handles:
    1. Encoding point clouds to latent representations
    2. Fusing SDF queries using union or blend operations
    3. Decoding fused SDFs back to 3D meshes
    """
    
    def __init__(self, model):
        """
        Initialize SDF fusion with a trained GenSDF model.
        
        Args:
            model: trained GenSDF model instance
        """
        self.model = model
        self.model.eval()
        
    def encode_pointcloud(self, pointcloud):
        """
        Encode a point cloud using GenSDF's encoder.
        
        Args:
            pointcloud: torch tensor of shape (1, N, 3) or (N, 3)
            
        Returns:
            Encoded representation (latent features) as a function of query points
        """
        if pointcloud.dim() == 2:
            pointcloud = pointcloud.unsqueeze(0)
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        pointcloud = pointcloud.to(device)
        
        # Return a function that computes SDF for any query points
        def sdf_function(query_points):
            """Query SDF values at given points"""
            if isinstance(query_points, np.ndarray):
                query_points = torch.from_numpy(query_points).float()
            
            if query_points.dim() == 2:
                query_points = query_points.unsqueeze(0)
            
            query_points = query_points.to(device)
            
            with torch.no_grad():
                sdf_values = self.model(pointcloud, query_points)
            
            if isinstance(sdf_values, torch.Tensor):
                sdf_values = sdf_values.cpu().numpy()
            
            return sdf_values.squeeze()
        
        return sdf_function
    
    def fuse_pointclouds(self, pc_a, pc_b, method='union', alpha=10):
        """
        Fuse two point clouds using specified fusion method.
        
        Args:
            pc_a: first point cloud, torch tensor of shape (1, N, 3) or (N, 3)
            pc_b: second point cloud, torch tensor of shape (1, M, 3) or (M, 3)
            method: fusion method, either 'union' or 'blend'
            alpha: blending parameter for 'blend' method (default=10)
            
        Returns:
            fused_sdf_function: a function that takes query points and returns fused SDF values
        """
        # Encode both point clouds
        print("Encoding first point cloud...")
        sdf_a = self.encode_pointcloud(pc_a)
        
        print("Encoding second point cloud...")
        sdf_b = self.encode_pointcloud(pc_b)
        
        # Create fused SDF function
        if method == 'union':
            print("Creating union of two SDFs...")
            def fused_sdf(x):
                return sdf_union(x, sdf_a, sdf_b)
        elif method == 'blend':
            print(f"Creating smooth blend of two SDFs (alpha={alpha})...")
            def fused_sdf(x):
                return sdf_blend(x, sdf_a, sdf_b, alpha=alpha)
        else:
            raise ValueError(f"Unknown fusion method: {method}. Use 'union' or 'blend'.")
        
        return fused_sdf
    
    def reconstruct_from_sdf(self, sdf_function, output_path, resolution=192, batch_size=64000):
        """
        Reconstruct a 3D mesh from an SDF function using marching cubes.
        
        This is similar to GenSDF's reconstruction but works with arbitrary SDF functions
        including fused SDFs.
        
        Args:
            sdf_function: function that takes query points and returns SDF values
            output_path: path to save the output PLY file
            resolution: grid resolution for marching cubes (default=192)
            batch_size: batch size for SDF queries (default=64000)
        """
        import skimage.measure
        import plyfile
        
        print(f"Reconstructing mesh with resolution={resolution}...")
        
        # Create 3D grid
        N = resolution
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)
        
        # Generate all grid points
        overall_index = torch.arange(0, N ** 3, 1, dtype=torch.long)
        samples = torch.zeros(N ** 3, 3)
        
        # Convert indices to 3D coordinates
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index // N) % N
        samples[:, 0] = (overall_index // (N * N)) % N
        
        # Convert to world coordinates
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[2]
        
        # Query SDF values in batches
        sdf_values = np.zeros(N ** 3)
        num_points = samples.shape[0]
        
        for i in range(0, num_points, batch_size):
            end_idx = min(i + batch_size, num_points)
            batch_points = samples[i:end_idx].numpy()
            sdf_values[i:end_idx] = sdf_function(batch_points)
            
            if (i // batch_size) % 10 == 0:
                progress = (end_idx / num_points) * 100
                print(f"Progress: {progress:.1f}%")
        
        # Reshape to 3D grid
        sdf_grid = sdf_values.reshape(N, N, N)
        
        print("Running marching cubes...")
        try:
            verts, faces, normals, values = skimage.measure.marching_cubes(
                sdf_grid, level=0.0, spacing=[voxel_size] * 3
            )
        except Exception as e:
            print(f"Marching cubes failed: {e}")
            return
        
        # Transform vertices to world coordinates
        mesh_points = np.zeros_like(verts)
        mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
        mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
        mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]
        
        # Save to PLY file
        print(f"Saving mesh to: {output_path}")
        num_verts = verts.shape[0]
        num_faces = faces.shape[0]
        
        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        for i in range(num_verts):
            verts_tuple[i] = tuple(mesh_points[i, :])
        
        faces_building = []
        for i in range(num_faces):
            faces_building.append(((faces[i, :].tolist(),)))
        faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])
        
        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")
        
        ply_data = plyfile.PlyData([el_verts, el_faces])
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        ply_data.write(output_path)
        print(f"Mesh saved with {num_verts} vertices and {num_faces} faces")


def create_fusion_example(model, pc1_path, pc2_path, output_dir, method='union', alpha=10, 
                          resolution=192, batch_size=64000):
    """
    Complete example workflow: load two point clouds, fuse them, and reconstruct.
    
    Args:
        model: trained GenSDF model
        pc1_path: path to first point cloud file (.xyz, .ply, or .csv)
        pc2_path: path to second point cloud file
        output_dir: directory to save outputs
        method: fusion method ('union' or 'blend')
        alpha: blending parameter for 'blend' method
        resolution: marching cubes resolution
        batch_size: SDF query batch size
    """
    import trimesh
    import pandas as pd
    
    # Load point clouds
    def load_pointcloud(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            pc = pd.read_csv(path, sep=',', header=None).values
            # If CSV has SDF values, only take surface points (sdv=0)
            if pc.shape[1] == 4:
                pc = pc[pc[:, -1] == 0][:, :3]
        elif ext == '.ply':
            pc = trimesh.load(path).vertices
        elif ext == '.xyz':
            pc = np.loadtxt(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return torch.from_numpy(pc).float()
    
    print(f"Loading point cloud 1 from: {pc1_path}")
    pc1 = load_pointcloud(pc1_path)
    print(f"  Loaded {pc1.shape[0]} points")
    
    print(f"Loading point cloud 2 from: {pc2_path}")
    pc2 = load_pointcloud(pc2_path)
    print(f"  Loaded {pc2.shape[0]} points")
    
    # Create fusion instance
    fusion = SDFFusion(model)
    
    # Fuse point clouds
    fused_sdf = fusion.fuse_pointclouds(pc1, pc2, method=method, alpha=alpha)
    
    # Reconstruct mesh from fused SDF
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"fused_{method}.ply")
    fusion.reconstruct_from_sdf(fused_sdf, output_path, resolution=resolution, batch_size=batch_size)
    
    print(f"\nFusion complete! Output saved to: {output_path}")


if __name__ == "__main__":
    print("This module provides SDF fusion utilities.")
    print("Import it in your scripts to use the fusion functions.")
    print("\nExample usage:")
    print("""
    from utils.sdf_fusion import SDFFusion, create_fusion_example
    
    # Load your trained GenSDF model
    model = ...  # Load your model here
    
    # Fuse two point clouds
    create_fusion_example(
        model=model,
        pc1_path='path/to/first.ply',
        pc2_path='path/to/second.ply',
        output_dir='output',
        method='union',  # or 'blend'
        alpha=10
    )
    """)
