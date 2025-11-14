#!/usr/bin/env python3
"""
Simple demonstration of SDF fusion operations (without requiring a trained model).

This script demonstrates the mathematical operations used in SDF fusion
and creates visualization data to understand the behavior.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def simple_sphere_sdf(center, radius):
    """Create a simple sphere SDF function."""
    def sdf(points):
        if isinstance(points, list):
            points = np.array(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        distances = np.linalg.norm(points - center, axis=-1)
        return distances - radius
    return sdf


def simple_box_sdf(center, size):
    """Create a simple box SDF function."""
    def sdf(points):
        if isinstance(points, list):
            points = np.array(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        # Distance from center
        d = np.abs(points - center) - size / 2
        
        # Outside distance
        outside = np.linalg.norm(np.maximum(d, 0), axis=-1)
        
        # Inside distance
        inside = np.minimum(np.max(d, axis=-1), 0)
        
        return outside + inside
    return sdf


def sdf_union(x, sdf_a, sdf_b):
    """SDF union: minimum of two SDFs."""
    return np.minimum(sdf_a(x), sdf_b(x))


def sdf_blend(x, sdf_a, sdf_b, alpha=10):
    """SDF smooth blend."""
    f1 = np.exp(-alpha * sdf_a(x))
    f2 = np.exp(-alpha * sdf_b(x))
    return -np.log(f1 + f2 + 1e-10) / alpha


def create_2d_visualization(output_dir):
    """Create 2D cross-section visualization of fusion methods."""
    print("Creating 2D visualization...")
    
    # Create two simple SDFs (circles in 2D)
    sdf1 = simple_sphere_sdf(center=np.array([-0.3, 0.0, 0.0]), radius=0.4)
    sdf2 = simple_sphere_sdf(center=np.array([0.3, 0.0, 0.0]), radius=0.4)
    
    # Create grid
    x = np.linspace(-1, 1, 200)
    y = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Evaluate SDFs
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
    
    sdf1_values = sdf1(points).reshape(X.shape)
    sdf2_values = sdf2(points).reshape(X.shape)
    union_values = sdf_union(points, sdf1, sdf2).reshape(X.shape)
    blend_values_5 = sdf_blend(points, sdf1, sdf2, alpha=5).reshape(X.shape)
    blend_values_10 = sdf_blend(points, sdf1, sdf2, alpha=10).reshape(X.shape)
    blend_values_20 = sdf_blend(points, sdf1, sdf2, alpha=20).reshape(X.shape)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot individual SDFs
    im1 = axes[0, 0].contourf(X, Y, sdf1_values, levels=20, cmap='RdBu')
    axes[0, 0].contour(X, Y, sdf1_values, levels=[0], colors='black', linewidths=2)
    axes[0, 0].set_title('SDF 1 (Left Circle)')
    axes[0, 0].axis('equal')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].contourf(X, Y, sdf2_values, levels=20, cmap='RdBu')
    axes[0, 1].contour(X, Y, sdf2_values, levels=[0], colors='black', linewidths=2)
    axes[0, 1].set_title('SDF 2 (Right Circle)')
    axes[0, 1].axis('equal')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot union
    im3 = axes[0, 2].contourf(X, Y, union_values, levels=20, cmap='RdBu')
    axes[0, 2].contour(X, Y, union_values, levels=[0], colors='black', linewidths=2)
    axes[0, 2].set_title('Union (Minimum)')
    axes[0, 2].axis('equal')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot blends with different alpha
    im4 = axes[1, 0].contourf(X, Y, blend_values_5, levels=20, cmap='RdBu')
    axes[1, 0].contour(X, Y, blend_values_5, levels=[0], colors='black', linewidths=2)
    axes[1, 0].set_title('Blend (alpha=5, smooth)')
    axes[1, 0].axis('equal')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].contourf(X, Y, blend_values_10, levels=20, cmap='RdBu')
    axes[1, 1].contour(X, Y, blend_values_10, levels=[0], colors='black', linewidths=2)
    axes[1, 1].set_title('Blend (alpha=10, moderate)')
    axes[1, 1].axis('equal')
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].contourf(X, Y, blend_values_20, levels=20, cmap='RdBu')
    axes[1, 2].contour(X, Y, blend_values_20, levels=[0], colors='black', linewidths=2)
    axes[1, 2].set_title('Blend (alpha=20, sharp)')
    axes[1, 2].axis('equal')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fusion_comparison_2d.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved 2D visualization to: {output_path}")
    plt.close()


def create_1d_profile(output_dir):
    """Create 1D profile showing SDF values along a line."""
    print("Creating 1D profile...")
    
    # Create two overlapping spheres
    sdf1 = simple_sphere_sdf(center=np.array([-0.3, 0.0, 0.0]), radius=0.4)
    sdf2 = simple_sphere_sdf(center=np.array([0.3, 0.0, 0.0]), radius=0.4)
    
    # Sample along x-axis
    x = np.linspace(-1, 1, 300)
    points = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=-1)
    
    sdf1_values = sdf1(points)
    sdf2_values = sdf2(points)
    union_values = sdf_union(points, sdf1, sdf2)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    plt.plot(x, sdf1_values, 'b-', label='SDF 1', linewidth=2)
    plt.plot(x, sdf2_values, 'r-', label='SDF 2', linewidth=2)
    plt.plot(x, union_values, 'g-', label='Union', linewidth=2, linestyle='--')
    
    # Add blends with different alpha values
    for alpha in [5, 10, 20]:
        blend_values = sdf_blend(points, sdf1, sdf2, alpha=alpha)
        plt.plot(x, blend_values, label=f'Blend (α={alpha})', linewidth=1.5, alpha=0.7)
    
    plt.axhline(y=0, color='k', linestyle=':', linewidth=1, label='Zero level (surface)')
    plt.xlabel('Position along x-axis')
    plt.ylabel('SDF Value')
    plt.title('SDF Fusion Methods - 1D Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, 'fusion_profile_1d.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved 1D profile to: {output_path}")
    plt.close()


def demonstrate_normalization(output_dir):
    """Demonstrate point cloud normalization."""
    print("Demonstrating normalization...")
    
    # Create a simple point cloud (cube)
    np.random.seed(42)
    n_points = 100
    original_pc = np.random.randn(n_points, 3) * 50 + np.array([100, 200, 150])
    
    # Normalize
    from utils.pointcloud_normalize import normalize_pointcloud
    normalized_pc, params = normalize_pointcloud(original_pc, return_params=True)
    
    # Create visualization
    fig = plt.figure(figsize=(14, 6))
    
    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_pc[:, 0], original_pc[:, 1], original_pc[:, 2], 
                c='blue', alpha=0.6, s=20)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Original Point Cloud\nCentroid: [{params["centroid"][0]:.1f}, {params["centroid"][1]:.1f}, {params["centroid"][2]:.1f}]')
    
    # Normalized
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(normalized_pc[:, 0], normalized_pc[:, 1], normalized_pc[:, 2], 
                c='red', alpha=0.6, s=20)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Normalized Point Cloud\nCentroid: [0.0, 0.0, 0.0]\nScale: {params["scale"]:.2f}')
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    
    # Draw bounding box for normalized
    bbox_corners = np.array([[-1, -1, -1], [1, 1, 1]])
    for i in [0, 1]:
        for j in [0, 1]:
            ax2.plot([bbox_corners[0, 0], bbox_corners[1, 0]], 
                    [bbox_corners[i, 1], bbox_corners[i, 1]], 
                    [bbox_corners[j, 2], bbox_corners[j, 2]], 'k--', alpha=0.3)
            ax2.plot([bbox_corners[i, 0], bbox_corners[i, 0]], 
                    [bbox_corners[0, 1], bbox_corners[1, 1]], 
                    [bbox_corners[j, 2], bbox_corners[j, 2]], 'k--', alpha=0.3)
            ax2.plot([bbox_corners[i, 0], bbox_corners[i, 0]], 
                    [bbox_corners[j, 1], bbox_corners[j, 1]], 
                    [bbox_corners[0, 2], bbox_corners[1, 2]], 'k--', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'normalization_demo.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved normalization demo to: {output_path}")
    plt.close()
    
    # Print statistics
    print(f"\n  Original point cloud stats:")
    print(f"    Min: {np.min(original_pc, axis=0)}")
    print(f"    Max: {np.max(original_pc, axis=0)}")
    print(f"    Centroid: {params['centroid']}")
    print(f"\n  Normalized point cloud stats:")
    print(f"    Min: {np.min(normalized_pc, axis=0)}")
    print(f"    Max: {np.max(normalized_pc, axis=0)}")
    print(f"    Centroid: {np.mean(normalized_pc, axis=0)}")
    print(f"    Bbox diagonal: {np.linalg.norm(np.max(normalized_pc, axis=0) - np.min(normalized_pc, axis=0)):.4f}")


def main():
    """Run all demonstrations."""
    output_dir = 'demo_output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("SDF Fusion and Normalization Demonstration")
    print("=" * 70)
    
    # Import sys to add path
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create visualizations
    create_2d_visualization(output_dir)
    create_1d_profile(output_dir)
    demonstrate_normalization(output_dir)
    
    print("\n" + "=" * 70)
    print("Demonstration complete!")
    print(f"Output files saved to: {output_dir}/")
    print("=" * 70)
    print("\nGenerated files:")
    for filename in os.listdir(output_dir):
        if filename.endswith('.png'):
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
