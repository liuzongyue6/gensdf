#!/usr/bin/env python3
"""
Test script for point cloud normalization utilities.
"""

import sys
import os
import tempfile
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.pointcloud_normalize import (
    normalize_pointcloud,
    save_transform_params,
    load_transform_params,
    apply_transform,
    inverse_transform
)


def test_normalize_basic():
    """Test basic normalization functionality."""
    print("Testing basic normalization...")
    
    # Create a simple point cloud
    pc = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0]
    ])
    
    # Normalize
    pc_norm, params = normalize_pointcloud(pc, return_params=True)
    
    # Check that centroid is at origin
    centroid = np.mean(pc_norm, axis=0)
    assert np.allclose(centroid, [0.0, 0.0, 0.0], atol=1e-10), f"Centroid not at origin: {centroid}"
    
    # Check that params contain expected keys
    assert 'centroid' in params, "Missing 'centroid' in params"
    assert 'scale' in params, "Missing 'scale' in params"
    
    print("  ✓ Basic normalization passed")
    return True


def test_normalize_bounding_box():
    """Test that bounding box diagonal is normalized to [-1, 1]."""
    print("Testing bounding box normalization...")
    
    # Create a point cloud with known bounding box
    pc = np.array([
        [-5.0, -5.0, -5.0],
        [5.0, 5.0, 5.0]
    ])
    
    pc_norm = normalize_pointcloud(pc, return_params=False)
    
    # Check bounding box
    bbox_min = np.min(pc_norm, axis=0)
    bbox_max = np.max(pc_norm, axis=0)
    
    # Diagonal should be close to 2 (from -1 to 1)
    diagonal = np.linalg.norm(bbox_max - bbox_min)
    expected_diagonal = 2.0
    
    assert np.isclose(diagonal, expected_diagonal, atol=1e-6), \
        f"Diagonal not normalized correctly: {diagonal} vs {expected_diagonal}"
    
    print("  ✓ Bounding box normalization passed")
    return True


def test_save_load_transform():
    """Test saving and loading transformation parameters."""
    print("Testing save/load transformation parameters...")
    
    pc = np.array([
        [10.0, 20.0, 30.0],
        [15.0, 25.0, 35.0],
        [20.0, 30.0, 40.0]
    ])
    
    _, params = normalize_pointcloud(pc, return_params=True)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        save_transform_params(params, temp_file)
        
        # Load back
        loaded_params = load_transform_params(temp_file)
        
        # Check that parameters match
        assert np.allclose(loaded_params['centroid'], params['centroid']), \
            "Loaded centroid doesn't match"
        assert np.isclose(loaded_params['scale'], params['scale']), \
            "Loaded scale doesn't match"
        
        print("  ✓ Save/load transformation passed")
        return True
    finally:
        os.unlink(temp_file)


def test_apply_transform():
    """Test applying transformation to a new point cloud."""
    print("Testing apply transformation...")
    
    # Original point cloud
    pc1 = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0]
    ])
    
    # Get normalization parameters from pc1
    _, params = normalize_pointcloud(pc1, return_params=True)
    
    # New point cloud with same geometry but shifted
    pc2 = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0]
    ])
    
    # Apply transformation
    pc2_transformed = apply_transform(pc2, params)
    
    # Should be the same as normalizing pc2 directly when geometries are identical
    pc2_normalized = normalize_pointcloud(pc2, return_params=False)
    
    assert np.allclose(pc2_transformed, pc2_normalized, atol=1e-6), \
        "Transformed point cloud doesn't match expected"
    
    print("  ✓ Apply transformation passed")
    return True


def test_inverse_transform():
    """Test inverse transformation."""
    print("Testing inverse transformation...")
    
    # Original point cloud
    pc_original = np.array([
        [5.0, 10.0, 15.0],
        [15.0, 20.0, 25.0],
        [25.0, 30.0, 35.0]
    ])
    
    # Normalize and get parameters
    pc_normalized, params = normalize_pointcloud(pc_original, return_params=True)
    
    # Apply inverse transformation
    pc_recovered = inverse_transform(pc_normalized, params)
    
    # Should recover the original point cloud
    assert np.allclose(pc_recovered, pc_original, atol=1e-6), \
        "Inverse transformation didn't recover original"
    
    print("  ✓ Inverse transformation passed")
    return True


def test_consistent_normalization():
    """Test that normalization is consistent across multiple point clouds."""
    print("Testing consistent normalization across point clouds...")
    
    # Reference point cloud
    pc_ref = np.array([
        [0.0, 0.0, 0.0],
        [100.0, 100.0, 100.0]
    ])
    
    # Get normalization parameters
    _, params = normalize_pointcloud(pc_ref, return_params=True)
    
    # New point cloud in same coordinate system
    pc_new = np.array([
        [50.0, 50.0, 50.0],
        [75.0, 75.0, 75.0]
    ])
    
    # Apply same transformation
    pc_new_transformed = apply_transform(pc_new, params)
    
    # Check that the transformation is consistent
    # The center of pc_ref is at (50, 50, 50), which should map to (0, 0, 0)
    expected_center = np.array([0.0, 0.0, 0.0])
    actual_center = apply_transform(np.array([[50.0, 50.0, 50.0]]), params)[0]
    
    assert np.allclose(actual_center, expected_center, atol=1e-6), \
        f"Reference center not at origin: {actual_center}"
    
    print("  ✓ Consistent normalization passed")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Point Cloud Normalization Tests")
    print("=" * 60)
    
    tests = [
        test_normalize_basic,
        test_normalize_bounding_box,
        test_save_load_transform,
        test_apply_transform,
        test_inverse_transform,
        test_consistent_normalization
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
