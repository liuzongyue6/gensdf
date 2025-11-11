#!/usr/bin/env python3
"""
Simple test script for CAE to CSV conversion functionality.
This test verifies the core functionality of the cae_to_csv.py script.
"""

import sys
import os
import tempfile
import numpy as np

# Add the preprocessing directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data', 'preprocessing'))

from cae_to_csv import parse_cae_file, recenter_and_normalize, cae_to_csv


def test_parse_cae_file():
    """Test parsing of CAE format."""
    print("Testing CAE parsing...")
    
    # Create a temporary CAE file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cae', delete=False) as f:
        f.write("GRID,4649,,2661.937,-433.604,113.7815\n")
        f.write("GRID,4650,,2662.825,-436.489,116.7319\n")
        f.write("GRID,4651,,2662.018,-433.853,116.4177\n")
        temp_file = f.name
    
    try:
        coordinates = parse_cae_file(temp_file)
        
        # Verify shape
        assert coordinates.shape == (3, 3), f"Expected shape (3, 3), got {coordinates.shape}"
        
        # Verify first point
        expected_first = np.array([2661.937, -433.604, 113.7815])
        np.testing.assert_array_almost_equal(coordinates[0], expected_first, decimal=4)
        
        print("✓ CAE parsing test passed")
        return True
    finally:
        os.unlink(temp_file)


def test_recenter_and_normalize():
    """Test recentering and normalization."""
    print("Testing recentering and normalization...")
    
    # Create simple test data
    coordinates = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0]
    ])
    
    result = recenter_and_normalize(coordinates)
    
    # After recentering, mean should be close to zero
    mean = np.mean(result, axis=0)
    np.testing.assert_array_almost_equal(mean, [0.0, 0.0, 0.0], decimal=10)
    
    print("✓ Recenter and normalize test passed")
    return True


def test_cae_to_csv_conversion():
    """Test full conversion pipeline."""
    print("Testing full CAE to CSV conversion...")
    
    # Create temporary CAE file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cae', delete=False) as f:
        f.write("GRID,4649,,2661.937,-433.604,113.7815\n")
        f.write("GRID,4650,,2662.825,-436.489,116.7319\n")
        f.write("GRID,4651,,2662.018,-433.853,116.4177\n")
        temp_cae = f.name
    
    # Create temporary CSV output file
    temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    
    try:
        # Convert
        cae_to_csv(temp_cae, temp_csv, recenter=True, normalize=True)
        
        # Verify output exists
        assert os.path.exists(temp_csv), "Output CSV file not created"
        
        # Load and verify CSV
        data = np.loadtxt(temp_csv, delimiter=',')
        
        # Should have 3 rows, 4 columns (x, y, z, sdv)
        assert data.shape == (3, 4), f"Expected shape (3, 4), got {data.shape}"
        
        # SDF values should all be 0
        np.testing.assert_array_equal(data[:, 3], [0.0, 0.0, 0.0])
        
        print("✓ Full conversion test passed")
        return True
    finally:
        os.unlink(temp_cae)
        if os.path.exists(temp_csv):
            os.unlink(temp_csv)


def test_cae_to_csv_no_normalization():
    """Test conversion without normalization."""
    print("Testing CAE to CSV conversion without normalization...")
    
    # Create temporary CAE file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cae', delete=False) as f:
        f.write("GRID,1,,1.0,2.0,3.0\n")
        f.write("GRID,2,,4.0,5.0,6.0\n")
        temp_cae = f.name
    
    # Create temporary CSV output file
    temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    
    try:
        # Convert without normalization
        cae_to_csv(temp_cae, temp_csv, recenter=False, normalize=False)
        
        # Load and verify CSV
        data = np.loadtxt(temp_csv, delimiter=',')
        
        # Coordinates should be unchanged (no normalization)
        expected = np.array([
            [1.0, 2.0, 3.0, 0.0],
            [4.0, 5.0, 6.0, 0.0]
        ])
        np.testing.assert_array_almost_equal(data, expected, decimal=6)
        
        print("✓ No normalization test passed")
        return True
    finally:
        os.unlink(temp_cae)
        if os.path.exists(temp_csv):
            os.unlink(temp_csv)


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running CAE to CSV conversion tests")
    print("=" * 60)
    
    tests = [
        test_parse_cae_file,
        test_recenter_and_normalize,
        test_cae_to_csv_conversion,
        test_cae_to_csv_no_normalization
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
