#!/usr/bin/env python3
"""
CAE Data Preprocessing Script

This script converts CAE format data to CSV format compatible with GenSDF training.

CAE Format:
    GRID,<id>,,<x>,<y>,<z>
    Example: GRID,4649,,2661.937,-433.604,113.7815

CSV Format:
    x,y,z,sdv
    where sdv is the signed distance value (0 for surface points)
"""

import argparse
import os
import numpy as np
import pandas as pd


def parse_cae_file(cae_file_path):
    """
    Parse CAE file and extract coordinates.
    
    Args:
        cae_file_path: Path to the CAE input file
        
    Returns:
        numpy array of shape (N, 3) containing x, y, z coordinates
    """
    coordinates = []
    
    with open(cae_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse line: GRID,<id>,,<x>,<y>,<z>
            parts = line.split(',')
            
            # Check if it's a valid GRID line
            if len(parts) >= 6 and parts[0] == 'GRID':
                try:
                    # Extract x, y, z coordinates (last 3 values)
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                    coordinates.append([x, y, z])
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line: {line}")
                    print(f"Error: {e}")
                    continue
    
    if len(coordinates) == 0:
        raise ValueError(f"No valid coordinates found in {cae_file_path}")
    
    return np.array(coordinates)


def recenter_and_normalize(coordinates):
    """
    Recenter and normalize coordinates following GenSDF preprocessing.
    
    Args:
        coordinates: numpy array of shape (N, 3)
        
    Returns:
        Recentered and normalized coordinates
    """
    # Recenter: subtract mean
    centered = coordinates - np.mean(coordinates, axis=0)
    
    # Normalize: divide by bounding box diagonal length
    bbox_min = np.min(centered, axis=0)
    bbox_max = np.max(centered, axis=0)
    bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
    
    if bbox_diagonal > 0:
        normalized = centered / bbox_diagonal
    else:
        normalized = centered
    
    return normalized


def cae_to_csv(cae_file_path, output_csv_path, recenter=True, normalize=True):
    """
    Convert CAE file to CSV format.
    
    Args:
        cae_file_path: Path to input CAE file
        output_csv_path: Path to output CSV file
        recenter: Whether to recenter the coordinates
        normalize: Whether to normalize the coordinates
    """
    print(f"Reading CAE file: {cae_file_path}")
    coordinates = parse_cae_file(cae_file_path)
    print(f"Parsed {len(coordinates)} points")
    
    # Apply recentering and normalization if requested
    if recenter or normalize:
        if recenter and normalize:
            print("Recentering and normalizing coordinates...")
        elif recenter:
            print("Recentering coordinates...")
        elif normalize:
            print("Normalizing coordinates...")
        coordinates = recenter_and_normalize(coordinates)
    
    # Add SDF values (0 for surface points from CAE data)
    # CAE data represents surface points, so SDF value is 0
    sdv = np.zeros((coordinates.shape[0], 1))
    
    # Combine coordinates and SDF values
    csv_data = np.hstack([coordinates, sdv])
    
    # Save to CSV
    print(f"Saving CSV file: {output_csv_path}")
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:  # 只有当目录路径不为空时才创建目录
        os.makedirs(output_dir, exist_ok=True)
    np.savetxt(output_csv_path, csv_data, delimiter=',', fmt='%.6f')
    
    print(f"Successfully converted {len(coordinates)} points to CSV format")
    print(f"Output saved to: {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert CAE format data to CSV format for GenSDF training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert single CAE file to CSV
    python cae_to_csv.py -i data.cae -o output.csv
    
    # Convert without recentering/normalization
    python cae_to_csv.py -i data.cae -o output.csv --no-recenter --no-normalize
    
    # Convert and save to specific directory
    python cae_to_csv.py -i data.cae -o data/acronym/MyObject/model/sdf_data.csv
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input CAE file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--no-recenter',
        action='store_true',
        help='Do not recenter the coordinates'
    )
    
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Do not normalize the coordinates'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Convert CAE to CSV
    cae_to_csv(
        args.input,
        args.output,
        recenter=not args.no_recenter,
        normalize=not args.no_normalize
    )


if __name__ == "__main__":
    main()
