# CAE to CSV Conversion Examples

This document provides examples of using the CAE to CSV preprocessing script.

## Example 1: Basic Usage

Convert a CAE file to CSV with default settings (recentering and normalization):

```bash
python data/preprocessing/cae_to_csv.py -i input.cae -o output.csv
```

## Example 2: Preserve Original Coordinates

Convert without recentering or normalization to preserve original coordinates:

```bash
python data/preprocessing/cae_to_csv.py -i input.cae -o output.csv --no-recenter --no-normalize
```

## Example 3: Batch Processing

Process multiple CAE files in a directory:

```bash
#!/bin/bash
# Process all .cae files in a directory

for cae_file in *.cae; do
    base_name=$(basename "$cae_file" .cae)
    python data/preprocessing/cae_to_csv.py -i "$cae_file" -o "${base_name}.csv"
    echo "Converted $cae_file to ${base_name}.csv"
done
```

## Example 4: Integration with GenSDF Data Structure

Convert and save directly to GenSDF data directory structure:

```bash
# For a new object in the acronym dataset
mkdir -p data/acronym/MyObject/instance_001
python data/preprocessing/cae_to_csv.py \
    -i my_object.cae \
    -o data/acronym/MyObject/instance_001/sdf_data.csv
```

## Example 5: Using with test_single.py

After converting CAE to CSV, you can use it with GenSDF's test_single.py:

```bash
# Convert CAE to CSV
python data/preprocessing/cae_to_csv.py -i my_object.cae -o my_object.csv

# Test with GenSDF
python test_single.py -f my_object.csv -o output_dir -e config/gensdf/semi -r last
```

## Sample Input (CAE Format)

```
GRID,4649,,2661.937,-433.604,113.7815
GRID,4650,,2662.825,-436.489,116.7319
GRID,4651,,2662.018,-433.853,116.4177
GRID,4652,,2663.893,-440.114,114.9757
GRID,4653,,2662.735,-436.186,114.3488
```

## Sample Output (CSV Format)

```
-0.040449,0.243181,-0.379674,0.000000
-0.020152,0.177241,-0.312239,0.000000
-0.038597,0.237490,-0.319420,0.000000
0.004258,0.094387,-0.352379,0.000000
-0.022209,0.184166,-0.366707,0.000000
```

Note: The coordinates are recentered and normalized by default. The last column is the signed distance value (0 for surface points).
