# Data Preprocessing Scripts

This directory contains preprocessing scripts to convert various data formats to the CSV format required by GenSDF.

## CAE to CSV Conversion

The `cae_to_csv.py` script converts CAE (Computer-Aided Engineering) format data to CSV format.

### CAE Format

CAE data typically follows this format:
```
GRID,<id>,,<x>,<y>,<z>
```

Example:
```
GRID,4649,,2661.937,-433.604,113.7815
GRID,4650,,2662.825,-436.489,116.7319
GRID,4651,,2662.018,-433.853,116.4177
```

### CSV Format

The output CSV format is:
```
x,y,z,sdv
```
where `sdv` is the signed distance value (0 for surface points from CAE data).

### Usage

```bash
# Basic usage - convert CAE to CSV with recentering and normalization
python cae_to_csv.py -i input_file.cae -o output_file.csv

# Convert without recentering/normalization
python cae_to_csv.py -i input_file.cae -o output_file.csv --no-recenter --no-normalize

# Convert and save to GenSDF data directory
python cae_to_csv.py -i data.cae -o ../acronym/MyObject/instance_id/sdf_data.csv
```

### Options

- `-i, --input`: Input CAE file path (required)
- `-o, --output`: Output CSV file path (required)
- `--no-recenter`: Skip recentering of coordinates
- `--no-normalize`: Skip normalization of coordinates

### Notes

- By default, the script applies recentering and normalization following GenSDF preprocessing conventions
- CAE surface points are assigned SDF value of 0
- The script automatically creates output directories if they don't exist

## Other Preprocessing Scripts

### sdf_gen.cpp
Generates signed distance field data from mesh files.

### sdf_gen_from_pc.cpp  
Generates signed distance field data from point clouds.

### preprocess_script.py
Batch processing script for preprocessing multiple objects.

### create_split.py
Creates train/test split files for datasets.
