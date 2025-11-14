# Point Cloud Normalization and SDF Fusion Guide

This guide explains how to use the new point cloud normalization and SDF fusion features in GenSDF.

## Overview

The new functionality provides:

1. **Point Cloud Normalization**: Normalize point clouds to a unified coordinate space
2. **Transformation Management**: Save and apply normalization parameters across multiple point clouds
3. **SDF Fusion**: Combine two point clouds using GenSDF's encoder-decoder architecture
4. **Mesh Reconstruction**: Generate 3D meshes from fused SDF representations

## 1. Point Cloud Normalization

### What it does

The normalization process:
1. Centers the point cloud's centroid at the origin (0, 0, 0)
2. Scales the bounding box diagonal to span the range [-1, 1]

This ensures that different parts can be operated on in a unified coordinate space, which is essential for accurate SDF fusion.

### Basic Usage

```python
from utils.pointcloud_normalize import normalize_pointcloud, save_transform_params

# Load your point cloud (Nx3 numpy array)
import numpy as np
pointcloud = np.loadtxt('input.xyz')

# Normalize and get transformation parameters
normalized_pc, transform_params = normalize_pointcloud(pointcloud, return_params=True)

# Save transformation parameters for later use
save_transform_params(transform_params, 'transform.json')

# Save normalized point cloud
np.savetxt('normalized.xyz', normalized_pc)
```

### Command Line Usage

```bash
# Normalize a point cloud and save parameters
python utils/pointcloud_normalize.py -i input.xyz -o normalized.xyz -p transform.json

# Apply saved transformation to a new point cloud
python utils/pointcloud_normalize.py -i new_input.xyz -o new_normalized.xyz --apply transform.json
```

### Apply Transformation to New Point Clouds

Once you have normalization parameters from a reference part, you can apply the same transformation to new parts:

```python
from utils.pointcloud_normalize import load_transform_params, apply_transform

# Load saved transformation parameters
transform_params = load_transform_params('transform.json')

# Load new point cloud
new_pc = np.loadtxt('new_part.xyz')

# Apply the same transformation
new_pc_normalized = apply_transform(new_pc, transform_params)
```

### Inverse Transformation

To convert normalized coordinates back to original space:

```python
from utils.pointcloud_normalize import inverse_transform

# Convert back to original coordinate system
original_pc = inverse_transform(normalized_pc, transform_params)
```

## 2. SDF Fusion

### Fusion Methods

Two fusion methods are provided:

#### Union (Minimum)
Takes the minimum SDF value at each point, creating a union of two shapes:

```python
from utils.sdf_fusion import sdf_union

# fused_sdf = sdf_union(query_points, sdf_part1, sdf_part2)
```

#### Smooth Blend
Creates a smooth transition between two shapes using exponential weighting:

```python
from utils.sdf_fusion import sdf_blend

# fused_sdf = sdf_blend(query_points, sdf_part1, sdf_part2, alpha=10)
# Higher alpha = sharper transition (more like union)
# Lower alpha = smoother blending
```

### Complete Fusion Workflow

```python
import torch
from model import GenSDF
from utils.sdf_fusion import SDFFusion

# 1. Load trained GenSDF model
model = GenSDF(specs, None).cuda()
checkpoint = torch.load('path/to/checkpoint.ckpt')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 2. Load and normalize your point clouds
pc1 = torch.from_numpy(normalized_pc1).float()
pc2 = torch.from_numpy(normalized_pc2).float()

# 3. Create fusion instance
fusion = SDFFusion(model)

# 4. Fuse point clouds
fused_sdf = fusion.fuse_pointclouds(
    pc1, pc2, 
    method='union',  # or 'blend'
    alpha=10         # only used for 'blend' method
)

# 5. Reconstruct 3D mesh from fused SDF
fusion.reconstruct_from_sdf(
    fused_sdf, 
    output_path='output/fused.ply',
    resolution=192,      # higher = better quality, more memory
    batch_size=64000     # lower if out of memory
)
```

## 3. Complete Example Script

A complete example script `fusion_example.py` is provided that handles the entire workflow:

```bash
# Basic fusion with union method
python fusion_example.py \
    -m config/gensdf/semi \
    -p1 data/part1.ply \
    -p2 data/part2.ply \
    -o output/fusion_result

# Fusion with smooth blending
python fusion_example.py \
    -m config/gensdf/semi \
    -p1 data/part1.ply \
    -p2 data/part2.ply \
    -o output/fusion_result \
    --method blend \
    --alpha 15

# Use existing transformation parameters
python fusion_example.py \
    -m config/gensdf/semi \
    -p1 data/part1.ply \
    -p2 data/part2.ply \
    -o output/fusion_result \
    --ref_transform transform.json

# Higher resolution reconstruction
python fusion_example.py \
    -m config/gensdf/semi \
    -p1 data/part1.ply \
    -p2 data/part2.ply \
    -o output/fusion_result \
    --resolution 256
```

### Command Line Options

- `-m, --model_dir`: Path to model directory with specs.json and checkpoint
- `-p1, --pointcloud1`: Path to first point cloud file (.ply, .xyz, or .csv)
- `-p2, --pointcloud2`: Path to second point cloud file
- `-o, --output`: Output directory for results
- `--method`: Fusion method: 'union' or 'blend' (default: union)
- `--alpha`: Blending parameter for blend method (default: 10)
- `--resolution`: Marching cubes resolution (default: 128)
- `--batch_size`: Batch size for SDF queries (default: 64000)
- `--checkpoint`: Checkpoint to load (default: last)
- `--ref_transform`: Use existing transformation parameters from JSON file
- `--save_transform`: Save transformation parameters to JSON file
- `--no_normalize`: Skip normalization (assume already normalized)

## 4. Workflow for Multiple Parts

### Scenario: Fusing Multiple Components

When working with multiple parts that need to be combined:

**Step 1: Normalize the first (reference) part**

```bash
python utils/pointcloud_normalize.py \
    -i data/reference_part.ply \
    -o data/reference_normalized.xyz \
    -p data/reference_transform.json
```

**Step 2: Apply the same transformation to other parts**

```bash
python utils/pointcloud_normalize.py \
    -i data/part2.ply \
    -o data/part2_normalized.xyz \
    --apply data/reference_transform.json

python utils/pointcloud_normalize.py \
    -i data/part3.ply \
    -o data/part3_normalized.xyz \
    --apply data/reference_transform.json
```

**Step 3: Fuse the normalized parts**

```bash
# Fuse part 1 and part 2
python fusion_example.py \
    -m config/gensdf/semi \
    -p1 data/reference_normalized.xyz \
    -p2 data/part2_normalized.xyz \
    -o output/fusion_1_2 \
    --no_normalize

# Fuse the result with part 3
# (You can use the output mesh and convert it to point cloud first)
```

## 5. Integration with CAE Data

The normalization follows the same format as `cae_to_csv.py`. To use CAE data:

```bash
# Convert CAE to CSV with normalization
python data/preprocessing/cae_to_csv.py \
    -i data/part.cae \
    -o data/part.csv

# The CSV format can be directly used with fusion_example.py
python fusion_example.py \
    -m config/gensdf/semi \
    -p1 data/part1.csv \
    -p2 data/part2.csv \
    -o output/result
```

## 6. Understanding the Decoder

GenSDF's decoder takes:
- **Input**: Concatenated latent features + 3D query coordinates
- **Output**: Signed distance value at each query point

The fusion process works by:
1. Encoding each point cloud to latent features using GenSDF's encoder
2. Computing SDF values for the same query points from both encodings
3. Combining SDF values using union or blend operations
4. Using marching cubes to extract the zero-level set (surface) from the fused SDF field

### Manual SDF Query

If you need more control:

```python
import torch
import numpy as np

# Query arbitrary points
query_points = torch.tensor([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [-0.5, -0.5, -0.5]
]).float().unsqueeze(0).cuda()

# Get SDF values
with torch.no_grad():
    sdf_values = model(pointcloud_tensor.cuda(), query_points)

print("SDF values:", sdf_values)
# Negative = inside object
# Positive = outside object
# ~0 = on surface
```

## 7. Testing

Run the test suite to verify functionality:

```bash
# Test point cloud normalization
python test_pointcloud_normalize.py

# Test CAE preprocessing (existing)
python test_cae_preprocessing.py
```

## 8. Tips and Best Practices

### Memory Management

- **Resolution**: Higher values (e.g., 256) produce better quality but require more GPU memory
- **Batch Size**: Lower this if you encounter out-of-memory errors
- **Point Sampling**: Use 1000-5000 points for faster processing

### Fusion Quality

- **Union method**: Best for non-overlapping parts or when you want clear boundaries
- **Blend method**: 
  - `alpha=5-10`: Very smooth transitions, good for organic shapes
  - `alpha=10-20`: Moderate smoothness
  - `alpha>20`: Sharp transitions, approaching union behavior

### Coordinate Systems

- Always ensure point clouds are in the same coordinate system before fusion
- Use the reference transformation approach for parts from the same assembly
- Check bounding box sizes - if parts have very different scales, normalize them first

## 9. Troubleshooting

**Problem**: Fused mesh looks wrong or parts are misaligned

**Solution**: Ensure both point clouds are properly normalized to the same coordinate space using a reference transformation.

---

**Problem**: Out of memory errors during reconstruction

**Solution**: Reduce `--resolution` (try 128 or 96) or `--batch_size` (try 32000).

---

**Problem**: Fusion result has artifacts

**Solution**: Try adjusting the `--alpha` parameter for blend method, or use union method instead.

---

**Problem**: Parts are too far apart after fusion

**Solution**: Check that normalization parameters are correct. Each part should be centered at origin with diagonal spanning [-1, 1].

## 10. File Formats

Supported input formats:
- **PLY**: Standard 3D mesh format (vertices only)
- **XYZ**: Simple text format with x, y, z coordinates (space or comma separated)
- **CSV**: Comma-separated format (can include SDF values in 4th column)

Output format:
- **PLY**: Standard 3D mesh format with vertices and faces

## References

- Original point cloud normalization logic: `data/preprocessing/cae_to_csv.py`
- GenSDF reconstruction: `model/gensdf/model.py` (reconstruct method)
- Marching cubes implementation: `utils/mesh.py`
