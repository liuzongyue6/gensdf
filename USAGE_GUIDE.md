# GenSDF 单个文件重建使用指南

## 基本用法

```bash
python test_single.py -f <输入文件> -o <输出目录> -e config/gensdf/semi -r last
```

## 新增参数说明

### 1. 分辨率控制 (`--resolution` 或 `-res`)
控制Marching Cubes的网格分辨率，越高质量越好但显存占用越大。

- **默认值**: 192
- **推荐范围**: 
  - 低分辨率: 64-128 (显存受限时)
  - 中等分辨率: 192-256 (平衡质量和速度)
  - 高分辨率: 256-512 (高质量，需要大显存)

**显存占用参考**:
- 128³ ≈ 2.1M 点
- 192³ ≈ 7.1M 点  
- 256³ ≈ 16.8M 点
- 512³ ≈ 134M 点 (需要大量显存!)

### 2. 批处理大小 (`--batch_size` 或 `-bs`)
控制每次推理的查询点数量。

- **默认值**: 64000
- **如果显存溢出**: 降低到 32000, 16000, 或更低
- **如果显存充足**: 可以提高到 128000 或更高以加速推理

### 3. 采样点数 (`--num_points` 或 `-np`)
从输入点云中采样的点数。

- **默认值**: 1000
- **推荐范围**: 500-5000
- 越多点数理论上效果越好，但会增加优化时间

### 4. 测试时优化
- **默认**: 启用 (推荐)
- **禁用优化**: 添加 `--no_test_optimize` (更快但质量稍低)

## 使用示例

### 示例 1: 默认设置（平衡质量和速度）
```bash
python test_single.py \
    -f BYD_brkd_free_format.csv \
    -o output_directory \
    -e config/gensdf/semi \
    -r last
```

### 示例 2: 高质量重建（需要更多显存）
```bash
python test_single.py \
    -f BYD_brkd_free_format.csv \
    -o output_directory \
    -e config/gensdf/semi \
    -r last \
    -res 256 \
    -bs 100000 \
    -np 2000
```

### 示例 3: 低显存模式
```bash
python test_single.py \
    -f BYD_brkd_free_format.csv \
    -o output_directory \
    -e config/gensdf/semi \
    -r last \
    -res 128 \
    -bs 32000 \
    -np 500
```

### 示例 4: 最高质量（需要大显存，如24GB+）
```bash
python test_single.py \
    -f BYD_brkd_free_format.csv \
    -o output_directory \
    -e config/gensdf/semi \
    -r last \
    -res 384 \
    -bs 200000 \
    -np 3000
```

### 示例 5: 快速预览（跳过优化）
```bash
python test_single.py \
    -f BYD_brkd_free_format.csv \
    -o output_directory \
    -e config/gensdf/semi \
    -r last \
    -res 128 \
    -np 500 \
    --no_test_optimize
```

## 输出文件

输出文件将保存在: `<输出目录>/<输入文件名>/reconstruct.ply`

例如，输入 `BYD_brkd_free_format.csv`，输出将是:
```
output_directory/BYD_brkd_free_format/reconstruct.ply
```

## 显存不足的解决方案

如果遇到 `CUDA out of memory` 错误，按以下顺序尝试：

1. **降低批处理大小**: `-bs 32000` → `-bs 16000` → `-bs 8000`
2. **降低分辨率**: `-res 192` → `-res 128` → `-res 96`
3. **减少采样点数**: `-np 1000` → `-np 500`
4. **设置环境变量**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   ```
5. **清理其他GPU进程**: `nvidia-smi` 查看并关闭其他占用显存的进程

## GPU显存参考表

| GPU型号 | 显存 | 推荐分辨率 | 推荐批处理 |
|---------|------|------------|------------|
| GTX 1080 Ti | 11GB | 128-192 | 32000-64000 |
| RTX 2080 Ti | 11GB | 128-192 | 32000-64000 |
| RTX 3090 | 24GB | 256-384 | 100000-200000 |
| RTX 4090 | 24GB | 256-512 | 100000-300000 |
| A100 | 40GB+ | 512+ | 200000+ |

## 常见问题

**Q: 分辨率从128提高到256，质量提升明显吗？**  
A: 是的，非常明显。256³是128³的8倍点数，可以捕捉更精细的细节。

**Q: 批处理大小影响结果质量吗？**  
A: 不影响。批处理大小只影响速度和显存占用，不影响最终质量。

**Q: 应该使用多少采样点？**  
A: 1000-2000点通常足够。如果输入点云很稀疏，可能需要更多点。

**Q: 测试时优化需要多长时间？**  
A: 默认400次迭代大约需要1-3分钟，取决于GPU性能。



## 环境配置

### 安装依赖

在使用 GenSDF 之前，需要确保正确安装了 CUDA 版本的 `torch-scatter`。

#### 1. 验证 PyTorch CUDA 是否可用

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU count:', torch.cuda.device_count())"
```

预期输出类似：

CUDA available: True
CUDA version: 11.3
GPU count: 1metric 官方仓库安装匹配 CUDA 版本的 torch-scatter，否则会报错 "RuntimeError: Not compiled with CUDA support"。

```bash
# 清除 pip 缓存
pip cache purge

# 卸载现有版本
pip uninstall torch-scatter -y

# 从 PyTorch Geometric 仓库安装（针对 PyTorch 1.11.0 + CUDA 11.3）
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html --no-cache-dir
```

**注意**: 
- 如果使用不同版本的 PyTorch/CUDA，需要调整 URL 中的版本号
- 必须使用 `-f` 参数指定 PyTorch Geometric 仓库，直接 `pip install torch-scatter` 会安装 CPU 版本
- `--no-cache-dir` 确保不使用缓存的旧版本

#### 3. 验证安装

```bash
python -c "import torch; from torch_scatter import scatter_max; x = torch.randn(10, 16).cuda(); idx = torch.randint(0, 4, (10,)).cuda(); result = scatter_max(x, idx, dim=0); print('CUDA scatter_max works!')"
```

如果成功打印 "CUDA scatter_max works!"，说明安装正确。

