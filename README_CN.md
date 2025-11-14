# GenSDF 点云归一化与融合功能

[English Guide](FUSION_GUIDE.md)

## 概述

本项目新增了以下功能：

1. **点云归一化**：将点云质心移至原点，包围盒对角线归一化到 [-1,1] 范围
2. **变换参数管理**：保存和应用归一化参数，用于多个零部件的统一处理
3. **SDF 融合**：使用 GenSDF 编码器-解码器架构融合两个点云
4. **网格重建**：从融合的 SDF 表示生成 3D 网格模型

## 快速开始

### 1. 点云归一化

```bash
# 归一化点云并保存变换参数
python utils/pointcloud_normalize.py -i 零件1.ply -o 归一化后.xyz -p 变换参数.json

# 对新零件应用相同的变换
python utils/pointcloud_normalize.py -i 零件2.ply -o 零件2归一化.xyz --apply 变换参数.json
```

### 2. 融合两个点云

```bash
# 基本融合（取最小值）
python fusion_example.py \
    -m config/gensdf/semi \
    -p1 零件1.ply \
    -p2 零件2.ply \
    -o 输出目录

# 平滑融合
python fusion_example.py \
    -m config/gensdf/semi \
    -p1 零件1.ply \
    -p2 零件2.ply \
    -o 输出目录 \
    --method blend \
    --alpha 10
```

## 功能详解

### 归一化处理

归一化过程包含两个步骤：

1. **中心化**：将点云的质心移动到原点 (0, 0, 0)
2. **缩放**：将包围盒对角线长度归一化，使其在 [-1, 1] 范围内

这确保了不同零部件可以在统一的坐标空间中操作，这对于准确的 SDF 融合至关重要。

#### Python 代码示例

```python
from utils.pointcloud_normalize import normalize_pointcloud, save_transform_params

# 加载点云 (Nx3 numpy数组)
import numpy as np
点云 = np.loadtxt('输入.xyz')

# 归一化并获取变换参数
归一化点云, 变换参数 = normalize_pointcloud(点云, return_params=True)

# 保存变换参数供后续使用
save_transform_params(变换参数, '变换参数.json')

# 保存归一化后的点云
np.savetxt('归一化.xyz', 归一化点云)
```

#### 应用变换到新零件

有了参考零部件的归一化参数后，可以对新零部件应用相同的变换：

```python
from utils.pointcloud_normalize import load_transform_params, apply_transform

# 加载已保存的变换参数
变换参数 = load_transform_params('变换参数.json')

# 加载新点云
新点云 = np.loadtxt('新零件.xyz')

# 应用相同的变换
新点云归一化 = apply_transform(新点云, 变换参数)
```

### SDF 融合方法

提供两种融合方法：

#### 1. Union（并集/取最小值）

在每个查询点取两个 SDF 值的最小值，创建两个形状的并集：

```python
from utils.sdf_fusion import sdf_union

# 融合后的SDF = sdf_union(查询点, sdf_零件1, sdf_零件2)
```

**适用场景**：
- 非重叠零部件
- 需要清晰边界的场合

#### 2. Blend（平滑混合）

使用指数加权创建两个形状之间的平滑过渡：

```python
from utils.sdf_fusion import sdf_blend

# 融合后的SDF = sdf_blend(查询点, sdf_零件1, sdf_零件2, alpha=10)
# alpha 值越高 = 过渡越锐利（更接近并集）
# alpha 值越低 = 过渡越平滑
```

**参数建议**：
- `alpha=5-10`：非常平滑的过渡，适合有机形状
- `alpha=10-20`：中等平滑度
- `alpha>20`：锐利过渡，接近并集效果

### 完整融合工作流程

```python
import torch
from model import GenSDF
from utils.sdf_fusion import SDFFusion

# 1. 加载训练好的 GenSDF 模型
model = GenSDF(specs, None).cuda()
checkpoint = torch.load('路径/到/checkpoint.ckpt')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 2. 加载并归一化点云
pc1 = torch.from_numpy(归一化点云1).float()
pc2 = torch.from_numpy(归一化点云2).float()

# 3. 创建融合实例
fusion = SDFFusion(model)

# 4. 融合点云
融合后的sdf = fusion.fuse_pointclouds(
    pc1, pc2, 
    method='union',  # 或 'blend'
    alpha=10         # 仅用于 'blend' 方法
)

# 5. 从融合的 SDF 重建 3D 网格
fusion.reconstruct_from_sdf(
    融合后的sdf, 
    output_path='输出/融合结果.ply',
    resolution=192,      # 越高 = 质量越好，内存占用越多
    batch_size=64000     # 如果内存不足，降低此值
)
```

## 使用 CAE 数据

归一化功能遵循与 `cae_to_csv.py` 相同的格式。要使用 CAE 数据：

```bash
# 将 CAE 转换为 CSV 并进行归一化
python data/preprocessing/cae_to_csv.py \
    -i 数据/零件.cae \
    -o 数据/零件.csv

# CSV 格式可直接用于 fusion_example.py
python fusion_example.py \
    -m config/gensdf/semi \
    -p1 数据/零件1.csv \
    -p2 数据/零件2.csv \
    -o 输出/结果
```

## GenSDF 解码器的工作原理

GenSDF 的解码器：
- **输入**：连接的潜在特征 + 3D 查询坐标
- **输出**：每个查询点的有符号距离值

融合过程的工作流程：

1. 使用 GenSDF 的编码器将每个点云编码为潜在特征
2. 对相同的查询点计算两个编码的 SDF 值
3. 使用并集或混合操作组合 SDF 值
4. 使用 marching cubes 从融合的 SDF 场中提取零等值面（表面）

### 手动 SDF 查询

如果需要更多控制：

```python
import torch
import numpy as np

# 查询任意点
查询点 = torch.tensor([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [-0.5, -0.5, -0.5]
]).float().unsqueeze(0).cuda()

# 获取 SDF 值
with torch.no_grad():
    sdf值 = model(点云张量.cuda(), 查询点)

print("SDF 值:", sdf值)
# 负值 = 物体内部
# 正值 = 物体外部
# ~0 = 在表面上
```

## 命令行选项

### fusion_example.py 参数

- `-m, --model_dir`：包含 specs.json 和 checkpoint 的模型目录路径
- `-p1, --pointcloud1`：第一个点云文件路径（.ply、.xyz 或 .csv）
- `-p2, --pointcloud2`：第二个点云文件路径
- `-o, --output`：结果输出目录
- `--method`：融合方法：'union' 或 'blend'（默认：union）
- `--alpha`：blend 方法的混合参数（默认：10）
- `--resolution`：marching cubes 分辨率（默认：128）
- `--batch_size`：SDF 查询的批次大小（默认：64000）
- `--checkpoint`：要加载的检查点（默认：last）
- `--ref_transform`：使用 JSON 文件中的现有变换参数
- `--save_transform`：将变换参数保存到 JSON 文件
- `--no_normalize`：跳过归一化（假设已归一化）

## 测试

运行测试套件验证功能：

```bash
# 测试点云归一化
python test_pointcloud_normalize.py

# 测试 CAE 预处理（现有）
python test_cae_preprocessing.py

# 运行融合演示（生成可视化）
python demo_fusion.py
```

## 演示可视化

运行 `demo_fusion.py` 会生成以下可视化文件：

1. **fusion_comparison_2d.png**：不同融合方法的 2D 对比
2. **fusion_profile_1d.png**：沿一条线的 SDF 值剖面
3. **normalization_demo.png**：归一化前后的点云对比

## 多零部件工作流程示例

### 场景：融合多个组件

当需要组合多个零部件时：

**步骤 1：归一化第一个（参考）零部件**

```bash
python utils/pointcloud_normalize.py \
    -i 数据/参考零件.ply \
    -o 数据/参考_归一化.xyz \
    -p 数据/参考_变换.json
```

**步骤 2：对其他零部件应用相同的变换**

```bash
python utils/pointcloud_normalize.py \
    -i 数据/零件2.ply \
    -o 数据/零件2_归一化.xyz \
    --apply 数据/参考_变换.json

python utils/pointcloud_normalize.py \
    -i 数据/零件3.ply \
    -o 数据/零件3_归一化.xyz \
    --apply 数据/参考_变换.json
```

**步骤 3：融合归一化后的零部件**

```bash
# 融合零件 1 和零件 2
python fusion_example.py \
    -m config/gensdf/semi \
    -p1 数据/参考_归一化.xyz \
    -p2 数据/零件2_归一化.xyz \
    -o 输出/融合_1_2 \
    --no_normalize

# 将结果与零件 3 融合
# （可以先将输出网格转换为点云）
```

## 提示和最佳实践

### 内存管理

- **分辨率**：更高的值（如 256）产生更好的质量，但需要更多 GPU 内存
- **批次大小**：如果遇到内存不足错误，降低此值
- **点采样**：使用 1000-5000 个点可以更快处理

### 融合质量

- **Union 方法**：适合非重叠零部件或需要清晰边界时
- **Blend 方法**：
  - `alpha=5-10`：非常平滑的过渡，适合有机形状
  - `alpha=10-20`：中等平滑度
  - `alpha>20`：锐利过渡，接近并集效果

### 坐标系统

- 融合前始终确保点云在同一坐标系统中
- 对来自同一装配体的零部件使用参考变换方法
- 检查包围盒大小 - 如果零部件比例差异很大，先进行归一化

## 故障排除

**问题**：融合的网格看起来不对或零部件未对齐

**解决方案**：确保两个点云都使用参考变换正确归一化到相同的坐标空间。

---

**问题**：重建期间内存不足错误

**解决方案**：降低 `--resolution`（尝试 128 或 96）或 `--batch_size`（尝试 32000）。

---

**问题**：融合结果有瑕疵

**解决方案**：尝试调整 blend 方法的 `--alpha` 参数，或改用 union 方法。

---

**问题**：融合后零部件距离太远

**解决方案**：检查归一化参数是否正确。每个零部件应该以原点为中心，对角线应该跨越 [-1, 1]。

## 支持的文件格式

支持的输入格式：
- **PLY**：标准 3D 网格格式（仅顶点）
- **XYZ**：简单文本格式，包含 x、y、z 坐标（空格或逗号分隔）
- **CSV**：逗号分隔格式（第 4 列可包含 SDF 值）

输出格式：
- **PLY**：标准 3D 网格格式，包含顶点和面

## 参考

- 原始点云归一化逻辑：`data/preprocessing/cae_to_csv.py`
- GenSDF 重建：`model/gensdf/model.py`（reconstruct 方法）
- Marching cubes 实现：`utils/mesh.py`
