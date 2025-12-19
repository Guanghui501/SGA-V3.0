# SLURM 提交脚本使用指南

## 📋 文件说明

本目录包含以下 SLURM 提交脚本：

1. **submit_train.sh** - 基础训练脚本（用于 train_folder.py）
2. **submit_train_crossmodal.sh** - 跨模态注意力训练脚本（使用 Cosine Scheduler）

## 🚀 快速开始

### 1. 修改脚本配置

在提交任务前，需要根据你的集群环境修改脚本中的以下部分：

#### a) SLURM 参数（根据集群资源调整）

```bash
#SBATCH --partition=gpu        # 修改为你的GPU分区名称
#SBATCH --gres=gpu:1           # GPU数量（如需多GPU改为 gpu:2, gpu:4 等）
#SBATCH --mem=32G              # 根据需要调整内存
#SBATCH --time=48:00:00        # 根据预计训练时间调整
```

#### b) 环境加载（取消注释并修改）

```bash
# 示例 - 根据你的集群环境修改
module load cuda/11.8
module load cudnn/8.6.0
module load anaconda3

# 激活conda环境
conda activate your_env_name
```

### 2. 提交任务

```bash
# 基础训练
sbatch submit_train.sh

# 跨模态注意力训练（推荐，使用Cosine Scheduler）
sbatch submit_train_crossmodal.sh
```

### 3. 查看任务状态

```bash
# 查看任务队列
squeue -u $USER

# 查看特定任务
squeue -j <job_id>

# 取消任务
scancel <job_id>

# 查看任务详情
scontrol show job <job_id>
```

### 4. 查看日志

```bash
# 实时查看输出日志
tail -f logs/train_cross_<job_id>.out

# 查看错误日志
tail -f logs/train_cross_<job_id>.err

# 查看所有日志
ls -lh logs/
```

## 🔧 常见配置修改

### 修改训练参数

编辑脚本中的训练命令部分：

```bash
python train_with_cross_modal_attention.py \
    --dataset 'jarvis' \              # 数据集: jarvis, mp, class
    --property 'total_energy' \       # 性质: formation_energy, band_gap, 等
    --epochs 1000 \                   # 训练轮数
    --batch_size 64 \                 # 批次大小
    --learning_rate 0.001 \           # 学习率
    --use_cross_modal True \          # 是否使用跨模态注意力
    --cross_modal_num_heads 4         # 注意力头数
```

### 修改GPU数量

```bash
# 单GPU
#SBATCH --gres=gpu:1

# 双GPU
#SBATCH --gres=gpu:2

# 四GPU（需要修改代码支持分布式训练）
#SBATCH --gres=gpu:4
```

### 修改内存和CPU

```bash
#SBATCH --mem=64G              # 增加到64GB
#SBATCH --cpus-per-task=16     # 增加到16核
```

## 📊 Scheduler 说明

### Cosine Scheduler（已启用，推荐）

脚本默认使用 `CosineAnnealingWarmRestarts`：
- **优点**: 减少过拟合，适合小数据集
- **配置**: 在 `config.py` 中已设置为 `"cosine"`
- **参数**: T_0=20（每20 epoch重启），eta_min=1e-6

如需切换回 OneCycle Scheduler，修改 `train_with_cross_modal_attention.py:574`:
```python
"scheduler": "onecycle",  # 从 "cosine" 改回 "onecycle"
```

## 🐛 故障排查

### 1. 任务被杀死（OOM）

增加内存：
```bash
#SBATCH --mem=64G
```

或减小 batch_size：
```bash
--batch_size 32
```

### 2. GPU未检测到

检查CUDA模块是否正确加载：
```bash
# 在脚本中添加调试信息
nvidia-smi
module list
echo $CUDA_VISIBLE_DEVICES
```

### 3. 环境未激活

确保 conda 环境正确激活：
```bash
# 方法1
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate your_env

# 方法2
source activate your_env
```

### 4. 预处理数据错误

如果使用预处理数据遇到 KeyError，确保使用最新代码（已修复）：
```bash
git pull origin claude/update-scheduler-cosine-gYTRr
```

## 📝 示例：不同数据集训练

### JARVIS - Formation Energy

```bash
python train_with_cross_modal_attention.py \
    --dataset 'jarvis' \
    --property 'formation_energy' \
    --epochs 1000 \
    --batch_size 64
```

### Material Project - Band Gap

```bash
python train_with_cross_modal_attention.py \
    --dataset 'mp' \
    --property 'band_gap' \
    --n_train 60000 \
    --n_val 5000 \
    --n_test 4132 \
    --epochs 1000 \
    --batch_size 64
```

### 使用预处理数据（推荐，速度更快）

```bash
python train_with_cross_modal_attention.py \
    --dataset 'jarvis' \
    --property 'formation_energy' \
    --use_preprocessed True \
    --preprocessed_dir 'preprocessed_data' \
    --epochs 1000
```

## 🔗 相关文件

- `config.py` - 训练配置（包含 Scheduler 选项）
- `train.py` - 训练主脚本（已添加 Cosine Scheduler）
- `train_with_cross_modal_attention.py` - 跨模态注意力训练脚本
- `data.py` - 数据加载（已修复 KeyError）

## 📞 需要帮助？

如有问题，检查：
1. 日志文件: `logs/train_cross_<job_id>.out`
2. 错误日志: `logs/train_cross_<job_id>.err`
3. Git 分支: `claude/update-scheduler-cosine-gYTRr`
