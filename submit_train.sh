#!/bin/bash
#SBATCH --job-name=sga_train          # 任务名称
#SBATCH --partition=gpu                # 分区名称（根据集群调整）
#SBATCH --nodes=1                      # 节点数
#SBATCH --ntasks=1                     # 任务数
#SBATCH --cpus-per-task=8              # 每个任务的CPU核心数
#SBATCH --gres=gpu:1                   # GPU数量
#SBATCH --mem=32G                      # 内存
#SBATCH --time=48:00:00                # 最大运行时间 (48小时)
#SBATCH --output=logs/train_%j.out     # 标准输出日志 (%j 会被替换为任务ID)
#SBATCH --error=logs/train_%j.err      # 错误输出日志

# 创建日志目录
mkdir -p logs

# 打印任务信息
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "================================================================"

# 加载必要的模块（根据集群环境调整）
# module load cuda/11.8
# module load cudnn/8.6.0
# module load anaconda3

# 激活conda环境（根据你的环境名称调整）
# source activate your_env_name
# 或者
# conda activate your_env_name

# 打印环境信息
echo ""
echo "Python版本:"
python --version
echo ""
echo "PyTorch版本:"
python -c "import torch; print(torch.__version__)"
echo "CUDA可用性:"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
echo "GPU数量:"
python -c "import torch; print('GPU Count:', torch.cuda.device_count())"
echo ""
echo "================================================================"

# 进入工作目录
cd $SLURM_SUBMIT_DIR

# 运行训练脚本
echo "开始训练..."
echo ""

python train_folder.py \
    --root_dir '../dataset/' \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --dataset 'Jarvis' \
    --property 'total_energy' \
    --epochs 1000 \
    --batch_size 64 \
    --resume 0

# 记录结束时间
echo ""
echo "================================================================"
echo "End Time: $(date)"
echo "================================================================"
