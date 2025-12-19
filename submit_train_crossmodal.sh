#!/bin/bash
#SBATCH --job-name=sga_crossmodal      # 任务名称
#SBATCH --partition=gpu                # 分区名称（根据集群调整）
#SBATCH --nodes=1                      # 节点数
#SBATCH --ntasks=1                     # 任务数
#SBATCH --cpus-per-task=8              # 每个任务的CPU核心数
#SBATCH --gres=gpu:1                   # GPU数量
#SBATCH --mem=32G                      # 内存
#SBATCH --time=48:00:00                # 最大运行时间 (48小时)
#SBATCH --output=logs/train_cross_%j.out     # 标准输出日志
#SBATCH --error=logs/train_cross_%j.err      # 错误输出日志

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
# 示例（根据你的集群环境取消注释并修改）:
# module purge
# module load cuda/11.8
# module load cudnn/8.6.0
# module load anaconda3/2023.03

# 激活conda环境（根据你的环境名称调整）
# 示例:
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate sga_env

# 打印环境信息
echo ""
echo "Python版本:"
python --version
echo ""
echo "PyTorch版本:"
python -c "import torch; print(torch.__version__)"
echo "CUDA可用性:"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "当前GPU设备:"
    python -c "import torch; print('Device Name:', torch.cuda.get_device_name(0))"
fi
echo ""
echo "================================================================"

# 进入工作目录
cd $SLURM_SUBMIT_DIR

# 运行训练脚本（使用 Cosine Scheduler）
echo "开始训练 (使用 CosineAnnealingWarmRestarts Scheduler)..."
echo ""

python train_with_cross_modal_attention.py \
    --root_dir '../dataset/' \
    --dataset 'jarvis' \
    --property 'total_energy' \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --epochs 1000 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --weight_decay 1e-5 \
    --resume 0 \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    --output_dir './output/'

# 记录结束时间
echo ""
echo "================================================================"
echo "End Time: $(date)"
echo "================================================================"
