#!/bin/bash
# TIGER-LLM GPU 训练启动脚本
# RTX 3060 6GB 优化配置

set -e

cd /home/vivwimp/TIGER

echo "=========================================="
echo "TIGER-LLM GPU 训练"
echo "=========================================="

PYTHON=/home/vivwimp/.venv/bin/python

# 检查 GPU 状态
echo ""
echo "GPU 状态:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo ""
echo "PyTorch CUDA 检查:"
$PYTHON -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
"

# 清理 GPU 缓存
echo ""
echo "清理 GPU 缓存..."
$PYTHON -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# 开始训练
echo ""
echo "开始训练..."
echo "配置:"
echo "  - Model: d_model=256, layers=4, heads=8"
echo "  - Batch size: 32"
echo "  - Gradient accumulation: 2"
echo "  - Mixed precision: FP16"
echo ""

cd model

# 使用优化的 GPU 训练脚本
$PYTHON main_llm_gpu.py \
    --dataset_path ../data/Tenrec \
    --code_path ../data/Tenrec/Tenrec_t5_rqvae.npy \
    --save_path ./ckpt/tiger_llm_gpu.pth \
    --num_epochs 50 \
    --batch_size 32 \
    --grad_accum_steps 2 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --max_seq_len 256 \
    --d_model 256 \
    --num_layers 4 \
    --num_heads 8 \
    --vocab_size 1300 \
    --beam_size 20 \
    --device cuda \
    --early_stop 10 \
    --num_workers 4 \
    --use_amp \
    --seed 2025

echo ""
echo "=========================================="
echo "训练完成！"
echo "模型: ./model/ckpt/tiger_llm_gpu.pth"
echo "日志: ./model/logs/tiger_llm_gpu.log"
echo "=========================================="

# 显示最终 GPU 状态
echo ""
nvidia-smi
