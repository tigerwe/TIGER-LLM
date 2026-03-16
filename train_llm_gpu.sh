#!/bin/bash
# TIGER-LLM GPU 训练脚本
# 针对 RTX 3060 6GB 显存优化

set -e

echo "=========================================="
echo "TIGER-LLM GPU 训练脚本"
echo "=========================================="

PYTHON=/home/vivwimp/.venv/bin/python
DATA_DIR=/home/vivwimp/TIGER/data/Tenrec

# 检查 GPU
$PYTHON -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')"

# 检查数据
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "错误：数据文件不存在"
    exit 1
fi

if [ ! -f "$DATA_DIR/Tenrec_t5_rqvae.npy" ]; then
    echo "错误：RQVAE 代码文件不存在"
    exit 1
fi

echo ""
echo "开始 GPU 训练..."
echo "配置: d_model=256, layers=4, batch_size=32 (适配 6GB 显存)"
echo ""

cd /home/vivwimp/TIGER/model

$PYTHON main_llm.py \
    --dataset_path $DATA_DIR \
    --code_path $DATA_DIR/Tenrec_t5_rqvae.npy \
    --save_path ./ckpt/tiger_llm_gpu.pth \
    --num_epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --max_seq_len 256 \
    --max_len 20 \
    --d_model 256 \
    --num_layers 4 \
    --num_heads 8 \
    --vocab_size 1300 \
    --beam_size 20 \
    --code_dim 5 \
    --device cuda \
    --early_stop 10 \
    --num_workers 4 \
    --seed 2025

echo ""
echo "=========================================="
echo "训练完成！模型保存在: ./model/ckpt/tiger_llm_gpu.pth"
echo "=========================================="
