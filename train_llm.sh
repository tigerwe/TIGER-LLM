#!/bin/bash
# TIGER-LLM 训练脚本 (Decoder-only)

set -e

echo "=========================================="
echo "TIGER-LLM (Decoder-only) 训练脚本"
echo "=========================================="

PYTHON=/home/vivwimp/.venv/bin/python
DATA_DIR=/home/vivwimp/TIGER/data/Tenrec

# 检查数据是否存在
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "错误：数据文件不存在，请先运行数据预处理"
    exit 1
fi

if [ ! -f "$DATA_DIR/Tenrec_t5_rqvae.npy" ]; then
    echo "错误：RQVAE 代码文件不存在"
    exit 1
fi

echo "开始训练 TIGER-LLM..."
cd /home/vivwimp/TIGER/model

$PYTHON main_llm.py \
    --dataset_path $DATA_DIR \
    --code_path $DATA_DIR/Tenrec_t5_rqvae.npy \
    --save_path ./ckpt/tiger_llm.pth \
    --num_epochs 30 \
    --batch_size 64 \
    --lr 1e-4 \
    --max_seq_len 512 \
    --max_len 20 \
    --d_model 256 \
    --num_layers 4 \
    --num_heads 8 \
    --vocab_size 1300 \
    --beam_size 20 \
    --code_dim 5 \
    --device cpu \
    --early_stop 5 \
    --num_workers 4

echo ""
echo "=========================================="
echo "训练完成！模型保存在: ./model/ckpt/tiger_llm.pth"
echo "=========================================="
