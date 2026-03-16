#!/bin/bash
# TIGER 快速训练脚本 (CPU 版本)

set -e

echo "=========================================="
echo "TIGER + Tenrec 快速训练脚本"
echo "=========================================="

PYTHON=/home/vivwimp/.venv/bin/python
DATA_DIR=/home/vivwimp/TIGER/data/Tenrec

# 1. 数据预处理（如果尚未完成）
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "[1/4] 数据预处理..."
    cd /home/vivwimp/TIGER/data
    $PYTHON prepare_tenrec.py \
        --input /home/vivwimp/Tenrec/ctr_data_1M.csv \
        --output ./Tenrec \
        --min_interactions 5 \
        --sample_ratio 0.05
else
    echo "[1/4] 数据已存在，跳过预处理"
fi

# 2. 训练 RQVAE（如果尚未完成）
if [ ! -f "$DATA_DIR/Tenrec_t5_rqvae.npy" ]; then
    echo "[2/4] 训练 RQVAE..."
    cd /home/vivwimp/TIGER/rqvae
    $PYTHON train_tenrec_rqvae.py \
        --data_path $DATA_DIR/item_emb.parquet \
        --output_dir $DATA_DIR \
        --ckpt_dir ./ckpt \
        --dataset_name Tenrec \
        --epochs 20 \
        --batch_size 512 \
        --lr 1e-3 \
        --device cpu
else
    echo "[2/4] RQVAE 代码已存在，跳过训练"
fi

# 3. 生成 RQVAE 代码（如果需要）
if [ ! -f "$DATA_DIR/Tenrec_t5_rqvae.npy" ]; then
    echo "[3/4] 生成 RQVAE 代码..."
    cd /home/vivwimp/TIGER/rqvae
    $PYTHON -c "
import torch
import numpy as np
from tqdm import tqdm
from datasets import EmbDataset
from torch.utils.data import DataLoader
from models.rqvae import RQVAE

checkpoint = torch.load('./ckpt/Tenrec/best_model.pth', map_location='cpu', weights_only=False)
args = checkpoint['args']
state_dict = checkpoint['state_dict']

data = EmbDataset('$DATA_DIR/item_emb.parquet')
data_loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=4)

model = RQVAE(
    in_dim=data.dim,
    num_emb_list=args.num_emb_list,
    e_dim=args.e_dim,
    layers=args.layers,
    dropout_prob=args.dropout_prob,
    bn=args.bn,
    loss_type=args.loss_type,
    quant_loss_weight=args.quant_loss_weight,
    kmeans_init=args.kmeans_init,
    kmeans_iters=args.kmeans_iters,
    sk_epsilons=args.sk_epsilons,
    sk_iters=args.sk_iters,
)
model.load_state_dict(state_dict)
model.eval()

all_indices = []
for d in tqdm(data_loader, desc='Generating codes'):
    with torch.no_grad():
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        all_indices.extend(indices.tolist())

codes_array = np.array(all_indices)
codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))
np.save('$DATA_DIR/Tenrec_t5_rqvae.npy', codes_array)
print(f'Codes saved to $DATA_DIR/Tenrec_t5_rqvae.npy')
"
else
    echo "[3/4] RQVAE 代码文件已存在"
fi

# 4. 训练 TIGER
echo "[4/4] 训练 TIGER 模型..."
cd /home/vivwimp/TIGER/model
$PYTHON main.py \
    --dataset_path $DATA_DIR \
    --code_path $DATA_DIR/Tenrec_t5_rqvae.npy \
    --save_path ./ckpt/tiger.pth \
    --num_epochs 10 \
    --batch_size 256 \
    --lr 1e-4 \
    --max_len 20 \
    --d_model 128 \
    --num_layers 4 \
    --num_decoder_layers 4 \
    --vocab_size 1281 \
    --device cpu \
    --early_stop 3

echo ""
echo "=========================================="
echo "训练完成！模型保存在: ./model/ckpt/tiger.pth"
echo "=========================================="
