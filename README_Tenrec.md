# TIGER + Tenrec 数据集使用指南

本项目已将 TIGER 适配到 Tenrec 数据集，可以使用 `Tenrec/ctr_data_1M.csv` 进行训练。

## 文件结构

```
TIGER/
├── data/
│   ├── prepare_tenrec.py       # 数据预处理脚本
│   └── Tenrec/                 # 生成的数据目录
│       ├── train.parquet       # 训练集
│       ├── valid.parquet       # 验证集
│       ├── test.parquet        # 测试集
│       ├── item_emb.parquet    # 物品语义嵌入
│       └── Tenrec_t5_rqvae.npy # RQVAE 生成的离散代码
├── rqvae/
│   └── train_tenrec_rqvae.py   # RQVAE 训练脚本
├── model/
│   ├── main.py                 # TIGER 主训练脚本
│   └── ckpt/
│       └── tiger.pth           # 最终模型文件
└── train_tenrec_tiger.py       # 完整训练流程脚本
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install pandas numpy pyarrow torch transformers scikit-learn tqdm
```

### 2. 数据预处理

```bash
cd TIGER/data
python prepare_tenrec.py \
  --input ../../Tenrec/ctr_data_1M.csv \
  --output ./Tenrec \
  --min_interactions 5 \
  --sample_ratio 0.05
```

参数说明：
- `--input`: Tenrec CSV 文件路径
- `--output`: 输出目录
- `--min_interactions`: 用户最少交互次数
- `--sample_ratio`: 采样比例（0-1），用于减少数据量

### 3. 训练 RQVAE

```bash
cd TIGER/rqvae
python train_tenrec_rqvae.py \
  --data_path ../data/Tenrec/item_emb.parquet \
  --output_dir ../data/Tenrec \
  --ckpt_dir ./ckpt \
  --dataset_name Tenrec \
  --epochs 50 \
  --batch_size 512 \
  --lr 1e-3 \
  --device cuda
```

### 4. 训练 TIGER

```bash
cd TIGER/model
python main.py \
  --dataset_path ../data/Tenrec \
  --code_path ../data/Tenrec/Tenrec_t5_rqvae.npy \
  --save_path ./ckpt/tiger.pth \
  --num_epochs 100 \
  --batch_size 256 \
  --lr 1e-4 \
  --max_len 20 \
  --d_model 128 \
  --num_layers 4 \
  --num_decoder_layers 4 \
  --vocab_size 1281 \
  --device cuda \
  --early_stop 10
```

### 5. 一键训练（完整流程）

```bash
cd TIGER
python train_tenrec_tiger.py \
  --csv_path ../Tenrec/ctr_data_1M.csv \
  --data_dir ./data/Tenrec \
  --tiger_ckpt_path ./model/ckpt/tiger.pth \
  --sample_ratio 0.05
```

## 主要修改说明

### 1. 数据预处理 (`prepare_tenrec.py`)

- 将 CSV 转换为 parquet 格式
- 构建用户交互序列
- 划分训练/验证/测试集
- 生成物品语义嵌入（使用确定性随机嵌入）

### 2. RQVAE 训练 (`train_tenrec_rqvae.py`)

- 适配 Tenrec 数据集路径
- 修复 `sk_epsilon` 为 None 时的 bug
- 修复 loss 计算方式

### 3. TIGER 模型 (`main.py`)

- 更新默认 `vocab_size` 为 1281（适应 5 维代码）

### 4. DataLoader (`dataloader.py`)

- 修复序列长度不一致的问题
- 使用 `F.pad` 对序列进行填充

### 5. VQ 模型 (`rqvae/models/vq.py`)

- 修复 `sk_epsilon` 为 None 时的比较错误

## 训练参数说明

### TIGER 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_epochs` | 训练轮数 | 100 |
| `--batch_size` | 批次大小 | 256 |
| `--lr` | 学习率 | 1e-4 |
| `--max_len` | 最大序列长度 | 20 |
| `--d_model` | 模型维度 | 128 |
| `--num_layers` | 编码器层数 | 4 |
| `--num_decoder_layers` | 解码器层数 | 4 |
| `--vocab_size` | 词汇表大小 | 1281 |
| `--beam_size` | Beam search 大小 | 30 |

### RQVAE 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_emb_list` | 每个 VQ 的嵌入数量 | [256,256,256,256] |
| `--e_dim` | 嵌入维度 | 64 |
| `--layers` | MLP 层维度 | [512,256,128] |
| `--epochs` | 训练轮数 | 50 |
| `--batch_size` | 批次大小 | 512 |

## 注意事项

1. **内存问题**: 原始数据有 1.2 亿行，建议使用 `--sample_ratio` 参数采样（如 0.05 表示使用 5% 数据）
2. **GPU 训练**: 使用 `--device cuda` 可以显著加速训练
3. **早停**: 默认启用早停机制（`--early_stop 10`），当验证集 NDCG@20 不再提升时停止
4. **模型保存**: 最佳模型会自动保存到 `--save_path` 指定的路径

## 结果评估

训练完成后，模型会自动输出以下指标：
- Recall@5, Recall@10, Recall@20
- NDCG@5, NDCG@10, NDCG@20

## 故障排除

### 1. 内存不足

减少 `--sample_ratio` 或 `--batch_size`。

### 2. 索引越界

确保 `--vocab_size` 设置为 1281（因为代码是 5 维的，每维最大 256）。

### 3. 序列长度不一致

`dataloader.py` 已修复此问题，使用 `F.pad` 对序列进行填充。
