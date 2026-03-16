# TIGER 复现指南

## 项目概述
TIGER (Recommender Systems with Generative Retrieval) 是一个使用生成式检索的推荐系统。

论文链接: https://arxiv.org/pdf/2305.05065

---

## 环境准备

### 1. 创建 Conda 环境
```bash
# 创建环境
conda create -n tiger python=3.10 -y
conda activate tiger

# 安装依赖
pip install torch transformers sentence-transformers pandas numpy polars pyarrow tqdm
```

---

## 数据准备

### 2. 下载 Amazon Review 2014 数据集

```bash
cd data

# 下载 Beauty 数据集
wget https://jmcauley.ucsd.edu/data/amazon/review_data/reviews_Beauty_5.json.gz
wget https://jmcauley.ucsd.edu/data/amazon/meta/meta_Beauty.json.gz

# 如果需要其他数据集
# wget https://jmcauley.ucsd.edu/data/amazon/review_data/reviews_Sports_and_Outdoors_5.json.gz
# wget https://jmcauley.ucsd.edu/data/amazon/meta/meta_Sports_and_Outdoors.json.gz
```

### 3. 数据预处理

运行 `data/process.ipynb` 中的代码：

**步骤 1**: 解压数据并转换为 JSON
```python
import json
import gzip
import os

dataset_name = "Beauty"
os.makedirs(dataset_name, exist_ok=True)

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))

# 解压评论数据
f = open(f"./{dataset_name}/{dataset_name}.json", 'w')
for l in parse(f"reviews_{dataset_name}_5.json.gz"):
    f.write(l + '\n')
f.close()
```

**步骤 2**: 处理评论数据，生成 train/valid/test.parquet
```python
import numpy as np
import pandas as pd

# 读取数据并创建用户-物品映射
# ... (详见 process.ipynb)
# 生成: train.parquet, valid.parquet, test.parquet, user_mapping.npy, item_mapping.npy
```

**步骤 3**: 处理元数据，生成物品语义嵌入
```python
from sentence_transformers import SentenceTransformer

# 下载 sentence-t5-base 模型
# modelscope download --model sentence-transformers/sentence-t5-base --local_dir ./sentence-t5-base

model = SentenceTransformer('./sentence-t5-base')

# 为每个物品生成语义嵌入
# 生成: item_emb.parquet
```

---

## 训练流程

### 4. 训练 RQVAE 模型

```bash
cd rqvae

python main.py \
    --data_path "../data/Beauty/item_emb.parquet" \
    --ckpt_dir "./ckpt/Beauty" \
    --epochs 3000 \
    --batch_size 1024 \
    --lr 1e-3 \
    --num_emb_list 256 256 256 \
    --e_dim 32 \
    --layers 512 256 128 64 \
    --device "cuda:0"
```

**参数说明**:
- `num_emb_list`: 每层 VQ 的 codebook 大小，这里是 3 层，每层 256 个 codeword
- `e_dim`: codebook 嵌入维度
- `layers`: RQVAE 的隐藏层维度

**预期输出**:
- 模型 checkpoint 保存在 `./ckpt/Beauty/{timestamp}/`
- 记录最佳 loss 和 collision rate

### 5. 生成离散代码

编辑 `generate_code.py`，修改以下参数：
```python
dataset = "Beauty"
ckpt_path = f"./ckpt/{dataset}/Jun-17-2025_15-21-52/best_collision_model.pth"  # 修改为实际的 checkpoint 路径
output_file = f"../data/{dataset}/{dataset}_t5_rqvae.npy"
device = torch.device("cuda:0")
```

运行代码生成：
```bash
python generate_code.py
```

**输出**:
- `../data/Beauty/Beauty_t5_rqvae.npy`: 物品的离散语义代码

### 6. 训练 TIGER 模型

```bash
cd model

python main.py \
    --dataset_path '../data/Beauty' \
    --code_path '../data/Beauty/Beauty_t5_rqvae.npy' \
    --batch_size 256 \
    --infer_size 96 \
    --num_epochs 200 \
    --lr 1e-4 \
    --device 'cuda' \
    --save_path './ckpt/tiger.pth' \
    --beam_size 30
```

**训练过程**:
- 每个 epoch 在验证集上评估 Recall@K 和 NDCG@K
- 使用 early stopping（默认 patience=10）
- 最佳模型保存在 `./ckpt/tiger.pth`

---

## 实验结果

### Beauty 数据集预期结果

| Metric    | Ours   | Paper  |
|-----------|--------|--------|
| Recall@5  | 0.0392 | 0.0454 |
| Recall@10 | 0.0594 | 0.0648 |
| NDCG@5    | 0.0257 | 0.0321 |
| NDCG@10   | 0.0321 | 0.0384 |

---

## 文件结构说明

```
TIGER/
├── data/                      # 数据目录
│   ├── Beauty/               # Beauty 数据集
│   │   ├── train.parquet     # 训练数据
│   │   ├── valid.parquet     # 验证数据
│   │   ├── test.parquet      # 测试数据
│   │   ├── item_emb.parquet  # 物品语义嵌入
│   │   ├── Beauty_t5_rqvae.npy  # 物品离散代码
│   │   ├── user_mapping.npy  # 用户 ID 映射
│   │   └── item_mapping.npy  # 物品 ID 映射
│   └── process.ipynb         # 数据预处理脚本
├── rqvae/                     # RQVAE 模型
│   ├── main.py               # 训练脚本
│   ├── generate_code.py      # 代码生成脚本
│   ├── trainer.py            # 训练器
│   ├── datasets.py           # 数据集定义
│   └── models/               # 模型定义
│       ├── rqvae.py
│       ├── rq.py
│       ├── vq.py
│       └── layers.py
└── model/                     # TIGER 模型
    ├── main.py               # 训练脚本
    ├── dataset.py            # 数据集定义
    ├── dataloader.py         # 数据加载器
    └── ckpt/                 # 模型保存目录
```

---

## 常见问题

### 1. CUDA 内存不足
- 减小 `batch_size`
- 使用更小的 `layers` 配置

### 2. Collision Rate 过高
- 增加 `num_emb_list` 中的值
- 增加 RQVAE 层数
- 调整 `sk_epsilons` 参数

### 3. 训练不稳定
- 调整学习率
- 增加 `warmup_epochs`
- 使用梯度裁剪

---

## 引用

```bibtex
@article{rajput2023recommender,
  title={Recommender Systems with Generative Retrieval},
  author={Rajput, Shashank and Mehta, Nikhil and Singh, Anima and Sarthi, Raghunandan and Heldt, Lucas and Hong, Lichan and Tay, Yi and Tran, Vinh and Samost, Jonah and Yang, Yinlink and others},
  journal={arXiv preprint arXiv:2305.05065},
  year={2023}
}
```
