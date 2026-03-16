# TIGER-LLM: Decoder-only + Prompt 改造说明

## 改造概述

将 TIGER 从 T5 (encoder-decoder) 架构改造为 **Decoder-only LLM (GPT 架构) + Prompt** 形式。

## 主要变化

### 1. 模型架构变化

| 特性 | 原 TIGER (T5) | TIGER-LLM (GPT) |
|------|--------------|-----------------|
| 架构 | Encoder-Decoder | Decoder-only |
| 基础模型 | T5ForConditionalGeneration | GPT2LMHeadModel |
| 注意力 | 双向 (encoder) + 因果 (decoder) | 纯因果 (causal) |
| 输入格式 | history → target | prompt + history → target |

### 2. 数据格式变化

**原 TIGER (T5)**:
```
Input (Encoder):  [PAD, PAD, code1, code2, code3]
Target (Decoder): [code4, EOS]
```

**TIGER-LLM (Prompt 形式)**:
```
Prompt: "User's past: [code1][code2][code3] Next:"
Input:  [BOS, code1, code2, code3, SEP, code4, EOS]
                 ↑___________________↑______↑
                 │    prompt         │ target│
                 └───────────────────┴───────┘
                        只预测这部分
```

### 3. Token 定义

| Token ID | 用途 |
|---------|------|
| 0 | PAD_TOKEN (填充) |
| 1 | BOS_TOKEN (序列开始) |
| 2 | EOS_TOKEN (序列结束) |
| 3 | SEP_TOKEN (分隔符) |
| 4-9 | 预留 |
| 10+ | 代码 token (code + offset) |

### 4. 训练方式变化

**原 TIGER**:
- Encoder 处理 history
- Decoder 自回归生成 target
- 使用交叉注意力

**TIGER-LLM**:
- 单一 Decoder 处理整个序列
- 因果掩码 (causal mask) 确保只看前面
- 只计算 target 部分的 loss (instruction tuning)

## 文件结构

```
TIGER/model/
├── tiger_llm.py          # Decoder-only 模型定义
├── main_llm.py           # 训练脚本
├── main.py               # 原 T5 版本（保留）
└── ckpt/
    ├── tiger.pth         # 原模型检查点
    └── tiger_llm.pth     # LLM 版本检查点
```

## 使用方法

### 1. 数据准备

使用之前生成的 Tenrec 数据：
```bash
# 确保数据已准备好
ls TIGER/data/Tenrec/
# 应该包含：train.parquet, valid.parquet, test.parquet, Tenrec_t5_rqvae.npy
```

### 2. 训练

```bash
cd TIGER
chmod +x train_llm.sh
./train_llm.sh
```

或手动运行：

```bash
cd TIGER/model
python main_llm.py \
    --dataset_path ../data/Tenrec \
    --code_path ../data/Tenrec/Tenrec_t5_rqvae.npy \
    --save_path ./ckpt/tiger_llm.pth \
    --num_epochs 30 \
    --batch_size 64 \
    --lr 1e-4 \
    --d_model 256 \
    --num_layers 4 \
    --num_heads 8 \
    --vocab_size 1300 \
    --device cuda
```

### 3. 关键参数说明

| 参数 | 说明 | 建议值 |
|-----|------|-------|
| `--d_model` | 模型维度 | 256-512 |
| `--num_layers` | 层数 | 4-8 |
| `--num_heads` | 注意力头数 | 8 |
| `--max_seq_len` | 最大序列长度 | 512 |
| `--vocab_size` | 词表大小 | 1300 (1281 + 特殊token) |

## Prompt 模板

默认模板：
```python
"User's past: {history} Next:"
```

可以自定义：
```bash
--prompt_template "Based on user's history {history}, recommend:"
```

## 与原版本的对比

### 优势

1. **更简洁**: 单一 Decoder，无需 encoder-decoder 交互
2. **更灵活**: 易于添加各种 prompt 工程技巧
3. **更现代**: 符合当前 LLM 的主流架构 (GPT 系列)
4. **可扩展**: 易于扩展到更大的模型规模

### 注意事项

1. **序列长度**: Decoder-only 需要处理更长的序列（包含 prompt）
2. **位置编码**: GPT2 使用可学习的位置编码，可能需要调整 `max_seq_len`
3. **内存占用**: 相同参数量下，decoder-only 可能占用更多内存

## 性能对比

由于架构变化，需要注意：
- Decoder-only 模型通常需要更多的训练步数
- 建议适当增大 batch_size 或使用梯度累积
- 可以尝试 LoRA/QLoRA 等参数高效微调方法

## 代码关键修改点

### 1. tiger_llm.py

```python
# 使用 GPT2 替代 T5
self.model = GPT2LMHeadModel(gpt2_config)

# 添加 prompt 模板
self.prompt_template = "User's past: {history} Next:"

# 构造 prompt 格式的输入
input_tokens = [BOS] + history_tokens + [SEP]
full_tokens = input_tokens + target_tokens + [EOS]

# 只预测 target 部分
labels = [-100] * len(input_tokens) + target_tokens + [EOS]
```

### 2. main_llm.py

```python
# 使用新的 Dataset 类
from tiger_llm import PromptGenRecDataset, collate_fn

# 训练时传入 labels
def train_epoch(model, train_loader, optimizer, device):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)  # 只计算 target 的 loss
        
        loss, logits = model(input_ids, attention_mask, labels)
```

## 扩展建议

### 1. 添加更多特殊 token

```python
# 可以添加用户画像、时间等信息
USER_TOKEN = 4      # 用户特征开始
TIME_TOKEN = 5      # 时间特征
CATEGORY_TOKEN = 6  # 类别特征
```

### 2. 使用 LoRA 微调

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### 3. 多轮对话形式

```python
# 将推荐变成多轮对话
prompt = """User: I watched {movie1}, {movie2}
System: Did you like {movie2}?
User: Yes, it was great!
System: Then you might like: {recommendation}"""
```

## 故障排除

### 1. 显存不足

```bash
# 减小 batch_size
--batch_size 32

# 或使用梯度累积
--gradient_accumulation_steps 2
```

### 2. 序列过长

```bash
# 减小 max_seq_len
--max_seq_len 256

# 或截断历史
--max_len 10
```

### 3. 生成质量差

```bash
# 增大 beam_size
--beam_size 50

# 或使用采样
--do_sample --top_p 0.9 --temperature 0.8
```
