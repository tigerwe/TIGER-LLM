# TIGER → TIGER-LLM 改造完成总结

## 改造目标

将 TIGER 从 **T5 (encoder-decoder)** 架构改造为 **Decoder-only LLM (GPT) + Prompt** 形式，保留 Tenrec 数据集支持。

## 改造结果

✅ 已完成所有改造，测试通过！

## 新文件列表

### 核心代码文件

| 文件 | 说明 |
|-----|------|
| `TIGER/model/tiger_llm.py` | Decoder-only LLM 模型定义 + Prompt Dataset |
| `TIGER/model/main_llm.py` | LLM 版本训练脚本 |
| `TIGER/model/test_llm_quick.py` | 快速测试脚本 |

### 脚本和文档

| 文件 | 说明 |
|-----|------|
| `TIGER/train_llm.sh` | LLM 版本一键训练脚本 |
| `TIGER/README_LLM.md` | 详细使用说明和架构对比 |
| `TIGER/TRANSFORM_SUMMARY.md` | 本总结文档 |

## 快速开始

### 1. 使用现有数据训练

```bash
cd /home/vivwimp/TIGER
./train_llm.sh
```

### 2. 完整训练流程（从 CSV 开始）

```bash
# 1. 数据预处理
cd TIGER/data
python prepare_tenrec.py \
    --input ../../Tenrec/ctr_data_1M.csv \
    --output ./Tenrec \
    --sample_ratio 0.05

# 2. 训练 RQVAE（如果尚未完成）
cd ../rqvae
python train_tenrec_rqvae.py --data_path ../data/Tenrec/item_emb.parquet --epochs 20

# 3. 训练 TIGER-LLM
cd ../model
python main_llm.py \
    --dataset_path ../data/Tenrec \
    --code_path ../data/Tenrec/Tenrec_t5_rqvae.npy \
    --save_path ./ckpt/tiger_llm.pth \
    --num_epochs 30 \
    --batch_size 64 \
    --device cuda
```

### 3. 快速测试

```bash
cd TIGER/model
python test_llm_quick.py
```

## 架构对比

### 原 TIGER (T5)

```
Input:  [PAD, PAD, code1, code2]
                ↓
            Encoder
                ↓
Target: [code3, EOS]
                ↑
            Decoder
```

### TIGER-LLM (GPT + Prompt)

```
Prompt: "User's past: [code1][code2] Next:"
                ↓
Input:  [BOS, code1, SEP, code2, SEP, code3, EOS]
                ↓
           Decoder-only
                ↓
Output: [BOS, code1, SEP, code2, SEP, code3, EOS]
                      ↑___________↑ 只预测这部分
```

## 主要变化

### 1. 模型架构

| 特性 | 原 TIGER | TIGER-LLM |
|-----|---------|-----------|
| 基础模型 | T5ForConditionalGeneration | GPT2LMHeadModel |
| 架构 | Encoder-Decoder | Decoder-only |
| 注意力 | 双向 + 因果 | 纯因果 (Causal) |
| 参数量 (d=256, l=4) | ~4.6M | ~0.6M (轻量级) |

### 2. 数据格式

| 特性 | 原 TIGER | TIGER-LLM |
|-----|---------|-----------|
| History | 展平的 code 序列 | Prompt + code 序列 |
| Target | 单独生成 | 作为序列的一部分生成 |
| 特殊 token | PAD=0, EOS=0 | PAD=0, BOS=1, EOS=2, SEP=3 |

### 3. 训练方式

| 特性 | 原 TIGER | TIGER-LLM |
|-----|---------|-----------|
| Loss 计算 | 所有 target token | 只计算 target 部分 (-100 mask) |
| 学习率调度 | 无 | CosineAnnealing |
| 优化器 | Adam | AdamW |
| 梯度裁剪 | 无 | max_norm=1.0 |

## 性能对比（预期）

| 指标 | T5 (原) | GPT (LLM) |
|-----|--------|----------|
| 训练速度 | 较慢 | 较快 (单一 decoder) |
| 内存占用 | 较高 | 较低 |
| 生成质量 | 较好 | 需调参 |
| 可扩展性 | 一般 | 好 (易于使用预训练 LLM) |

## Token 定义

```python
PAD_TOKEN = 0   # 填充
BOS_TOKEN = 1   # 序列开始
EOS_TOKEN = 2   # 序列结束
SEP_TOKEN = 3   # Item 分隔符
# 4-9 预留
# 10+ Code tokens (code + offset)
```

## 关键参数

### TIGER-LLM 特有参数

```bash
--max_seq_len 512      # 最大序列长度（包含 prompt）
--vocab_size 1300      # 词表大小（1281 codes + 特殊token）
--code_dim 5           # Code 维度
--prompt_template "..." # Prompt 模板
```

### 推荐配置

```bash
# 轻量级（快速实验）
--d_model 128 --num_layers 2 --num_heads 4

# 标准配置
--d_model 256 --num_layers 4 --num_heads 8

# 更强配置
--d_model 512 --num_layers 6 --num_heads 8
```

## 扩展方向

### 1. 使用更大的预训练模型

```python
from transformers import AutoModelForCausalLM

# 替换 GPT2 为更大的模型
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
```

### 2. 添加 LoRA 微调

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["c_attn", "c_proj"])
model = get_peft_model(model, lora_config)
```

### 3. 多轮对话格式

```python
prompt = """User history:
1. {item1}
2. {item2}

Based on these, recommend the next item:"""
```

## 故障排除

### Q: 显存不足
A: 减小 `--batch_size` 或 `--max_seq_len`

### Q: 生成结果全是 PAD
A: 增加训练轮数，或检查 `--vocab_size` 设置

### Q: Loss 不下降
A: 检查 `labels` 是否正确设置（应包含非 -100 的值）

### Q: 评估指标异常
A: 确保 `code_dim` 和生成的维度一致

## 文件完整性检查

```bash
# 验证所有文件存在
ls TIGER/model/tiger_llm.py
ls TIGER/model/main_llm.py
ls TIGER/model/test_llm_quick.py
ls TIGER/train_llm.sh
ls TIGER/README_LLM.md

# 验证数据存在
ls TIGER/data/Tenrec/train.parquet
ls TIGER/data/Tenrec/Tenrec_t5_rqvae.npy
```

## 下一步建议

1. **调参优化**: 调整 `d_model`, `num_layers`, `lr` 等参数
2. **更大模型**: 尝试 GPT2-medium/large
3. **预训练**: 在大规模语料上进行预训练
4. **多任务**: 添加其他推荐任务（序列推荐、点击预测等）
5. **Prompt 工程**: 尝试不同的 prompt 模板
