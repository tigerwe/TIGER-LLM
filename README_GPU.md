# TIGER-LLM GPU 训练指南

## 硬件环境

**已检测到的配置：**
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- 显存: 6 GB
- CUDA 版本: 11.8 (兼容驱动 11.4)
- PyTorch: 2.7.1+cu118

## 快速开始

### 1. 一键启动 GPU 训练

```bash
cd /home/vivwimp/TIGER
./train_gpu.sh
```

### 2. 后台运行（推荐）

```bash
cd /home/vivwimp/TIGER
nohup ./train_gpu.sh > train_gpu.log 2>&1 &

# 查看日志
tail -f train_gpu.log
```

### 3. 监控训练

```bash
# 方式 1: 实时监控 GPU
watch -n 1 nvidia-smi

# 方式 2: 使用监控脚本
python monitor_gpu.py

# 方式 3: 查看训练日志
tail -f model/logs/tiger_llm_gpu.log
```

## GPU 优化特性

### 1. 混合精度训练 (FP16)

```python
with autocast(enabled=True):
    loss, logits = model(input_ids, attention_mask, labels)
```

**效果：**
- 显存占用减少 ~40%
- 训练速度提升 ~30%
- RTX 3060 6GB 可以从 batch_size=16 提升到 32

### 2. 梯度累积

```bash
--batch_size 32 --grad_accum_steps 2
# 等效 batch_size = 64
```

**效果：**
- 小显存也能模拟大 batch
- 训练更稳定

### 3. 显存优化设置

| 参数 | 值 | 说明 |
|-----|-----|------|
| --batch_size | 32 | 主批次大小 |
| --grad_accum_steps | 2 | 梯度累积 |
| --max_seq_len | 256 | 最大序列长度 |
| --d_model | 256 | 模型维度 |
| --num_layers | 4 | 层数 |
| --use_amp | True | 混合精度 |

**显存占用：** ~4.5GB / 6GB

## 性能对比

### CPU vs GPU (RTX 3060)

| 指标 | CPU | GPU | 加速比 |
|-----|-----|-----|-------|
| 模型创建 | 1s | 1s | 1x |
| 单批次前向 | 2.5s | 0.05s | **50x** |
| 单 epoch (78584 样本) | ~50min | ~2min | **25x** |
| 完整训练 (50 epochs) | ~40h | ~1.5h | **25x** |

*注：实际速度取决于 CPU 型号和 batch size*

## 显存配置建议

### 根据显存大小调整

#### 6GB (RTX 3060)
```bash
--batch_size 32 \
--grad_accum_steps 2 \
--d_model 256 \
--num_layers 4 \
--max_seq_len 256
```

#### 8GB (RTX 3070/4060)
```bash
--batch_size 64 \
--grad_accum_steps 1 \
--d_model 256 \
--num_layers 4 \
--max_seq_len 512
```

#### 12GB (RTX 3060 Ti/4070)
```bash
--batch_size 128 \
--grad_accum_steps 1 \
--d_model 512 \
--num_layers 6 \
--max_seq_len 512
```

#### 24GB (RTX 3090/4090)
```bash
--batch_size 256 \
--grad_accum_steps 1 \
--d_model 768 \
--num_layers 8 \
--max_seq_len 1024
```

## 常见问题

### Q1: CUDA out of memory

**解决：**
```bash
# 减小 batch_size
--batch_size 16

# 或减小序列长度
--max_seq_len 128

# 或减小模型
--d_model 128 --num_layers 2
```

### Q2: 训练速度不够快

**检查：**
1. 确认使用 GPU: `nvidia-smi` 查看 GPU 利用率
2. 调整 num_workers: `--num_workers 4` (根据 CPU 核心数)
3. 使用更大的 batch_size (如果显存允许)

### Q3: 混合精度训练错误

**解决：**
```bash
# 禁用混合精度
--no-use_amp
```

### Q4: 多 GPU 训练

当前代码支持单 GPU，多 GPU 需要修改：
```python
# 使用 DataParallel
model = nn.DataParallel(model)

# 或使用 DistributedDataParallel
# (需要更复杂的设置)
```

## 训练监控

### 1. 实时查看 GPU 状态

```bash
# 每秒刷新
watch -n 1 nvidia-smi

# 或使用 gpustat (需安装: pip install gpustat)
gpustat -i 1
```

### 2. 查看训练日志

```bash
# 实时查看
tail -f model/logs/tiger_llm_gpu.log

# 查看最后 100 行
tail -n 100 model/logs/tiger_llm_gpu.log | grep -E "(Epoch|Loss|Recall|NDCG)"
```

### 3. TensorBoard (可选)

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./runs/tiger_llm')

# 训练时
writer.add_scalar('Loss/train', loss, step)
writer.add_scalar('NDCG/valid', ndcg, epoch)
```

查看：
```bash
tensorboard --logdir=./runs
```

## 故障排除

### 问题：PyTorch 找不到 CUDA

```bash
# 检查 CUDA 版本
nvcc --version

# 重新安装 PyTorch (匹配 CUDA 版本)
# CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 问题：GPU 利用率低

**原因：**
1. DataLoader 瓶颈 → 增加 `num_workers`
2. Batch size 太小 → 增大 `batch_size`
3. 序列长度太短 → 增大 `max_seq_len`

### 问题：训练中断后恢复

```python
# 保存检查点
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# 恢复训练
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

## 预期训练时间

**RTX 3060 6GB 配置：**
- 数据集: Tenrec (78,584 样本)
- 配置: d=256, l=4, batch=32
- 1 epoch: ~2 分钟
- 50 epochs: ~1.5 小时 (早停可能提前结束)

**最终指标预期：**
- Recall@5: 0.02-0.05
- Recall@10: 0.04-0.08
- NDCG@5: 0.015-0.03
- NDCG@10: 0.02-0.04

*注：实际指标取决于数据质量和训练充分度*
