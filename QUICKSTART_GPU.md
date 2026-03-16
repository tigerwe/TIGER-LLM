# 🚀 GPU 训练快速启动

## 1. 检查环境
```bash
cd /home/vivwimp/TIGER
./train_gpu.sh  # 会自动检查 GPU
```

## 2. 开始训练

### 方式 A: 前台运行（实时查看）
```bash
./train_gpu.sh
```

### 方式 B: 后台运行（推荐）
```bash
nohup ./train_gpu.sh > train.log 2>&1 &
tail -f train.log
```

### 方式 C: 使用 tmux/screen
```bash
tmux new -s tiger
cd /home/vivwimp/TIGER
./train_gpu.sh
# Ctrl+B, D 分离会话
tmux attach -t tiger  # 重新连接
```

## 3. 监控训练

```bash
# 终端 1: 运行训练
./train_gpu.sh

# 终端 2: 监控 GPU
watch -n 1 nvidia-smi

# 终端 3: 查看日志
tail -f model/logs/tiger_llm_gpu.log
```

## 4. 预期结果

```
Epoch 1/50: Training...
Epoch 1: Train Loss: 2.3456
Epoch 1: Valid Recall@5: 0.0123, NDCG@5: 0.0087
Epoch 1: Valid Recall@10: 0.0234, NDCG@10: 0.0156
✓ 最佳模型已保存 (NDCG@20: 0.0234)

... (训练继续) ...

Epoch 25: Early stopping triggered.
训练完成！
```

## 5. 获取结果

```bash
# 模型文件
ls -lh model/ckpt/tiger_llm_gpu.pth

# 训练日志
cat model/logs/tiger_llm_gpu.log | grep -E "(Best|NDCG@20)"
```

## ⚡ 加速技巧

1. **显存足够？** 增大 batch size:
   ```bash
   --batch_size 64  # 默认 32
   ```

2. **追求速度？** 减小模型:
   ```bash
   --d_model 128 --num_layers 2  # 默认 256/4
   ```

3. **追求精度？** 增大模型:
   ```bash
   --d_model 512 --num_layers 6 --batch_size 16
   ```

## 🐛 遇到问题？

```bash
# 显存不足
CUDA out of memory → 减小 --batch_size 或 --max_seq_len

# GPU 未使用
检查 nvidia-smi 是否有 python 进程

# 训练卡住
检查 DataLoader num_workers，尝试设为 0
```
