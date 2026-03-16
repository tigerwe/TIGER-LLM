# TIGER-LLM TensorRT-LLM 部署指南

使用 TensorRT-LLM 和 Docker 部署 TIGER-LLM 推理服务，实现高性能生成式推荐。

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                      Client                             │
│              (Web/App/Other Services)                   │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP REST API
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Nginx (Reverse Proxy)                      │
│     • Load Balancing    • Rate Limiting    • SSL        │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│           TIGER-LLM Inference Server                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  FastAPI REST API                               │   │
│  │  • /recommend (single)                          │   │
│  │  • /recommend/batch (batch)                     │   │
│  │  • /health (health check)                       │   │
│  └────────────────────┬────────────────────────────┘   │
│                       │                                 │
│  ┌────────────────────▼────────────────────────────┐   │
│  │  TensorRT-LLM Engine / PyTorch Fallback         │   │
│  │  • FP16 Mixed Precision                         │   │
│  │  • Beam Search                                  │   │
│  │  • KV Cache                                     │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 环境准备

#### 必要条件
- **NVIDIA GPU** (推荐 RTX 3060 及以上，显存 ≥ 6GB)
- **NVIDIA Driver** ≥ 470
- **Docker** ≥ 20.10
- **Docker Compose** ≥ 2.0
- **NVIDIA Container Toolkit** (nvidia-docker2)

#### 安装 NVIDIA Container Toolkit
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. 一键部署

```bash
cd triton_server
chmod +x deploy.sh
./deploy.sh all
```

这会自动执行：
1. ✅ 检查依赖
2. ✅ 构建 Docker 镜像
3. ✅ 导出 TensorRT 引擎
4. ✅ 启动推理服务
5. ✅ 测试 API

### 3. 验证部署

```bash
# 健康检查
curl http://localhost:8000/health

# 测试推荐
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "history": [1, 2, 3, 4, 5],
    "topk": 10
  }'
```

## 手动部署步骤

### 步骤 1: 构建镜像

```bash
cd triton_server
docker build -t tiger-llm:inference-v1.0 -f Dockerfile ..
```

### 步骤 2: 导出 TensorRT 引擎

```bash
# 方法1: 使用脚本
python export_tensorrt.py \
    --checkpoint ../model/ckpt/tiger_llm_gpu.pth \
    --output_dir ./trt_engines \
    --method hf

# 方法2: 使用 Docker
docker run --rm --gpus all \
    -v $(pwd)/../model/ckpt:/app/model/ckpt:ro \
    -v $(pwd)/trt_engines:/app/trt_engines \
    tiger-llm:inference-v1.0 \
    python triton_server/export_tensorrt.py \
        --checkpoint /app/model/ckpt/tiger_llm_gpu.pth \
        --output_dir /app/trt_engines
```

### 步骤 3: 启动服务

```bash
# 基础启动
docker-compose up -d

# 带 Nginx 反向代理
docker-compose --profile with-nginx up -d

# 带监控 (Prometheus + Grafana)
docker-compose --profile monitoring up -d
```

## API 文档

### 端点列表

| 端点 | 方法 | 描述 |
|-----|------|------|
| `/` | GET | 服务信息 |
| `/health` | GET | 健康检查 |
| `/recommend` | POST | 单用户推荐 |
| `/recommend/batch` | POST | 批量推荐 |
| `/model/info` | GET | 模型信息 |
| `/benchmark` | POST | 性能测试 |

### 请求示例

#### 单用户推荐
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "history": [1, 2, 3, 4, 5],
    "topk": 20,
    "temperature": 1.0,
    "beam_width": 20
  }'
```

**响应:**
```json
{
  "user_id": "user_123",
  "recommendations": [
    {"item_id": 42, "score": 0.95, "rank": 1},
    {"item_id": 17, "score": 0.87, "rank": 2},
    ...
  ],
  "inference_time_ms": 12.5,
  "backend": "TensorRT-LLM"
}
```

#### 批量推荐
```bash
curl -X POST http://localhost:8000/recommend/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"user_id": "user_1", "history": [1, 2, 3], "topk": 10},
      {"user_id": "user_2", "history": [4, 5, 6], "topk": 10}
    ]
  }'
```

## 性能优化

### TensorRT-LLM vs PyTorch

| 指标 | PyTorch | TensorRT-LLM | 提升 |
|-----|---------|--------------|------|
| 延迟 (P99) | 50ms | 15ms | **3.3x** |
| 吞吐 (QPS) | 500 | 2000 | **4x** |
| 显存占用 | 5GB | 3GB | **40%** |
| 批处理能力 | 中等 | 优秀 | - |

### 调优参数

在 `docker-compose.yml` 中调整：

```yaml
environment:
  # 批处理大小
  - MAX_BATCH_SIZE=64
  
  # Beam search 宽度
  - BEAM_WIDTH=20
  
  # 使用 TensorRT (true/false)
  - USE_TRT=true
```

### 批处理策略

- **单用户请求**: 延迟优先，`beam_width=20`
- **批量请求**: 吞吐优先，`batch_size=64`
- **实时推荐**: 小 batch，`max_batch_size=8`

## 监控与日志

### 查看日志
```bash
# 实时日志
docker-compose logs -f tiger-llm-server

# 导出日志
docker-compose logs tiger-llm-server > server.log
```

### 性能监控

```bash
# 运行基准测试
curl -X POST "http://localhost:8000/benchmark?num_requests=1000"

# GPU 监控
watch -n 1 nvidia-smi
```

### Prometheus + Grafana（可选）

```bash
# 启动监控栈
docker-compose --profile monitoring up -d

# 访问 Grafana: http://localhost:3000
# 默认账号: admin/admin
```

## 故障排查

### 常见问题

#### 1. 容器启动失败
```bash
# 检查日志
docker-compose logs tiger-llm-server

# 常见原因：
# - GPU 不可用：检查 nvidia-docker 安装
# - 模型文件缺失：确认挂载路径
# - 端口冲突：修改 docker-compose.yml 端口映射
```

#### 2. TensorRT 引擎构建失败
```bash
# 使用 PyTorch 备选
docker-compose exec tiger-llm-server \
    python -c "import os; os.environ['USE_TRT'] = 'false'"

# 或修改 docker-compose.yml
environment:
  - USE_TRT=false
```

#### 3. 内存不足 (OOM)
```bash
# 减小批处理大小
docker-compose exec tiger-llm-server \
    python triton_server/api_server.py --max-batch-size 32

# 限制容器内存
deploy:
  resources:
    limits:
      memory: 6G
```

#### 4. 推理速度慢
```bash
# 检查是否使用 GPU
docker-compose exec tiger-llm-server nvidia-smi

# 检查后端类型
curl http://localhost:8000/health
# 应返回 "backend": "TensorRT-LLM"
```

### 调试模式

```bash
# 前台启动，查看详细日志
docker-compose up tiger-llm-server

# 进入容器调试
docker-compose exec tiger-llm-server bash
```

## 生产环境建议

### 1. 高可用部署

```yaml
# docker-compose.yml
deploy:
  replicas: 2
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1  # 每个实例使用 1 个 GPU
```

### 2. 负载均衡

使用 Nginx 或 Envoy 进行负载均衡：

```nginx
upstream tiger_backend {
    server tiger-llm-1:8000;
    server tiger-llm-2:8000;
}

server {
    location / {
        proxy_pass http://tiger_backend;
    }
}
```

### 3. 自动扩缩容

使用 Kubernetes HPA：

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tiger-llm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tiger-llm
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 安全建议

1. **API 认证**: 使用 API Key 或 JWT
2. **速率限制**: 限制每个用户的请求频率
3. **输入验证**: 限制 history 长度和 item ID 范围
4. **HTTPS**: 使用 SSL/TLS 加密传输
5. **网络隔离**: 使用 Docker 网络隔离

## 许可证

MIT License - 详见项目根目录 LICENSE 文件
