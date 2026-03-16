#!/bin/bash
# TIGER-LLM TensorRT-LLM 部署脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="tiger-llm:inference-v1.0"
CONTAINER_NAME="tiger-llm-inference"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  TIGER-LLM Deployment Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查命令
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查依赖
check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    if ! command_exists docker; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        echo -e "${RED}Error: Docker Compose is not installed${NC}"
        exit 1
    fi
    
    # 检查 NVIDIA Docker 运行时
    if ! docker info | grep -q "nvidia"; then
        echo -e "${YELLOW}Warning: NVIDIA Docker runtime not detected${NC}"
        echo "You may need to install nvidia-docker2 for GPU support"
    fi
    
    echo -e "${GREEN}✓ Dependencies OK${NC}"
}

# 检查模型文件
check_model() {
    echo -e "${YELLOW}Checking model files...${NC}"
    
    MODEL_FILE="$PROJECT_DIR/model/ckpt/tiger_llm_gpu.pth"
    if [ ! -f "$MODEL_FILE" ]; then
        echo -e "${RED}Error: Model file not found: $MODEL_FILE${NC}"
        echo "Please train the model first:"
        echo "  cd $PROJECT_DIR && ./train_gpu.sh"
        exit 1
    fi
    
    CODE_FILE="$PROJECT_DIR/data/Tenrec/Tenrec_t5_rqvae.npy"
    if [ ! -f "$CODE_FILE" ]; then
        echo -e "${RED}Error: Code file not found: $CODE_FILE${NC}"
        echo "Please run RQVAE training first"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Model files OK${NC}"
}

# 构建镜像
build_image() {
    echo -e "${YELLOW}Building Docker image...${NC}"
    
    cd "$PROJECT_DIR"
    
    docker build \
        -f triton_server/Dockerfile \
        -t $IMAGE_NAME \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        .
    
    echo -e "${GREEN}✓ Image built: $IMAGE_NAME${NC}"
}

# 导出 TensorRT 引擎
export_engine() {
    echo -e "${YELLOW}Exporting TensorRT engine...${NC}"
    
    cd "$PROJECT_DIR"
    
    # 使用临时容器导出引擎
    docker run --rm \
        --gpus all \
        -v "$PROJECT_DIR/model/ckpt:/app/model/ckpt:ro" \
        -v "$PROJECT_DIR/data/Tenrec:/app/data/Tenrec:ro" \
        -v "$SCRIPT_DIR/trt_engines:/app/trt_engines" \
        -w /app \
        $IMAGE_NAME \
        python triton_server/export_tensorrt.py \
            --checkpoint /app/model/ckpt/tiger_llm_gpu.pth \
            --output_dir /app/trt_engines \
            --method hf
    
    echo -e "${GREEN}✓ TensorRT engine exported${NC}"
}

# 启动服务
start_service() {
    echo -e "${YELLOW}Starting inference service...${NC}"
    
    cd "$SCRIPT_DIR"
    
    # 创建必要的目录
    mkdir -p logs trt_engines
    
    # 启动容器
    docker-compose up -d
    
    echo -e "${GREEN}✓ Service started${NC}"
    echo ""
    echo -e "${GREEN}API Endpoint: http://localhost:8000${NC}"
    echo -e "${GREEN}Health Check: http://localhost:8000/health${NC}"
    echo -e "${GREEN}API Docs: http://localhost:8000/docs${NC}"
}

# 停止服务
stop_service() {
    echo -e "${YELLOW}Stopping inference service...${NC}"
    
    cd "$SCRIPT_DIR"
    docker-compose down
    
    echo -e "${GREEN}✓ Service stopped${NC}"
}

# 查看日志
view_logs() {
    cd "$SCRIPT_DIR"
    docker-compose logs -f tiger-llm-server
}

# 测试服务
test_service() {
    echo -e "${YELLOW}Testing inference service...${NC}"
    
    # 等待服务启动
    echo "Waiting for service to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null; then
            echo -e "${GREEN}✓ Service is ready${NC}"
            break
        fi
        sleep 1
    done
    
    # 测试推荐接口
    echo "Testing recommend API..."
    curl -X POST http://localhost:8000/recommend \
        -H "Content-Type: application/json" \
        -d '{
            "user_id": "test_user",
            "history": [1, 2, 3, 4, 5],
            "topk": 10
        }' | python -m json.tool
}

# 性能测试
benchmark() {
    echo -e "${YELLOW}Running benchmark...${NC}"
    
    curl -X POST "http://localhost:8000/benchmark?num_requests=100" | python -m json.tool
}

# 显示帮助
show_help() {
    echo "Usage: ./deploy.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup       - Check dependencies and model files"
    echo "  build       - Build Docker image"
    echo "  export      - Export TensorRT engine"
    echo "  start       - Start inference service"
    echo "  stop        - Stop inference service"
    echo "  restart     - Restart inference service"
    echo "  logs        - View service logs"
    echo "  test        - Test inference service"
    echo "  benchmark   - Run performance benchmark"
    echo "  all         - Full deployment (build + export + start + test)"
    echo "  help        - Show this help message"
}

# 主逻辑
case "${1:-all}" in
    setup)
        check_dependencies
        check_model
        ;;
    build)
        check_dependencies
        build_image
        ;;
    export)
        export_engine
        ;;
    start)
        check_model
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        start_service
        ;;
    logs)
        view_logs
        ;;
    test)
        test_service
        ;;
    benchmark)
        benchmark
        ;;
    all)
        check_dependencies
        check_model
        build_image
        export_engine
        start_service
        sleep 5
        test_service
        ;;
    help)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
