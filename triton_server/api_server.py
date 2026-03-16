"""
TIGER-LLM REST API Server
基于 FastAPI 的推理服务
"""
import os
import sys
import asyncio
import uvicorn
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np

sys.path.append(str(Path(__file__).parent))
from inference_server import TIGERLLMInference, InferenceConfig


# ============== 数据模型 ==============

class RecommendRequest(BaseModel):
    """推荐请求"""
    user_id: Optional[str] = Field(None, description="用户ID（可选）")
    history: List[int] = Field(..., description="用户历史item ID列表", example=[1, 2, 3, 4, 5])
    topk: int = Field(20, ge=1, le=100, description="推荐数量")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="采样温度")
    beam_width: int = Field(20, ge=1, le=50, description="Beam search宽度")


class RecommendResponse(BaseModel):
    """推荐响应"""
    user_id: Optional[str] = None
    recommendations: List[Dict] = Field(..., description="推荐结果列表")
    inference_time_ms: float = Field(..., description="推理时间（毫秒）")
    backend: str = Field(..., description="推理后端")


class BatchRecommendRequest(BaseModel):
    """批量推荐请求"""
    requests: List[RecommendRequest]


class BatchRecommendResponse(BaseModel):
    """批量推荐响应"""
    results: List[RecommendResponse]
    total_time_ms: float


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    backend: str
    max_batch_size: int
    model_loaded: bool


# ============== 全局状态 ==============

inference_engine: Optional[TIGERLLMInference] = None
engine_config: Optional[InferenceConfig] = None


# ============== 生命周期管理 ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global inference_engine, engine_config
    
    # 启动时加载模型
    print("=" * 60)
    print("Starting TIGER-LLM Inference Server")
    print("=" * 60)
    
    # 从环境变量读取配置
    engine_dir = os.getenv('TRT_ENGINE_DIR', './trt_engines')
    max_batch_size = int(os.getenv('MAX_BATCH_SIZE', '64'))
    beam_width = int(os.getenv('BEAM_WIDTH', '20'))
    use_trt = os.getenv('USE_TRT', 'true').lower() == 'true'
    
    engine_config = InferenceConfig(
        engine_dir=engine_dir,
        max_batch_size=max_batch_size,
        beam_width=beam_width
    )
    
    try:
        inference_engine = TIGERLLMInference(engine_config, use_trt=use_trt)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠️  Failed to load TensorRT model: {e}")
        print("Falling back to PyTorch...")
        inference_engine = TIGERLLMInference(engine_config, use_trt=False)
    
    print("=" * 60)
    
    yield
    
    # 关闭时清理
    print("Shutting down server...")
    inference_engine = None


# ============== 创建 FastAPI 应用 ==============

app = FastAPI(
    title="TIGER-LLM Inference API",
    description="Generative Recommendation API using Decoder-Only LLM",
    version="1.0.0",
    lifespan=lifespan
)


# ============== API 端点 ==============

@app.get("/", response_model=Dict)
async def root():
    """根路径"""
    return {
        "service": "TIGER-LLM Inference API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    global inference_engine, engine_config
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return HealthResponse(
        status="healthy",
        backend="TensorRT-LLM" if inference_engine.use_trt else "PyTorch",
        max_batch_size=engine_config.max_batch_size,
        model_loaded=True
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    单用户推荐
    
    - **history**: 用户历史交互的item ID列表
    - **topk**: 返回的推荐数量（默认20）
    - **temperature**: 采样温度（默认1.0）
    - **beam_width**: Beam search宽度（默认20）
    """
    global inference_engine
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    try:
        recommendations = inference_engine.recommend(
            history_items=request.history,
            topk=request.topk,
            temperature=request.temperature,
            beam_width=request.beam_width
        )
        
        inference_time = (time.time() - start_time) * 1000
        
        # 格式化结果
        rec_list = [
            {"item_id": int(item_id), "score": float(score), "rank": i+1}
            for i, (item_id, score) in enumerate(recommendations)
        ]
        
        return RecommendResponse(
            user_id=request.user_id,
            recommendations=rec_list,
            inference_time_ms=inference_time,
            backend="TensorRT-LLM" if inference_engine.use_trt else "PyTorch"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/recommend/batch", response_model=BatchRecommendResponse)
async def recommend_batch(request: BatchRecommendRequest):
    """
    批量推荐
    
    一次处理多个用户的推荐请求，提高效率
    """
    global inference_engine
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    results = []
    
    # 目前使用顺序处理，后续可以优化为真正的 batch
    for req in request.requests:
        try:
            rec_result = await recommend(req)
            results.append(rec_result)
        except Exception as e:
            # 记录错误但继续处理
            results.append(RecommendResponse(
                user_id=req.user_id,
                recommendations=[],
                inference_time_ms=0,
                backend="error",
                error=str(e)
            ))
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchRecommendResponse(
        results=results,
        total_time_ms=total_time
    )


@app.get("/model/info")
async def model_info():
    """获取模型信息"""
    global inference_engine, engine_config
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "backend": "TensorRT-LLM" if inference_engine.use_trt else "PyTorch",
        "max_batch_size": engine_config.max_batch_size,
        "beam_width": engine_config.beam_width,
        "max_input_len": engine_config.max_input_len,
        "max_output_len": engine_config.max_output_len,
        "temperature": engine_config.temperature,
        "top_p": engine_config.top_p,
        "codebook_size": inference_engine.CODEBOOK_SIZE,
        "num_codebooks": inference_engine.NUM_CODEBOOKS,
    }


@app.post("/benchmark")
async def run_benchmark(num_requests: int = 100):
    """
    运行性能基准测试
    
    - **num_requests**: 测试请求数量（默认100）
    """
    global inference_engine
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import io
    import sys
    from contextlib import redirect_stdout
    
    # 捕获 benchmark 输出
    f = io.StringIO()
    with redirect_stdout(f):
        inference_engine.benchmark(num_requests=num_requests)
    
    output = f.getvalue()
    
    return {
        "benchmark_output": output,
        "num_requests": num_requests
    }


# ============== 错误处理 ==============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# ============== 启动入口 ==============

def main():
    """启动服务器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TIGER-LLM API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host')
    parser.add_argument('--port', type=int, default=8000, help='Port')
    parser.add_argument('--workers', type=int, default=1, help='Workers')
    parser.add_argument('--reload', action='store_true', help='Auto reload')
    
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )


if __name__ == '__main__':
    main()
