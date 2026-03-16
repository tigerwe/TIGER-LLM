"""
TIGER-LLM TensorRT-LLM 推理服务
支持 batch inference 和 beam search
"""
import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

# 尝试导入 TensorRT-LLM
try:
    import tensorrt_llm
    from tensorrt_llm.runtime import GenerationSession, ModelConfig, SamplingConfig
    from tensorrt_llm._utils import torch_dtype_to_trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("⚠️  TensorRT-LLM not available, using PyTorch fallback")

sys.path.append(str(Path(__file__).parent.parent))
from model.tiger_llm import TIGER_LLM


@dataclass
class InferenceConfig:
    """推理配置"""
    engine_dir: str
    max_batch_size: int = 64
    max_input_len: int = 256
    max_output_len: int = 5  # code dimension
    beam_width: int = 20
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0


class TIGERLLMInference:
    """
    TIGER-LLM 推理引擎
    支持 TensorRT-LLM 和 PyTorch 两种后端
    """
    
    # 特殊 token ID
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    SEP_TOKEN = 3
    CODEBOOK_SIZE = 256
    NUM_CODEBOOKS = 5
    
    def __init__(self, config: InferenceConfig, use_trt: bool = True):
        self.config = config
        self.use_trt = use_trt and TRT_AVAILABLE
        
        # 加载 code 映射
        self._load_code_mapping()
        
        # 初始化推理引擎
        if self.use_trt:
            self._init_tensorrt_engine()
        else:
            self._init_pytorch_engine()
        
        print(f"✓ Inference engine initialized")
        print(f"  Backend: {'TensorRT-LLM' if self.use_trt else 'PyTorch'}")
        print(f"  Max batch: {config.max_batch_size}")
        print(f"  Beam width: {config.beam_width}")
    
    def _load_code_mapping(self):
        """加载 item-to-code 映射"""
        # 加载 RQVAE codes
        code_path = Path(__file__).parent.parent / 'data' / 'Tenrec' / 'Tenrec_t5_rqvae.npy'
        if code_path.exists():
            self.codes_data = np.load(code_path, allow_pickle=True)
            self.num_items = len(self.codes_data)
        else:
            print(f"⚠️  Code file not found: {code_path}")
            self.codes_data = None
            self.num_items = 100000  # 默认值
        
        # 构建 code-to-item 映射
        self.code_to_item = {}
        if self.codes_data is not None:
            for idx, code in enumerate(self.codes_data):
                offsets = [int(c) + i * self.CODEBOOK_SIZE + 10 
                          for i, c in enumerate(code)]
                self.code_to_item[tuple(offsets)] = idx + 1
    
    def _init_tensorrt_engine(self):
        """初始化 TensorRT-LLM 引擎"""
        print(f"Loading TensorRT-LLM engine from: {self.config.engine_dir}")
        
        # 加载引擎
        engine_path = os.path.join(self.config.engine_dir, 'rank0.engine')
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        
        # 读取引擎配置
        config_path = os.path.join(self.config.engine_dir, 'config.json')
        with open(config_path, 'r') as f:
            engine_config = json.load(f)
        
        # 创建模型配置
        self.model_config = ModelConfig(
            vocab_size=engine_config['vocab_size'],
            num_layers=engine_config['num_layers'],
            num_heads=engine_config['num_heads'],
            hidden_size=engine_config['hidden_size'],
            max_batch_size=self.config.max_batch_size,
            max_input_len=self.config.max_input_len,
            max_output_len=self.config.max_output_len,
            data_type=engine_config.get('data_type', 'fp16')
        )
        
        # 创建生成会话
        self.session = GenerationSession(self.model_config, engine_path)
        
        print(f"✓ TensorRT engine loaded")
    
    def _init_pytorch_engine(self):
        """初始化 PyTorch 引擎（备选）"""
        print(f"Loading PyTorch model from: {self.config.engine_dir}")
        
        # 加载模型
        checkpoint_path = Path(__file__).parent.parent / 'model' / 'ckpt' / 'tiger_llm_gpu.pth'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        self.model = TIGER_LLM(checkpoint['config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.cuda().eval()
        
        print(f"✓ PyTorch model loaded")
    
    def _item_to_code(self, item_id: int) -> List[int]:
        """将 item ID 转换为 code tokens"""
        if self.codes_data is not None:
            code_idx = (item_id - 1) % self.num_items
            code = self.codes_data[code_idx]
            offsets = [int(c) + i * self.CODEBOOK_SIZE + 10 
                      for i, c in enumerate(code)]
            return offsets
        else:
            # 简化版本：生成随机 code
            import random
            random.seed(item_id)
            return [random.randint(10, 265) for _ in range(self.NUM_CODEBOOKS)]
    
    def _codes_to_tokens(self, codes: List[List[int]]) -> List[int]:
        """将 code 列表转换为 token 序列"""
        tokens = []
        for code in codes:
            tokens.extend(code)
            tokens.append(self.SEP_TOKEN)
        return tokens[:-1] if tokens else tokens  # 移除最后一个 SEP
    
    def _build_prompt(self, history_items: List[int]) -> torch.Tensor:
        """构建输入 prompt"""
        history_codes = [self._item_to_code(item) for item in history_items]
        history_tokens = self._codes_to_tokens(history_codes)
        
        # [BOS] + history + [SEP]
        prompt_tokens = [self.BOS_TOKEN] + history_tokens + [self.SEP_TOKEN]
        
        return torch.tensor([prompt_tokens], dtype=torch.long)
    
    def _tokens_to_code(self, tokens: List[int]) -> List[int]:
        """将 tokens 转换回 code"""
        # 移除偏移量
        codes = []
        for i, token in enumerate(tokens[:self.NUM_CODEBOOKS]):
            if token >= 10:
                code = (token - 10 - i * self.CODEBOOK_SIZE)
                codes.append(max(0, min(255, code)))
            else:
                codes.append(0)
        return codes
    
    def _code_to_item(self, code: List[int]) -> int:
        """将 code 转换为 item ID"""
        code_tuple = tuple(code)
        if code_tuple in self.code_to_item:
            return self.code_to_item[code_tuple]
        
        # 近似匹配
        best_match = 1
        best_score = float('inf')
        for c, item_id in self.code_to_item.items():
            score = sum(abs(a - b) for a, b in zip(c, code))
            if score < best_score:
                best_score = score
                best_match = item_id
        
        return best_match
    
    def recommend(self, 
                  history_items: List[int], 
                  topk: int = 20,
                  temperature: float = None,
                  beam_width: int = None) -> List[Tuple[int, float]]:
        """
        为用户生成推荐
        
        Args:
            history_items: 用户历史交互的 item ID 列表
            topk: 推荐数量
            temperature: 采样温度
            beam_width: beam search 宽度
        
        Returns:
            [(item_id, score), ...] 按分数排序的推荐列表
        """
        if temperature is None:
            temperature = self.config.temperature
        if beam_width is None:
            beam_width = self.config.beam_width
        
        # 构建输入
        input_ids = self._build_prompt(history_items).cuda()
        
        if self.use_trt:
            return self._recommend_trt(input_ids, topk, temperature, beam_width)
        else:
            return self._recommend_pytorch(input_ids, topk, temperature, beam_width)
    
    def _recommend_trt(self, input_ids: torch.Tensor, topk: int, 
                       temperature: float, beam_width: int) -> List[Tuple[int, float]]:
        """使用 TensorRT-LLM 推理"""
        batch_size = input_ids.shape[0]
        input_lengths = torch.tensor([input_ids.shape[1]] * batch_size, dtype=torch.int32)
        
        # 创建采样配置
        sampling_config = SamplingConfig(
            beam_width=beam_width,
            temperature=temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k
        )
        
        # 生成
        output_ids = self.session.decode(
            input_ids,
            input_lengths,
            sampling_config,
            max_output_len=self.config.max_output_len
        )
        
        # 解析输出
        # output_ids shape: (batch_size, beam_width, seq_len)
        recommendations = []
        seen_items = set()
        
        for beam_idx in range(min(beam_width, output_ids.shape[1])):
            tokens = output_ids[0, beam_idx, :].cpu().tolist()
            
            # 转换为 code
            code = self._tokens_to_code(tokens)
            
            # 转换为 item
            item_id = self._code_to_item(code)
            
            if item_id not in seen_items:
                # 分数 (使用 beam 排名作为近似)
                score = 1.0 / (beam_idx + 1)
                recommendations.append((item_id, score))
                seen_items.add(item_id)
                
                if len(recommendations) >= topk:
                    break
        
        return recommendations
    
    def _recommend_pytorch(self, input_ids: torch.Tensor, topk: int,
                          temperature: float, beam_width: int) -> List[Tuple[int, float]]:
        """使用 PyTorch 推理（备选）"""
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=self.config.max_output_len,
                num_beams=beam_width,
                num_return_sequences=min(beam_width, topk),
                temperature=temperature,
                early_stopping=True
            )
        
        # 解析输出
        recommendations = []
        input_len = input_ids.shape[1]
        
        for i in range(outputs.shape[0]):
            tokens = outputs[i, input_len:].cpu().tolist()
            code = self._tokens_to_code(tokens)
            item_id = self._code_to_item(code)
            score = 1.0 / (i + 1)
            recommendations.append((item_id, score))
        
        return recommendations[:topk]
    
    def recommend_batch(self, 
                       history_items_list: List[List[int]],
                       topk: int = 20) -> List[List[Tuple[int, float]]]:
        """
        Batch 推荐
        
        Args:
            history_items_list: 多个用户的历史交互列表
            topk: 每个用户的推荐数量
        
        Returns:
            [[(item_id, score), ...], ...] 每个用户的推荐列表
        """
        results = []
        for history_items in history_items_list:
            recs = self.recommend(history_items, topk)
            results.append(recs)
        return results
    
    def benchmark(self, num_requests: int = 100, batch_size: int = 1):
        """性能基准测试"""
        print(f"\nBenchmarking with {num_requests} requests (batch={batch_size})...")
        
        # 准备测试数据
        import random
        test_histories = [
            [random.randint(1, 10000) for _ in range(random.randint(5, 20))]
            for _ in range(num_requests)
        ]
        
        # 预热
        for _ in range(10):
            self.recommend(test_histories[0], topk=20)
        
        # 测试
        start_time = time.time()
        
        if batch_size == 1:
            for history in test_histories:
                self.recommend(history, topk=20)
        else:
            for i in range(0, num_requests, batch_size):
                batch = test_histories[i:i+batch_size]
                self.recommend_batch(batch, topk=20)
        
        total_time = time.time() - start_time
        
        print(f"Results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Requests: {num_requests}")
        print(f"  Batch size: {batch_size}")
        print(f"  Throughput: {num_requests/total_time:.2f} req/s")
        print(f"  Latency: {total_time*1000/num_requests:.2f} ms/req")


if __name__ == '__main__':
    # 测试
    config = InferenceConfig(
        engine_dir='./trt_engines',
        max_batch_size=64,
        beam_width=20
    )
    
    engine = TIGERLLMInference(config, use_trt=False)
    
    # 单条测试
    history = [1, 2, 3, 4, 5]  # 示例历史
    recommendations = engine.recommend(history, topk=10)
    print(f"\nRecommendations for user with history {history}:")
    for item_id, score in recommendations:
        print(f"  Item {item_id}: {score:.4f}")
    
    # 性能测试
    engine.benchmark(num_requests=100)
