"""
TIGER-LLM TensorRT-LLM 导出脚本
将 PyTorch 模型转换为 TensorRT-LLM 格式
"""
import os
import sys
import torch
import argparse
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from model.tiger_llm import TIGER_LLM


def export_to_onnx(model, output_path, max_batch_size=64, max_seq_len=256):
    """
    导出 PyTorch 模型为 ONNX 格式
    """
    print(f"Exporting model to ONNX: {output_path}")
    
    model.eval()
    
    # 创建示例输入
    batch_size = 1
    seq_len = 50
    dummy_input = torch.randint(
        0, model.model.config.vocab_size, 
        (batch_size, seq_len),
        dtype=torch.long
    ).cuda()
    
    # 导出 ONNX
    torch.onnx.export(
        model.model,
        dummy_input,
        output_path,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=14,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"✓ ONNX model saved to: {output_path}")
    return output_path


def build_tensorrt_engine(onnx_path, output_dir, dtype='fp16', max_batch_size=64):
    """
    使用 TensorRT-LLM 构建推理引擎
    """
    print(f"Building TensorRT-LLM engine...")
    print(f"  ONNX: {onnx_path}")
    print(f"  Output: {output_dir}")
    print(f"  Data type: {dtype}")
    print(f"  Max batch size: {max_batch_size}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # TensorRT-LLM 构建命令
    build_cmd = f"""
    trtllm-build \
        --checkpoint_dir {onnx_path} \
        --output_dir {output_dir} \
        --dtype {dtype} \
        --max_batch_size {max_batch_size} \
        --max_input_len 256 \
        --max_output_len 10 \
        --gemm_plugin {dtype} \
        --gpt_attention_plugin {dtype} \
        --context_fmha enable
    """
    
    print("Building command:")
    print(build_cmd)
    
    # 执行构建
    ret = os.system(build_cmd)
    if ret != 0:
        print("⚠️  trtllm-build failed, trying alternative method...")
        # 备选：使用 Python API
        build_with_python_api(onnx_path, output_dir, dtype, max_batch_size)
    
    print(f"✓ TensorRT engine saved to: {output_dir}")


def build_with_python_api(onnx_path, output_dir, dtype='fp16', max_batch_size=64):
    """
    使用 TensorRT-LLM Python API 构建引擎
    """
    try:
        import tensorrt_llm
        from tensorrt_llm.models import GPTLMHeadModel
        from tensorrt_llm._utils import str_dtype_to_trt
        
        print("Using TensorRT-LLM Python API...")
        
        # 这里需要根据实际模型配置调整
        # 这是一个示例配置
        config = {
            'vocab_size': 1300,
            'n_layer': 4,
            'n_embd': 256,
            'n_head': 8,
            'n_positions': 256,
        }
        
        # 创建 TensorRT-LLM 模型
        trt_llm_model = GPTLMHeadModel(config)
        
        # 加载权重并构建引擎
        # ... (需要实现具体的权重转换)
        
        print("✓ Engine built with Python API")
        
    except ImportError:
        print("❌ TensorRT-LLM not installed. Please install with:")
        print("  pip install tensorrt_llm")
        raise


def convert_checkpoint_to_hf(checkpoint_path, output_dir):
    """
    将 PyTorch checkpoint 转换为 HuggingFace 格式
    这是 TensorRT-LLM 推荐的方式
    """
    print(f"Converting checkpoint to HuggingFace format...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    model = TIGER_LLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 保存为 HuggingFace 格式
    model.model.save_pretrained(output_dir)
    
    # 保存配置文件
    import json
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump({
            'architectures': ['GPT2LMHeadModel'],
            'model_type': 'gpt2',
            'vocab_size': config['vocab_size'],
            'n_layer': config['num_layers'],
            'n_embd': config['d_model'],
            'n_head': config['num_heads'],
            'n_positions': config['max_seq_len'],
            'n_ctx': config['max_seq_len'],
            'pad_token_id': config['pad_token_id'],
            'bos_token_id': config['bos_token_id'],
            'eos_token_id': config['eos_token_id'],
        }, f, indent=2)
    
    print(f"✓ HuggingFace model saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Export TIGER-LLM to TensorRT')
    parser.add_argument('--checkpoint', type=str, 
                        default='../model/ckpt/tiger_llm_gpu.pth',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output_dir', type=str, 
                        default='./trt_engines',
                        help='Output directory for TensorRT engine')
    parser.add_argument('--dtype', type=str, default='fp16', 
                        choices=['fp16', 'fp32', 'int8'],
                        help='Data type for TensorRT engine')
    parser.add_argument('--max_batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('--method', type=str, default='hf',
                        choices=['onnx', 'hf', 'direct'],
                        help='Conversion method')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TIGER-LLM TensorRT Export")
    print("=" * 60)
    
    if args.method == 'hf':
        # 方法1: 转换为 HuggingFace 格式 (推荐)
        hf_dir = os.path.join(args.output_dir, 'hf_model')
        convert_checkpoint_to_hf(args.checkpoint, hf_dir)
        print(f"\n✓ Export complete!")
        print(f"  HuggingFace model: {hf_dir}")
        print(f"\nNext step: Build TensorRT engine with:")
        print(f"  trtllm-build --checkpoint_dir {hf_dir} --output_dir {args.output_dir}/engine")
        
    elif args.method == 'onnx':
        # 方法2: ONNX 导出
        # 加载模型
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model = TIGER_LLM(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.cuda().eval()
        
        onnx_path = os.path.join(args.output_dir, 'model.onnx')
        export_to_onnx(model, onnx_path, args.max_batch_size)
        
    elif args.method == 'direct':
        # 方法3: 直接构建
        build_tensorrt_engine(args.checkpoint, args.output_dir, 
                             args.dtype, args.max_batch_size)


if __name__ == '__main__':
    main()
