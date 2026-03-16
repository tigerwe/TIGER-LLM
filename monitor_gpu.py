#!/usr/bin/env python3
"""
GPU 训练监控脚本
实时监控显存使用和训练进度
"""
import subprocess
import time
import os
import sys


def get_gpu_info():
    """获取 GPU 信息"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
            return {
                'utilization': float(util),
                'memory_used': float(mem_used) / 1024,  # GB
                'memory_total': float(mem_total) / 1024,  # GB
                'temperature': float(temp)
            }
    except Exception as e:
        print(f"获取 GPU 信息失败: {e}")
    return None


def get_training_log(log_file):
    """读取训练日志的最后几行"""
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                return lines[-5:] if len(lines) >= 5 else lines
    except Exception:
        pass
    return []


def clear_screen():
    """清屏"""
    os.system('clear' if os.name == 'posix' else 'cls')


def main():
    log_file = './model/logs/tiger_llm_gpu.log'
    
    print("GPU 训练监控")
    print("按 Ctrl+C 退出")
    print("-" * 60)
    
    try:
        while True:
            gpu_info = get_gpu_info()
            
            if gpu_info:
                print(f"\rGPU: {gpu_info['utilization']:5.1f}% | "
                      f"显存: {gpu_info['memory_used']:.2f}/{gpu_info['memory_total']:.2f} GB | "
                      f"温度: {gpu_info['temperature']:.0f}°C", end='', flush=True)
            else:
                print("\r无法获取 GPU 信息", end='', flush=True)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n监控结束")


if __name__ == "__main__":
    main()
