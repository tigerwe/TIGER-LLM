#!/usr/bin/env python3
"""
完整的 Tenrec TIGER 训练脚本
执行流程:
1. 数据预处理 (csv -> parquet)
2. 生成语义嵌入
3. 训练 RQVAE
4. 生成离散代码
5. 训练 TIGER 模型
"""
import os
import sys
import argparse
import subprocess


def run_command(cmd, description):
    """运行命令并输出结果"""
    print(f"\n{'='*60}")
    print(f"步骤: {description}")
    print(f"命令: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"错误: {description} 失败!")
        sys.exit(1)
    print(f"✓ {description} 完成")


def main():
    parser = argparse.ArgumentParser(description="Train TIGER on Tenrec dataset")
    
    # 数据路径
    parser.add_argument('--csv_path', type=str, default='../Tenrec/ctr_data_1M.csv',
                        help='Path to Tenrec ctr_data_1M.csv')
    parser.add_argument('--data_dir', type=str, default='./data/Tenrec',
                        help='Directory to store processed data')
    parser.add_argument('--rqvae_ckpt_dir', type=str, default='./rqvae/ckpt/Tenrec',
                        help='Directory to save RQVAE checkpoints')
    parser.add_argument('--tiger_ckpt_path', type=str, default='./model/ckpt/tiger.pth',
                        help='Path to save final TIGER model')
    
    # 数据预处理参数
    parser.add_argument('--min_interactions', type=int, default=5,
                        help='Minimum interactions per user')
    
    # RQVAE 参数
    parser.add_argument('--rqvae_epochs', type=int, default=50)
    parser.add_argument('--rqvae_batch_size', type=int, default=256)
    parser.add_argument('--rqvae_lr', type=float, default=1e-3)
    
    # TIGER 参数
    parser.add_argument('--tiger_epochs', type=int, default=100)
    parser.add_argument('--tiger_batch_size', type=int, default=256)
    parser.add_argument('--tiger_lr', type=float, default=1e-4)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    # 控制执行哪些步骤
    parser.add_argument('--skip_data_prep', action='store_true',
                        help='Skip data preparation if already done')
    parser.add_argument('--skip_rqvae', action='store_true',
                        help='Skip RQVAE training if already done')
    parser.add_argument('--skip_tiger', action='store_true',
                        help='Skip TIGER training')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TIGER 训练流程 - Tenrec 数据集")
    print("="*60)
    
    abs_csv_path = os.path.abspath(args.csv_path)
    abs_data_dir = os.path.abspath(args.data_dir)
    
    # 步骤 1: 数据预处理
    if not args.skip_data_prep:
        cmd = f"cd {os.path.dirname(os.path.abspath(__file__))}/data && python prepare_tenrec.py " \
              f"--input '{abs_csv_path}' " \
              f"--output '{abs_data_dir}' " \
              f"--min_interactions {args.min_interactions}"
        run_command(cmd, "数据预处理")
    else:
        print("\n跳过数据预处理")
    
    # 步骤 2: 训练 RQVAE
    if not args.skip_rqvae:
        rqvae_data_path = os.path.join(abs_data_dir, 'item_emb.parquet')
        cmd = f"cd {os.path.dirname(os.path.abspath(__file__))}/rqvae && python train_tenrec_rqvae.py " \
              f"--data_path '{rqvae_data_path}' " \
              f"--output_dir '{abs_data_dir}' " \
              f"--ckpt_dir '{os.path.abspath(args.rqvae_ckpt_dir)}' " \
              f"--dataset_name Tenrec " \
              f"--epochs {args.rqvae_epochs} " \
              f"--batch_size {args.rqvae_batch_size} " \
              f"--lr {args.rqvae_lr} " \
              f"--device {args.device}"
        run_command(cmd, "RQVAE 训练")
    else:
        print("\n跳过 RQVAE 训练")
    
    # 步骤 3: 训练 TIGER
    if not args.skip_tiger:
        code_path = os.path.join(abs_data_dir, 'Tenrec_t5_rqvae.npy')
        tiger_ckpt_dir = os.path.dirname(os.path.abspath(args.tiger_ckpt_path))
        os.makedirs(tiger_ckpt_dir, exist_ok=True)
        
        cmd = f"cd {os.path.dirname(os.path.abspath(__file__))}/model && python main.py " \
              f"--dataset_path '{abs_data_dir}' " \
              f"--code_path '{code_path}' " \
              f"--save_path '{os.path.abspath(args.tiger_ckpt_path)}' " \
              f"--num_epochs {args.tiger_epochs} " \
              f"--batch_size {args.tiger_batch_size} " \
              f"--lr {args.tiger_lr} " \
              f"--max_len {args.max_len} " \
              f"--d_model {args.d_model} " \
              f"--num_layers {args.num_layers} " \
              f"--num_decoder_layers {args.num_decoder_layers} " \
              f"--device {args.device}"
        run_command(cmd, "TIGER 训练")
    else:
        print("\n跳过 TIGER 训练")
    
    print("\n" + "="*60)
    print("全部流程完成!")
    if not args.skip_tiger:
        print(f"模型已保存到: {args.tiger_ckpt_path}")
    print("="*60)


if __name__ == "__main__":
    main()
