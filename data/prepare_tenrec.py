"""
将 Tenrec 的 ctr_data_1M.csv 转换为 TIGER 需要的格式
优化版本：使用分块读取处理大文件
"""
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import json


def process_tenrec_data(csv_path, output_dir, min_interactions=5, sample_ratio=0.1):
    """
    处理 Tenrec CTR 数据，转换为 TIGER 需要的格式
    
    Args:
        csv_path: ctr_data_1M.csv 路径
        output_dir: 输出目录
        min_interactions: 用户最少交互次数
        sample_ratio: 采样比例（0-1），用于减少数据量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在加载数据: {csv_path}")
    print(f"采样比例: {sample_ratio}")
    
    # 使用分块读取处理大文件
    chunksize = 1000000  # 每次读取 100万行
    user_clicks = defaultdict(list)
    
    # 首先统计总行数
    print("统计总行数...")
    total_rows = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, na_values=['\\N', 'NULL', '']):
        total_rows += len(chunk)
    print(f"总行数: {total_rows}")
    
    # 计算采样后的目标行数
    target_rows = int(total_rows * sample_ratio)
    print(f"目标采样行数: {target_rows}")
    
    # 读取并采样数据
    current_rows = 0
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize, na_values=['\\N', 'NULL', ''])):
        print(f"处理第 {i+1} 个块...")
        
        # 在当前块中进行采样
        chunk_sample_size = int(len(chunk) * sample_ratio)
        if chunk_sample_size > 0:
            chunk = chunk.sample(n=chunk_sample_size, random_state=42)
        
        # 过滤掉有缺失值的行
        chunk = chunk.dropna(subset=['user_id', 'item_id', 'click'])
        
        # 只保留点击过的物品
        chunk = chunk[chunk['click'] == 1]
        
        # 收集用户点击序列
        for _, row in chunk.iterrows():
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            user_clicks[user_id].append(item_id)
            current_rows += 1
        
        if current_rows >= target_rows:
            break
    
    print(f"采样后数据量: {current_rows} 行")
    print(f"用户数: {len(user_clicks)}")
    
    # 过滤掉交互过少的用户
    user_sequences = {u: seq for u, seq in user_clicks.items() if len(seq) >= min_interactions}
    
    print(f"过滤后用户数 (>= {min_interactions} 次交互): {len(user_sequences)}")
    
    # 获取所有唯一的物品 ID
    all_items = set()
    for seq in user_sequences.values():
        all_items.update(seq)
    print(f"物品数: {len(all_items)}")
    
    # 划分训练/验证/测试集 (leave-one-out)
    train_data = []
    val_data = []
    test_data = []
    
    for user_id, seq in user_sequences.items():
        if len(seq) >= 3:
            # 训练集: 除最后两个外的所有物品
            train_history = seq[:-2]
            train_target = seq[-2]
            train_data.append({
                'user': user_id,
                'history': train_history,
                'target': train_target
            })
            
            # 验证集: 除最后一个外的所有物品
            val_history = seq[:-1]
            val_target = seq[-1]
            val_data.append({
                'user': user_id,
                'history': val_history,
                'target': val_target
            })
            
            # 测试集: 所有物品
            test_history = seq[:-1]
            test_target = seq[-1]
            test_data.append({
                'user': user_id,
                'history': test_history,
                'target': test_target
            })
    
    # 转换为 DataFrame
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)
    
    print(f"\n训练集: {len(train_df)} 样本")
    print(f"验证集: {len(val_df)} 样本")
    print(f"测试集: {len(test_df)} 样本")
    
    # 保存为 parquet
    train_df.to_parquet(os.path.join(output_dir, 'train.parquet'), index=False)
    val_df.to_parquet(os.path.join(output_dir, 'valid.parquet'), index=False)
    test_df.to_parquet(os.path.join(output_dir, 'test.parquet'), index=False)
    
    print(f"\n数据已保存到: {output_dir}")
    
    # 保存数据集信息
    info = {
        'num_users': len(user_sequences),
        'num_items': len(all_items),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'sample_ratio': sample_ratio,
        'min_interactions': min_interactions
    }
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    return sorted(list(all_items))


def generate_item_embeddings(item_ids, output_path, embedding_dim=768):
    """
    为物品生成随机语义嵌入（简化版本，因为没有 metadata）
    实际应用中可以使用 item_id 的特征或其他信息
    
    Args:
        item_ids: 物品 ID 列表
        output_path: 输出路径
        embedding_dim: 嵌入维度
    """
    print(f"\n生成物品语义嵌入...")
    print(f"物品数: {len(item_ids)}, 嵌入维度: {embedding_dim}")
    
    # 使用固定的随机种子保证可重复性
    embeddings = []
    for item_id in item_ids:
        # 基于 item_id 生成确定性的随机嵌入
        np.random.seed(item_id % 100000)  # 限制种子范围
        emb = np.random.randn(embedding_dim).astype(np.float32)
        # 归一化
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        embeddings.append({
            'ItemID': item_id,
            'embedding': emb.tolist()
        })
    
    emb_df = pd.DataFrame(embeddings)
    emb_df.to_parquet(output_path, index=False)
    
    print(f"物品嵌入已保存到: {output_path}")
    print(f"嵌入形状: ({len(emb_df)}, {embedding_dim})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Tenrec data for TIGER")
    parser.add_argument('--input', type=str, default='../../Tenrec/ctr_data_1M.csv',
                        help='Path to ctr_data_1M.csv')
    parser.add_argument('--output', type=str, default='./Tenrec',
                        help='Output directory')
    parser.add_argument('--min_interactions', type=int, default=5,
                        help='Minimum interactions per user')
    parser.add_argument('--sample_ratio', type=float, default=0.05,
                        help='Sampling ratio to reduce data size (0-1)')
    args = parser.parse_args()
    
    # 处理数据
    item_ids = process_tenrec_data(args.input, args.output, args.min_interactions, args.sample_ratio)
    
    # 生成物品嵌入
    emb_path = os.path.join(args.output, 'item_emb.parquet')
    generate_item_embeddings(item_ids, emb_path)
    
    print("\n数据准备完成!")
