"""
为 Tenrec 数据集训练 RQVAE 模型
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from time import time

from datasets import EmbDataset
from models.rqvae import RQVAE


def check_collision(all_indices_str):
    """检查是否有冲突的 code"""
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str):
    """统计每个 code 出现的次数"""
    from collections import defaultdict
    indices_count = defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    """获取冲突的物品"""
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []
    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


def train_rqvae(args):
    """训练 RQVAE 模型"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    print(f"加载数据: {args.data_path}")
    data = EmbDataset(args.data_path)
    data_loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
    model = model.to(device)
    
    print(f"模型参数:\n{model}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 创建检查点目录
    ckpt_dir = os.path.join(args.ckpt_dir, args.dataset_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    best_loss = float('inf')
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0
        
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            out, rq_loss, indices = model(batch)
            loss, recon_loss = model.compute_loss(out, rq_loss, xs=batch)
            quant_loss = rq_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_quant_loss += quant_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'quant': f'{quant_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(data_loader)
        avg_recon = total_recon_loss / len(data_loader)
        avg_quant = total_quant_loss / len(data_loader)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, Quant={avg_quant:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'args': args,
                'loss': best_loss,
            }, ckpt_path)
            print(f"最佳模型已保存到: {ckpt_path}")
    
    return model, ckpt_dir


def generate_codes(args, ckpt_dir):
    """生成物品离散代码"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载最佳模型
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
    print(f"\n加载模型: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    
    # 加载数据
    data = EmbDataset(args.data_path)
    data_loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # 生成代码
    all_indices = []
    all_indices_str = []
    
    with torch.no_grad():
        for d in tqdm(data_loader, desc="生成代码"):
            d = d.to(device)
            indices = model.get_indices(d, use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            
            for index in indices:
                code = [f"<a_{int(index[0])}>", f"<b_{int(index[1])}>", 
                        f"<c_{int(index[2])}>", f"<d_{int(index[3])}>"]
                all_indices.append(code)
                all_indices_str.append(str(code))
    
    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)
    
    # 尝试解决冲突
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    
    tt = 0
    while tt < 30 and not check_collision(all_indices_str):
        collision_item_groups = get_collision_item(all_indices_str)
        print(f"第 {tt+1} 轮，冲突组数: {len(collision_item_groups)}")
        
        for collision_items in collision_item_groups:
            d = torch.stack([data[i] for i in collision_items]).to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            
            for item, index in zip(collision_items, indices):
                code = [f"<a_{int(index[0])}>", f"<b_{int(index[1])}>",
                        f"<c_{int(index[2])}>", f"<d_{int(index[3])}>"]
                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1
    
    print(f"最终冲突率: {(len(all_indices_str) - len(set(all_indices_str.tolist()))) / len(all_indices_str):.4f}")
    
    # 转换为数字代码
    codes = []
    for indices in all_indices.tolist():
        code = [int(item.split('_')[1].strip('>')) for item in indices]
        codes.append(code)
    
    codes_array = np.array(codes)
    
    # 添加额外维度解决冲突
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))
    
    # 解决剩余冲突
    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    
    if len(duplicates) > 0:
        print(f"解决 {len(duplicates)} 个重复代码...")
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i
    
    # 保存代码
    output_file = os.path.join(args.output_dir, f'{args.dataset_name}_t5_rqvae.npy')
    np.save(output_file, codes_array)
    print(f"\n代码已保存到: {output_file}")
    print(f"代码形状: {codes_array.shape}")
    print(f"前5个代码:\n{codes_array[:5]}")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RQVAE for Tenrec")
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='../data/Tenrec/item_emb.parquet')
    parser.add_argument('--output_dir', type=str, default='../data/Tenrec')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt')
    parser.add_argument('--dataset_name', type=str, default='Tenrec')
    
    # 模型参数
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256, 256])
    parser.add_argument('--e_dim', type=int, default=64)
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128])
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--bn', action='store_true', default=True)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    # 损失参数
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    
    # K-means 参数
    parser.add_argument('--kmeans_init', action='store_true', default=True)
    parser.add_argument('--kmeans_iters', type=int, default=100)
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[None, None, None, None])
    parser.add_argument('--sk_iters', type=int, default=100)
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("训练 RQVAE 模型")
    print("=" * 50)
    
    # 训练模型
    model, ckpt_dir = train_rqvae(args)
    
    # 生成代码
    print("\n" + "=" * 50)
    print("生成离散代码")
    print("=" * 50)
    code_path = generate_codes(args, ckpt_dir)
    
    print("\n完成!")
