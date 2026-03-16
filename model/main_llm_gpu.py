"""
TIGER-LLM GPU 训练脚本 (优化版)
支持混合精度训练和梯度累积
"""
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import argparse
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from tiger_llm import TIGER_LLM, PromptGenRecDataset, collate_fn


def calculate_pos_index(preds, labels, maxk=20):
    """Calculate the position index of the ground truth items."""
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    
    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
        cur_label = labels[i].tolist()
        for j in range(maxk):
            cur_pred = preds[i, j].tolist()
            if cur_pred == cur_label:
                pos_index[i, j] = True
                break
    return pos_index


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()


def train_epoch(model, train_loader, optimizer, device, scheduler=None, 
                use_amp=True, grad_accum_steps=1):
    """
    训练一个 epoch (支持混合精度)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    scaler = GradScaler() if use_amp else None
    
    progress_bar = tqdm(train_loader, desc="Training")
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 混合精度前向
        with autocast(enabled=use_amp):
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            # 梯度累积：损失除以累积步数
            loss = loss / grad_accum_steps
        
        # 混合精度反向
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 梯度累积更新
        if (step + 1) % grad_accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        
        total_loss += loss.item() * grad_accum_steps
        num_batches += 1
        
        # 更新进度条
        progress_bar.set_postfix({'loss': f'{loss.item() * grad_accum_steps:.4f}'})
    
    return total_loss / num_batches


def evaluate(model, eval_loader, topk_list, beam_size, device, code_dim=5, use_amp=True, sep_token_id=3):
    """
    评估模型性能
    """
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)
            
            # 找到 SEP token 位置，只保留 prompt 部分用于生成
            # prompt: [BOS] + history + [SEP]
            prompt_input_ids = []
            prompt_attention_mask = []
            for i in range(input_ids.shape[0]):
                # 找到 SEP token (id=3)
                sep_positions = (input_ids[i] == sep_token_id).nonzero(as_tuple=True)[0]
                if len(sep_positions) > 0:
                    sep_pos = sep_positions[0].item() + 1  # 包含 SEP
                else:
                    sep_pos = input_ids.shape[1]  # 如果没找到，用全部
                
                prompt_input_ids.append(input_ids[i, :sep_pos])
                prompt_attention_mask.append(attention_mask[i, :sep_pos])
            
            # 填充到相同长度
            max_prompt_len = max(x.shape[0] for x in prompt_input_ids)
            padded_input_ids = torch.full((len(prompt_input_ids), max_prompt_len), 
                                         0, dtype=torch.long, device=device)
            padded_attention_mask = torch.zeros((len(prompt_input_ids), max_prompt_len), 
                                               dtype=torch.long, device=device)
            
            for i, (ids, mask) in enumerate(zip(prompt_input_ids, prompt_attention_mask)):
                padded_input_ids[i, :len(ids)] = ids
                padded_attention_mask[i, :len(mask)] = mask
            
            with autocast(enabled=use_amp):
                outputs = model.generate(
                    input_ids=padded_input_ids,
                    attention_mask=padded_attention_mask,
                    max_length=code_dim,
                    num_beams=beam_size
                )
            
            # 处理输出
            batch_size = input_ids.shape[0]
            input_len = input_ids.shape[1]
            
            # 确保输出长度正确
            if outputs.shape[1] > input_len:
                generated = outputs[:, input_len:]
            else:
                # 如果输出太短，用 PAD 填充
                gen_len = max(code_dim, 1)
                generated = torch.full((outputs.shape[0], gen_len), 0, 
                                      dtype=torch.long, device=device)
            
            # reshape 为 (batch_size, beam_size, gen_len)
            gen_len = generated.shape[1]
            try:
                generated = generated.view(batch_size, beam_size, gen_len)
            except RuntimeError:
                # 如果 reshape 失败，调整形状
                total_beams = generated.shape[0]
                actual_beams = total_beams // batch_size
                if actual_beams > 0:
                    generated = generated[:batch_size * actual_beams]
                    generated = generated.view(batch_size, actual_beams, gen_len)
                else:
                    continue
            
            # 计算命中情况
            pos_index = calculate_pos_index(generated, labels, maxk=min(beam_size, generated.shape[1]))
            
            for k in topk_list:
                if k <= pos_index.shape[1]:
                    recall = recall_at_k(pos_index, k).mean().item()
                    ndcg = ndcg_at_k(pos_index, k).mean().item()
                    recalls['Recall@' + str(k)].append(recall)
                    ndcgs['NDCG@' + str(k)].append(ndcg)
    
    avg_recalls = {k: sum(v) / len(v) if v else 0.0 for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) if v else 0.0 for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="TIGER-LLM GPU Training")
    
    # 模型参数
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--grad_accum_steps', type=int, default=2, help='梯度累积步数')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--use_amp', action='store_true', default=True, help='使用混合精度')
    
    # 数据参数
    parser.add_argument('--dataset_path', type=str, default='../data/Tenrec')
    parser.add_argument('--code_path', type=str, default='../data/Tenrec/Tenrec_t5_rqvae.npy')
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--prompt_template', type=str, default="User's past: {history} Next:")
    
    # Token IDs
    parser.add_argument('--pad_token_id', type=int, default=0)
    parser.add_argument('--bos_token_id', type=int, default=1)
    parser.add_argument('--eos_token_id', type=int, default=2)
    parser.add_argument('--sep_token_id', type=int, default=3)
    parser.add_argument('--vocab_size', type=int, default=1300)
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_path', type=str, default='./logs/tiger_llm_gpu.log')
    parser.add_argument('--save_path', type=str, default='./ckpt/tiger_llm_gpu.pth')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--topk_list', nargs='+', type=int, default=[5, 10, 20])
    parser.add_argument('--beam_size', type=int, default=20)
    parser.add_argument('--code_dim', type=int, default=5)
    
    config = vars(parser.parse_args())
    
    # 检查 GPU
    if not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU 训练")
        config['device'] = 'cpu'
        config['use_amp'] = False
    else:
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 设置日志
    os.makedirs(os.path.dirname(config['log_path']), exist_ok=True)
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    
    logging.basicConfig(
        filename=config['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Configuration: {config}")
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设备
    device = torch.device(config['device'])
    
    # 创建数据集
    print("加载数据集...")
    train_dataset = PromptGenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], 'train.parquet'),
        code_path=config['code_path'],
        mode='train',
        max_len=config['max_seq_len'],
        prompt_template=config['prompt_template'],
        PAD_TOKEN=config['pad_token_id'],
        BOS_TOKEN=config['bos_token_id'],
        EOS_TOKEN=config['eos_token_id'],
        SEP_TOKEN=config['sep_token_id']
    )
    
    valid_dataset = PromptGenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], 'valid.parquet'),
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_seq_len'],
        prompt_template=config['prompt_template'],
        PAD_TOKEN=config['pad_token_id'],
        BOS_TOKEN=config['bos_token_id'],
        EOS_TOKEN=config['eos_token_id'],
        SEP_TOKEN=config['sep_token_id']
    )
    
    test_dataset = PromptGenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], 'test.parquet'),
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_seq_len'],
        prompt_template=config['prompt_template'],
        PAD_TOKEN=config['pad_token_id'],
        BOS_TOKEN=config['bos_token_id'],
        EOS_TOKEN=config['eos_token_id'],
        SEP_TOKEN=config['sep_token_id']
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(valid_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    logging.info(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    
    # 创建模型
    print("\n创建模型...")
    model = TIGER_LLM(config)
    print(model.n_parameters)
    logging.info(model.n_parameters)
    
    model.to(device)
    
    # 优化器和学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    total_steps = len(train_loader) * config['num_epochs'] // config['grad_accum_steps']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        total_steps=total_steps,
        pct_start=0.1,  # warmup 10%
        anneal_strategy='cos'
    )
    
    # 训练循环
    print(f"\n开始训练 {config['num_epochs']} epochs...")
    print(f"Batch size: {config['batch_size']}, 梯度累积: {config['grad_accum_steps']}")
    print(f"等效 batch size: {config['batch_size'] * config['grad_accum_steps']}")
    
    best_ndcg = 0.0
    early_stop_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, device, scheduler,
            use_amp=config['use_amp'],
            grad_accum_steps=config['grad_accum_steps']
        )
        
        print(f"训练 Loss: {train_loss:.4f}")
        logging.info(f"Train Loss: {train_loss:.4f}")
        
        # 评估
        avg_recalls, avg_ndcgs = evaluate(
            model, valid_loader, config['topk_list'], 
            config['beam_size'], device, config['code_dim'],
            use_amp=config['use_amp'],
            sep_token_id=config['sep_token_id']
        )
        
        print(f"验证 - Recall@5: {avg_recalls['Recall@5']:.4f}, NDCG@5: {avg_ndcgs['NDCG@5']:.4f}")
        print(f"验证 - Recall@10: {avg_recalls['Recall@10']:.4f}, NDCG@10: {avg_ndcgs['NDCG@10']:.4f}")
        print(f"验证 - Recall@20: {avg_recalls['Recall@20']:.4f}, NDCG@20: {avg_ndcgs['NDCG@20']:.4f}")
        logging.info(f"Valid - Recalls: {avg_recalls}, NDCGs: {avg_ndcgs}")
        
        # 保存最佳模型
        if avg_ndcgs['NDCG@20'] > best_ndcg:
            best_ndcg = avg_ndcgs['NDCG@20']
            early_stop_counter = 0
            
            # 测试集评估
            test_recalls, test_ndcgs = evaluate(
                model, test_loader, config['topk_list'],
                config['beam_size'], device, config['code_dim'],
                use_amp=config['use_amp'],
                sep_token_id=config['sep_token_id']
            )
            
            print(f"测试 - Recall@5: {test_recalls['Recall@5']:.4f}, NDCG@5: {test_ndcgs['NDCG@5']:.4f}")
            print(f"测试 - Recall@20: {test_recalls['Recall@20']:.4f}, NDCG@20: {test_ndcgs['NDCG@20']:.4f}")
            logging.info(f"Test - Recalls: {test_recalls}, NDCGs: {test_ndcgs}")
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ndcg': best_ndcg,
                'config': config
            }, config['save_path'])
            print(f"✓ 最佳模型已保存 (NDCG@20: {best_ndcg:.4f})")
        else:
            early_stop_counter += 1
            print(f"早停计数: {early_stop_counter}/{config['early_stop']}")
            
            if early_stop_counter >= config['early_stop']:
                print("早停触发，训练结束")
                break
    
    # 训练结束，保存最终模型（如果没有保存过）
    if not os.path.exists(config['save_path']):
        print("\n保存最终模型...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config
        }, config['save_path'])
        print(f"✓ 最终模型已保存: {config['save_path']}")
    
    print("\n训练完成！")
    print(f"模型路径: {config['save_path']}")


if __name__ == "__main__":
    main()
