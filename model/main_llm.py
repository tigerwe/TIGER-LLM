"""
TIGER-LLM 训练脚本
使用 Decoder-only LLM + Prompt 形式
"""
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from tiger_llm import TIGER_LLM, PromptGenRecDataset, collate_fn


def calculate_pos_index(preds, labels, maxk=20):
    """
    Calculate the position index of the ground truth items.
    
    Args:
      preds: The predicted token sequences, of shape (batch_size, maxk, code_dim).
      labels: The ground truth token sequences, of shape (batch_size, code_dim).
    
    Returns:
      A boolean tensor of shape (batch_size, maxk) indicating whether the
      prediction at each position is correct.
    """
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


def train_epoch(model, train_loader, optimizer, device, scheduler=None):
    """
    训练一个 epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, eval_loader, topk_list, beam_size, device, code_dim=5):
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
            labels = batch['target'].to(device)  # (batch_size, code_dim)
            
            # 生成推荐
            # 先找到 prompt 结束位置（SEP token 之后）
            # 这里简化处理，假设输入已经构造好了
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=code_dim,
                num_beams=beam_size
            )
            
            # 只取生成的部分（去掉输入 prompt）
            # 生成的 tokens 长度应该是 code_dim
            batch_size = input_ids.shape[0]
            generated = outputs[:, input_ids.shape[1]:]  # (batch_size * beam_size, gen_len)
            
            # reshape 为 (batch_size, beam_size, gen_len)
            gen_len = generated.shape[1]
            generated = generated.view(batch_size, beam_size, gen_len)
            
            # 计算命中情况
            pos_index = calculate_pos_index(generated, labels, maxk=beam_size)
            
            for k in topk_list:
                recall = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()
                recalls['Recall@' + str(k)].append(recall)
                ndcgs['NDCG@' + str(k)].append(ndcg)
    
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="TIGER-LLM configuration")
    
    # 模型参数
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    
    # 数据参数
    parser.add_argument('--dataset_path', type=str, default='../data/Tenrec', help='Dataset path')
    parser.add_argument('--code_path', type=str, default='../data/Tenrec/Tenrec_t5_rqvae.npy', help='Code path')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum history length')
    parser.add_argument('--prompt_template', type=str, default="User's past: {history} Next:", help='Prompt template')
    
    # Token IDs (需要预留 0-9 给特殊 token)
    parser.add_argument('--pad_token_id', type=int, default=0)
    parser.add_argument('--bos_token_id', type=int, default=1)
    parser.add_argument('--eos_token_id', type=int, default=2)
    parser.add_argument('--sep_token_id', type=int, default=3)
    parser.add_argument('--vocab_size', type=int, default=1300, help='Vocabulary size (1281 + special tokens)')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--log_path', type=str, default='./logs/tiger_llm.log', help='Log path')
    parser.add_argument('--save_path', type=str, default='./ckpt/tiger_llm.pth', help='Save path')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--topk_list', nargs='+', type=int, default=[5, 10, 20], help='Top-k values')
    parser.add_argument('--beam_size', type=int, default=20, help='Beam size')
    parser.add_argument('--code_dim', type=int, default=5, help='Code dimension')
    
    config = vars(parser.parse_args())
    
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
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # 创建数据集
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
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Valid dataset size: {len(valid_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")
    
    # 创建模型
    model = TIGER_LLM(config)
    print(model.n_parameters)
    logging.info(model.n_parameters)
    
    model.to(device)
    
    # 优化器和学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # 训练循环
    best_ndcg = 0.0
    early_stop_counter = 0
    
    for epoch in range(config['num_epochs']):
        logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        logging.info(f"Training loss: {train_loss:.4f}")
        print(f"Epoch {epoch + 1}: Training loss = {train_loss:.4f}")
        
        # 评估
        avg_recalls, avg_ndcgs = evaluate(
            model, valid_loader, config['topk_list'], 
            config['beam_size'], device, config['code_dim']
        )
        
        logging.info(f"Validation - Recalls: {avg_recalls}")
        logging.info(f"Validation - NDCGs: {avg_ndcgs}")
        print(f"Validation - Recalls: {avg_recalls}")
        print(f"Validation - NDCGs: {avg_ndcgs}")
        
        # 保存最佳模型
        if avg_ndcgs['NDCG@20'] > best_ndcg:
            best_ndcg = avg_ndcgs['NDCG@20']
            early_stop_counter = 0
            
            # 在测试集上评估
            test_recalls, test_ndcgs = evaluate(
                model, test_loader, config['topk_list'],
                config['beam_size'], device, config['code_dim']
            )
            logging.info(f"Test - Recalls: {test_recalls}")
            logging.info(f"Test - NDCGs: {test_ndcgs}")
            print(f"Test - Recalls: {test_recalls}")
            print(f"Test - NDCGs: {test_ndcgs}")
            
            torch.save(model.state_dict(), config['save_path'])
            logging.info(f"Best model saved to {config['save_path']}")
        else:
            early_stop_counter += 1
            logging.info(f"No improvement. Early stop counter: {early_stop_counter}")
            
            if early_stop_counter >= config['early_stop']:
                logging.info("Early stopping triggered.")
                break
    
    print("Training completed!")


if __name__ == "__main__":
    main()
