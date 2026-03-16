"""
快速测试 TIGER-LLM 训练流程
"""
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tiger_llm import TIGER_LLM, PromptGenRecDataset, collate_fn
import os


def test_model():
    print("=" * 60)
    print("TIGER-LLM 快速测试")
    print("=" * 60)
    
    # 配置
    config = {
        'num_layers': 2,
        'num_heads': 4,
        'd_model': 128,
        'max_seq_len': 200,
        'vocab_size': 1300,
        'pad_token_id': 0,
        'bos_token_id': 1,
        'eos_token_id': 2,
        'dropout_rate': 0.1,
    }
    
    device = torch.device('cpu')
    
    # 1. 测试模型创建
    print("\n[1/5] 创建模型...")
    model = TIGER_LLM(config)
    print(model.n_parameters)
    model.to(device)
    print("✓ 模型创建成功")
    
    # 2. 测试数据集
    print("\n[2/5] 加载数据集...")
    data_dir = '../data/Tenrec'
    
    train_dataset = PromptGenRecDataset(
        dataset_path=os.path.join(data_dir, 'train.parquet'),
        code_path=os.path.join(data_dir, 'Tenrec_t5_rqvae.npy'),
        mode='train',
        max_len=config['max_seq_len'],
        PAD_TOKEN=0, BOS_TOKEN=1, EOS_TOKEN=2, SEP_TOKEN=3
    )
    
    print(f"  训练集大小: {len(train_dataset)}")
    
    # 取前 100 个样本用于快速测试
    train_subset = torch.utils.data.Subset(train_dataset, range(min(100, len(train_dataset))))
    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    print(f"  使用子集大小: {len(train_subset)}")
    print("✓ 数据集加载成功")
    
    # 3. 测试前向传播
    print("\n[3/5] 测试前向传播...")
    batch = next(iter(train_loader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  labels shape: {labels.shape}")
    
    with torch.no_grad():
        loss, logits = model(input_ids, attention_mask, labels)
    
    print(f"  loss: {loss.item():.4f}")
    print(f"  logits shape: {logits.shape}")
    print("✓ 前向传播成功")
    
    # 4. 测试训练步骤
    print("\n[4/5] 测试训练步骤...")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        loss, _ = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        
        if i == 0:
            print(f"  初始 loss: {loss.item():.4f}")
    
    print(f"  训练后 loss: {loss.item():.4f}")
    print("✓ 训练步骤成功")
    
    # 5. 测试生成
    print("\n[5/5] 测试生成...")
    model.eval()
    
    # 构造输入 prompt
    test_input = batch['input_ids'][:1].to(device)
    test_mask = batch['attention_mask'][:1].to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            test_input,
            test_mask,
            max_length=5,
            num_beams=5
        )
    
    print(f"  输入 shape: {test_input.shape}")
    print(f"  输出 shape: {outputs.shape}")
    print(f"  生成的 tokens: {outputs[0].tolist()[-10:]}")
    print("✓ 生成成功")
    
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
