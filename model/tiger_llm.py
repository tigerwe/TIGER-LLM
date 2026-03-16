"""
TIGER-LLM: Decoder-only LLM + Prompt 形式
使用 GPT2 架构替代 T5
"""
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


class TIGER_LLM(nn.Module):
    """
    TIGER with Decoder-only LLM (GPT2 architecture)
    """
    def __init__(self, config: Dict[str, Any]):
        super(TIGER_LLM, self).__init__()
        
        gpt2_config = GPT2Config(
            n_layer=config['num_layers'],
            n_head=config['num_heads'],
            n_embd=config['d_model'],
            n_positions=config['max_seq_len'],
            n_ctx=config['max_seq_len'],
            vocab_size=config['vocab_size'],
            pad_token_id=config['pad_token_id'],
            bos_token_id=config['bos_token_id'],
            eos_token_id=config['eos_token_id'],
            activation_function='gelu',
        )
        
        self.model = GPT2LMHeadModel(gpt2_config)
        self.prompt_template = config.get('prompt_template', 
            "User's past: {history} Recommendation:")
        
    @property
    def n_parameters(self):
        """Calculates the number of trainable parameters."""
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.model.get_input_embeddings().parameters())
        return (
            f'#Embedding parameters: {emb_params}\n'
            f'#Non-embedding parameters: {total_params - emb_params}\n'
            f'#Total trainable parameters: {total_params}\n'
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None):
        """
        Forward pass for causal language modeling
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len), -100 for positions not to compute loss
        
        Returns:
            loss, logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits
    
    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                 max_length: int = 10, num_beams: int = 20, **kwargs):
        """
        Generate recommendations using beam search
        
        Args:
            input_ids: Input tensor with prompt
            attention_mask: Attention mask
            max_length: Maximum length of generated sequence
            num_beams: Number of beams for beam search
        
        Returns:
            Generated token sequences
        """
        # 确保 pad_token_id 设置正确，防止生成非法 token
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
            early_stopping=True,
            # 限制生成的 token 范围
            forced_eos_token_id=self.model.config.eos_token_id,
            **kwargs
        )


class PromptGenRecDataset(torch.utils.data.Dataset):
    """
    Dataset for Prompt-based Generative Recommendation
    将历史交互编码为 Prompt 格式
    """
    def __init__(self, dataset_path, code_path, mode, max_len, 
                 prompt_template="User's past: {history} Next:", 
                 codebook_size=256, num_codebooks=5,
                 PAD_TOKEN=0, BOS_TOKEN=1, EOS_TOKEN=2, SEP_TOKEN=3):
        """
        Args:
            dataset_path: Path to parquet file
            code_path: Path to code mapping
            mode: 'train' or 'evaluation'
            max_len: Maximum sequence length
            prompt_template: Template for prompt
            codebook_size: Size of each codebook
            num_codebooks: Number of codebooks (dimensions)
            PAD_TOKEN: Padding token ID
            BOS_TOKEN: Beginning of sequence token
            EOS_TOKEN: End of sequence token
            SEP_TOKEN: Separator token
        """
        self.dataset_path = dataset_path
        self.code_path = code_path
        self.mode = mode
        self.max_len = max_len
        self.prompt_template = prompt_template
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.PAD_TOKEN = PAD_TOKEN
        self.BOS_TOKEN = BOS_TOKEN
        self.EOS_TOKEN = EOS_TOKEN
        self.SEP_TOKEN = SEP_TOKEN
        
        # 加载 code 数据
        self.codes_data = np.load(code_path, allow_pickle=True)
        self.num_items = len(self.codes_data)
        
        # 加载并处理数据
        self.data = self._prepare_data()
    
    def _get_item_code(self, item_id):
        """获取 item 的 code，带偏移"""
        # item_id 可能很大（原始ID），需要映射到 code 索引
        # 假设 code 索引 = (item_id - 1) % num_items
        code_idx = (item_id - 1) % self.num_items
        code = self.codes_data[code_idx]
        
        # 添加偏移量：每个 codebook 的 token 范围不重叠
        # +10 为特殊 token 预留空间
        offsets = [int(c) + i * self.codebook_size + 10 
                  for i, c in enumerate(code)]
        return offsets
    
    def _prepare_data(self):
        """Process dataset into prompt format"""
        import pandas as pd
        
        df = pd.read_parquet(self.dataset_path)
        processed_data = []
        
        for _, row in df.iterrows():
            history = list(row['history'])
            target = row['target']
            
            # 获取 code
            history_codes = [self._get_item_code(h) for h in history]
            target_code = self._get_item_code(target)
            
            if self.mode == 'train':
                # 训练时使用 sliding window
                for i in range(1, len(history)):
                    prompt_history = history_codes[:i]
                    prompt_target = history_codes[i]
                    processed_data.append({
                        'history': prompt_history,
                        'target': prompt_target
                    })
            else:
                # 评估模式
                processed_data.append({
                    'history': history_codes,
                    'target': target_code
                })
        
        return processed_data
    
    def _codes_to_tokens(self, codes):
        """将多维 code 展开为 token 序列"""
        tokens = []
        for code in codes:
            tokens.extend(code)
            tokens.append(self.SEP_TOKEN)  # 每个 item 之间用 SEP 分隔
        return tokens[:-1] if tokens else tokens  # 移除最后一个 SEP
    
    def __getitem__(self, index):
        """Get a single data item in prompt format"""
        item = self.data[index]
        
        # 构建序列
        history_tokens = self._codes_to_tokens(item['history'])
        target_tokens = item['target'] if isinstance(item['target'], list) else list(item['target'])
        
        # 输入序列: BOS + history + SEP
        input_tokens = [self.BOS_TOKEN] + history_tokens + [self.SEP_TOKEN]
        
        # 完整序列（用于训练）: BOS + history + SEP + target + EOS
        full_tokens = input_tokens + target_tokens + [self.EOS_TOKEN]
        
        # 截断
        if len(full_tokens) > self.max_len:
            # 保留 BOS，截断前面的历史
            excess = len(full_tokens) - self.max_len
            history_tokens = history_tokens[excess:]
            input_tokens = [self.BOS_TOKEN] + history_tokens + [self.SEP_TOKEN]
            full_tokens = input_tokens + target_tokens + [self.EOS_TOKEN]
        
        # 创建 attention mask
        attention_mask = [1] * len(full_tokens)
        
        # 创建 labels: -100 表示不计算 loss（instruction tuning）
        # 只预测 target 部分
        labels = [-100] * len(input_tokens) + target_tokens + [self.EOS_TOKEN]
        
        # 填充
        padding_length = self.max_len - len(full_tokens)
        if padding_length > 0:
            full_tokens = full_tokens + [self.PAD_TOKEN] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length
        else:
            full_tokens = full_tokens[:self.max_len]
            attention_mask = attention_mask[:self.max_len]
            labels = labels[:self.max_len]
        
        return {
            'input_ids': torch.tensor(full_tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'target': torch.tensor(target_tokens, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.data)


def collate_fn(batch, pad_token=0):
    """
    Collate function for DataLoader
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'target': targets
    }


if __name__ == "__main__":
    # Test
    config = {
        'num_layers': 4,
        'num_heads': 8,
        'd_model': 256,
        'max_seq_len': 512,
        'vocab_size': 1281 + 10,
        'pad_token_id': 0,
        'bos_token_id': 1,
        'eos_token_id': 2,
        'dropout_rate': 0.1,
    }
    
    model = TIGER_LLM(config)
    print(model.n_parameters)
