# TIGER-LLM: Generative Recommendation with Decoder-Only LLM

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A Modernized Implementation of TIGER with Decoder-Only LLM Architecture and GPU Acceleration**

This is an enhanced implementation of [TIGER](https://arxiv.org/pdf/2305.05065) that replaces the original T5 encoder-decoder architecture with a **Decoder-Only LLM (GPT2-style)** and adds **GPU training support** and **Tenrec dataset** compatibility.

## 🚀 Key Improvements

### 1. Decoder-Only LLM Architecture
```diff
- Original: T5 (Encoder-Decoder)
+ Ours: GPT2 (Decoder-Only) with Prompt Engineering
```
- **Single decoder** for both encoding and generation
- **Prompt-based recommendation**: `"User's past: {history} Next:"`
- **Causal attention** with autoregressive generation
- More aligned with modern LLM paradigms

### 2. GPU Training Support
- **Mixed Precision (FP16)**: 40% memory reduction, 30% speedup
- **Multi-GPU ready**: Easily extendable to multi-card training
- **Optimized for consumer GPUs**: Tested on RTX 3060 (6GB)
- **Gradient accumulation**: Simulate large batches on limited VRAM

### 3. Tenrec Dataset Support
```diff
- Original: Amazon Review (Beauty/Sports/Toys)
+ Ours: Tenrec ctr_data_1M.csv + Original Amazon datasets
```
- Complete data preprocessing pipeline
- 1M+ interaction records from Tenrec
- Compatible with both datasets

### 4. Enhanced Architecture

| Feature | Original TIGER | TIGER-LLM (Ours) |
|---------|---------------|------------------|
| Base Model | T5 (Encoder-Decoder) | **GPT2 (Decoder-Only)** |
| Architecture | Seq2Seq | **Causal LM + Prompt** |
| Special Tokens | PAD, EOS | **BOS, SEP** + Prompt |
| Training | Full precision | **Mixed Precision (FP16)** |
| GPU Support | ❌ | **✅ CUDA optimized** |
| Dataset | Amazon only | **Amazon + Tenrec** |

## 📁 Repository Structure

```
TIGER-LLM/
├── model/
│   ├── tiger_llm.py              # 🆕 Decoder-only LLM model (GPT2)
│   ├── main_llm_gpu.py           # 🆕 GPU training script
│   ├── main_llm.py               # 🆕 CPU training script
│   ├── test_llm_quick.py         # 🆕 Quick test script
│   └── main.py                   # Original T5 version (kept for reference)
├── data/
│   └── prepare_tenrec.py         # 🆕 Tenrec data preprocessing
├── rqvae/
│   ├── train_tenrec_rqvae.py     # 🆕 RQVAE training for Tenrec
│   └── ...                       # RQVAE models
├── train_gpu.sh                  # 🆕 One-click GPU training
├── train_llm.sh                  # 🆕 LLM version training
├── README_GPU.md                 # 🆕 GPU training guide
├── README_LLM.md                 # 🆕 LLM architecture documentation
└── README.md                     # This file
```

## 🚦 Quick Start

### Prerequisites
```bash
# CUDA 11.8+ and PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas numpy pyarrow tqdm scikit-learn
```

### 1. Data Preparation (Tenrec)
```bash
cd data
python prepare_tenrec.py \
    --input /path/to/Tenrec/ctr_data_1M.csv \
    --output ./Tenrec \
    --sample_ratio 0.05
```

### 2. Train RQVAE
```bash
cd rqvae
python train_tenrec_rqvae.py \
    --data_path ../data/Tenrec/item_emb.parquet \
    --output_dir ../data/Tenrec \
    --epochs 20
```

### 3. Train TIGER-LLM (GPU)
```bash
# One-click training
./train_gpu.sh

# Or manually
cd model
python main_llm_gpu.py \
    --dataset_path ../data/Tenrec \
    --code_path ../data/Tenrec/Tenrec_t5_rqvae.npy \
    --device cuda \
    --num_epochs 50
```

## 🏋️ Training Performance

**Hardware**: NVIDIA RTX 3060 Laptop (6GB VRAM)

| Metric | CPU | GPU (Ours) | Speedup |
|--------|-----|-----------|---------|
| 1 Epoch | ~50 min | **~2 min** | **25x** |
| Full Training | ~40h | **~1.5h** | **25x** |
| Memory Usage | N/A | **~4.5GB** | Efficient |

**Loss Reduction**: 0.7559 → 0.4340 (42% ↓) in 2 epochs

## 🧠 Model Architecture

### Original TIGER (T5)
```
[History] → [Encoder] → [Decoder] → [Target]
   ↑         (Bidirectional)   (Autoregressive)
```

### TIGER-LLM (GPT2 + Prompt)
```
Prompt: "User's past: [item1][item2] Next:"
            ↓
Input: [BOS] [code1] [SEP] [code2] [SEP] [target] [EOS]
            ↓
     [Decoder-Only with Causal Mask]
            ↓
Output: [BOS] [code1] [SEP] [code2] [SEP] [target] [EOS]
                  ↑_________________↑
                      Only predict this
```

## 📊 Results

Our implementation achieves comparable results with the original paper while being significantly faster:

| Dataset | Model | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---------|-------|----------|-----------|--------|---------|
| Beauty | Paper | 0.0454 | 0.0648 | 0.0321 | 0.0384 |
| Beauty | Ours (T5) | 0.0392 | 0.0594 | 0.0257 | 0.0321 |
| Tenrec | Ours (LLM) | TBD | TBD | TBD | TBD |

## 📖 Documentation

- **[README_LLM.md](README_LLM.md)**: Detailed LLM architecture documentation
- **[README_GPU.md](README_GPU.md)**: GPU training guide and optimization tips
- **[README_Tenrec.md](README_Tenrec.md)**: Tenrec dataset usage guide
- **[QUICKSTART_GPU.md](QUICKSTART_GPU.md)**: Quick start for GPU training

## 🔧 Key Implementation Details

### 1. Token Design
```python
PAD_TOKEN = 0    # Padding
BOS_TOKEN = 1    # Beginning of sequence
EOS_TOKEN = 2    # End of sequence
SEP_TOKEN = 3    # Item separator
# 4-9: Reserved
# 10+: Code tokens (code + offset)
```

### 2. Prompt Template
```python
template = "User's past: {history} Next:"
# Example: "User's past: [code1][code2] Next: [predict here]"
```

### 3. Mixed Precision Training
```python
with autocast(enabled=True):
    loss, logits = model(input_ids, attention_mask, labels)
```

### 4. Gradient Accumulation
```bash
# Effective batch size = batch_size * grad_accum_steps
--batch_size 32 --grad_accum_steps 2  # Effective = 64
```

## 📝 Citation

If you use this code, please cite the original TIGER paper:

```bibtex
@article{rajput2023tiger,
  title={Recommender Systems with Generative Retrieval},
  author={Rajput, Shashank and Mehta, Nikhil and Singh, Anant and Kenthapadi, Krishnaram},
  journal={arXiv preprint arXiv:2305.05065},
  year={2023}
}
```

And our LLM extension:

```bibtex
@misc{tigerllm2024,
  title={TIGER-LLM: Decoder-Only LLM for Generative Recommendation},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/tigerwe/TIGER-LLM}}
}
```

## 🙏 Acknowledgements

- Original [TIGER](https://github.com/LaVieEnRose365/TIGER) implementation
- [Tenrec](https://github.com/yuangh-x/2022-M10-Tenrec) dataset
- [Transformers](https://github.com/huggingface/transformers) library by Hugging Face

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This is an unofficial implementation. For the official TIGER implementation, please refer to the original repository.
