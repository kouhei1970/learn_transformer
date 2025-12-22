# learn_transformer

Transformerを理解するために自作するプロジェクトです。

## 概要

このリポジトリは、Transformerアーキテクチャを深く理解するために、ゼロから実装していくプロジェクトです。

## 目的

- Transformerの仕組みを理解する
- Attention機構の実装を学ぶ
- 自然言語処理の基礎を身につける

## プロジェクト構成

```
learn_transformer/
├── src/                    # ソースコード
│   ├── attention.py        # Self-Attention & Multi-Head Attention実装
│   ├── position_encoding.py # Position Encoding実装
│   ├── feed_forward.py     # Feed Forward Network実装
│   ├── encoder.py          # Encoder実装
│   ├── decoder.py          # Decoder実装
│   └── transformer.py      # 完全なTransformer実装
├── notebooks/              # Jupyter Notebook
│   ├── 01_self_attention_demo.ipynb      # Self-Attentionのデモ
│   ├── 02_multi_head_attention_demo.ipynb # Multi-Head Attentionのデモ
│   ├── 03_position_encoding_demo.ipynb   # Position Encodingのデモ
│   ├── 04_feed_forward_demo.ipynb        # Feed Forward Networkのデモ
│   ├── 05_encoder_demo.ipynb             # Encoderのデモ
│   ├── 06_decoder_demo.ipynb             # Decoderのデモ
│   ├── 07_transformer_demo.ipynb         # 完全なTransformerのデモ
│   ├── 08_diverse_tasks_demo.ipynb       # 様々なタスクでの学習デモ
│   ├── QandA_01_attention.ipynb          # Q&A: Attention基礎 (Q1-Q4)
│   ├── QandA_02_multihead.ipynb          # Q&A: Multi-Head Attention (Q5-Q16)
│   ├── QandA_03_architecture.ipynb       # Q&A: アーキテクチャ (Q17-Q31)
│   └── QandA_04_experiments.ipynb        # Q&A: 実験・応用
├── tests/                  # テストコード
├── requirements.txt        # 依存パッケージ
└── README.md
```

## セットアップ

```bash
# 依存パッケージのインストール
pip install -r requirements.txt
```

## 進捗

実装予定の要素:
- [x] Self-Attention機構 (`src/attention.py`)
- [x] Multi-Head Attention (`src/attention.py`)
- [x] Position Encoding (`src/position_encoding.py`)
- [x] Feed Forward Network (`src/feed_forward.py`)
- [x] Encoder (`src/encoder.py`)
  - Layer Normalization
  - Residual Connection
  - EncoderLayer / Encoder / TransformerEncoder
- [x] Decoder (`src/decoder.py`)
  - Causal Mask（未来を見ないマスク）
  - Cross-Attention（Encoder出力への注意）
  - DecoderLayer / Decoder / TransformerDecoder
- [x] 完全なTransformerモデル (`src/transformer.py`)
  - Encoder + Decoder統合
  - 自動マスク生成
  - Greedy / サンプリング生成
  - Top-K / Top-P サンプリング

## 使い方

### 1. Self-Attentionの実装を確認

```python
from src.attention import SelfAttention
import torch

# モデルのインスタンス化
d_model = 64
model = SelfAttention(d_model)

# 入力データ（バッチサイズ=2, シーケンス長=5, 次元=64）
x = torch.randn(2, 5, d_model)

# 順伝播
output, attention_weights = model(x)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
```

### 2. デモノートブック

`notebooks/01_self_attention_demo.ipynb`でSelf-Attentionの動作を詳しく学べます：
- Query, Key, Valueの計算過程
- Attention Weightsの可視化
- 簡単な学習タスク（数列コピー）

```bash
jupyter notebook notebooks/01_self_attention_demo.ipynb
```

## 参考文献

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
