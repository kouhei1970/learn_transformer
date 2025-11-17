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
│   └── attention.py        # Self-Attention & Multi-Head Attention実装
├── notebooks/              # Jupyter Notebook
│   ├── 01_self_attention_demo.ipynb   # Self-Attentionのデモ
│   ├── 02_multi_head_attention_demo.ipynb  # Multi-Head Attentionのデモ
│   └── QandA.ipynb         # 学習中の質問と回答
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
- [ ] Position Encoding
- [ ] Feed Forward Network
- [ ] Encoder
- [ ] Decoder
- [ ] 完全なTransformerモデル

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
