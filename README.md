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
│   ├── 09_addition_improvement.ipynb     # Additionタスク改善実験
│   ├── 10_pytorch_vs_custom.ipynb        # PyTorch版 vs 自作版 性能比較
│   ├── tutorial_transformer.ipynb        # Transformerチュートリアル（番外）
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
- [x] PyTorch組み込みTransformerとの比較 (`notebooks/10_pytorch_vs_custom.ipynb`)

## 使い方

### クイックスタート（チュートリアル）

Transformerの使い方を学ぶには、チュートリアルノートブックを参照してください：

```bash
jupyter notebook notebooks/tutorial_transformer.ipynb
```

チュートリアルでは以下を解説しています：
- モデルの作成とパラメータ設定
- データの準備方法
- 学習と推論（生成）
- 実践例（コピータスク、加算タスク）
- Tips & トラブルシューティング

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

## PyTorch版との性能比較

自作TransformerとPyTorch組み込み`nn.Transformer`の比較結果（`10_pytorch_vs_custom.ipynb`）:

| 項目 | 自作版 | PyTorch版 | 備考 |
|------|--------|-----------|------|
| パラメータ数 | 241,842 | 243,378 | 差分0.64% |
| コピータスク精度 | 100% | 99.6% | 同等 |
| 推論速度 | 327ms | 131ms | **PyTorchが2.49倍高速** |

**結論**: 機能的には同等だが、PyTorch版は推論が約2.5倍高速。教育目的には自作版、本番環境にはPyTorch版を推奨。

## 参考文献

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
