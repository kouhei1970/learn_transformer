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
│   ├── transformer.py      # 完全なTransformer実装
│   ├── tokenizer.py        # 日本語トークナイザー
│   ├── dataset.py          # データセット処理
│   ├── train_chat.py       # チャットモデル学習
│   └── chat_engine.py      # 推論エンジン
├── chat_app.py             # Web UI
├── models/                 # 学習済みモデル保存先
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
│   ├── pytorch_basics.ipynb              # PyTorch入門（番外）
│   ├── QandA_01_attention.ipynb          # Q&A: Attention基礎 (Q1-Q4)
│   ├── QandA_02_multihead.ipynb          # Q&A: Multi-Head Attention (Q5-Q20)
│   ├── QandA_03_architecture.ipynb       # Q&A: アーキテクチャ (Q17-Q31)
│   ├── QandA_04_experiments.ipynb        # Q&A: 実験・応用 (Q21-Q25)
│   └── QandA_05_llm_scaling.ipynb        # Q&A: LLM・スケーリング (Q32-)
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

## チャットツール

自作Transformerを使った日本語チャットツールを実装しています。

### セットアップ

```bash
# 追加の依存パッケージをインストール
pip install sentencepiece datasets gradio
```

### クイックスタート（デモモード）

まずは動作確認だけしたい場合、未学習モデルでWeb UIを試せます：

```bash
python chat_app.py --demo
```

ブラウザで http://localhost:7860 を開くとチャット画面が表示されます。

**注意**: デモモードは未学習モデルなので、意味のある応答は生成されません。

### 学習してから使う（推奨）

#### Step 1: モデルを学習

```bash
# 標準的な学習（GPU推奨、数時間〜半日）
python -m src.train_chat --epochs 100 --batch_size 32

# 小さいモデルで試す場合（CPU可、数十分）
python -m src.train_chat \
    --epochs 50 \
    --batch_size 16 \
    --d_model 256 \
    --num_layers 4 \
    --max_samples 1000

# フルオプション
python -m src.train_chat \
    --epochs 100 \
    --batch_size 32 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    --d_ff 2048 \
    --vocab_size 32000 \
    --max_len 256 \
    --save_dir models/chat
```

#### Step 2: 学習済みモデルでチャット

```bash
python chat_app.py --model models/chat/best_model.pt
```

### Web UIの操作

| パラメータ | 説明 | 推奨値 |
|-----------|------|--------|
| **Temperature** | 低いと確定的、高いとランダム | 0.7〜1.0 |
| **Top-K** | 確率上位K個のトークンのみ使用（0で無効） | 50 |
| **Top-P** | 累積確率がPになるまでのトークンを使用（1.0で無効） | 0.9 |
| **Greedy** | チェックすると常に最も確率の高いトークンを選択 | OFF |

**Tips**:
- 応答が繰り返しになる場合 → Temperatureを上げる（1.0〜1.5）
- 応答が支離滅裂な場合 → Temperatureを下げる（0.5〜0.7）またはGreedyをON
- 多様な応答が欲しい場合 → Top-Kを小さく（10〜30）、Top-Pを小さく（0.7〜0.8）

### コマンドラインオプション

```bash
# ポート番号を変更
python chat_app.py --demo --port 8080

# 公開リンクを作成（外部からアクセス可能）
python chat_app.py --demo --share
```

### Pythonから直接使う

```python
from src.chat_engine import ChatEngine

# 学習済みモデルをロード
engine = ChatEngine.from_checkpoint("models/chat/best_model.pt")

# チャット
response = engine.chat("こんにちは")
print(response)

# パラメータを指定
response = engine.chat(
    "Transformerについて教えて",
    max_len=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
```

### ファイル構成

```
learn_transformer/
├── src/
│   ├── tokenizer.py      # 日本語トークナイザー（SentencePiece）
│   ├── dataset.py        # データセット処理
│   ├── train_chat.py     # 学習スクリプト
│   └── chat_engine.py    # 推論エンジン
├── chat_app.py           # Gradio Web UI
└── models/               # 学習済みモデル保存先
    └── chat/
        ├── best_model.pt     # ベストモデル
        ├── final_model.pt    # 最終モデル
        ├── config.json       # 設定ファイル
        └── tokenizer/        # トークナイザーファイル
```

### 学習シーケンス

学習スクリプトの実行フローを示します。

```
┌─────────────────────────────────────────────────────────────┐
│  python -m src.train_chat --epochs 100 --batch_size 32      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. 初期化                                                   │
│     ├── デバイス検出 (CUDA / MPS / CPU)                      │
│     ├── 保存ディレクトリ作成 (models/chat/)                  │
│     └── 設定をconfig.jsonに保存                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. トークナイザー準備                                        │
│     ├── 既存があれば読み込み                                  │
│     └── なければ新規学習（SentencePiece BPE）                 │
│         └── tokenizer.model を保存                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. データセット準備                                          │
│     ├── Hugging Faceからデータ取得                           │
│     ├── Train/Val分割 (90%/10%)                              │
│     ├── トークン化                                           │
│     │   ├── src: 入力文 → トークンID                         │
│     │   ├── tgt_input: <bos> + 応答文                        │
│     │   └── tgt_output: 応答文 + <eos>                       │
│     └── DataLoader作成                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. モデル作成                                                │
│     ├── Transformer(src_vocab, tgt_vocab, d_model, ...)     │
│     ├── AdamW optimizer                                      │
│     ├── CosineAnnealingWarmRestarts scheduler               │
│     └── CrossEntropyLoss (PADトークン無視)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. 学習ループ (epoch 1 → N)                                 │
│     │                                                        │
│     │  ┌──────────────────────────────────────────────────┐ │
│     │  │  5a. 訓練フェーズ                                 │ │
│     │  │      for batch in train_loader:                  │ │
│     │  │        ├── 順伝播: logits = model(src, tgt)      │ │
│     │  │        ├── 損失計算: CrossEntropyLoss            │ │
│     │  │        ├── 逆伝播: loss.backward()               │ │
│     │  │        ├── 勾配クリッピング                       │ │
│     │  │        └── パラメータ更新: optimizer.step()       │ │
│     │  └──────────────────────────────────────────────────┘ │
│     │                         │                              │
│     │                         ▼                              │
│     │  ┌──────────────────────────────────────────────────┐ │
│     │  │  5b. 検証フェーズ                                 │ │
│     │  │      with torch.no_grad():                       │ │
│     │  │        └── 損失・精度を計算                       │ │
│     │  └──────────────────────────────────────────────────┘ │
│     │                         │                              │
│     │                         ▼                              │
│     │  ┌──────────────────────────────────────────────────┐ │
│     │  │  5c. 保存判定                                     │ │
│     │  │      ├── val_loss改善 → best_model.pt 保存       │ │
│     │  │      └── N epoch経過 → checkpoint_epochN.pt 保存 │ │
│     │  └──────────────────────────────────────────────────┘ │
│     │                         │                              │
│     └─────────────────────────┴── 次のepochへ ───────────────┘
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  6. 学習完了                                                  │
│     ├── final_model.pt 保存                                  │
│     ├── training_history.json 保存                           │
│     └── 結果サマリー表示                                      │
└─────────────────────────────────────────────────────────────┘
```

#### 学習の中断と再開

- **中断**: `Ctrl+C` で中断できます
- **途中経過**: `best_model.pt` と `checkpoint_epochN.pt` は自動保存されます
- **再開**: `--resume` オプションで途中から再開できます

```bash
# 途中から再開
python -m src.train_chat --resume models/chat/checkpoint_epoch50.pt --epochs 100

# より頻繁に保存（5エポックごと）
python -m src.train_chat --epochs 100 --save_every 5
```

#### 各フェーズの所要時間目安

| フェーズ | GPU環境 | CPU環境 |
|---------|---------|---------|
| 1. 初期化 | 数秒 | 数秒 |
| 2. トークナイザー学習 | 1〜5分 | 1〜5分 |
| 3. データセット準備 | 1〜2分 | 1〜2分 |
| 4. モデル作成 | 数秒 | 数秒 |
| 5. 学習ループ（100エポック） | 数時間 | 数日（非推奨） |
| 6. 完了処理 | 数秒 | 数秒 |

### 注意事項

- このモデルは教育目的で作成されており、商用LLMほどの品質は期待できません
- GPU環境での学習を推奨（CPU環境では小さいモデルを使用してください）
- 学習データは [kunishou/databricks-dolly-15k-ja](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja) を使用

## 参考文献

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
