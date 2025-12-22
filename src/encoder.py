"""
Transformer Encoder の実装

Encoderは以下の要素で構成されます：
1. Multi-Head Attention（自己注意機構）
2. Feed Forward Network（位置ごとの全結合層）
3. Layer Normalization（層正規化）
4. Residual Connection（残差接続）

各EncoderLayerは上記を組み合わせ、複数のLayerを積み重ねて
Encoder全体を構成します。

参考: "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
"""

import torch
import torch.nn as nn
import copy

from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .position_encoding import PositionalEncoding


class LayerNorm(nn.Module):
    """
    Layer Normalization（層正規化）

    各サンプルの特徴次元に沿って正規化を行います。
    Batch Normalizationと異なり、バッチサイズに依存しません。

    数式:
        LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β

    ここで:
    - μ: 特徴次元の平均
    - σ²: 特徴次元の分散
    - γ, β: 学習可能なパラメータ（スケールとシフト）
    - ε: 数値安定性のための小さな値

    なぜLayer Normalizationを使うのか:
    1. 学習の安定化: 各層の入力分布を正規化
    2. 勾配消失/爆発の防止: 値のスケールを一定に保つ
    3. バッチサイズに依存しない: 推論時も安定
    """

    def __init__(self, d_model, eps=1e-6):
        """
        Args:
            d_model (int): モデルの次元数
            eps (float): 数値安定性のための小さな値
        """
        super().__init__()

        # 学習可能なパラメータ
        self.gamma = nn.Parameter(torch.ones(d_model))   # スケール（初期値1）
        self.beta = nn.Parameter(torch.zeros(d_model))   # シフト（初期値0）
        self.eps = eps

    def forward(self, x):
        """
        Layer Normalizationの順伝播

        Args:
            x (torch.Tensor): 入力 [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: 正規化後の出力 [batch_size, seq_len, d_model]
        """
        # 最後の次元（特徴次元）に沿って平均と分散を計算
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # 正規化してスケール・シフトを適用
        normalized = (x - mean) / (std + self.eps)
        return self.gamma * normalized + self.beta


class ResidualConnection(nn.Module):
    """
    Residual Connection（残差接続）+ Layer Normalization

    残差接続は、入力をそのまま出力に加算する仕組みです：
        output = LayerNorm(x + sublayer(x))

    または（Pre-LN形式）：
        output = x + sublayer(LayerNorm(x))

    なぜ残差接続を使うのか:
    1. 勾配の流れを改善: 深いネットワークでも勾配が消失しにくい
    2. 恒等写像を学習しやすい: 何もしないことを学習できる
    3. 学習の安定化: 各層の変化を小さく保てる

    このクラスはPost-LN形式（元の論文）を実装しています。
    """

    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model (int): モデルの次元数
            dropout (float): ドロップアウト率
        """
        super().__init__()

        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        残差接続の順伝播

        Args:
            x (torch.Tensor): 入力
            sublayer (callable): サブレイヤー（Attention or FFN）

        Returns:
            torch.Tensor: 残差接続後の出力
        """
        # Post-LN: LayerNorm(x + sublayer(x))
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer（1層分）

    構造:
        入力
          ↓
        ┌─────────────────┐
        │ Multi-Head      │
        │ Self-Attention  │
        └────────┬────────┘
          ↓      │
        Add & Norm ←─┘ (残差接続)
          ↓
        ┌─────────────────┐
        │ Feed Forward    │
        │ Network         │
        └────────┬────────┘
          ↓      │
        Add & Norm ←─┘ (残差接続)
          ↓
        出力

    各サブレイヤー後に:
    1. Dropout
    2. 残差接続（入力を加算）
    3. Layer Normalization
    """

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        """
        Args:
            d_model (int): モデルの次元数
            num_heads (int): Attentionヘッド数
            d_ff (int): FFNの中間層次元（デフォルトは4×d_model）
            dropout (float): ドロップアウト率
        """
        super().__init__()

        if d_ff is None:
            d_ff = d_model * 4

        # サブレイヤー
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # 残差接続 + Layer Normalization（2つ必要）
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask=None):
        """
        EncoderLayerの順伝播

        Args:
            x (torch.Tensor): 入力 [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): Attentionマスク

        Returns:
            torch.Tensor: 出力 [batch_size, seq_len, d_model]
        """
        # 1. Self-Attention + 残差接続
        # lambdaでAttentionをラップ（残差接続に渡すため）
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, mask)[0])

        # 2. Feed Forward + 残差接続
        x = self.residual2(x, self.feed_forward)

        return x


class Encoder(nn.Module):
    """
    Transformer Encoder（全体）

    構造:
        入力トークン
          ↓
        ┌─────────────────┐
        │ Token Embedding │
        └────────┬────────┘
          ↓      +
        ┌─────────────────┐
        │ Position        │
        │ Encoding        │
        └────────┬────────┘
          ↓
        ┌─────────────────┐
        │ Encoder Layer 1 │
        ├─────────────────┤
        │ Encoder Layer 2 │
        ├─────────────────┤
        │      ...        │
        ├─────────────────┤
        │ Encoder Layer N │
        └────────┬────────┘
          ↓
        出力

    元の論文では N=6 層を使用。
    """

    def __init__(self, d_model, num_heads, num_layers, d_ff=None,
                 max_len=5000, dropout=0.1):
        """
        Args:
            d_model (int): モデルの次元数
            num_heads (int): Attentionヘッド数
            num_layers (int): Encoderレイヤーの数
            d_ff (int): FFNの中間層次元（デフォルトは4×d_model）
            max_len (int): 最大シーケンス長
            dropout (float): ドロップアウト率
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Position Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # N層のEncoder Layer
        # 各層は独立したパラメータを持つ
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最終Layer Normalization（オプション、Pre-LN形式で必要）
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Encoderの順伝播

        Args:
            x (torch.Tensor): 入力埋め込み [batch_size, seq_len, d_model]
                              （Token Embeddingの出力を想定）
            mask (torch.Tensor, optional): Attentionマスク

        Returns:
            torch.Tensor: Encoder出力 [batch_size, seq_len, d_model]
        """
        # 1. Position Encodingを加算
        x = self.pos_encoding(x)

        # 2. N層のEncoder Layerを順に適用
        for layer in self.layers:
            x = layer(x, mask)

        # 3. 最終正規化
        x = self.norm(x)

        return x


class TransformerEncoder(nn.Module):
    """
    完全なTransformer Encoder（埋め込み層を含む）

    入力トークンID → Embedding → Encoder → 出力

    これは分類タスクやBERTのような用途で使用されます。
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff=None, max_len=5000, dropout=0.1, pad_idx=0):
        """
        Args:
            vocab_size (int): 語彙サイズ
            d_model (int): モデルの次元数
            num_heads (int): Attentionヘッド数
            num_layers (int): Encoderレイヤーの数
            d_ff (int): FFNの中間層次元
            max_len (int): 最大シーケンス長
            dropout (float): ドロップアウト率
            pad_idx (int): パディングトークンのインデックス
        """
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # Encoder
        self.encoder = Encoder(d_model, num_heads, num_layers,
                               d_ff, max_len, dropout)

    def forward(self, src, src_mask=None):
        """
        TransformerEncoderの順伝播

        Args:
            src (torch.Tensor): 入力トークンID [batch_size, seq_len]
            src_mask (torch.Tensor, optional): ソースマスク

        Returns:
            torch.Tensor: Encoder出力 [batch_size, seq_len, d_model]
        """
        # パディングマスクを作成（指定がない場合）
        # MultiHeadAttention内でunsqueeze(1)されるため、[batch_size, seq_len, seq_len]で作成
        if src_mask is None:
            # パディング位置は0、それ以外は1
            # [batch_size, seq_len] → [batch_size, 1, seq_len]
            seq_len = src.size(1)
            pad_mask = (src != self.pad_idx).unsqueeze(1)
            # [batch_size, 1, seq_len] → [batch_size, seq_len, seq_len]
            src_mask = pad_mask.expand(-1, seq_len, -1)

        # Embedding（スケーリング付き）
        # 論文では sqrt(d_model) でスケーリング
        x = self.embedding(src) * (self.d_model ** 0.5)

        # Encoder
        output = self.encoder(x, src_mask)

        return output


# テスト用のコード
if __name__ == "__main__":
    # デバイスの設定
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # パラメータ
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 4
    num_layers = 3
    vocab_size = 1000

    print("=" * 70)
    print("Layer Normalization Test")
    print("=" * 70)

    x = torch.randn(batch_size, seq_len, d_model).to(device)
    layer_norm = LayerNorm(d_model).to(device)
    output_ln = layer_norm(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_ln.shape}")
    print(f"Output mean (should be ~0): {output_ln.mean(dim=-1)[0, :3].detach().cpu().numpy()}")
    print(f"Output std (should be ~1): {output_ln.std(dim=-1)[0, :3].detach().cpu().numpy()}")

    print("\n" + "=" * 70)
    print("Encoder Layer Test")
    print("=" * 70)

    encoder_layer = EncoderLayer(d_model, num_heads).to(device)
    output_layer = encoder_layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_layer.shape}")

    print("\n" + "=" * 70)
    print("Full Encoder Test")
    print("=" * 70)

    encoder = Encoder(d_model, num_heads, num_layers).to(device)
    output_encoder = encoder(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_encoder.shape}")
    print(f"Number of layers: {num_layers}")

    print("\n" + "=" * 70)
    print("Transformer Encoder (with Embedding) Test")
    print("=" * 70)

    transformer_encoder = TransformerEncoder(
        vocab_size, d_model, num_heads, num_layers
    ).to(device)

    # トークンIDを入力
    src = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    output_full = transformer_encoder(src)

    print(f"Input token IDs shape: {src.shape}")
    print(f"Output shape: {output_full.shape}")

    print("\n" + "=" * 70)
    print("Parameter Count")
    print("=" * 70)

    encoder_params = sum(p.numel() for p in encoder.parameters())
    full_params = sum(p.numel() for p in transformer_encoder.parameters())

    print(f"Encoder (without embedding): {encoder_params:,} parameters")
    print(f"TransformerEncoder (with embedding): {full_params:,} parameters")
    print(f"  - Embedding: {vocab_size * d_model:,} parameters")
