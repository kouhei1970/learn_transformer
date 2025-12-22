"""
Transformer Decoder の実装

Decoderは以下の要素で構成されます：
1. Masked Self-Attention（未来を見ない自己注意機構）
2. Cross-Attention（Encoder出力への注意機構）
3. Feed Forward Network（位置ごとの全結合層）
4. Layer Normalization（層正規化）
5. Residual Connection（残差接続）

DecoderはEncoderと似ていますが、2つの重要な違いがあります：
1. Masked Self-Attention: 未来のトークンを見ないようにマスク
2. Cross-Attention: Encoder出力をK, Vとして使用

参考: "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
"""

import torch
import torch.nn as nn
import math

from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .position_encoding import PositionalEncoding
from .encoder import LayerNorm, ResidualConnection


def generate_causal_mask(size, device=None):
    """
    Causal Mask（因果マスク）を生成

    未来のトークンを見ないようにするための下三角行列マスク。

    Args:
        size (int): シーケンス長
        device: デバイス

    Returns:
        torch.Tensor: Causal Mask [size, size] (dtype=bool)
            True = 参照可能, False = マスク（参照不可）

    例（size=4の場合）:
        [[True,  False, False, False],
         [True,  True,  False, False],
         [True,  True,  True,  False],
         [True,  True,  True,  True ]]

    位置iからは位置0~iまでしか見えない（未来は見えない）
    """
    mask = torch.tril(torch.ones(size, size, device=device)).bool()
    return mask


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer（1層分）

    構造:
        入力
          ↓
        ┌─────────────────┐
        │ Masked          │
        │ Self-Attention  │  ← 未来を見ないマスク付き
        └────────┬────────┘
          ↓      │
        Add & Norm ←─┘ (残差接続)
          ↓
        ┌─────────────────┐
        │ Cross-Attention │  ← Q=Decoder, K,V=Encoder出力
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
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # 残差接続 + Layer Normalization（3つ必要）
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        DecoderLayerの順伝播

        Args:
            x (torch.Tensor): Decoder入力 [batch_size, tgt_len, d_model]
            encoder_output (torch.Tensor): Encoder出力 [batch_size, src_len, d_model]
            src_mask (torch.Tensor, optional): ソース側のマスク（パディング用）
            tgt_mask (torch.Tensor, optional): ターゲット側のマスク（Causal + パディング）

        Returns:
            torch.Tensor: 出力 [batch_size, tgt_len, d_model]
        """
        # 1. Masked Self-Attention + 残差接続
        # 自分自身への注意（未来は見ない）
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, tgt_mask)[0])

        # 2. Cross-Attention + 残差接続
        # Q=Decoder状態, K,V=Encoder出力
        x = self.residual2(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask)[0])

        # 3. Feed Forward + 残差接続
        x = self.residual3(x, self.feed_forward)

        return x


class Decoder(nn.Module):
    """
    Transformer Decoder（全体）

    構造:
        入力埋め込み
          ↓
        + Position Encoding
          ↓
        ┌─────────────────┐
        │ Decoder Layer 1 │
        ├─────────────────┤
        │ Decoder Layer 2 │
        ├─────────────────┤
        │      ...        │
        ├─────────────────┤
        │ Decoder Layer N │
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
            num_layers (int): Decoderレイヤーの数
            d_ff (int): FFNの中間層次元（デフォルトは4×d_model）
            max_len (int): 最大シーケンス長
            dropout (float): ドロップアウト率
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Position Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # N層のDecoder Layer
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最終Layer Normalization
        self.norm = LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Decoderの順伝播

        Args:
            x (torch.Tensor): 入力埋め込み [batch_size, tgt_len, d_model]
            encoder_output (torch.Tensor): Encoder出力 [batch_size, src_len, d_model]
            src_mask (torch.Tensor, optional): ソース側のマスク
            tgt_mask (torch.Tensor, optional): ターゲット側のマスク

        Returns:
            torch.Tensor: Decoder出力 [batch_size, tgt_len, d_model]
        """
        # 1. Position Encodingを加算
        x = self.pos_encoding(x)

        # 2. N層のDecoder Layerを順に適用
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        # 3. 最終正規化
        x = self.norm(x)

        return x


class TransformerDecoder(nn.Module):
    """
    完全なTransformer Decoder（埋め込み層と出力層を含む）

    入力トークンID → Embedding → Decoder → 線形変換 → 語彙確率

    これは言語モデルや翻訳タスクのデコーダーとして使用されます。
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff=None, max_len=5000, dropout=0.1, pad_idx=0):
        """
        Args:
            vocab_size (int): 語彙サイズ
            d_model (int): モデルの次元数
            num_heads (int): Attentionヘッド数
            num_layers (int): Decoderレイヤーの数
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

        # Decoder
        self.decoder = Decoder(d_model, num_heads, num_layers,
                               d_ff, max_len, dropout)

        # 出力層（語彙サイズへの射影）
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        TransformerDecoderの順伝播

        Args:
            tgt (torch.Tensor): ターゲットトークンID [batch_size, tgt_len]
            encoder_output (torch.Tensor): Encoder出力 [batch_size, src_len, d_model]
            src_mask (torch.Tensor, optional): ソースマスク
            tgt_mask (torch.Tensor, optional): ターゲットマスク

        Returns:
            torch.Tensor: 語彙上の確率分布 [batch_size, tgt_len, vocab_size]
        """
        tgt_len = tgt.size(1)

        # Causal Maskを作成（指定がない場合）
        if tgt_mask is None:
            # 1. Causal Mask（未来を見ない）
            causal_mask = generate_causal_mask(tgt_len, tgt.device)

            # 2. パディングマスク
            pad_mask = (tgt != self.pad_idx).unsqueeze(1)  # [batch, 1, tgt_len]

            # 3. 結合: Causal AND パディング
            tgt_mask = causal_mask.unsqueeze(0) & pad_mask  # [batch, tgt_len, tgt_len]

        # Embedding（スケーリング付き）
        x = self.embedding(tgt) * (self.d_model ** 0.5)

        # Decoder
        decoder_output = self.decoder(x, encoder_output, src_mask, tgt_mask)

        # 出力層（語彙への射影）
        logits = self.output_projection(decoder_output)

        return logits


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
    src_len = 10
    tgt_len = 8
    d_model = 64
    num_heads = 4
    num_layers = 3
    vocab_size = 1000

    print("=" * 70)
    print("Causal Mask Test")
    print("=" * 70)

    mask = generate_causal_mask(5)
    print("Causal Mask (size=5):")
    print(mask)

    print("\n" + "=" * 70)
    print("Decoder Layer Test")
    print("=" * 70)

    # ダミーの入力
    x = torch.randn(batch_size, tgt_len, d_model).to(device)
    encoder_output = torch.randn(batch_size, src_len, d_model).to(device)

    decoder_layer = DecoderLayer(d_model, num_heads).to(device)
    output = decoder_layer(x, encoder_output)

    print(f"Decoder input shape: {x.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Decoder layer output shape: {output.shape}")

    print("\n" + "=" * 70)
    print("Full Decoder Test")
    print("=" * 70)

    decoder = Decoder(d_model, num_heads, num_layers).to(device)
    output = decoder(x, encoder_output)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of layers: {num_layers}")

    print("\n" + "=" * 70)
    print("Transformer Decoder (with Embedding) Test")
    print("=" * 70)

    transformer_decoder = TransformerDecoder(
        vocab_size, d_model, num_heads, num_layers
    ).to(device)

    # トークンIDを入力
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len)).to(device)
    logits = transformer_decoder(tgt, encoder_output)

    print(f"Target token IDs shape: {tgt.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Output logits shape: {logits.shape}")

    print("\n" + "=" * 70)
    print("Parameter Count")
    print("=" * 70)

    decoder_params = sum(p.numel() for p in decoder.parameters())
    full_params = sum(p.numel() for p in transformer_decoder.parameters())

    print(f"Decoder (without embedding): {decoder_params:,} parameters")
    print(f"TransformerDecoder (with embedding): {full_params:,} parameters")
