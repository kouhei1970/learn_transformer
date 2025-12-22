"""
Feed Forward Network（位置ごとの全結合層）の実装

TransformerのEncoderとDecoderでは、Attention層の後に
Feed Forward Network (FFN) が適用されます。

FFNは各位置（トークン）に対して独立に同じ変換を適用します。
これにより、Attentionで集約した情報を非線形変換で処理します。

参考: "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network

    2層の全結合層で構成され、間に活性化関数を挟みます：
        FFN(x) = max(0, xW₁ + b₁)W₂ + b₂  (論文ではReLU)
        FFN(x) = GELU(xW₁ + b₁)W₂ + b₂    (最近のモデルではGELU)

    特徴：
    1. 各位置に独立に適用（位置間の情報交換はAttentionが担当）
    2. 中間層の次元を拡大（通常4倍）して表現力を高める
    3. 非線形変換によりモデルの表現力を向上

    なぜ必要か：
    - Attentionは本質的に線形変換（重み付き和）
    - FFNで非線形性を導入し、複雑なパターンを学習可能に
    """

    def __init__(self, d_model, d_ff=None, dropout=0.1, activation='relu'):
        """
        Args:
            d_model (int): モデルの次元数（入力・出力の次元）
            d_ff (int): 中間層の次元数（デフォルトは d_model * 4）
            dropout (float): ドロップアウト率
            activation (str): 活性化関数 ('relu' or 'gelu')
        """
        super().__init__()

        # 中間層の次元（デフォルトは4倍）
        if d_ff is None:
            d_ff = d_model * 4

        self.d_model = d_model
        self.d_ff = d_ff

        # 2層の線形変換
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

        # 活性化関数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        """
        Feed Forward Networkの順伝播

        Args:
            x (torch.Tensor): 入力 [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: 出力 [batch_size, seq_len, d_model]
        """
        # 1層目: d_model → d_ff（拡大）+ 活性化関数
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # 2層目: d_ff → d_model（縮小）
        x = self.linear2(x)
        x = self.dropout(x)

        return x


class GatedFeedForward(nn.Module):
    """
    Gated Feed Forward Network (SwiGLU variant)

    LLaMA, PaLMなどの最新モデルで使用されるゲート機構付きFFN：
        FFN(x) = (xW₁ ⊙ SiLU(xW_gate)) W₂

    ⊙はelement-wise乗算、SiLUはSwish活性化関数。

    ゲート機構により、情報の流れを制御し、より表現力を高めます。
    """

    def __init__(self, d_model, d_ff=None, dropout=0.1):
        """
        Args:
            d_model (int): モデルの次元数
            d_ff (int): 中間層の次元数（デフォルトは d_model * 4）
            dropout (float): ドロップアウト率
        """
        super().__init__()

        if d_ff is None:
            d_ff = d_model * 4

        self.d_model = d_model
        self.d_ff = d_ff

        # 3つの線形変換（ゲート用に1つ追加）
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)

        self.dropout = nn.Dropout(dropout)

        # SiLU (Swish) 活性化関数
        self.silu = nn.SiLU()

    def forward(self, x):
        """
        Gated FFNの順伝播

        Args:
            x (torch.Tensor): 入力 [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: 出力 [batch_size, seq_len, d_model]
        """
        # ゲート機構: gate = SiLU(x @ W_gate)
        gate = self.silu(self.w_gate(x))

        # 通常の変換: hidden = x @ W1
        hidden = self.w1(x)

        # ゲートで制御: hidden = hidden ⊙ gate
        hidden = hidden * gate

        # 出力: output = hidden @ W2
        output = self.w2(hidden)
        output = self.dropout(output)

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
    d_ff = 256  # 4倍

    print("=" * 70)
    print("Feed Forward Network Test")
    print("=" * 70)

    # 入力
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    print(f"Input shape: {x.shape}")

    # 標準FFN (ReLU)
    ffn_relu = FeedForward(d_model, d_ff, activation='relu').to(device)
    output_relu = ffn_relu(x)
    print(f"FFN (ReLU) output shape: {output_relu.shape}")

    # 標準FFN (GELU)
    ffn_gelu = FeedForward(d_model, d_ff, activation='gelu').to(device)
    output_gelu = ffn_gelu(x)
    print(f"FFN (GELU) output shape: {output_gelu.shape}")

    # Gated FFN (SwiGLU)
    gated_ffn = GatedFeedForward(d_model, d_ff).to(device)
    output_gated = gated_ffn(x)
    print(f"Gated FFN output shape: {output_gated.shape}")

    print("\n" + "=" * 70)
    print("Parameter Comparison")
    print("=" * 70)

    ffn_params = sum(p.numel() for p in ffn_relu.parameters())
    gated_params = sum(p.numel() for p in gated_ffn.parameters())

    print(f"Standard FFN parameters: {ffn_params:,}")
    print(f"  - linear1: {d_model} × {d_ff} + {d_ff} = {d_model * d_ff + d_ff:,}")
    print(f"  - linear2: {d_ff} × {d_model} + {d_model} = {d_ff * d_model + d_model:,}")

    print(f"\nGated FFN parameters: {gated_params:,}")
    print(f"  - w1: {d_model} × {d_ff} = {d_model * d_ff:,}")
    print(f"  - w_gate: {d_model} × {d_ff} = {d_model * d_ff:,}")
    print(f"  - w2: {d_ff} × {d_model} = {d_ff * d_model:,}")

    print(f"\nGated FFN has ~1.5x more parameters but often better performance")
