"""
Position Encoding（位置エンコーディング）の実装

Transformerは再帰構造を持たないため、シーケンス内の位置情報を
明示的に与える必要があります。Position Encodingは各位置に
固有のベクトルを加算することで、モデルに位置情報を伝えます。

参考: "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding（正弦波による位置エンコーディング）

    各位置posと各次元iに対して、以下の式で位置エンコーディングを計算:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    なぜSin/Cosを使うのか:
    1. 任意の固定オフセットkに対して、PE(pos+k)はPE(pos)の線形関数で表現できる
       → モデルが相対位置を学習しやすい
    2. 学習不要で、任意の長さのシーケンスに対応可能
    3. 各次元が異なる周波数を持ち、多様なスケールの位置情報を表現

    周波数の意味:
    - 低次元（i小）: 高周波数 → 近い位置の違いを捉える
    - 高次元（i大）: 低周波数 → 遠い位置の違いを捉える
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model (int): モデルの次元数（埋め込みベクトルの次元）
            max_len (int): 対応可能な最大シーケンス長
            dropout (float): ドロップアウト率
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Position Encodingを事前計算（学習時に毎回計算しなくて良い）
        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # position: [max_len, 1] - 各位置のインデックス
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: [d_model/2] - 各次元の周波数を決める項
        # 10000^(2i/d_model) を計算
        # = exp(2i * (-log(10000) / d_model))
        # = exp(2i * log(10000^(-1/d_model)))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数次元: sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数次元: cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # バッチ次元を追加: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # bufferとして登録（パラメータではないが、モデルと一緒に保存される）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        入力にPosition Encodingを加算

        Args:
            x (torch.Tensor): 入力埋め込み [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Position Encoding加算後 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)

        # 入力のシーケンス長に合わせてPEをスライス
        # self.pe: [1, max_len, d_model] → [1, seq_len, d_model]
        # ブロードキャストにより全バッチに同じPEが加算される
        x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)

    def get_encoding(self, seq_len):
        """
        Position Encodingを取得（可視化用）

        Args:
            seq_len (int): 取得するシーケンス長

        Returns:
            torch.Tensor: Position Encoding [seq_len, d_model]
        """
        return self.pe[0, :seq_len, :].clone()


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding（学習可能な位置エンコーディング）

    位置ごとに学習可能なパラメータを持つ方式。
    BERTなどで使用されている。

    特徴:
    - データから最適な位置表現を学習
    - max_lenを超えるシーケンスには対応不可
    - シンプルで実装が容易
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model (int): モデルの次元数
            max_len (int): 対応可能な最大シーケンス長
            dropout (float): ドロップアウト率
        """
        super(LearnedPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # 学習可能な位置埋め込み
        self.pe = nn.Embedding(max_len, d_model)

        # 位置インデックス用のbuffer
        self.register_buffer(
            'position_ids',
            torch.arange(max_len).unsqueeze(0)  # [1, max_len]
        )

    def forward(self, x):
        """
        入力にLearned Position Encodingを加算

        Args:
            x (torch.Tensor): 入力埋め込み [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Position Encoding加算後 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)

        # 位置インデックスを取得して埋め込みを計算
        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.pe(position_ids)

        x = x + position_embeddings

        return self.dropout(x)


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

    print("=" * 70)
    print("Sinusoidal Positional Encoding Test")
    print("=" * 70)

    # 入力（通常は埋め込み層の出力）
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    print(f"Input shape: {x.shape}")

    # Positional Encoding
    pos_encoder = PositionalEncoding(d_model).to(device)
    output = pos_encoder(x)
    print(f"Output shape: {output.shape}")

    # Position Encodingの値を確認
    pe = pos_encoder.get_encoding(seq_len)
    print(f"\nPosition Encoding shape: {pe.shape}")
    print(f"PE[0, :8] (position 0, first 8 dims):\n{pe[0, :8].cpu().numpy()}")
    print(f"PE[1, :8] (position 1, first 8 dims):\n{pe[1, :8].cpu().numpy()}")

    print("\n" + "=" * 70)
    print("Learned Positional Encoding Test")
    print("=" * 70)

    learned_pos_encoder = LearnedPositionalEncoding(d_model).to(device)
    output_learned = learned_pos_encoder(x)
    print(f"Output shape: {output_learned.shape}")

    # パラメータ数の比較
    sinusoidal_params = sum(p.numel() for p in pos_encoder.parameters())
    learned_params = sum(p.numel() for p in learned_pos_encoder.parameters())

    print(f"\nSinusoidal PE parameters: {sinusoidal_params:,} (no learnable params)")
    print(f"Learned PE parameters: {learned_params:,}")
