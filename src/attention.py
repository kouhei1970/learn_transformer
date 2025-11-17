"""
Self-Attention機構の実装

Transformerの核となるSelf-Attention（自己注意機構）を実装します。
Attention機構は、入力シーケンスの各要素が他の要素とどれだけ関連しているかを計算し、
それに基づいて重み付き和を取る仕組みです。

参考: "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    Attentionの基本となる機構で、以下の計算を行います:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    ここで:
    - Q (Query): 「何を探しているか」を表すベクトル
    - K (Key): 「何を持っているか」を表すベクトル
    - V (Value): 実際の情報を持つベクトル
    - d_k: Keyの次元数（スケーリング係数として使用）
    """
    
    def __init__(self, dropout=0.1):
        """
        Args:
            dropout (float): Attention重みに適用するドロップアウト率
        """
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Scaled Dot-Product Attentionの順伝播
        
        Args:
            query (torch.Tensor): Query行列 [batch_size, seq_len, d_k]
            key (torch.Tensor): Key行列 [batch_size, seq_len, d_k]
            value (torch.Tensor): Value行列 [batch_size, seq_len, d_v]
            mask (torch.Tensor, optional): マスク [batch_size, seq_len, seq_len]
                                          Trueの位置は-infにマスクされる
        
        Returns:
            output (torch.Tensor): Attention適用後の出力 [batch_size, seq_len, d_v]
            attention_weights (torch.Tensor): Attention重み [batch_size, seq_len, seq_len]
        """
        # Keyの次元数を取得（スケーリングに使用）
        d_k = query.size(-1)
        
        # Step 1: QとKの内積を計算
        # query: [batch, seq_len, d_k]
        # key.transpose(-2, -1): [batch, d_k, seq_len]
        # scores: [batch, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Step 2: sqrt(d_k)でスケーリング
        # これにより、内積が大きくなりすぎてsoftmaxの勾配が消失するのを防ぐ
        scores = scores / math.sqrt(d_k)
        
        # Step 3: マスクの適用（オプション）
        # 例: 未来の情報を見ないようにする（デコーダー）、パディングを無視する、など
        if mask is not None:
            # マスクされた位置を-infにすることで、softmax後に0になる
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 4: Softmaxで正規化してAttention重みを計算
        # 各行（各Query位置）について、全てのKey位置への重みの和が1になる
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 5: ドロップアウト適用（学習時の正則化）
        attention_weights = self.dropout(attention_weights)
        
        # Step 6: Attention重みとValueの重み付き和を計算
        # attention_weights: [batch, seq_len, seq_len]
        # value: [batch, seq_len, d_v]
        # output: [batch, seq_len, d_v]
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class SelfAttention(nn.Module):
    """
    Self-Attention Layer
    
    入力から線形変換でQuery, Key, Valueを生成し、
    Scaled Dot-Product Attentionを適用します。
    
    「Self」とは、同じ入力シーケンスから Q, K, V を生成することを意味します。
    これにより、シーケンス内の各要素が他の全要素との関係性を学習できます。
    """
    
    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model (int): モデルの次元数（入力・出力の次元）
            dropout (float): ドロップアウト率
        """
        super(SelfAttention, self).__init__()
        
        self.d_model = d_model
        
        # 入力からQuery, Key, Valueを生成するための線形変換
        # 各変換は独立したパラメータを持つ
        # bias=False: 最新のベストプラクティスに従い、バイアスなしで実装
        # (Attentionは相対的な関係性を捉えるため、絶対的なオフセットは不要)
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        
        # Scaled Dot-Product Attention
        self.attention = ScaledDotProductAttention(dropout)
        
        # Attention適用後の出力を変換する線形層
        self.output_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        """
        Self-Attentionの順伝播
        
        Args:
            x (torch.Tensor): 入力 [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): Attentionマスク
        
        Returns:
            output (torch.Tensor): 出力 [batch_size, seq_len, d_model]
            attention_weights (torch.Tensor): Attention重み [batch_size, seq_len, seq_len]
        """
        # Step 1: 入力から Q, K, V を線形変換で生成
        # 同じ入力 x から生成するため「Self」Attention
        query = self.query_linear(x)  # [batch, seq_len, d_model]
        key = self.key_linear(x)      # [batch, seq_len, d_model]
        value = self.value_linear(x)  # [batch, seq_len, d_model]
        
        # Step 2: Scaled Dot-Product Attentionを適用
        attention_output, attention_weights = self.attention(
            query, key, value, mask
        )
        
        # Step 3: 出力を線形変換
        output = self.output_linear(attention_output)
        
        return output, attention_weights


# テスト用のシンプルな例
if __name__ == "__main__":
    # デバイスの設定（macOS GPU対応）
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")
    
    # ハイパーパラメータ
    batch_size = 2    # バッチサイズ
    seq_len = 5       # シーケンス長
    d_model = 64      # モデルの次元数
    
    # ランダムな入力を生成
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    print(f"Input shape: {x.shape}")
    
    # Self-Attentionモデルを作成
    self_attention = SelfAttention(d_model).to(device)
    
    # 順伝播
    output, attention_weights = self_attention(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"\nAttention weights (first sample):")
    print(attention_weights[0].detach().cpu().numpy())
    print(f"\nAttention weights sum per row (should be ~1.0):")
    print(attention_weights[0].sum(dim=-1).detach().cpu().numpy())
