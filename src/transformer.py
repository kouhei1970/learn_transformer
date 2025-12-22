"""
完全なTransformerモデルの実装

Encoder と Decoder を統合した Sequence-to-Sequence モデル。
翻訳、要約、質問応答などのタスクに使用できます。

構造:
    入力（ソース）     入力（ターゲット）
         ↓                  ↓
    ┌─────────┐        ┌─────────┐
    │ Encoder │───────→│ Decoder │
    └─────────┘  K,V   └────┬────┘
                            ↓
                      出力（予測）

参考: "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder, generate_causal_mask
from .position_encoding import PositionalEncoding


class Transformer(nn.Module):
    """
    完全なTransformerモデル（Encoder-Decoder構造）

    入力:
        - src: ソースシーケンス（例: 日本語文）
        - tgt: ターゲットシーケンス（例: 英語文）

    出力:
        - 各位置での語彙上の確率分布
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_len=5000,
        dropout=0.1,
        src_pad_idx=0,
        tgt_pad_idx=0,
        share_embedding=False,
    ):
        """
        Args:
            src_vocab_size (int): ソース言語の語彙サイズ
            tgt_vocab_size (int): ターゲット言語の語彙サイズ
            d_model (int): モデルの次元数（デフォルト: 512）
            num_heads (int): Attentionヘッド数（デフォルト: 8）
            num_encoder_layers (int): Encoderの層数（デフォルト: 6）
            num_decoder_layers (int): Decoderの層数（デフォルト: 6）
            d_ff (int): FFNの中間層次元（デフォルト: 2048）
            max_len (int): 最大シーケンス長
            dropout (float): ドロップアウト率
            src_pad_idx (int): ソース側のパディングトークンID
            tgt_pad_idx (int): ターゲット側のパディングトークンID
            share_embedding (bool): ソースとターゲットで埋め込みを共有するか
        """
        super().__init__()

        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        # ソース側の埋め込み
        self.src_embedding = nn.Embedding(
            src_vocab_size, d_model, padding_idx=src_pad_idx
        )

        # ターゲット側の埋め込み（共有または独立）
        if share_embedding and src_vocab_size == tgt_vocab_size:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(
                tgt_vocab_size, d_model, padding_idx=tgt_pad_idx
            )

        # Position Encoding（ソースとターゲットで共有）
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder
        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
        )

        # Decoder
        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
        )

        # 出力層（語彙サイズへの射影）
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # パラメータの初期化
        self._init_parameters()

    def _init_parameters(self):
        """パラメータをXavier初期化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        """
        ソース側のパディングマスクを作成

        Args:
            src: ソーストークンID [batch_size, src_len]

        Returns:
            マスク [batch_size, 1, src_len]
            True = 有効なトークン, False = パディング
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1)
        return src_mask

    def make_tgt_mask(self, tgt):
        """
        ターゲット側のマスクを作成（Causal + パディング）

        Args:
            tgt: ターゲットトークンID [batch_size, tgt_len]

        Returns:
            マスク [batch_size, tgt_len, tgt_len]
        """
        tgt_len = tgt.size(1)

        # 1. Causal Mask（未来を見ない）
        causal_mask = generate_causal_mask(tgt_len, tgt.device)

        # 2. パディングマスク
        pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1)  # [batch, 1, tgt_len]

        # 3. 結合
        tgt_mask = causal_mask.unsqueeze(0) & pad_mask  # [batch, tgt_len, tgt_len]

        return tgt_mask

    def encode(self, src, src_mask=None):
        """
        ソースシーケンスをエンコード

        Args:
            src: ソーストークンID [batch_size, src_len]
            src_mask: ソースマスク（オプション）

        Returns:
            Encoder出力 [batch_size, src_len, d_model]
        """
        # 埋め込み（スケーリング付き）
        src_embedded = self.src_embedding(src) * (self.d_model ** 0.5)

        # Position Encodingを加算
        src_embedded = self.pos_encoding(src_embedded)

        # Encoder
        encoder_output = self.encoder(src_embedded, src_mask)

        return encoder_output

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        ターゲットシーケンスをデコード

        Args:
            tgt: ターゲットトークンID [batch_size, tgt_len]
            encoder_output: Encoder出力 [batch_size, src_len, d_model]
            src_mask: ソースマスク（オプション）
            tgt_mask: ターゲットマスク（オプション）

        Returns:
            Decoder出力 [batch_size, tgt_len, d_model]
        """
        # 埋め込み（スケーリング付き）
        tgt_embedded = self.tgt_embedding(tgt) * (self.d_model ** 0.5)

        # Decoder（Position EncodingはDecoder内部で適用）
        decoder_output = self.decoder(
            tgt_embedded, encoder_output, src_mask, tgt_mask
        )

        return decoder_output

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Transformerの順伝播

        Args:
            src: ソーストークンID [batch_size, src_len]
            tgt: ターゲットトークンID [batch_size, tgt_len]
            src_mask: ソースマスク（オプション、自動生成可能）
            tgt_mask: ターゲットマスク（オプション、自動生成可能）

        Returns:
            出力logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # マスクの自動生成
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)

        # Encode
        encoder_output = self.encode(src, src_mask)

        # Decode
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # 出力層
        logits = self.output_projection(decoder_output)

        return logits

    def generate(
        self,
        src,
        max_len=50,
        start_token_id=1,
        end_token_id=2,
        temperature=1.0,
        top_k=None,
        top_p=None,
    ):
        """
        自己回帰的にシーケンスを生成（推論時）

        Args:
            src: ソーストークンID [batch_size, src_len]
            max_len: 最大生成長
            start_token_id: 開始トークンのID（<start>）
            end_token_id: 終了トークンのID（<end>）
            temperature: サンプリング温度（1.0=そのまま、<1で確定的、>1でランダム）
            top_k: Top-Kサンプリング（Noneで無効）
            top_p: Top-Pサンプリング/Nucleus（Noneで無効）

        Returns:
            生成されたトークンID [batch_size, generated_len]
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        # ソースをエンコード
        src_mask = self.make_src_mask(src)
        encoder_output = self.encode(src, src_mask)

        # 開始トークンで初期化
        generated = torch.full(
            (batch_size, 1), start_token_id, dtype=torch.long, device=device
        )

        # 終了フラグ
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            # ターゲットマスクを作成
            tgt_mask = self.make_tgt_mask(generated)

            # デコード
            decoder_output = self.decode(
                generated, encoder_output, src_mask, tgt_mask
            )

            # 最後の位置の出力を取得
            logits = self.output_projection(decoder_output[:, -1, :])  # [batch, vocab]

            # 温度でスケーリング
            logits = logits / temperature

            # Top-Kサンプリング
            if top_k is not None:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < threshold,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            # Top-Pサンプリング（Nucleus）
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                    :, :-1
                ].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            # 確率に変換してサンプリング
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 終了トークンをチェック
            finished = finished | (next_token.squeeze(-1) == end_token_id)

            # 生成済みシーケンスに追加
            generated = torch.cat([generated, next_token], dim=1)

            # 全バッチが終了したら終了
            if finished.all():
                break

        return generated

    def greedy_decode(self, src, max_len=50, start_token_id=1, end_token_id=2):
        """
        貪欲法（Greedy）でシーケンスを生成

        Args:
            src: ソーストークンID [batch_size, src_len]
            max_len: 最大生成長
            start_token_id: 開始トークンのID
            end_token_id: 終了トークンのID

        Returns:
            生成されたトークンID [batch_size, generated_len]
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        # ソースをエンコード
        src_mask = self.make_src_mask(src)
        encoder_output = self.encode(src, src_mask)

        # 開始トークンで初期化
        generated = torch.full(
            (batch_size, 1), start_token_id, dtype=torch.long, device=device
        )

        for _ in range(max_len - 1):
            # ターゲットマスクを作成
            tgt_mask = self.make_tgt_mask(generated)

            # デコード
            decoder_output = self.decode(
                generated, encoder_output, src_mask, tgt_mask
            )

            # 最後の位置の出力を取得し、argmaxで次のトークンを決定
            logits = self.output_projection(decoder_output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)

            # 生成済みシーケンスに追加
            generated = torch.cat([generated, next_token], dim=1)

            # 全バッチが終了トークンを生成したら終了
            if (next_token.squeeze(-1) == end_token_id).all():
                break

        return generated


def count_parameters(model):
    """モデルのパラメータ数をカウント"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    batch_size = 2
    src_len = 10
    tgt_len = 8

    print("=" * 70)
    print("Transformer Model Test")
    print("=" * 70)

    # モデルの作成
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    ).to(device)

    print(f"\nModel Parameters: {count_parameters(model):,}")

    # ダミー入力
    src = torch.randint(1, src_vocab_size, (batch_size, src_len)).to(device)
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len)).to(device)

    print(f"\nSource shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")

    # 順伝播
    print("\n" + "=" * 70)
    print("Forward Pass Test")
    print("=" * 70)

    model.train()
    logits = model(src, tgt)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: [batch_size={batch_size}, tgt_len={tgt_len}, vocab_size={tgt_vocab_size}]")

    # 生成テスト
    print("\n" + "=" * 70)
    print("Generation Test (Greedy)")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        generated = model.greedy_decode(src, max_len=20)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens (first batch): {generated[0].tolist()}")

    # パラメータの内訳
    print("\n" + "=" * 70)
    print("Parameter Breakdown")
    print("=" * 70)

    embedding_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "embedding" in name
    )
    encoder_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "encoder" in name and "embedding" not in name
    )
    decoder_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "decoder" in name and "embedding" not in name
    )
    output_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "output_projection" in name
    )

    total = count_parameters(model)
    print(f"Embedding:  {embedding_params:>12,} ({100*embedding_params/total:.1f}%)")
    print(f"Encoder:    {encoder_params:>12,} ({100*encoder_params/total:.1f}%)")
    print(f"Decoder:    {decoder_params:>12,} ({100*decoder_params/total:.1f}%)")
    print(f"Output:     {output_params:>12,} ({100*output_params/total:.1f}%)")
    print(f"{'─'*40}")
    print(f"Total:      {total:>12,}")
