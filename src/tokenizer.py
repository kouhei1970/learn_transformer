"""
日本語対応トークナイザー

SentencePieceを使用した日本語トークナイザーの実装。
学習済みモデルの読み込みと、データからの新規学習の両方をサポート。
"""

import os
import tempfile
from typing import List, Optional, Union
import sentencepiece as spm


class JapaneseTokenizer:
    """
    日本語対応のSentencePieceトークナイザー

    特殊トークン:
        - <pad> (id=0): パディング
        - <bos> (id=1): 文頭 (Beginning of Sentence)
        - <eos> (id=2): 文末 (End of Sentence)
        - <unk> (id=3): 未知語 (Unknown)

    使用例:
        # 既存モデルを読み込む場合
        tokenizer = JapaneseTokenizer.from_pretrained("path/to/model.model")

        # 新規学習する場合
        tokenizer = JapaneseTokenizer.train(
            texts=["こんにちは", "今日は良い天気ですね"],
            vocab_size=8000,
            model_prefix="my_tokenizer"
        )

        # エンコード・デコード
        token_ids = tokenizer.encode("こんにちは")
        text = tokenizer.decode(token_ids)
    """

    # 特殊トークンID
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, model_path: str):
        """
        トークナイザーを初期化

        Args:
            model_path: SentencePieceモデルファイルのパス (.model)
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.model_path = model_path

    @classmethod
    def from_pretrained(cls, model_path: str) -> "JapaneseTokenizer":
        """
        学習済みモデルを読み込む

        Args:
            model_path: モデルファイルのパス

        Returns:
            JapaneseTokenizer インスタンス
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return cls(model_path)

    @classmethod
    def train(
        cls,
        texts: List[str],
        vocab_size: int = 8000,
        model_prefix: str = "tokenizer",
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
        save_dir: Optional[str] = None
    ) -> "JapaneseTokenizer":
        """
        テキストデータからトークナイザーを学習

        Args:
            texts: 学習用テキストのリスト
            vocab_size: 語彙サイズ
            model_prefix: モデルファイルの接頭辞
            model_type: モデルタイプ (bpe, unigram, char, word)
            character_coverage: 文字カバー率（日本語は0.9995推奨）
            save_dir: モデル保存ディレクトリ（Noneの場合は一時ディレクトリ）

        Returns:
            JapaneseTokenizer インスタンス
        """
        # 保存ディレクトリの設定
        if save_dir is None:
            save_dir = tempfile.mkdtemp()
        os.makedirs(save_dir, exist_ok=True)

        # テキストを一時ファイルに書き込む
        input_file = os.path.join(save_dir, "train_data.txt")
        with open(input_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text.strip() + "\n")

        # モデルパス
        model_prefix_path = os.path.join(save_dir, model_prefix)

        # SentencePieceモデルを学習
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix_path,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            pad_id=cls.PAD_ID,
            bos_id=cls.BOS_ID,
            eos_id=cls.EOS_ID,
            unk_id=cls.UNK_ID,
            pad_piece="<pad>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            unk_piece="<unk>",
        )

        model_path = model_prefix_path + ".model"
        print(f"Tokenizer trained and saved to: {model_path}")

        return cls(model_path)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False
    ) -> List[int]:
        """
        テキストをトークンIDのリストに変換

        Args:
            text: 入力テキスト
            add_bos: 先頭に<bos>トークンを追加
            add_eos: 末尾に<eos>トークンを追加

        Returns:
            トークンIDのリスト
        """
        token_ids = self.sp.EncodeAsIds(text)

        if add_bos:
            token_ids = [self.BOS_ID] + token_ids
        if add_eos:
            token_ids = token_ids + [self.EOS_ID]

        return token_ids

    def decode(
        self,
        token_ids: Union[List[int], "torch.Tensor"],
        skip_special_tokens: bool = True
    ) -> str:
        """
        トークンIDのリストをテキストに変換

        Args:
            token_ids: トークンIDのリスト
            skip_special_tokens: 特殊トークンをスキップするか

        Returns:
            デコードされたテキスト
        """
        # Tensorの場合はリストに変換
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        if skip_special_tokens:
            # 特殊トークンを除去
            token_ids = [
                tid for tid in token_ids
                if tid not in [self.PAD_ID, self.BOS_ID, self.EOS_ID]
            ]

        return self.sp.DecodeIds(token_ids)

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False
    ) -> List[List[int]]:
        """
        複数のテキストをバッチでエンコード

        Args:
            texts: テキストのリスト
            add_bos: 先頭に<bos>トークンを追加
            add_eos: 末尾に<eos>トークンを追加

        Returns:
            トークンIDのリストのリスト
        """
        return [
            self.encode(text, add_bos=add_bos, add_eos=add_eos)
            for text in texts
        ]

    def decode_batch(
        self,
        batch_token_ids: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        複数のトークンIDリストをバッチでデコード

        Args:
            batch_token_ids: トークンIDリストのリスト
            skip_special_tokens: 特殊トークンをスキップするか

        Returns:
            デコードされたテキストのリスト
        """
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in batch_token_ids
        ]

    def tokenize(self, text: str) -> List[str]:
        """
        テキストをトークン文字列のリストに変換

        Args:
            text: 入力テキスト

        Returns:
            トークン文字列のリスト
        """
        return self.sp.EncodeAsPieces(text)

    @property
    def vocab_size(self) -> int:
        """語彙サイズを返す"""
        return self.sp.GetPieceSize()

    @property
    def pad_token_id(self) -> int:
        """パディングトークンIDを返す"""
        return self.PAD_ID

    @property
    def bos_token_id(self) -> int:
        """BOSトークンIDを返す"""
        return self.BOS_ID

    @property
    def eos_token_id(self) -> int:
        """EOSトークンIDを返す"""
        return self.EOS_ID

    @property
    def unk_token_id(self) -> int:
        """UNKトークンIDを返す"""
        return self.UNK_ID

    def id_to_piece(self, token_id: int) -> str:
        """トークンIDをトークン文字列に変換"""
        return self.sp.IdToPiece(token_id)

    def piece_to_id(self, piece: str) -> int:
        """トークン文字列をトークンIDに変換"""
        return self.sp.PieceToId(piece)

    def save(self, save_dir: str, model_prefix: str = "tokenizer"):
        """
        トークナイザーを保存

        Args:
            save_dir: 保存ディレクトリ
            model_prefix: モデルファイルの接頭辞
        """
        import shutil
        os.makedirs(save_dir, exist_ok=True)

        # モデルファイルをコピー
        dest_path = os.path.join(save_dir, f"{model_prefix}.model")
        shutil.copy(self.model_path, dest_path)

        # vocab ファイルもコピー（存在する場合）
        vocab_path = self.model_path.replace(".model", ".vocab")
        if os.path.exists(vocab_path):
            dest_vocab_path = os.path.join(save_dir, f"{model_prefix}.vocab")
            shutil.copy(vocab_path, dest_vocab_path)

        print(f"Tokenizer saved to: {save_dir}")


def create_tokenizer_from_dataset(
    texts: List[str],
    vocab_size: int = 32000,
    save_dir: str = "models/tokenizer"
) -> JapaneseTokenizer:
    """
    データセットからトークナイザーを作成するヘルパー関数

    Args:
        texts: 学習用テキストのリスト
        vocab_size: 語彙サイズ
        save_dir: 保存ディレクトリ

    Returns:
        JapaneseTokenizer インスタンス
    """
    print(f"Training tokenizer with vocab_size={vocab_size}...")
    print(f"Number of training texts: {len(texts)}")

    tokenizer = JapaneseTokenizer.train(
        texts=texts,
        vocab_size=vocab_size,
        model_prefix="tokenizer",
        model_type="bpe",
        character_coverage=0.9995,
        save_dir=save_dir
    )

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    return tokenizer


if __name__ == "__main__":
    # テスト用のサンプルテキスト
    sample_texts = [
        "こんにちは。今日は良い天気ですね。",
        "Transformerは自然言語処理で広く使われています。",
        "日本語のトークナイザーを作成しています。",
        "機械学習は面白い分野です。",
        "人工知能の発展は目覚ましいものがあります。",
    ]

    # トークナイザーを学習
    tokenizer = JapaneseTokenizer.train(
        texts=sample_texts,
        vocab_size=500,  # サンプルなので小さい語彙サイズ
        model_prefix="test_tokenizer",
        save_dir="models/test_tokenizer"
    )

    # テスト
    test_text = "こんにちは。今日はTransformerについて学びましょう。"

    print(f"\nOriginal text: {test_text}")
    print(f"Tokens: {tokenizer.tokenize(test_text)}")

    token_ids = tokenizer.encode(test_text, add_bos=True, add_eos=True)
    print(f"Token IDs: {token_ids}")

    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: {decoded}")

    print(f"\nVocab size: {tokenizer.vocab_size}")
    print(f"PAD ID: {tokenizer.pad_token_id}")
    print(f"BOS ID: {tokenizer.bos_token_id}")
    print(f"EOS ID: {tokenizer.eos_token_id}")
