"""
チャット学習用データセット処理

Hugging Faceのdatasetsを使用して日本語対話データをロード・前処理する。
"""

import os
from typing import List, Tuple, Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class ChatDataset(Dataset):
    """
    チャット学習用データセット

    各サンプルは (input, response) のペアで構成される。
    Transformerの学習形式に変換:
        - src: 入力文（ユーザーの発言）
        - tgt_input: <bos> + 応答文[:-1]（Decoder入力）
        - tgt_output: 応答文[1:] + <eos>（教師信号）

    Args:
        inputs: 入力文（ユーザー発言）のリスト
        responses: 応答文のリスト
        tokenizer: JapaneseTokenizerインスタンス
        max_len: 最大シーケンス長
    """

    def __init__(
        self,
        inputs: List[str],
        responses: List[str],
        tokenizer,
        max_len: int = 512
    ):
        self.inputs = inputs
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_len = max_len

        assert len(inputs) == len(responses), \
            f"Length mismatch: {len(inputs)} inputs vs {len(responses)} responses"

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        インデックスに対応するサンプルを返す

        Returns:
            dict with keys:
                - src: 入力トークンID [src_len]
                - tgt_input: デコーダー入力 [tgt_len]
                - tgt_output: 教師信号 [tgt_len]
        """
        input_text = self.inputs[idx]
        response_text = self.responses[idx]

        # 入力をエンコード
        src_ids = self.tokenizer.encode(input_text)
        response_ids = self.tokenizer.encode(response_text)

        # 長さ制限
        src_ids = src_ids[:self.max_len]
        response_ids = response_ids[:self.max_len - 1]  # BOS/EOS分の余裕

        # Decoder入力: <bos> + response
        tgt_input_ids = [self.tokenizer.bos_token_id] + response_ids

        # 教師信号: response + <eos>
        tgt_output_ids = response_ids + [self.tokenizer.eos_token_id]

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt_input": torch.tensor(tgt_input_ids, dtype=torch.long),
            "tgt_output": torch.tensor(tgt_output_ids, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    バッチをパディングして揃える

    Args:
        batch: ChatDatasetから取得したサンプルのリスト
        pad_id: パディングトークンID

    Returns:
        パディング済みのバッチ
    """
    src_list = [item["src"] for item in batch]
    tgt_input_list = [item["tgt_input"] for item in batch]
    tgt_output_list = [item["tgt_output"] for item in batch]

    # パディング
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_id)
    tgt_input_padded = pad_sequence(tgt_input_list, batch_first=True, padding_value=pad_id)
    tgt_output_padded = pad_sequence(tgt_output_list, batch_first=True, padding_value=pad_id)

    return {
        "src": src_padded,
        "tgt_input": tgt_input_padded,
        "tgt_output": tgt_output_padded,
    }


def create_dataloader(
    dataset: ChatDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_id: int = 0
) -> DataLoader:
    """
    DataLoaderを作成

    Args:
        dataset: ChatDatasetインスタンス
        batch_size: バッチサイズ
        shuffle: シャッフルするか
        num_workers: ワーカー数
        pad_id: パディングトークンID

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id)
    )


def load_dolly_ja(
    max_samples: Optional[int] = None,
    split: str = "train"
) -> Tuple[List[str], List[str]]:
    """
    kunishou/databricks-dolly-15k-ja データセットをロード

    Args:
        max_samples: 最大サンプル数（Noneで全件）
        split: データ分割

    Returns:
        (inputs, responses) のタプル
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading kunishou/databricks-dolly-15k-ja...")
    dataset = load_dataset("kunishou/databricks-dolly-15k-ja", split=split)

    inputs = []
    responses = []

    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        # instructionを入力、outputを応答として使用
        instruction = item.get("instruction", "")
        context = item.get("input", "")
        output = item.get("output", "")

        # コンテキストがある場合は連結
        if context:
            input_text = f"{instruction}\n\n{context}"
        else:
            input_text = instruction

        if input_text and output:
            inputs.append(input_text)
            responses.append(output)

    print(f"Loaded {len(inputs)} samples")
    return inputs, responses


def load_japanese_chat_data(
    dataset_name: str = "kunishou/databricks-dolly-15k-ja",
    max_samples: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """
    日本語チャットデータをロード

    Args:
        dataset_name: データセット名
        max_samples: 最大サンプル数

    Returns:
        (inputs, responses) のタプル
    """
    if dataset_name == "kunishou/databricks-dolly-15k-ja":
        return load_dolly_ja(max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def prepare_chat_data(
    tokenizer,
    dataset_name: str = "kunishou/databricks-dolly-15k-ja",
    max_samples: Optional[int] = None,
    max_len: int = 512,
    batch_size: int = 32,
    train_ratio: float = 0.9
) -> Tuple[DataLoader, DataLoader]:
    """
    チャットデータを準備してDataLoaderを返す

    Args:
        tokenizer: JapaneseTokenizerインスタンス
        dataset_name: データセット名
        max_samples: 最大サンプル数
        max_len: 最大シーケンス長
        batch_size: バッチサイズ
        train_ratio: 訓練データの割合

    Returns:
        (train_loader, val_loader) のタプル
    """
    # データをロード
    inputs, responses = load_japanese_chat_data(
        dataset_name=dataset_name,
        max_samples=max_samples
    )

    # 訓練/検証に分割
    split_idx = int(len(inputs) * train_ratio)

    train_inputs = inputs[:split_idx]
    train_responses = responses[:split_idx]
    val_inputs = inputs[split_idx:]
    val_responses = responses[split_idx:]

    print(f"Train samples: {len(train_inputs)}")
    print(f"Validation samples: {len(val_inputs)}")

    # データセット作成
    train_dataset = ChatDataset(
        train_inputs, train_responses, tokenizer, max_len=max_len
    )
    val_dataset = ChatDataset(
        val_inputs, val_responses, tokenizer, max_len=max_len
    )

    # DataLoader作成
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pad_id=tokenizer.pad_token_id
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pad_id=tokenizer.pad_token_id
    )

    return train_loader, val_loader


def get_all_texts_for_tokenizer(
    dataset_name: str = "kunishou/databricks-dolly-15k-ja",
    max_samples: Optional[int] = None
) -> List[str]:
    """
    トークナイザー学習用に全テキストを取得

    Args:
        dataset_name: データセット名
        max_samples: 最大サンプル数

    Returns:
        全テキストのリスト
    """
    inputs, responses = load_japanese_chat_data(
        dataset_name=dataset_name,
        max_samples=max_samples
    )

    # 入力と応答を連結
    all_texts = inputs + responses
    print(f"Total texts for tokenizer training: {len(all_texts)}")
    return all_texts


if __name__ == "__main__":
    # テスト用
    from tokenizer import JapaneseTokenizer

    # サンプルデータ
    sample_inputs = [
        "こんにちは",
        "今日の天気は？",
        "Transformerについて教えて",
    ]
    sample_responses = [
        "こんにちは！何かお手伝いできることはありますか？",
        "今日は晴れです。気温は20度くらいでしょう。",
        "Transformerは、Attention機構を使ったニューラルネットワークです。",
    ]

    # トークナイザーを学習
    all_texts = sample_inputs + sample_responses
    tokenizer = JapaneseTokenizer.train(
        texts=all_texts,
        vocab_size=500,
        model_prefix="test_tokenizer",
        save_dir="models/test_tokenizer"
    )

    # データセット作成
    dataset = ChatDataset(
        sample_inputs, sample_responses, tokenizer, max_len=64
    )

    print(f"\nDataset size: {len(dataset)}")

    # サンプル取得
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  src: {sample['src'].tolist()}")
    print(f"  tgt_input: {sample['tgt_input'].tolist()}")
    print(f"  tgt_output: {sample['tgt_output'].tolist()}")

    # デコード確認
    print(f"\n  src decoded: {tokenizer.decode(sample['src'])}")
    print(f"  tgt decoded: {tokenizer.decode(sample['tgt_output'])}")

    # DataLoader テスト
    loader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        pad_id=tokenizer.pad_token_id
    )

    for batch in loader:
        print(f"\nBatch shapes:")
        print(f"  src: {batch['src'].shape}")
        print(f"  tgt_input: {batch['tgt_input'].shape}")
        print(f"  tgt_output: {batch['tgt_output'].shape}")
        break
