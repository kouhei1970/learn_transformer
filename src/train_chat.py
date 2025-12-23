"""
チャットモデル学習スクリプト

自作Transformerを使って日本語チャットモデルを学習する。

使用例:
    python -m src.train_chat --epochs 100 --batch_size 32

    または

    python src/train_chat.py --epochs 100 --batch_size 32
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.transformer import Transformer, count_parameters
from src.tokenizer import JapaneseTokenizer, create_tokenizer_from_dataset
from src.dataset import (
    prepare_chat_data,
    get_all_texts_for_tokenizer,
    load_japanese_chat_data
)


def get_device():
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA GPU ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    return device


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    clip_grad=1.0
):
    """1エポック分の学習"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        src = batch["src"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)

        # 順伝播
        logits = model(src, tgt_input)

        # 損失計算（パディングを無視）
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )

        # 逆伝播
        optimizer.zero_grad()
        loss.backward()

        # 勾配クリッピング
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        # 統計
        total_loss += loss.item() * src.size(0)

        # 精度計算（パディングを除く）
        predictions = logits.argmax(dim=-1)
        mask = tgt_output != 0  # PAD_ID = 0
        correct = ((predictions == tgt_output) & mask).sum().item()
        tokens = mask.sum().item()
        total_correct += correct
        total_tokens += tokens

        # プログレスバー更新
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct/tokens*100:.1f}%"
        })

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device):
    """検証データで評価"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            src = batch["src"].to(device)
            tgt_input = batch["tgt_input"].to(device)
            tgt_output = batch["tgt_output"].to(device)

            # 順伝播
            logits = model(src, tgt_input)

            # 損失計算
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )

            total_loss += loss.item() * src.size(0)

            # 精度計算
            predictions = logits.argmax(dim=-1)
            mask = tgt_output != 0
            correct = ((predictions == tgt_output) & mask).sum().item()
            tokens = mask.sum().item()
            total_correct += correct
            total_tokens += tokens

    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    return avg_loss, accuracy


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    tokenizer_path,
    config,
    save_path
):
    """チェックポイントを保存"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "tokenizer_path": tokenizer_path,
        "config": config,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(model, optimizer, load_path, device):
    """チェックポイントを読み込み"""
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")
    return epoch, loss


def train(
    # データ設定
    dataset_name="kunishou/databricks-dolly-15k-ja",
    max_samples=None,
    max_len=256,

    # モデル設定
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    dropout=0.1,
    vocab_size=32000,

    # 学習設定
    epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    warmup_steps=1000,
    clip_grad=1.0,

    # 保存設定
    save_dir="models/chat",
    save_every=10,

    # その他
    resume_from=None,
):
    """
    チャットモデルを学習

    Args:
        dataset_name: 使用するデータセット名
        max_samples: 最大サンプル数（Noneで全件）
        max_len: 最大シーケンス長
        d_model: モデル次元
        num_heads: Attentionヘッド数
        num_layers: Encoder/Decoderの層数
        d_ff: FFN中間層次元
        dropout: ドロップアウト率
        vocab_size: 語彙サイズ
        epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        warmup_steps: ウォームアップステップ数
        clip_grad: 勾配クリッピングの閾値
        save_dir: モデル保存ディレクトリ
        save_every: 何エポックごとに保存するか
        resume_from: 再開するチェックポイントのパス
    """
    # デバイス
    device = get_device()

    # 保存ディレクトリ作成
    os.makedirs(save_dir, exist_ok=True)
    tokenizer_dir = os.path.join(save_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    # 設定を保存
    config = {
        "dataset_name": dataset_name,
        "max_samples": max_samples,
        "max_len": max_len,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "d_ff": d_ff,
        "dropout": dropout,
        "vocab_size": vocab_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "clip_grad": clip_grad,
    }

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"Config saved: {config_path}")

    # トークナイザーの準備
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.model")

    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = JapaneseTokenizer.from_pretrained(tokenizer_path)
    else:
        print("Training new tokenizer...")
        all_texts = get_all_texts_for_tokenizer(
            dataset_name=dataset_name,
            max_samples=max_samples
        )
        tokenizer = create_tokenizer_from_dataset(
            texts=all_texts,
            vocab_size=vocab_size,
            save_dir=tokenizer_dir
        )

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # データローダーの準備
    print("\nPreparing data loaders...")
    train_loader, val_loader = prepare_chat_data(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        max_samples=max_samples,
        max_len=max_len,
        batch_size=batch_size,
        train_ratio=0.9
    )

    # モデルの作成
    print("\nCreating model...")
    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout,
        src_pad_idx=tokenizer.pad_token_id,
        tgt_pad_idx=tokenizer.pad_token_id,
        share_embedding=True,  # 同じ語彙なので共有
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # 最適化
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # 学習率スケジューラ
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # 損失関数（パディングを無視）
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # チェックポイントから再開
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        start_epoch, _ = load_checkpoint(model, optimizer, resume_from, device)
        start_epoch += 1  # 次のエポックから開始

    # 学習ログ
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # 学習ループ
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        # 学習
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, clip_grad
        )

        # 検証
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # スケジューラ更新
        scheduler.step()

        # ログ記録
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        print(f"Time: {epoch_time:.1f}s, LR: {scheduler.get_last_lr()[0]:.2e}")

        # ベストモデルを保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(save_dir, "best_model.pt")
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                tokenizer_path, config, best_path
            )

        # 定期的に保存
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                tokenizer_path, config, checkpoint_path
            )

    # 最終モデルを保存
    final_path = os.path.join(save_dir, "final_model.pt")
    save_checkpoint(
        model, optimizer, epochs - 1, val_loss,
        tokenizer_path, config, final_path
    )

    # 学習履歴を保存
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {save_dir}")
    print("=" * 60)

    return model, tokenizer, history


def main():
    parser = argparse.ArgumentParser(description="Train chat model")

    # データ設定
    parser.add_argument(
        "--dataset", type=str, default="kunishou/databricks-dolly-15k-ja",
        help="Dataset name"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples (None for all)"
    )
    parser.add_argument(
        "--max_len", type=int, default=256,
        help="Maximum sequence length"
    )

    # モデル設定
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")

    # 学習設定
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping")

    # 保存設定
    parser.add_argument("--save_dir", type=str, default="models/chat", help="Save directory")
    parser.add_argument("--save_every", type=int, default=10, help="Save every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    train(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        max_len=args.max_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        vocab_size=args.vocab_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clip_grad=args.clip_grad,
        save_dir=args.save_dir,
        save_every=args.save_every,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
