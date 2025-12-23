"""
チャット推論エンジン

学習済みモデルを使ってテキストベースのチャットを行う。

使用例:
    from src.chat_engine import ChatEngine

    # モデルをロード
    engine = ChatEngine.from_checkpoint("models/chat/best_model.pt")

    # チャット
    response = engine.chat("こんにちは")
    print(response)
"""

import os
import sys
import json
from typing import Optional, List, Dict, Any

import torch

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.transformer import Transformer
from src.tokenizer import JapaneseTokenizer


class ChatEngine:
    """
    チャット推論エンジン

    学習済みのTransformerモデルとトークナイザーを使って、
    テキストベースの対話を行う。

    Attributes:
        model: 学習済みTransformerモデル
        tokenizer: JapaneseTokenizerインスタンス
        device: 計算デバイス（CPU/GPU）
        config: モデル設定
    """

    def __init__(
        self,
        model: Transformer,
        tokenizer: JapaneseTokenizer,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        ChatEngineを初期化

        Args:
            model: 学習済みTransformerモデル
            tokenizer: トークナイザー
            device: デバイス
            config: 設定辞書
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config

        # モデルを評価モードに
        self.model.eval()

        # 会話履歴（オプション）
        self.conversation_history: List[Dict[str, str]] = []

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ) -> "ChatEngine":
        """
        チェックポイントからChatEngineを作成

        Args:
            checkpoint_path: チェックポイントファイルのパス
            device: デバイス（Noneで自動検出）

        Returns:
            ChatEngineインスタンス
        """
        # デバイス
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        print(f"Loading checkpoint from: {checkpoint_path}")
        print(f"Using device: {device}")

        # チェックポイントをロード
        checkpoint = torch.load(checkpoint_path, map_location=device)

        config = checkpoint["config"]
        tokenizer_path = checkpoint["tokenizer_path"]

        # トークナイザーをロード
        print(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = JapaneseTokenizer.from_pretrained(tokenizer_path)

        # モデルを作成
        model = Transformer(
            src_vocab_size=tokenizer.vocab_size,
            tgt_vocab_size=tokenizer.vocab_size,
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_encoder_layers=config["num_layers"],
            num_decoder_layers=config["num_layers"],
            d_ff=config["d_ff"],
            max_len=config["max_len"],
            dropout=config["dropout"],
            src_pad_idx=tokenizer.pad_token_id,
            tgt_pad_idx=tokenizer.pad_token_id,
            share_embedding=True,
        ).to(device)

        # 重みをロード
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print(f"Model loaded successfully (epoch {checkpoint['epoch']})")

        return cls(model, tokenizer, device, config)

    @classmethod
    def from_directory(
        cls,
        model_dir: str,
        model_name: str = "best_model.pt",
        device: Optional[torch.device] = None
    ) -> "ChatEngine":
        """
        ディレクトリからChatEngineを作成

        Args:
            model_dir: モデルディレクトリ
            model_name: モデルファイル名
            device: デバイス

        Returns:
            ChatEngineインスタンス
        """
        checkpoint_path = os.path.join(model_dir, model_name)
        return cls.from_checkpoint(checkpoint_path, device)

    def chat(
        self,
        user_input: str,
        max_len: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        use_greedy: bool = False
    ) -> str:
        """
        ユーザー入力に対して応答を生成

        Args:
            user_input: ユーザーの入力テキスト
            max_len: 最大生成長
            temperature: サンプリング温度
            top_k: Top-Kサンプリング
            top_p: Top-Pサンプリング
            use_greedy: Greedy Decodingを使用するか

        Returns:
            生成された応答テキスト
        """
        # 入力をトークン化
        input_ids = self.tokenizer.encode(user_input)
        src = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # 生成
        with torch.no_grad():
            if use_greedy:
                generated = self.model.greedy_decode(
                    src,
                    max_len=max_len,
                    start_token_id=self.tokenizer.bos_token_id,
                    end_token_id=self.tokenizer.eos_token_id
                )
            else:
                generated = self.model.generate(
                    src,
                    max_len=max_len,
                    start_token_id=self.tokenizer.bos_token_id,
                    end_token_id=self.tokenizer.eos_token_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

        # デコード
        output_ids = generated[0].tolist()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # 会話履歴に追加
        self.conversation_history.append({
            "user": user_input,
            "assistant": response
        })

        return response

    def chat_with_context(
        self,
        user_input: str,
        context_turns: int = 3,
        **kwargs
    ) -> str:
        """
        会話履歴を含めて応答を生成

        Args:
            user_input: ユーザーの入力テキスト
            context_turns: 含める過去の会話ターン数
            **kwargs: chat()に渡す追加引数

        Returns:
            生成された応答テキスト
        """
        # 過去の会話を連結
        context_parts = []
        recent_history = self.conversation_history[-context_turns:]

        for turn in recent_history:
            context_parts.append(f"ユーザー: {turn['user']}")
            context_parts.append(f"アシスタント: {turn['assistant']}")

        context_parts.append(f"ユーザー: {user_input}")
        context_parts.append("アシスタント:")

        full_input = "\n".join(context_parts)

        return self.chat(full_input, **kwargs)

    def clear_history(self):
        """会話履歴をクリア"""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """会話履歴を取得"""
        return self.conversation_history.copy()

    def interactive_chat(
        self,
        greeting: str = "チャットを開始します。終了するには 'quit' と入力してください。",
        prompt: str = "You: ",
        **kwargs
    ):
        """
        インタラクティブなチャットセッションを開始

        Args:
            greeting: 開始時のメッセージ
            prompt: ユーザー入力のプロンプト
            **kwargs: chat()に渡す追加引数
        """
        print(greeting)
        print("-" * 50)

        while True:
            try:
                user_input = input(prompt).strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("チャットを終了します。")
                    break

                if not user_input:
                    continue

                response = self.chat(user_input, **kwargs)
                print(f"Bot: {response}")
                print()

            except KeyboardInterrupt:
                print("\nチャットを終了します。")
                break
            except Exception as e:
                print(f"エラーが発生しました: {e}")

    def generate_multiple(
        self,
        user_input: str,
        num_responses: int = 3,
        **kwargs
    ) -> List[str]:
        """
        複数の応答候補を生成

        Args:
            user_input: ユーザーの入力テキスト
            num_responses: 生成する応答数
            **kwargs: chat()に渡す追加引数

        Returns:
            応答テキストのリスト
        """
        responses = []
        for _ in range(num_responses):
            response = self.chat(user_input, **kwargs)
            responses.append(response)
            # 履歴から削除（重複を避けるため）
            self.conversation_history.pop()

        return responses


def demo_chat():
    """デモ用のチャット（モデルがない場合のテスト）"""
    print("=" * 60)
    print("Chat Engine Demo (without trained model)")
    print("=" * 60)

    # ダミーのトークナイザーとモデルを作成
    sample_texts = [
        "こんにちは",
        "今日は良い天気ですね",
        "Transformerについて教えて",
        "ありがとうございます",
    ]

    print("\nTraining demo tokenizer...")
    tokenizer = JapaneseTokenizer.train(
        texts=sample_texts,
        vocab_size=500,
        model_prefix="demo_tokenizer",
        save_dir="models/demo"
    )

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # デモ用のモデル（未学習）
    device = torch.device("cpu")

    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=64,
        num_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        max_len=64,
        dropout=0.1,
    ).to(device)

    config = {
        "d_model": 64,
        "num_heads": 2,
        "num_layers": 2,
        "d_ff": 128,
        "max_len": 64,
        "dropout": 0.1,
    }

    engine = ChatEngine(model, tokenizer, device, config)

    print("\nNote: This is an untrained model, so responses will be random.")
    print("-" * 60)

    # テスト
    test_input = "こんにちは"
    print(f"Input: {test_input}")

    response = engine.chat(test_input, max_len=20, use_greedy=True)
    print(f"Response (untrained): {response}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chat Engine")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo with untrained model"
    )

    args = parser.parse_args()

    if args.demo or args.model is None:
        demo_chat()
    else:
        # 学習済みモデルでチャット
        engine = ChatEngine.from_checkpoint(args.model)
        engine.interactive_chat()
