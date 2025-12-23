"""
チャットWeb UI

Gradioを使用したチャットインターフェース。

使用方法:
    # 学習済みモデルを使用
    python chat_app.py --model models/chat/best_model.pt

    # デモモード（未学習モデル）
    python chat_app.py --demo
"""

import os
import sys
import argparse

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import torch

try:
    import gradio as gr
    GRADIO_VERSION = int(gr.__version__.split('.')[0])
except ImportError:
    print("Gradio is not installed. Please run: pip install gradio")
    sys.exit(1)

from src.chat_engine import ChatEngine
from src.transformer import Transformer
from src.tokenizer import JapaneseTokenizer


# グローバル変数（Gradioコールバック用）
chat_engine = None


def create_demo_engine():
    """デモ用のエンジンを作成（未学習モデル）"""
    # デモ用にもう少し多くのテキストを用意
    sample_texts = [
        "こんにちは",
        "今日は良い天気ですね",
        "Transformerについて教えて",
        "ありがとうございます",
        "日本語のチャットボットです",
        "機械学習は面白いです",
        "自然言語処理を学んでいます",
        "深層学習のモデルを作っています",
        "Attention機構について説明します",
        "Encoderは入力を処理します",
        "Decoderは出力を生成します",
        "ニューラルネットワークの学習",
        "パラメータを最適化します",
        "勾配降下法で学習します",
        "損失関数を最小化します",
        "バッチサイズは32です",
        "エポック数は100です",
        "学習率を調整します",
        "過学習を防ぎます",
        "データセットを準備します",
    ]

    # トークナイザーを学習
    tokenizer_dir = os.path.join(project_root, "models", "demo")
    os.makedirs(tokenizer_dir, exist_ok=True)

    # 既存のトークナイザーがあれば読み込む
    tokenizer_path = os.path.join(tokenizer_dir, "demo_tokenizer.model")
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = JapaneseTokenizer.from_pretrained(tokenizer_path)
    else:
        # 語彙サイズをテキスト量に合わせて小さく
        tokenizer = JapaneseTokenizer.train(
            texts=sample_texts,
            vocab_size=150,  # 小さい語彙サイズ
            model_prefix="demo_tokenizer",
            save_dir=tokenizer_dir
        )

    # デバイス
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # モデル（小さめ、未学習）
    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        max_len=128,
        dropout=0.1,
    ).to(device)

    config = {
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,
        "d_ff": 256,
        "max_len": 128,
        "dropout": 0.1,
    }

    return ChatEngine(model, tokenizer, device, config)


def chat_response(message, history, temperature, top_k, top_p, use_greedy):
    """
    チャットの応答を生成

    Args:
        message: ユーザーのメッセージ
        history: 会話履歴
        temperature: サンプリング温度
        top_k: Top-K
        top_p: Top-P
        use_greedy: Greedy Decodingを使用

    Returns:
        応答テキスト
    """
    global chat_engine

    if chat_engine is None:
        return "エラー: モデルがロードされていません。"

    try:
        response = chat_engine.chat(
            message,
            max_len=100,
            temperature=temperature,
            top_k=int(top_k) if top_k > 0 else None,
            top_p=top_p if top_p < 1.0 else None,
            use_greedy=use_greedy
        )
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"エラーが発生しました: {str(e)}"


def clear_history():
    """会話履歴をクリア"""
    global chat_engine
    if chat_engine:
        chat_engine.clear_history()
    return []


def create_interface():
    """Gradioインターフェースを作成"""

    with gr.Blocks(title="自作Transformer チャット") as demo:

        gr.Markdown("""
        # 自作Transformer チャット

        スクラッチで実装したTransformerモデルを使ったチャットボットです。

        **注意**: このモデルは教育目的で作成されており、
        商用LLMほどの品質は期待できません。
        """)

        with gr.Row():
            with gr.Column(scale=3):
                # Gradioバージョンに応じてChatbotを作成
                chatbot = gr.Chatbot(
                    label="チャット",
                    height=400,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="メッセージを入力",
                        placeholder="こんにちは...",
                        scale=4
                    )
                    send_btn = gr.Button("送信", scale=1, variant="primary")

                clear_btn = gr.Button("会話をクリア")

            with gr.Column(scale=1):
                gr.Markdown("### 生成パラメータ")

                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature",
                    info="高いほどランダム"
                )

                top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Top-K",
                    info="0で無効"
                )

                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-P (Nucleus)",
                    info="1.0で無効"
                )

                use_greedy = gr.Checkbox(
                    label="Greedy Decoding",
                    value=False,
                    info="チェックするとサンプリングを無効化"
                )

                gr.Markdown("""
                ### 使い方
                - **Temperature**: 低いと確定的、高いとランダム
                - **Top-K**: 確率上位K個のトークンのみ使用
                - **Top-P**: 累積確率がPになるまでのトークンを使用
                - **Greedy**: 常に最も確率の高いトークンを選択
                """)

        # イベントハンドラ
        def respond(message, chat_history, temp, k, p, greedy):
            if not message.strip():
                return "", chat_history

            response = chat_response(message, chat_history, temp, k, p, greedy)
            # Gradio 5/6の新形式: role/contentを持つ辞書のリスト
            chat_history = chat_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
            return "", chat_history

        # 送信ボタン
        send_btn.click(
            respond,
            [msg, chatbot, temperature, top_k, top_p, use_greedy],
            [msg, chatbot]
        )

        # Enterキー
        msg.submit(
            respond,
            [msg, chatbot, temperature, top_k, top_p, use_greedy],
            [msg, chatbot]
        )

        # クリアボタン
        def clear_chat():
            clear_history()
            return []

        clear_btn.click(clear_chat, outputs=chatbot)

        # モデル情報
        gr.Markdown("""
        ---
        ### モデル情報

        このチャットボットは以下のコンポーネントで構成されています:

        - **Transformer**: Encoder-Decoder構造
        - **トークナイザー**: SentencePiece (日本語対応)
        - **学習データ**: 日本語対話データセット

        ソースコード: `learn_transformer/` プロジェクト
        """)

    return demo


def main():
    global chat_engine

    parser = argparse.ArgumentParser(description="Chat Web UI")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with demo (untrained) model"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public link"
    )

    args = parser.parse_args()

    print(f"Gradio version: {gr.__version__}")

    # モデルをロード
    if args.demo or args.model is None:
        print("Running in demo mode (untrained model)")
        print("Note: Responses will be random/nonsensical")
        chat_engine = create_demo_engine()
    else:
        if not os.path.exists(args.model):
            print(f"Error: Model not found at {args.model}")
            print("Use --demo to run with an untrained model")
            sys.exit(1)

        print(f"Loading model from: {args.model}")
        chat_engine = ChatEngine.from_checkpoint(args.model)

    # インターフェースを作成して起動
    demo = create_interface()

    print(f"\nStarting server on port {args.port}...")
    demo.launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
