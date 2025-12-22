# Transformer学習ログ

## 2025年12月22日

### 完了した内容

#### QandA分割
巨大化したQandA.ipynbを4つのファイルに分割：
- `QandA_01_attention.ipynb`: Attention基礎 (Q1-Q4)
- `QandA_02_multihead.ipynb`: Multi-Head Attention (Q5-Q14)
- `QandA_03_architecture.ipynb`: アーキテクチャ (Q15-Q28)
- `QandA_04_experiments.ipynb`: 実験・応用

#### コーディングスタイル修正
- ✅ `super(ClassName, self)` → `super()` に統一（Python 3スタイル）
- 全ファイル（attention.py, encoder.py, feed_forward.py, position_encoding.py）を修正
- Jupyterのモジュールキャッシュ問題を回避

#### 新規Q&A追加（QandA_03_architecture.ipynb）
- **Q21**: Add & Normの「Add」とは何か（残差接続）
- **Q22**: なぜ「残差」と呼ぶのか、英語では？（Residual Connection）
- **Q23**: ResNetとは何か
- **Q24**: Encoder出力の意味とDecoderでの使われ方（Cross-Attention）
- **Q25**: Encoderは複数のEncoder Layerが直列接続されているのか
- **Q26**: 「文脈を考慮した表現」とは具体的に何か
- **Q27**: 同じ単語が複数回出現する場合の区別方法
- **Q28**: 複数の特徴量ベクトルが1つの単語に結びつくのか

#### 学習ポイント
- **残差接続**: ResNet由来、`F(x) = H(x) - x`という差分を学習
- **文脈表現**: 同じ単語でも周囲の単語によって異なるベクトルになる
- **Encoder→Decoder**: Cross-AttentionでK, Vとして使用、勾配も逆伝播
- **出力変換**: 多対一の写像（異なるベクトル→同じ単語）

### 次のステップ
- [ ] Decoder
- [ ] Cross-Attention
- [ ] 完全なTransformerモデル

---

## 2025年12月21日（続き）

### 完了した内容

#### 実装
- ✅ Encoder（エンコーダー）の実装 (`src/encoder.py`)
  - LayerNorm: 層正規化（学習の安定化）
  - ResidualConnection: 残差接続（勾配の流れを改善）
  - EncoderLayer: 1層分のEncoder（Attention + FFN）
  - Encoder: N層のEncoderLayer
  - TransformerEncoder: 埋め込み層を含む完全版
- ✅ 05_encoder_demo.ipynb 完了

#### 学習ポイント
- **Layer Normalization**: 特徴次元に沿って正規化、バッチサイズに依存しない
- **Residual Connection**: 入力をそのまま出力に加算、勾配のショートカット
- **Encoder構造**: [Self-Attention → Add & Norm → FFN → Add & Norm] × N層
- **パラメータ数**: 語彙サイズが大きいとEmbedding層が支配的

---

## 2025年12月21日

### 完了した内容

#### 実装
- ✅ Feed Forward Network（FFN）の実装 (`src/feed_forward.py`)
  - 標準FFN（ReLU/GELU）
  - Gated FFN（SwiGLU、LLaMA/PaLM方式）
- ✅ 04_feed_forward_demo.ipynb 完了

#### 学習ポイント
- FFNの役割（Attentionは線形、FFNで非線形性を導入）
- Position-wise（各トークンに独立に適用）
- 中間層の拡大（d_model → 4×d_model → d_model）
- 活性化関数の比較（ReLU vs GELU vs SwiGLU）
- パラメータ配分（FFNがTransformer層の約2/3を占める）

#### 学習記録（QandA.ipynb）
- **Q19**: FFNの数式は「フィードフォワード」を示しているのか？
  - 中身は2層MLP（線形→活性化→線形）
  - 「フィードフォワード」は情報の流れ方（一方向、再帰なし）を表す用語
  - Attentionと区別するための名前（トークン間の相互作用なし）
- **Q20**: 「1層あたりのパラメータ配分」の「1層」とは？
  - Attention + FFN のセット（= 1つのEncoder/Decoderブロック）
  - Transformerは複数の層を積み重ねた構造（元論文では6層）

---

## 2025年12月20日

### 完了した内容

#### 実装
- ✅ Position Encoding（位置エンコーディング）の実装 (`src/position_encoding.py`)
  - Sinusoidal Positional Encoding（論文オリジナル）
  - Learned Positional Encoding（BERT/GPT方式）
- ✅ 03_position_encoding_demo.ipynb 完了

#### 学習記録（QandA.ipynb）
- **Q17**: Position Encodingは何次元を使うのか？
  - d_model次元すべてを使用
  - 複数周波数の組み合わせで位置を一意に特定（二進数と類似）
  - トークン埋め込みに加算して位置情報を重畳
- **Q18**: Position Encodingは全要素に加法的に加わるのか？
  - はい、element-wiseに加算
  - 連結(concat)ではなく加算を使う理由（効率性、十分な性能）

#### その他の理解ポイント
- なぜ位置情報が必要か（Transformerは並列処理のため位置情報が失われる）
- Sin/Cosを使う理由（相対位置が線形変換・回転で表現可能）
- 周波数の意味（低次元=高周波、高次元=低周波）
- Sinusoidal vs Learned の比較

---

## 2025年11月19日

### 完了した内容

#### 実装
- ✅ Self-Attention機構の実装 (`src/attention.py`)
- ✅ Multi-Head Attention機構の実装 (`src/attention.py`)
- ✅ 01_self_attention_demo.ipynb 完了
- ✅ 02_multi_head_attention_demo.ipynb 完了

#### 学習記録（QandA.ipynb）
質問と回答を以下の通り記録：

- **Q1**: nn.Linearとは？
- **Q2**: 線形変換におけるバイアスの役割
- **Q3**: Softmaxの役割
- **Q4**: Self-Attentionで使われているテンソルの各次元の意味
- **Q5**: なぜsqrt(d_k)でスケーリングするのか
- **Q6**: masked_fillの-1e9の意味
- **Q7**: Attention重みの可視化の意味
- **Q8**: transpose(-2, -1)の意味
- **Q9**: Attentionにバイアスを使わない理由
- **Q10**: bias=Falseがベストプラクティスの理由
- **Q11**: Concatとは何か
- **Q12**: なぜQ, K, Vに線形変換を通すのか
- **Q13**: Multi-Head AttentionでさらにQ, K, Vを線形変換するのはなぜ？
- **Q14**: Multi-Head AttentionのQ, K, Vも入力の線形変換と考えて良いか？
- **Q15**: Attentionは本質的にQ, K, Vの関数であり、入力xは必要ない？
- **Q16**: Attention WeightsとVの積で、行と列のどちらに意味があるのか？

### 重要な理解ポイント

1. **Attention機構の本質**
   - Attentionは`Attention(Q, K, V)`というQ, K, Vの純粋な関数
   - 入力xは、Q, K, Vを生成する手段の一つ（Self-Attentionの場合）
   - Cross-Attentionでは異なる入力からQ, K, Vを生成

2. **Multi-Head Attentionの構造**
   - 1つの大きな線形変換で全headをまとめて処理
   - 各headは出力の異なる部分を担当（独立したパラメータ）
   - パラメータ数はSingle-Headと同じ

3. **行列計算の理解**
   - Transformerでは「行」に意味がある（各行が1つのトークン）
   - バッチ処理のため、行ベクトル形式を採用
   - 通常の列ベクトル形式とは転置の関係

### 次回の学習予定

#### 次のステップ: Position Encoding
- [ ] Position Encodingの理論と実装
- [ ] なぜ位置情報が必要か
- [ ] Sin/Cos関数を使った位置エンコーディング
- [ ] 03_position_encoding_demo.ipynb の作成

#### その後の予定
- [ ] Feed Forward Network
- [ ] Encoderブロックの構築
- [ ] Decoderブロックの構築
- [ ] 完全なTransformerモデル
- [ ] テキスト生成の実装

### 参考資料
- 論文: "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
- src/attention.py: 実装済みのコード
- QandA.ipynb: 全ての質問と回答

### メモ
- 理解は順調に進んでいる
- 各質問への回答で理論と実装の両方を確認済み
- 次は位置情報をどう扱うかが重要なテーマ
