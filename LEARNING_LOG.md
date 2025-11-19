# Transformer学習ログ

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
