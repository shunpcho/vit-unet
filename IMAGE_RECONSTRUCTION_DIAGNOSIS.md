# ViT-UNet 画像再構成の問題診断と解決履歴

## 発見された主要な問題と解決策

### 1. **スキップ接続の Value（V）が間違っていた**

**場所**: [model.py#L673](src/vit_unet/models/model.py#L673)

**問題**:

```python
# 修正前
self.SkipConnections[...](encoder_skip[skip_idx], x_patch, x_patch)
#                         Q: encoder          K: decoder   V: decoder ❌
```

**修正**:

```python
# 修正後
self.SkipConnections[...](encoder_skip[skip_idx], x_patch, encoder_skip[skip_idx])
#                         Q: encoder          K: decoder   V: encoder ✓
```

**影響**:

- Attention メカニズムで、Value がデコーダ自身になっていたため、エンコーダからの情報が効果的に伝わっていなかった
- スキップ接続の本来の目的（エンコーダの高解像度情報をデコーダに伝える）が機能していなかった

---

### 2. **出力層の Sigmoid 飽和問題**

**場所**: [model.py#L630-647](src/vit_unet/models/model.py#L630-647)

#### 初期の問題: 情報損失

- デコーダの最終出力は正規化された特徴空間（平均 0、標準偏差 1）
- これを直接 clamp すると、56.83%の値が 0 に、3.08%が 1 に飽和
- 情報の大幅な損失が発生

#### 第 1 回修正: Sigmoid の追加

```python
self.output_conv2d = torch.nn.Sequential(
    torch.nn.Conv2d(3, 3, 3, padding="same"),
    torch.nn.ReLU(inplace=True),
    torch.nn.Conv2d(3, 3, 1),
    torch.nn.Sigmoid()  # 正規化された特徴を[0,1]に変換
)
```

**結果**:

- ✓ 情報損失は解消（100%の値が正常範囲）
- ❌ **新しい問題発生**: Sigmoid 入力が 0 付近に集中（平均 0.05）し、出力が全て 0.5 付近（薄紫の一様な画像）に

#### 第 2 回修正: 残差接続の導入

**診断結果**:

- Sigmoid の前の値が 0 付近に集中（60%以上が[-0.5, 0.5]）
- Sigmoid(0) ≈ 0.5 のため、すべてのピクセルが 0.5 付近に飽和
- デノイジングタスクでは出力 ≈ 入力なので、残差学習が適切

**最終的な解決策**:

```python
# Sigmoidを削除し、残差（修正量）を予測
self.output_conv2d = torch.nn.Sequential(
    torch.nn.Conv2d(3, 3, 3, padding="same"),
    torch.nn.ReLU(inplace=True),
    torch.nn.Conv2d(3, 3, 1),
    # 活性化なし - 残差は正負どちらも可能
)

# 最終処理で残差を加算
x_restored = x_input + scale * residual
x_restored = torch.clamp(x_restored, 0.0, 1.0)
```

**影響**:

- **第 1 回修正後**: 薄紫の一様な画像（平均 0.51、標準偏差 0.15）
- **第 2 回修正後**: 入力の統計を保持（平均 0.26→0.27、標準偏差 0.24→0.25）

---

### 3. **残差スケーリング係数の問題**

#### 問題: 固定スケーリングでは不十分

```python
# 固定スケーリング（第2回修正）
x_restored = x_input + 0.1 * residual
```

**問題点**:

- データセットごとにノイズレベルが異なる
- 0.1 が大きすぎる場合: ランダムな残差が入力を悪化させる
- 0.1 が小さすぎる場合: ノイズ除去が不十分（出力 ≈ 入力）
- 手動チューニングが必要

**診断結果**（ランダム初期化モデル）:

- 生の MSE（ノイズ画像 vs クリーン画像）: 0.000129
- 出力の MSE（出力 vs クリーン画像）: 0.002947
- ❌ 出力が入力より悪化（ノイズ除去できていない）

#### 最終的な解決策: 学習可能なスケーリング

```python
# 初期化時
self.residual_scale = torch.nn.Parameter(torch.tensor(0.1))

# 最終処理
x_restored = x_input + self.residual_scale * residual
```

**利点**:

- ✓ モデルが最適なスケーリング係数を自動学習
- ✓ データセットのノイズレベルに自動適応
- ✓ ハイパーパラメータチューニング不要
- ✓ 勾配によって最適化される（requires_grad=True）

**検証結果**:

- パラメータ作成: ✓ 正常
- 勾配計算: ✓ 正常（grad=0.060380）
- Forward pass: ✓ 正常

---

### 4. **Upsampling の位置**

**問題**:
デコーダ層の処理後、スキップ接続内で upsampling していたため、処理の順序が不明確だった

**修正**:

```python
# 修正後
x_patch = dec(x_patch)  # デコーダ層
if (i + 1) % self.depth_te == 0:
    x_patch = upsampling(x_patch, self.num_channels)  # 先にアップサンプル
    x_patch = self._apply_skip_connection(x_patch, encoder_skip, i)  # スキップ接続適用
```

---

## 検証結果の変遷

### 初期状態（修正前）

```
Clamping statistics:
  Values < 0: 111730 (56.83%) ❌
  Values > 1: 6048 (3.08%)   ❌
  Values in [0,1]: 78830 (40.10%)
出力: 一様なランダムノイズ、入力との相関なし
```

### 第 1 回修正後（Sigmoid 追加）

```
Clamping statistics:
  Values < 0: 0 (0.00%)      ✓
  Values > 1: 0 (0.00%)      ✓
  Values in [0,1]: 196608 (100.00%) ✓
出力: 薄紫の一様な画像（すべて0.5付近）
```

### 第 2 回修正後（残差接続）

```
Residual statistics:
  Raw residual: [-2.20, 2.93], Mean: 0.029, Std: 0.54
  Scaled (0.1x): [-0.22, 0.29], Mean: 0.003, Std: 0.054
出力: 入力の統計を保持、小さな修正を適用
```

### 最終版（学習可能なスケーリング）

```
Parameters:
  residual_scale: 0.1000 (初期値)
  requires_grad: True ✓
  gradient: 0.060380 ✓
出力: 学習中に自動調整（期待値: 0.01～1.0）
```

---

## 今後の対応

### ⚠️ 重要: モデルの再学習が必要

現在の学習済みモデルは、以下の理由で使用できません：

1. **出力層の構造が変更された**

   - 旧: `Conv2d(3, 3, 3)` + `Sigmoid`（2 層）
   - 新: `Conv2d + ReLU + Conv2d`（3 層）+ 学習可能スケーリング
   - パラメータの互換性なし

2. **残差接続の追加**

   - 入力画像を最終層に渡す必要がある
   - forward メソッドのシグネチャ変更

3. **モデルアーキテクチャの不整合**
   - 保存されたモデル: 8 ヘッド（base モデル）
   - 現在の設定: 4 ヘッド（lite モデル）

### 推奨される学習手順

1. **新しいアーキテクチャで学習を開始**

   ```bash
   cd /home/s.chochi/ai-works/vit-unet
   python -m vit_unet.train train \
       --input_folder=/home/s.chochi/ai-works/denoiser/data/CC15 \
       --n_epochs=50 \
       --model_string=lite \
       --lr=1e-5 \
       --batch_size=4
   ```

2. **学習中に監視すべき項目**

   - WandB で学習曲線を確認
   - **`residual_scale`の値をログ**: ノイズレベルに応じて調整されるか確認
   - 出力画像が入力と異なるか（デノイジングが機能しているか）
   - PSNR が改善しているか

3. **期待される`residual_scale`の挙動**

   - 初期値: 0.1
   - ノイズが小さいデータ: 0.01 ～ 0.05 に減少
   - ノイズが大きいデータ: 0.3 ～ 1.0 に増加
   - 学習が進むにつれて安定

4. **評価**
   - PSNR 値の確認
   - 視覚的な画像品質の確認
   - 入力と出力の差分を可視化

---

## 技術的詳細

### Sigmoid 飽和問題の原因

**なぜ Sigmoid が失敗したか**:

```
デコーダ出力 → 正規化（平均0、標準偏差1）
    ↓
Conv2d層（重みがランダムまたは小さい）
    ↓
出力が0付近に集中（標準偏差0.7程度）
    ↓
Sigmoid(0) = 0.5
    ↓
すべてのピクセルが0.5付近 = 薄紫の一様な画像
```

### 残差接続の利点

**デノイジングタスクに最適な理由**:

1. **出力 ≈ 入力**: ノイズ除去では、出力は入力とほぼ同じ
2. **小さな修正**: モデルは「ノイズ」だけを学習すればよい
3. **学習の安定性**: 小さな残差を学習する方が、絶対値を学習するより簡単
4. **勾配流**: 残差接続により勾配が直接入力に流れる

### 学習可能なスケーリングの重要性

**なぜ固定値ではダメか**:

- ノイズレベルはデータセットによって異なる
- 学習の進行に応じて最適な値が変わる
- 過学習防止のために動的調整が必要

**数学的な説明**:

```
output = input + α * residual

α: 学習可能なスケーリング係数
- α→0: output≈input（ノイズ除去なし）
- α→1: 残差をそのまま適用
- 最適なα: データのノイズ標準偏差に依存
```

### Skip Connection のアテンション

正しい実装:

- **Q (Query)**: エンコーダの特徴（「何を探すか」）
- **K (Key)**: デコーダの特徴（「どこを見るか」）
- **V (Value)**: エンコーダの特徴（「何を取り出すか」）← これが重要！

間違った実装では、V がデコーダになっていたため、エンコーダの情報が活用されていなかった。

---

## デバッグツール

以下のスクリプトで各段階を検証可能：

1. **`debug_output.py`**: モデル出力と内部表現の確認
2. **`debug_final_processing.py`**: 最終処理の各ステップを追跡
3. **`debug_sigmoid_issue.py`**: Sigmoid 飽和問題の診断
4. **`diagnose_residual_scale.py`**: 残差スケーリングの診断
5. **`test_patch_unpatch.py`**: パッチング操作の正確性検証
6. **`test_residual.py`**: 残差接続のテスト
7. **`test_learnable_scale.py`**: 学習可能スケーリングの検証

---

## 参考画像

- `results/debug_output.png`: 各種入力での出力比較
- `results/final_processing_debug.png`: 最終処理の各ステップの可視化
- `results/sigmoid_issue_debug.png`: Sigmoid 飽和問題の可視化
- `results/residual_test.png`: 残差接続のテスト結果
- `results/patch_unpatch_test_*.png`: パッチ/アンパッチの正確性検証

---

## まとめ

### 解決した問題

1. ✅ スキップ接続の Value（V）を修正
2. ✅ Sigmoid 飽和による薄紫画像問題を残差接続で解決
3. ✅ 残差スケーリング係数を学習可能にして自動調整
4. ✅ Upsampling の処理順序を明確化

### 最終的なアーキテクチャの特徴

- **残差接続**: `output = input + α * residual`
- **学習可能 α**: モデルが最適値を自動学習
- **適切な出力層**: 活性化なしで残差を予測
- **正しいスキップ接続**: エンコーダ特徴を適切に活用

### 期待される改善

- 入力画像の構造を保持しながらノイズ除去
- データセットに応じた自動調整
- 学習の安定性向上
- 勾配流の改善
