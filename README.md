# Biomarker_Optimization
変更記録
1. データ読み込み（cardio_train.csv）
2. BMI = weight/(height/100)**2、年齢（日→年）を追加
3. カテゴリ変数をLabelEncoderで数値化
4. 連続変数（age_years, height, weight, ap_hi, ap_lo, BMI）をStandardScalerで標準化
5. データを訓練:検証:テスト＝60%:20%:20%に分割
6. PyTorch Dataset/DataLoader を定義
7. TabTransformer モデル構築（categories, num_continuous, dim, depth, heads…）
8. BCEWithLogitsLoss＋AdamW（weight_decay含む）で最適化
9. train_with_validation で
   - 各エポック訓練 → バリデーション精度測定 → EarlyStopping（閾値0.75）
   - 訓練損失・訓練精度・検証精度を記録
10. テストデータで最終評価（Accuracy, F1, Precision, Recall）
11. 学習曲線（訓練損失・訓練＆検証精度）を横並びでプロット
12. 混同行列＋ROC曲線をセル内数値付きで自動表示
