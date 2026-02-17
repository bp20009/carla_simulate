# evaluation_accident / 事故評価ツール

## Overview / 概要

**EN**  
This directory contains post-processing scripts for accident experiments (event extraction, risk estimation, and plotting).

**JA**  
このディレクトリには、事故実験の後処理スクリプト（イベント抽出、リスク推定、可視化）が含まれます。

## Inputs / 入力データ

**EN**
- Experiment logs under `results_grid_accident` or `results_grid_accident_multi`
- Per run: `logs/actor.csv`, `logs/collisions.csv`, optionally `logs/meta.json`
- Baseline CSV: `exp_future/collisions_exp_accident.csv`

**JA**
- `results_grid_accident` または `results_grid_accident_multi` 配下の実験ログ
- 各runに `logs/actor.csv`, `logs/collisions.csv`（必要に応じて `logs/meta.json`）
- baseline CSV: `exp_future/collisions_exp_accident.csv`

## One-shot for Multi-Accident / 複数事故の一括解析

**EN**  
Run all accident directories (`accident_*_pf_*`) in one command.

**JA**  
`accident_*_pf_*` を自動検出し、全事故を一括解析します。

```bash
python3 evaluation_accident/run_multi_accident_analysis.py \
  --multi-root results_grid_accident_multi \
  --baseline-csv exp_future/collisions_exp_accident.csv \
  --dt 0.1 \
  --risk-mode speed
```

Main outputs / 主な出力:
- `results_grid_accident_multi/analysis_multi/analysis_index.csv`
- `results_grid_accident_multi/analysis_multi/predicted_risk_summary_global.csv`
- `results_grid_accident_multi/analysis_multi/near_miss_and_collisions_summary_global.csv`
- `results_grid_accident_multi/analysis_multi/fig_collision_summary_global.pdf`
- `results_grid_accident_multi/analysis_multi/fig_risk_summary_global.pdf`
- `results_grid_accident_multi/analysis_multi/<accident_tag>/...`

Options / オプション:
- `--risk-mode speed|pair`
- `--methods autopilot lstm` (filter)
- `--fail-fast`

## Step-by-step Workflow / 手動ステップ実行

### 1) Extract event CSVs / イベントCSV抽出

```bash
python evaluation_accident/summarize_hotspots_with_nearmiss.py \
  --root results_grid_accident \
  --base-payload-frame 25411 \
  --dt 0.1 \
  --threshold-m 5.0 \
  --include-nearmiss \
  --include-collisions
```

### 2) Plot mean events by lead / lead別平均回数図

```bash
python evaluation_accident/plot_mean_events_by_lead.py \
  --events-dir out_hotspots_events_pairs_YYYYmmdd_HHMMSS
```

### 3) Scatter PDFs / 散布図PDF

```bash
python evaluation_accident/plot_events_scatter_pdf.py \
  --events-dir out_hotspots_events_pairs_YYYYmmdd_HHMMSS \
  --make-merged-methods \
  --merged-target all
```

### 4) Predicted risk / 事故リスク推定

Speed variant / 速度版:

```bash
python evaluation_accident/predicted_risk_with_speed.py \
  --baseline-csv exp_future/collisions_exp_accident.csv \
  --base-frame 25411 \
  --base-dir results_grid_accident \
  --dt 0.1 \
  --use-switch-window \
  --window-from-switch-sec 10.0 \
  --out-per-run predicted_risk_per_run_3.csv \
  --out-summary predicted_risk_summary_3.csv
```

Pair variant / ペア版:

```bash
python evaluation_accident/predicted_risk_with_pair.py \
  --baseline-csv exp_future/collisions_exp_accident.csv \
  --base-frame 25411 \
  --base-dir results_grid_accident \
  --dt 0.1 \
  --use-switch-window \
  --window-from-switch-sec 10.0 \
  --plot-targets \
  --out-per-run predicted_risk_per_run_4.csv \
  --out-summary predicted_risk_summary_4.csv
```

### 5) Final summary figures / 要約図出力

```bash
python evaluation_accident/plot_results_figures.py \
  --near-miss-collisions near_miss_and_collisions_per_method.csv \
  --risk-summary predicted_risk_summary_4.csv \
  --out-collision-pdf fig_collision_summary.pdf \
  --out-risk-pdf fig_risk_summary.pdf
```

## Script Map / スクリプト一覧

- `run_multi_accident_analysis.py`: one-shot pipeline for multi-accident / 複数事故一括実行
- `summarize_hotspots_with_nearmiss.py`: near-miss/collision event extraction / イベント抽出
- `summarize_near_miss_and_collisions_from_dirs.py`: per-run/per-method summary / run・手法集計
- `predicted_risk_with_speed.py`: risk rate + speed plots / リスク率+速度
- `predicted_risk_with_pair.py`: risk rate + speed/headway pairs / リスク率+速度/車間距離
- `plot_results_figures.py`: paper-ready summary figures / 要約図
- `plot_mean_events_by_lead.py`, `plot_events_scatter_pdf.py`: event visualization / 可視化
- `analyze_collision_intensity.py`, `plot_intensity.py`: intensity analysis / 強度分析

## Notes / 注意

**EN**
- Keep `--dt` and frame assumptions aligned with experiment settings.
- For multi-accident analysis, use global summary CSV/PDF for final comparison.

**JA**
- `--dt` とフレーム前提は実験時設定と揃えてください。
- 複数事故比較の最終評価は global 集約CSV/PDF を参照してください。
