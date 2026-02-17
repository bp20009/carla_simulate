# evaluation_accident

予測実験（autopilot / lstm）の評価・可視化用スクリプト群です。  
`results_grid_accident` 系の実験ログを集計し、CSVとPDFを作成します。

## 前提

- 実験結果ディレクトリ（例: `results_grid_accident` または `results_grid_accident_multi`）
- 各runに以下があること
  - `logs/actor.csv`
  - `logs/collisions.csv`
  - （必要に応じて）`logs/meta.json`
- baseline事故CSV（例: `exp_future/collisions_exp_accident.csv`）

想定構造:

```text
<base_dir>/
  autopilot/
    lead_1/rep_1/logs/{actor.csv,collisions.csv,meta.json}
  lstm/
    lead_1/rep_1/logs/{actor.csv,collisions.csv,meta.json}
```

または multi-accident の場合:

```text
<base_dir>/
  accident_1_pf_25411/
    autopilot/lead_1/rep_1/logs/{actor.csv,collisions.csv,meta.json}
    lstm/lead_1/rep_1/logs/{actor.csv,collisions.csv,meta.json}
```

## 典型ワークフロー

## results_grid_accident_multi を一発解析する

事故ごとのサブディレクトリ（`accident_*_pf_*`）を自動検出して、
イベント抽出→集計→predicted risk→要約図まで順番に実行します。

```bash
python evaluation_accident/run_multi_accident_analysis.py \
  --multi-root results_grid_accident_multi \
  --baseline-csv exp_future/collisions_exp_accident.csv \
  --dt 0.1 \
  --risk-mode speed
```

主な出力:
- `results_grid_accident_multi/analysis_multi/analysis_index.csv`
- `results_grid_accident_multi/analysis_multi/predicted_risk_summary_all_accidents.csv`
- `results_grid_accident_multi/analysis_multi/near_miss_and_collisions_per_method_all_accidents.csv`
- `results_grid_accident_multi/analysis_multi/predicted_risk_summary_global.csv`（全事故をmethod×leadで1つに集約）
- `results_grid_accident_multi/analysis_multi/near_miss_and_collisions_summary_global.csv`（全事故をmethod×leadで1つに集約）
- `results_grid_accident_multi/analysis_multi/fig_collision_summary_global.pdf`（全事故統合の衝突強度図）
- `results_grid_accident_multi/analysis_multi/fig_risk_summary_global.pdf`（全事故統合の事故リスク図）
- `results_grid_accident_multi/analysis_multi/<accident_tag>/...`（事故ごとの詳細PDF/CSV）

補足:
- `--risk-mode pair` を指定すると `predicted_risk_with_pair.py` を使います。
- 失敗時に停止したい場合は `--fail-fast` を付けてください。

## 1) ニアミス・衝突イベントを抽出（イベントCSV生成）

```bash
python evaluation_accident/summarize_hotspots_with_nearmiss.py \
  --root results_grid_accident \
  --base-payload-frame 25411 \
  --dt 0.1 \
  --threshold-m 5.0 \
  --include-nearmiss \
  --include-collisions
```

出力先（例）:
- `out_hotspots_events_pairs_YYYYmmdd_HHMMSS/events_nearmiss.csv`
- `out_hotspots_events_pairs_YYYYmmdd_HHMMSS/events_collision.csv`
- `out_hotspots_events_pairs_YYYYmmdd_HHMMSS/events_all.csv`

## 2) lead別平均回数を図化

```bash
python evaluation_accident/plot_mean_events_by_lead.py \
  --events-dir out_hotspots_events_pairs_YYYYmmdd_HHMMSS
```

## 3) run別散布図（必要ならmethod重ね描き）

```bash
python evaluation_accident/plot_events_scatter_pdf.py \
  --events-dir out_hotspots_events_pairs_YYYYmmdd_HHMMSS \
  --make-merged-methods \
  --merged-target all
```

## 4) 危険領域侵入率（predicted risk）を算出

軽量版（侵入率 + 速度プロット）:

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

拡張版（対象車両全員 + ペア車間距離まで）:

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

## 5) 論文用の要約図（衝突強度・侵入率）を出力

`near_miss_and_collisions_per_method.csv` と `predicted_risk_summary_*.csv` を使います。

```bash
python evaluation_accident/plot_results_figures.py \
  --near-miss-collisions near_miss_and_collisions_per_method.csv \
  --risk-summary predicted_risk_summary_4.csv \
  --out-collision-pdf fig_collision_summary.pdf \
  --out-risk-pdf fig_risk_summary.pdf
```

## 補助スクリプト

- `analyze_collision_intensity.py`
  - 強度分布の解析・閾値スイープ
- `plot_intensity.py`
  - 強度ヒストグラム可視化
- `summarize_counts_mean_over_rep.py`
  - rep平均（ニアミス/衝突）の集計
- `plot_collision_opponents_after_switch.py`
  - 切替後衝突相手の解析
- `plot_switch_trajectories_from_actor_csvs.py`
  - 切替前後の軌跡可視化
- `plot_lane_deviation_fig.py`
  - 車線逸脱系メトリクス可視化
- `plot_speedup_from_summaries.py`
  - time acceleration系summaryの比較図

## 注意点

- `--base-payload-frame` と `--dt` は実験時の設定に合わせてください。
- multi-accident結果を評価する場合は、事故ごとのサブディレクトリ（例: `accident_1_pf_*`）単位で `--root` / `--base-dir` を切り替えるのが安全です。
- 出力列名はスクリプトごとに異なるため、次段の入力CSV指定を必ず確認してください。
