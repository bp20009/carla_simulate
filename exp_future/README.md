# exp_future / 未来実験スクリプト

## Overview / 概要

**EN**  
`exp_future` contains experiment runners and utilities for autopilot/LSTM future-phase evaluation in CARLA.

**JA**  
`exp_future` には、CARLAでの Autopilot/LSTM 未来フェーズ評価向けの実験ランナーと補助ツールが入っています。

## Main Scripts / 主なスクリプト

- `experiment_grid.py`  
  **EN**: Generic grid runner for lead/repetition experiments.  
  **JA**: lead・反復回数などのグリッド探索を行う汎用ランナー。

- `run_future_grid.py`  
  **EN**: Lightweight sweep around a target frame (autopilot vs lstm).  
  **JA**: ターゲットフレーム周辺で autopilot と lstm を比較する軽量スイープ。

- `batch_run_and_analyze_decel.py`  
  **EN**: Batch execution + post-switch deceleration analysis.  
  **JA**: バッチ実行と切替後減速度解析をまとめて実行。

- `measure_time_acceleration.py`  
  **EN**: Measures simulation speedup under different actor/render settings.  
  **JA**: アクター数・描画条件ごとのシミュレーション加速率を計測。

- `train_traj_lstm.py`  
  **EN**: Trainer for trajectory LSTM model used by replay scripts.  
  **JA**: リプレイスクリプトで使う軌跡LSTMの学習スクリプト。

## Quick Examples / 実行例

### Time Acceleration Benchmark / 時間加速ベンチマーク

```bash
python exp_future/measure_time_acceleration.py \
  --duration 15 \
  --actor-counts 0:50:10 \
  --output results/time_accel_benchmark.csv
```

### Deceleration Batch + Analysis / 減速度バッチ解析

```bash
python exp_future/batch_run_and_analyze_decel.py \
  --replay-script scripts/udp_replay/replay_from_udp_carla_pred.py \
  --sender-script send_data/send_udp_frames_from_csv.py \
  --csv-path send_data/exp_accident.csv \
  --runs 10 \
  --tracking-sec 30 --future-sec 10 \
  --fixed-delta 0.1 --poll-interval 0.1 \
  --center-payload-frame 25411 \
  --pre-sec 60 --post-sec 30 \
  --outdir results_decel
```

### Lightweight Future Grid / 軽量Futureグリッド

```bash
python exp_future/run_future_grid.py \
  --csv send_data/exp_accident.csv \
  --replay-script scripts/udp_replay/replay_from_udp_carla_pred.py \
  --sender-script send_data/send_udp_frames_from_csv.py \
  --runs 10 \
  --target-frame 25411 \
  --fixed-delta 0.1 \
  --outdir results_future_grid
```

## Windows Batch Helpers / Windowsバッチ補助

- `run_time_accel_benchmark.bat`
- `run_decel_batch.bat`
- `run_future_grid.bat` (at repository root)
- `run_future_grid_10_accidents.bat` (at repository root, multi-accident)

## Notes / 注意

**EN**
- Match `--fixed-delta`, `--dt`, and frame assumptions across runner/sender/replay.
- Start CARLA before scripts unless the batch file handles startup.
- Use deterministic seeds (`--tm-seed`, `--base-seed`) for reproducibility.

**JA**
- runner / sender / replay 間で `--fixed-delta`・`--dt`・フレーム前提を揃えてください。
- バッチで起動管理しない場合は、CARLAを先に起動してください。
- 再現性を重視する場合は `--tm-seed`・`--base-seed` を固定してください。
