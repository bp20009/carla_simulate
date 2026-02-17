# CARLA Simulation Utilities / CARLAシミュレーションユーティリティ

## Overview / 概要

**EN**  
This repository provides practical tools for replaying external tracking data in CARLA, logging actor states, and running accident-focused autopilot/LSTM experiments.

**JA**  
このリポジトリは、外部トラッキングデータをCARLAへリプレイし、アクター状態を記録し、事故評価向けのAutopilot/LSTM実験を実行するための実用ツール群を提供します。

## Runbook / 手順書

**EN**  
Detailed execution and experiment procedures are available at:
- `docs/EXECUTION_AND_EXPERIMENT_RUNBOOK_JA.md`

**JA**  
実行方法と実験手順の詳細は以下を参照してください:
- `docs/EXECUTION_AND_EXPERIMENT_RUNBOOK_JA.md`

## Included Tools / 収録ツール

**EN**
- UDP replay receivers that map incoming tracking payloads to CARLA actors.
- CSV/UDP sender tools for frame-based datasets.
- State streaming and trajectory visualization utilities.
- Batch experiment runners (grid sweep, deceleration analysis, extraction).
- Sweep/result analysis utilities.

**JA**
- 受信したトラッキングペイロードをCARLAアクターへ反映するUDPリプレイ受信ツール
- フレーム形式データセット向けのCSV/UDP送信ツール
- 状態ストリーミングと軌跡可視化ツール
- バッチ実験ランナー（グリッド探索、減速度解析、抽出）
- スイープ結果の分析ツール

## Requirements / 必要条件

**EN**
- Reachable CARLA server.
- Python 3.9+ recommended.
- Python dependencies:
  - `carla` (Python API version must match the running CARLA server version exactly)
  - `matplotlib` (plot/animation)
  - `pillow` (optional, GIF output)
  - `ffmpeg` on `PATH` (optional, MP4 output)

**JA**
- このマシンから接続可能なCARLAサーバ
- Python 3.9以上推奨
- Python依存関係:
  - `carla`（Python APIは実行中のCARLAサーバと同一バージョンである必要があります）
  - `matplotlib`（プロット/アニメーション）
  - `pillow`（任意、GIF出力）
  - `ffmpeg` を `PATH` 上に配置（任意、MP4出力）

**EN**  
Version rule: install the exact CARLA Python API that matches your server version.
Example: CARLA server `0.9.13` requires `carla==0.9.13` (not `0.9.16`).

**JA**  
バージョン規則: CARLA Python APIはサーバ版と完全一致でインストールしてください。
例: サーバが `0.9.13` の場合、`carla==0.9.13` が必要です（`0.9.16` は不可）。

Install / インストール:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# install CARLA API with exact server version, e.g.:
# pip install "carla==0.9.13"
```

## Quick Start / クイックスタート

### 1) Smoke Test Spawn / スモークテスト用スポーン

**EN**  
`scripts/autopilot_simulation.py` currently spawns:
- 1 autopilot vehicle
- 1 walker with AI controller
- 1 cyclist actor with AI controller

and keeps them alive for `--duration` seconds.

**JA**  
`scripts/autopilot_simulation.py` は現在、以下をスポーンします。
- オートパイロット車両 1台
- AIコントローラ付き歩行者 1体
- AIコントローラ付き自転車アクター 1体

`--duration` 秒だけ維持して終了します。

```bash
python scripts/autopilot_simulation.py \
  --host 127.0.0.1 \
  --port 2000 \
  --duration 30 \
  --seed 42
```

### 2) Stream State to CSV / 状態をCSVへストリーム

```bash
python scripts/vehicle_state_stream.py \
  --host 127.0.0.1 \
  --port 2000 \
  --mode wait \
  --include-velocity \
  --frame-elapsed \
  --wall-clock \
  --output vehicle_states.csv
```

**EN** Important options:
- `--role-prefix`: filter by `role_name` prefix (e.g. `udp_replay:`).
- `--control-state-file`: merge control mode overrides from JSON.
- `--include-object-id`, `--include-monotonic`, `--include-tick-wall-dt`: extra timing/ID columns.
- `--timing-output`: write per-frame timing CSV.

**JA** 主なオプション:
- `--role-prefix`: `role_name` の接頭辞でフィルタ（例: `udp_replay:`）
- `--control-state-file`: JSONから制御モード上書きを取り込む
- `--include-object-id`, `--include-monotonic`, `--include-tick-wall-dt`: 追加ID/時間列を出力
- `--timing-output`: フレームごとの処理時間CSVを出力

### 3) Visualize Trajectories / 軌跡可視化

Static plot / 静的プロット:

```bash
python scripts/plot_vehicle_trajectories.py vehicle_states.csv --only vehicle --save trajectories.png
```

Animation / アニメーション:

```bash
python scripts/animate_vehicle_trajectories.py vehicle_states.csv trajectories.mp4 --fps 15 --history 60
```

## UDP Data Pipeline / UDPデータパイプライン

### Send CSV Rows / CSV行送信

```bash
python send_data/send_udp_from_csv.py data.csv --host 127.0.0.1 --port 5005
```

**EN**
- Sends each row as JSON by default.
- Use `--message-column` to send a single column as payload.
- Use `--interval` or `--delay-column` to control pacing.

**JA**
- デフォルトでは各行をJSONとして送信
- `--message-column` で特定カラムのみ送信
- `--interval` または `--delay-column` で送信間隔を制御

### Convert to Reduced Frame CSV / 簡易フレームCSVへ変換

```bash
python scripts/convert_vehicle_state_csv.py vehicle_states.csv vehicle_states_reduced.csv
```

Schema / スキーマ:
- `frame,id,type,x,y,z`

### Send Grouped Frame Payloads / フレーム単位でまとめて送信

```bash
python send_data/send_udp_frames_from_csv.py vehicle_states_reduced.csv \
  --host 127.0.0.1 --port 5005 --interval 0.1 --frame-stride 1
```

Useful options / 主なオプション:
- `--start-frame`, `--end-frame`
- `--frame-stride`
- `--max-actors`

## Replay Receivers / リプレイ受信ツール

### Basic Replay / 基本リプレイ

```bash
python scripts/udp_replay/replay_from_udp.py \
  --carla-host 127.0.0.1 \
  --carla-port 2000 \
  --listen-port 5005 \
  --enable-completion
```

### Prediction Replay + Handoff / 予測リプレイ + 制御切替

```bash
python scripts/udp_replay/replay_from_udp_carla_pred.py \
  --carla-host 127.0.0.1 \
  --carla-port 2000 \
  --listen-port 5005 \
  --switch-payload-frame 25411 \
  --end-payload-frame 25511 \
  --control-state-file /tmp/control_state.json \
  --metadata-output logs/meta.json \
  --actor-log logs/actor.csv \
  --id-map-file logs/id_map.csv
```

**EN**
- Replays frame-based actor states from UDP.
- Can hand control to autopilot at a specified payload frame.
- Emits collision logs and metadata for experiment analysis.

**JA**
- UDPで受けたフレーム単位アクター状態をCARLAに再生
- 指定ペイロードフレームでAutopilotへ制御移譲可能
- 衝突ログとメタデータを出力し、実験分析に利用可能

Other variants / 他バリアント:
- `scripts/udp_replay/replay_from_udp_future_exp.py`
- `scripts/udp_replay/replay_from_udp_lstm.py`

## Experiment Utilities / 実験ユーティリティ

### Future Accident Extraction / 未来フェーズ事故抽出

```bash
python scripts/extract_future_accidents.py results_grid_accident \
  --threshold 1000 \
  --require-is-accident \
  --out results_grid_accident/future_accidents_ge1000.csv
```

### Grid/Batch Runners / グリッド・バッチ実行

- `exp_future/experiment_grid.py`
- `exp_future/run_future_grid.py`
- `exp_future/batch_run_and_analyze_decel.py`
- `exp_future/measure_time_acceleration.py`
- `exp_future/train_traj_lstm.py`
- `evaluation_accident/run_multi_accident_analysis.py` (one-shot analysis for `results_grid_accident_multi`)

Windows batch wrappers / Windowsバッチ:
- `run_future_grid.bat`
- `run_future_grid_10_accidents.bat`
- `run_decel_batch.bat`
- `run_extract_future_accidents.bat`
- `run_sweep.bat`

**EN**
- `run_future_grid_10_accidents.bat` runs multi-accident sweeps using `ACCIDENT_PF_LIST` (10 frames).
- Current defaults run a symmetric sender window (`PRE_SEC=30`, `POST_SEC=30`) around each target frame.
- CARLA restart timing is per-method block (every 10 reps), not after both methods combined.
- `ACCIDENT_PF_LIST` tokens are normalized at runtime, so both space-separated and comma-mixed entries are accepted.

**JA**
- `run_future_grid_10_accidents.bat` は `ACCIDENT_PF_LIST`（10フレーム）を対象に複数事故のスイープを実行します。
- 現在のデフォルトは、各対象フレームに対して対称ウィンドウ（`PRE_SEC=30`, `POST_SEC=30`）です。
- CARLAの再起動タイミングは method 単位（10反復ごと）で、2method合算の20反復後ではありません。
- `ACCIDENT_PF_LIST` は実行時に正規化されるため、スペース区切りだけでなくカンマ混在の記述も扱えます。
- 複数事故結果の一括解析には `evaluation_accident/run_multi_accident_analysis.py` を使用できます。

**EN Note**  
`run_future_grid_2_5_10.bat` currently contains memo/instruction text mixed with script content, so treat it as reference text unless rewritten as a clean `.bat` file.

**JA 注記**  
`run_future_grid_2_5_10.bat` は現状、メモ/説明文とスクリプトが混在しているため、純粋な `.bat` として再作成するまでは参照用ファイルとして扱ってください。

## Sweep Analysis / スイープ分析

### Analyze Sweep Results / スイープ結果分析

```bash
python analyze_sweep.py \
  --outdir sweep_results_YYYYMMDD_HHMMSS \
  --ref send_data/exp_300.csv \
  --match-key external_id \
  --kind vehicle \
  --overlay
```

**EN**  
Generates per-run trajectory/timing plots, actor-wise RMSE/coverage metrics, and summary CSVs.

**JA**  
ランごとの軌跡・時間プロット、アクター別RMSE/カバレッジ指標、サマリCSVを生成します。

Additional helpers / 補助ツール:
- `check_runs.py`
- `scripts/actor_quality_metrics.py`

## Repository Layout / ディレクトリ構成

- `/scripts`: runtime tools / 実行ツール
- `/scripts/udp_replay`: replay receivers and helpers / リプレイ受信・補助
- `/send_data`: UDP senders and CSV inputs / 送信ツールと入力CSV
- `/exp_future`: experiment orchestration / 実験オーケストレーション
- `/evaluation_accident`: accident evaluation plotting scripts / 事故評価プロット
- `/tests`: automated tests / テスト

## Testing / テスト

```bash
pytest -q
```

**EN**  
If the CARLA Python API is missing, scripts/tests importing `carla` may fail.
If the CARLA Python API version does not exactly match the CARLA server version,
connection/runtime behavior can fail.
For non-CARLA validation, integration tests with a mock UDP server and pseudo client are available:

```bash
pytest -q -k "udp_integration_mock or send_udp"
```

**JA**  
CARLA Python APIが未導入の場合、`carla` をimportするスクリプト/テストは失敗する可能性があります。
また、CARLA Python APIとCARLAサーバのバージョンが一致しない場合、
接続や実行時動作が失敗する可能性があります。
CARLA不要で確認したい場合は、モックUDPサーバ＋擬似クライアントの統合テストを実行できます:

```bash
pytest -q -k "udp_integration_mock or send_udp"
```

## Tips / 補足

**EN**
- Use `python <script> --help` for authoritative CLI options.
- Keep CARLA running unless a batch script starts/stops it explicitly.
- Set seeds (`--seed`, `--tm-seed`, `--base-seed`) for reproducibility.
- Contact us : bp20009@shibaura-it.ac.jp
  
If you use this code in your research, please cite:

S.\ Kamioka, T.\ Yamazaki, and T.\ Miyoshi, ``Towards urban digital twins: a framework integrating CARLA with 3D city models,'' IEEE International Conference on Consumer Technology - Pacific (ICCT-Pacific2026), Yamaguchi, Japan, 4 pages, March\ 2026.

**JA**
- CLIオプションの最新情報は `python <script> --help` を参照してください。
- バッチが明示的に起動/停止しない限り、CARLAは事前起動しておいてください。
- 再現性のため、`--seed` / `--tm-seed` / `--base-seed` を固定してください。

- 何か不明点などあれば bp20009@shibaura-it.ac.jp まで
  
本コードを用いた研究成果を公表する場合は，以下の論文を引用してください．

  S.\ Kamioka, T.\ Yamazaki, and T.\ Miyoshi, ``Towards urban digital twins: a framework integrating CARLA with 3D city models,'' IEEE International Conference on Consumer Technology - Pacific (ICCT-Pacific2026), Yamaguchi, Japan, 4 pages, March\ 2026.

  Multimedia Information NETwork Lab.

  Shibaura Institute of Technology, Japan
  
  マルチメディア情報通信研究室　芝浦工業大学
