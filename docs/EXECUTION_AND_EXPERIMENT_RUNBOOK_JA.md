# CARLA実行・実験手順書（Runbook）

この手順書は、`/Users/mafuyu/Prog/carla_simulate` のスクリプトを使って、

- 動作確認（スモークテスト）
- UDPリプレイ実験（autopilot/lstm）
- 減速度評価
- 事故抽出
- スイープ分析

を再現するための実務向けガイドです。

## 1. 前提条件

- CARLAサーバへ接続できること（通常 `127.0.0.1:2000`）
- Python 3.9+ 推奨
- 依存パッケージ:
  - `carla>=0.9.13`
  - 必要に応じて `matplotlib`, `pillow`
- 実験入力CSV（例: `send_data/exp_accident.csv`）が存在すること

初期セットアップ:

```bash
cd /Users/mafuyu/Prog/carla_simulate
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. まずやる動作確認（5分）

### 2.1 CARLA接続と簡易スポーン

```bash
python scripts/autopilot_simulation.py --host 127.0.0.1 --port 2000 --duration 10 --seed 42
```

期待結果:
- エラーなく終了する
- CARLA上で車両/歩行者/自転車が短時間スポーンする

### 2.2 状態ストリーム出力

```bash
python scripts/vehicle_state_stream.py \
  --host 127.0.0.1 --port 2000 \
  --mode wait \
  --include-velocity --frame-elapsed --wall-clock \
  --output /tmp/vehicle_states.csv
```

`Ctrl+C` で停止後、`/tmp/vehicle_states.csv` が作成されていればOKです。

## 3. UDPリプレイ最小実行（手動2プロセス）

### 3.1 送信用CSVを作る（必要時）

`vehicle_state_stream.py` 出力を使う場合:

```bash
python scripts/convert_vehicle_state_csv.py /tmp/vehicle_states.csv /tmp/vehicle_states_reduced.csv
```

### 3.2 受信側を起動（Terminal A）

```bash
python scripts/udp_replay/replay_from_udp_carla_pred.py \
  --carla-host 127.0.0.1 --carla-port 2000 \
  --listen-host 0.0.0.0 --listen-port 5005 \
  --switch-payload-frame 25411 \
  --end-payload-frame 25511 \
  --metadata-output /tmp/replay_meta.json \
  --actor-log /tmp/replay_actor.csv \
  --id-map-file /tmp/replay_id_map.csv \
  --control-state-file /tmp/control_state.json \
  --enable-completion
```

### 3.3 送信側を起動（Terminal B）

```bash
python send_data/send_udp_frames_from_csv.py send_data/exp_accident.csv \
  --host 127.0.0.1 --port 5005 \
  --interval 0.1 --frame-stride 1
```

期待結果:
- 受信側が終了または安定稼働し、`/tmp/replay_meta.json` などのログが生成される

## 4. 本実験フロー

## 4-A. グリッド実験（autopilot vs lstm）

Windows標準フローは `run_future_grid.bat` を使います。

```bat
run_future_grid.bat
```

引数指定例:

```bat
run_future_grid.bat send_data\exp_accident.csv results_grid_accident
```

出力（主なもの）:
- `results_grid_accident/summary_grid.csv`
- `results_grid_accident/<method>/lead_<N>/rep_<N>/logs/meta.json`
- `results_grid_accident/<method>/lead_<N>/rep_<N>/logs/collisions.csv`
- `results_grid_accident/<method>/lead_<N>/rep_<N>/logs/actor.csv`

注意:
- `run_future_grid.bat` は CARLA再起動制御を含むため、`CARLA_ROOT` などの環境に合わせた編集が必要です。
- `run_future_grid_2_5_10.bat` は説明文混在のため、そのまま実行用としては推奨しません。

## 4-B. 減速度評価バッチ

簡易実行（Windowsバッチ）:

```bat
run_decel_batch.bat
```

クロスプラットフォーム実行例（Python直実行）:

```bash
python exp_future/batch_run_and_analyze_decel.py \
  --replay-script scripts/udp_replay/replay_from_udp_carla_pred.py \
  --sender-script send_data/send_udp_frames_from_csv.py \
  --csv-path send_data/exp_accident.csv \
  --runs 10 \
  --tracking-sec 30 --future-sec 10 \
  --fixed-delta 0.1 --poll-interval 0.1 \
  --sender-interval 0.1 \
  --switch-eval-ticks 2 \
  --center-payload-frame 25411 --pre-sec 60 --post-sec 30 \
  --outdir results_decel
```

出力:
- `results_decel/run_0001/logs/{meta.json,actor.csv,id_map.csv,replay.log,sender.log}`
- `results_decel/run_0001/decel_after_switch_actors.csv`
- `results_decel/summary_runs.csv`

## 4-C. N/Ts スイープ（通信・記録品質評価）

Windows:

```bat
run_sweep.bat
```

出力:
- `sweep_results_<date>_<time>/N*_Ts*/replay_state.csv`
- `sweep_results_<date>_<time>/N*_Ts*/stream_timing.csv`
- `sweep_results_<date>_<time>/receiver_*.log`

## 5. 実験後の集計・分析

### 5.1 未来フェーズ事故抽出

```bash
python scripts/extract_future_accidents.py results_grid_accident \
  --threshold 1000 \
  --require-is-accident \
  --out results_grid_accident/future_accidents_ge1000.csv
```

またはWindows:

```bat
run_extract_future_accidents.bat results_grid_accident
```

### 5.2 スイープ分析

```bash
python analyze_sweep.py \
  --outdir sweep_results_YYYYMMDD_HHMMSS \
  --ref send_data/exp_300.csv \
  --match-key external_id \
  --kind vehicle \
  --overlay
```

出力:
- `.../analysis_<timestamp>/summary_runs.csv`
- `.../analysis_<timestamp>/runs/<tag>/...pdf`

### 5.3 クイック健全性チェック

```bash
python check_runs.py
```

## 6. 典型的な運用順（推奨）

1. `autopilot_simulation.py` で接続確認
2. 小規模手動リプレイ（2プロセス）でログ生成確認
3. `run_future_grid.bat` で本番グリッド実験
4. `extract_future_accidents.py` で事故イベント抽出
5. 必要なら `batch_run_and_analyze_decel.py` で減速度比較
6. `analyze_sweep.py` で品質評価可視化

## 7. トラブルシュート

- 送信が0フレーム:
  - 入力CSVの `frame` 列有無、`--start-frame/--end-frame` 範囲、`--frame-stride` を確認
- CARLA接続失敗:
  - `--carla-host/--carla-port` とサーバ起動状態を確認
- ログが空:
  - 受信側起動後に送信側を開始（起動順）
  - `--startup-delay` を増やす
- 実験が止まらない/長すぎる:
  - `--max-runtime`, `--end-payload-frame` を設定
- 再現性がぶれる:
  - `--tm-seed` / `--seed` / `BASE_SEED` を固定

## 8. 参照ファイル

- `README.md`
- `exp_future/README.md`
- `evaluation_accident/README.md`
- `run_future_grid.bat`
- `run_decel_batch.bat`
- `run_sweep.bat`
