@echo off
setlocal enabledelayedexpansion

REM --- CARLA は別起動しておく ---

REM --- 実行パラメータ ---
set RUNNER_SCRIPT=exp_future\batch_run_and_analyze_decel.py
set REPLAY_SCRIPT=scripts\udp_replay\replay_from_udp_carla_pred.py
set SENDER_SCRIPT=send_data\send_udp_frames_from_csv.py
set CSV=send_data\exp_accident.csv

set RUNS=5
set WINDOW_SEC=2.0

REM 追跡30s + 未来10s など（あなたの実験に合わせる）
set TRACKING_SEC=30
set FUTURE_SEC=10

python %RUNNER_SCRIPT% ^
  --replay-script %REPLAY_SCRIPT% ^
  --sender-script %SENDER_SCRIPT% ^
  --csv-path %CSV% ^
  --runs %RUNS% ^
  --window-sec %WINDOW_SEC% ^
  --tracking-sec %TRACKING_SEC% ^
  --future-sec %FUTURE_SEC% ^
  --fixed-delta 0.1 ^
  --poll-interval 0.1 ^
  --sender-interval 0.1 ^
  --tm-seed 20009 ^
  --outdir results

endlocal
