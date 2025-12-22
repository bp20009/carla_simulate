@echo off
setlocal enabledelayedexpansion

REM --- 注意 ---
REM * CARLA は別起動しておく（本バッチは CARLA を起動しません）。
REM * CSV/ポート設定は sender-script 側の入力 CSV と送信先ポートに合わせる。
REM * sender-script は send_data\send_udp_frames_from_csv.py を使う。

REM --- 実行パラメータ（必要に応じて編集） ---
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
set "REPLAY_SCRIPT=%SCRIPT_DIR%batch_run_and_analyze_decel.py"
set "SENDER_SCRIPT=%ROOT_DIR%\send_data\send_udp_frames_from_csv.py"
REM CSV は実在するファイルに差し替える（例: %ROOT_DIR%\data\exp_accident.csv）
set "CSV=%ROOT_DIR%\path\to\your_decel.csv"
set "RUNS=3"
set "WINDOW_SEC=10"

if not exist "%CSV%" (
  echo CSV not found: "%CSV%"
  exit /b 1
)

python "%REPLAY_SCRIPT%" ^
--replay-script "%REPLAY_SCRIPT%" ^
--sender-script "%SENDER_SCRIPT%" ^
--csv "%CSV%" ^
--runs "%RUNS%" ^
--window-sec "%WINDOW_SEC%"

endlocal
