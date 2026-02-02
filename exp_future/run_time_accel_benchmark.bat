@echo off
setlocal enabledelayedexpansion

REM --- 注意 ---
REM * CARLA は別起動しておく（本バッチは CARLA を起動しません）。
REM * 必要に応じて host / port / duration / actor counts を調整してください。

REM --- 実行パラメータ（必要に応じて編集） ---
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
set "PY_SCRIPT=%SCRIPT_DIR%measure_time_acceleration.py"
set "CARLA_HOST=127.0.0.1"
set "CARLA_PORT=2000"
set "DURATION_SEC=15"
set "ACTOR_COUNTS=0:50:10"
set "OUTPUT_CSV=%ROOT_DIR%\results\time_accel_benchmark.csv"

python "%PY_SCRIPT%" ^
  --host "%CARLA_HOST%" ^
  --port "%CARLA_PORT%" ^
  --duration "%DURATION_SEC%" ^
  --actor-counts "%ACTOR_COUNTS%" ^
  --output "%OUTPUT_CSV%"

endlocal
