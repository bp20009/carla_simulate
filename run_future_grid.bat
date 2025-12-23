@echo off
cd /d "%~dp0"
setlocal enabledelayedexpansion

if "%~1"=="" (
  set "CSV_PATH=%~dp0send_data\exp_accident.csv"
  set "OUTDIR=%~dp0results_grid_accident"
) else (
  set "CSV_PATH=%~1"
  set "OUTDIR=%~2"
  if "%OUTDIR%"=="" set "OUTDIR=results_grid"
)

set "REPLAY_SCRIPT=scripts\udp_replay\replay_from_udp_future_exp.py"
set "SENDER_SCRIPT=send_data\send_udp_frames_from_csv.py"
set "LSTM_MODEL=scripts\udp_replay\traj_lstm.pt"
set "LSTM_DEVICE=cpu"
set "META_TOOL=scripts\udp_replay\meta_tools.py"

set "FIXED_DELTA=0.1"
set "POLL_INTERVAL=0.1"
set "TRACKING_SEC=30"
set "FUTURE_SEC=10"
set "LEAD_MIN=1"
set "LEAD_MAX=10"
set "REPS=5"
set "BASE_SEED=20009"
set "STARTUP_DELAY=1"

set "CARLA_HOST=127.0.0.1"
set "CARLA_PORT=2000"
set "LISTEN_HOST=0.0.0.0"
set "LISTEN_PORT=5005"
set "SENDER_HOST=127.0.0.1"
set "SENDER_PORT=5005"

set /a "MAX_RUNTIME=%TRACKING_SEC%+%FUTURE_SEC%"
set /a "WAIT_SEC=%MAX_RUNTIME%+30"
if not defined CALIB_MAX_RUNTIME set "CALIB_MAX_RUNTIME=%MAX_RUNTIME%"
set /a "CALIB_WAIT_SEC=%CALIB_MAX_RUNTIME%+30"

set "CALIB_LOGS=%OUTDIR%\calibration\logs"
set "CALIB_META=%CALIB_LOGS%\meta.json"
set "CALIB_COLL=%CALIB_LOGS%\collisions.csv"

mkdir "%CALIB_LOGS%" >nul 2>&1

echo [calibration] starting...
for /f %%p in ('python -c "import subprocess,sys; p=subprocess.Popen([sys.executable]+sys.argv[1:]); print(p.pid)" ^
  "%REPLAY_SCRIPT%" --carla-host "%CARLA_HOST%" --carla-port "%CARLA_PORT%" --listen-host "%LISTEN_HOST%" --listen-port "%LISTEN_PORT%" ^
  --poll-interval "%POLL_INTERVAL%" --fixed-delta "%FIXED_DELTA%" --max-runtime "%CALIB_MAX_RUNTIME%" --tm-seed "%BASE_SEED%" ^
  --future-mode none --metadata-output "%CALIB_META%" --collision-log "%CALIB_COLL%"') do set "REPLAY_PID=%%p"

timeout /t %STARTUP_DELAY% /nobreak >nul
python "%SENDER_SCRIPT%" "%CSV_PATH%" --host "%SENDER_HOST%" --port "%SENDER_PORT%" --interval "%FIXED_DELTA%"
call :wait_for_pid %REPLAY_PID% %CALIB_WAIT_SEC%
if errorlevel 1 (
  taskkill /PID %REPLAY_PID% /T /F >nul 2>&1
)

for /f %%a in ('python "%META_TOOL%" first_accident_pf "%CALIB_META%"') do set "ACCIDENT_PF=%%a"

if "%ACCIDENT_PF%"=="" (
  echo Calibration failed: no accidents in %CALIB_META%
  exit /b 1
)

set "SUMMARY=%OUTDIR%\summary_grid.csv"
echo method,lead_sec,rep,seed,switch_payload_frame,ran_ok,accident_after_switch,first_accident_payload_frame,status,accident_payload_frame_ref > "%SUMMARY%"

for %%M in (autopilot lstm) do (
  for /L %%L in (%LEAD_MIN%,1,%LEAD_MAX%) do (
      for /f %%s in ('python "%META_TOOL%" switch_pf "%ACCIDENT_PF%" "%%L" "%FIXED_DELTA%"') do set "SWITCH_PF=%%s"

    for /L %%R in (1,1,%REPS%) do (
      set /a "SEED=%BASE_SEED%+%%R"
      set "RUN_DIR=%OUTDIR%\%%M\lead_%%L\rep_%%R"
      set "RUN_LOGS=!RUN_DIR!\logs"
      set "RUN_META=!RUN_LOGS!\meta.json"
      set "RUN_COLL=!RUN_LOGS!\collisions.csv"
      set "RUN_ACTOR=!RUN_LOGS!\actor.csv"
      set "RUN_IDMAP=!RUN_LOGS!\id_map.csv"

      mkdir "!RUN_LOGS!" >nul 2>&1

      set "RAN_OK=1"
      set "STATUS=ok"

      set "LSTM_ARGS="
      if /I "%%M"=="lstm" (
        set "LSTM_ARGS=--lstm-model ""!LSTM_MODEL!"" --lstm-device !LSTM_DEVICE!"
      )

      for /f %%p in ('python -c "import subprocess,sys; p=subprocess.Popen([sys.executable]+sys.argv[1:]); print(p.pid)" ^
        "%REPLAY_SCRIPT%" --carla-host "%CARLA_HOST%" --carla-port "%CARLA_PORT%" --listen-host "%LISTEN_HOST%" --listen-port "%LISTEN_PORT%" ^
        --poll-interval "%POLL_INTERVAL%" --fixed-delta "%FIXED_DELTA%" --max-runtime "%MAX_RUNTIME%" --tm-seed "!SEED!" ^
        --future-mode "%%M" --switch-payload-frame "!SWITCH_PF!" --metadata-output "!RUN_META!" --collision-log "!RUN_COLL!" ^
        --actor-log "!RUN_ACTOR!" --id-map-file "!RUN_IDMAP!" !LSTM_ARGS!') do set "REPLAY_PID=%%p"

      timeout /t %STARTUP_DELAY% /nobreak >nul
      python "%SENDER_SCRIPT%" "%CSV_PATH%" --host "%SENDER_HOST%" --port "%SENDER_PORT%" --interval "%FIXED_DELTA%"
      call :wait_for_pid !REPLAY_PID! %WAIT_SEC%
      if errorlevel 1 (
        set "RAN_OK=0"
        set "STATUS=timeout"
        taskkill /PID !REPLAY_PID! /T /F >nul 2>&1
      )

      for /f %%f in ('python "%META_TOOL%" first_accident_pf "!RUN_META!"') do set "FIRST_ACC_PF=%%f"

      for /f %%a in ('python "%META_TOOL%" accident_after_switch "!RUN_META!" "!SWITCH_PF!"') do set "AFTER_SWITCH=%%a"

      if "!FIRST_ACC_PF!"=="" set "FIRST_ACC_PF="
      if "!AFTER_SWITCH!"=="" set "AFTER_SWITCH=0"

      echo %%M,%%L,%%R,!SEED!,!SWITCH_PF!,!RAN_OK!,!AFTER_SWITCH!,!FIRST_ACC_PF!,!STATUS!,%ACCIDENT_PF%>> "%SUMMARY%"
    )
  )
)

echo Wrote: %SUMMARY%
exit /b 0

:wait_for_pid
setlocal
set "PID=%~1"
set /a "REMAIN=%~2"
:wait_loop
tasklist /fi "PID eq %PID%" | findstr /i "%PID%" >nul
if errorlevel 1 (
  endlocal & exit /b 0
)
if %REMAIN% LEQ 0 (
  endlocal & exit /b 1
)
set /a REMAIN-=1
timeout /t 1 /nobreak >nul
goto wait_loop
