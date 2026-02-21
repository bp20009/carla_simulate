@echo off
cd /d "%~dp0"
setlocal enabledelayedexpansion

REM Usage:
REM   run_future_grid_10_accidents_rerun_timeouts.bat [SUMMARY_IN] [CSV_PATH] [OUTDIR]
REM Defaults:
REM   SUMMARY_IN = .\results_grid_accident_multi\summary_grid_multi_accidents.csv
REM   CSV_PATH   = .\send_data\exp_accident.csv
REM   OUTDIR     = .\results_grid_accident_multi

if "%~1"=="" (
  set "SUMMARY_IN=%~dp0results_grid_accident_multi\summary_grid_multi_accidents.csv"
) else (
  set "SUMMARY_IN=%~1"
)

if "%~2"=="" (
  set "CSV_PATH=%~dp0send_data\exp_accident.csv"
) else (
  set "CSV_PATH=%~2"
)

if "%~3"=="" (
  set "OUTDIR=%~dp0results_grid_accident_multi"
) else (
  set "OUTDIR=%~3"
)

if not exist "%SUMMARY_IN%" (
  echo [ERROR] summary not found: "%SUMMARY_IN%"
  exit /b 1
)

if not exist "%CSV_PATH%" (
  echo [ERROR] csv not found: "%CSV_PATH%"
  exit /b 1
)

set "REPLAY_SCRIPT=scripts\udp_replay\replay_from_udp_future_exp.py"
set "SENDER_SCRIPT=send_data\send_udp_frames_from_csv.py"
set "LSTM_MODEL=scripts\udp_replay\traj_lstm.pt"
set "LSTM_DEVICE=cpu"
set "META_TOOL=scripts\udp_replay\meta_tools.py"

set "FIXED_DELTA=0.1"
set "PRE_SEC=30"
set "POST_SEC=30"
set "PF_PER_SEC=10"
set "POLL_INTERVAL=0.1"
set "TRACKING_SEC=30"
set "FUTURE_SEC=10"
set "BUFFER_PF_AFTER=2"
set "STARTUP_DELAY=2"
set "MAX_RUNTIME=100"

set /a "WAIT_BASE=%MAX_RUNTIME%+30"
set /a "WAIT_SEC=%WAIT_BASE%"

set "CARLA_ROOT=D:\Carla-0.10.0-Win64-Shipping"
set "CARLA_EXE=%CARLA_ROOT%\CarlaUnreal.exe"
set "CARLA_BOOT_WAIT=60"
set "CARLA_BOOT_TIMEOUT=300"
set "CARLA_WARMUP_SEC=90"

set "CARLA_HOST=127.0.0.1"
set "CARLA_PORT=2000"
set "LISTEN_HOST=0.0.0.0"
set "LISTEN_PORT=5005"
set "SENDER_HOST=127.0.0.1"
set "SENDER_PORT=5005"
set "PY=python"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"
set "SUMMARY_OUT=%OUTDIR%\summary_rerun_timeouts.csv"
echo accident_idx,accident_payload_frame_ref,accident_tag,method,lead_sec,rep,seed,switch_payload_frame,ran_ok,accident_after_switch,first_accident_payload_frame,status,source_summary> "%SUMMARY_OUT%"

echo =========================================================
echo Summary input : %SUMMARY_IN%
echo CSV path      : %CSV_PATH%
echo Output dir    : %OUTDIR%
echo Summary out   : %SUMMARY_OUT%
echo =========================================================

set /a "TOTAL_ROWS=0"
set /a "TIMEOUT_ROWS=0"
set /a "RERUN_OK=0"
set /a "RERUN_FAIL=0"

for /f "usebackq skip=1 tokens=1-12 delims=," %%a in ("%SUMMARY_IN%") do (
  set /a "TOTAL_ROWS+=1"
  set "ACC_IDX=%%a"
  set "ACCIDENT_PF=%%b"
  set "ACC_TAG=%%c"
  set "METHOD=%%d"
  set "LEAD=%%e"
  set "REP=%%f"
  set "SEED=%%g"
  set "SWITCH_PF=%%h"
  set "SRC_STATUS=%%l"

  if /I "!SRC_STATUS!"=="timeout" (
    set /a "TIMEOUT_ROWS+=1"
    echo ---------------------------------------------------------
    echo [RERUN !TIMEOUT_ROWS!] accident=!ACCIDENT_PF! tag=!ACC_TAG! method=!METHOD! lead=!LEAD! rep=!REP! seed=!SEED!

    set /a "START_FRAME=ACCIDENT_PF-(PRE_SEC*PF_PER_SEC)"
    set /a "END_FRAME=ACCIDENT_PF+(POST_SEC*PF_PER_SEC)"
    if !START_FRAME! LSS 0 set "START_FRAME=0"

    for /f %%w in ('
      powershell -NoProfile -Command ^
        "$ErrorActionPreference='Stop';" ^
        "$delta=%FIXED_DELTA%;" ^
        "$start=!START_FRAME!;" ^
        "$end=!END_FRAME!;" ^
        "$sendDuration=($end - $start + 1) * $delta;" ^
        "$wait=[int][math]::Ceiling($sendDuration + %STARTUP_DELAY% + 5);" ^
        "$waitBound=[int][math]::Max($wait,%WAIT_BASE%);" ^
        "Write-Output $waitBound"
    ') do set "WAIT_SEC=%%w"

    set /a "SWITCH_PF_EVAL=SWITCH_PF+BUFFER_PF_AFTER"

    set "RUN_DIR=%OUTDIR%\!ACC_TAG!\!METHOD!\lead_!LEAD!\rep_!REP!"
    set "RUN_LOGS=!RUN_DIR!\logs"
    set "RUN_META=!RUN_LOGS!\meta.json"
    set "RUN_COLL=!RUN_LOGS!\collisions.csv"
    set "RUN_ACTOR=!RUN_LOGS!\actor.csv"
    set "RUN_IDMAP=!RUN_LOGS!\id_map.csv"
    mkdir "!RUN_LOGS!" >nul 2>&1

    set "RAN_OK=1"
    set "STATUS=ok"
    set "AFTER_SWITCH=0"
    set "FIRST_ACC_PF="

    echo Restarting CARLA server ...
    call :restart_carla
    call :wait_carla_ready %CARLA_HOST% %CARLA_PORT% %CARLA_BOOT_TIMEOUT%
    if errorlevel 1 (
      echo [ERROR] CARLA not ready
      set "RAN_OK=0"
      set "STATUS=carla_not_ready"
    )

    if "!RAN_OK!"=="1" (
      echo Warmup wait %CARLA_WARMUP_SEC%s ...
      timeout /t %CARLA_WARMUP_SEC% /nobreak >nul

      set "RECV_OUT=!RUN_LOGS!\receiver.out.log"
      set "RECV_ERR=!RUN_LOGS!\receiver.err.log"
      set "PID_FILE=!RUN_LOGS!\replay.pid"
      del /q "!PID_FILE!" >nul 2>&1

      powershell -NoProfile -Command ^
        "$ErrorActionPreference='Stop';" ^
        "$argsList=@('%REPLAY_SCRIPT%','--carla-host','%CARLA_HOST%','--carla-port','%CARLA_PORT%','--listen-host','%LISTEN_HOST%','--listen-port','%LISTEN_PORT%','--poll-interval','%POLL_INTERVAL%','--fixed-delta','%FIXED_DELTA%','--max-runtime','%MAX_RUNTIME%','--tm-seed','!SEED!','--future-mode','!METHOD!','--switch-payload-frame','!SWITCH_PF!','--metadata-output','!RUN_META!','--collision-log','!RUN_COLL!','--actor-log','!RUN_ACTOR!','--id-map-file','!RUN_IDMAP!','--enable-completion');" ^
        "if ('!METHOD!' -eq 'lstm') { $argsList += @('--lstm-model','%LSTM_MODEL%','--lstm-device','%LSTM_DEVICE%','--lstm-sample-interval','%FIXED_DELTA%') };" ^
        "$p=Start-Process -FilePath '%PY%' -ArgumentList $argsList -RedirectStandardOutput '!RECV_OUT!' -RedirectStandardError '!RECV_ERR!' -NoNewWindow -PassThru;" ^
        "Set-Content -Path '!PID_FILE!' -Value $p.Id -NoNewline;"

      set "REPLAY_PID="
      if exist "!PID_FILE!" set /p REPLAY_PID=<"!PID_FILE!"
      echo(!REPLAY_PID!| findstr /r "^[0-9][0-9]*$" >nul
      if errorlevel 1 (
        echo [ERROR] REPLAY_PID invalid: "!REPLAY_PID!"
        if exist "!RECV_OUT!" type "!RECV_OUT!"
        if exist "!RECV_ERR!" type "!RECV_ERR!"
        set "RAN_OK=0"
        set "STATUS=pid_invalid"
      )
    )

    if "!RAN_OK!"=="1" (
      timeout /t %STARTUP_DELAY% /nobreak >nul
      python "%SENDER_SCRIPT%" "%CSV_PATH%" --host "%SENDER_HOST%" --port "%SENDER_PORT%" --interval "%FIXED_DELTA%" --start-frame "!START_FRAME!" --end-frame "!END_FRAME!" --log-level INFO
      call :wait_for_pid !REPLAY_PID! %WAIT_SEC%
      if errorlevel 1 (
        set "RAN_OK=0"
        set "STATUS=timeout"
        taskkill /PID !REPLAY_PID! /T /F >nul 2>&1
      )
    )

    if exist "!RUN_META!" (
      for /f %%x in ('python "%META_TOOL%" first_accident_pf_after_switch "!RUN_META!" "!SWITCH_PF_EVAL!"') do set "FIRST_ACC_PF=%%x"
      for /f %%y in ('python "%META_TOOL%" accident_after_switch "!RUN_META!" "!SWITCH_PF_EVAL!"') do set "AFTER_SWITCH=%%y"
    )
    if "!AFTER_SWITCH!"=="" set "AFTER_SWITCH=0"

    echo !ACC_IDX!,!ACCIDENT_PF!,!ACC_TAG!,!METHOD!,!LEAD!,!REP!,!SEED!,!SWITCH_PF!,!RAN_OK!,!AFTER_SWITCH!,!FIRST_ACC_PF!,!STATUS!,!SUMMARY_IN!>> "%SUMMARY_OUT%"
    if "!RAN_OK!"=="1" (
      set /a "RERUN_OK+=1"
    ) else (
      set /a "RERUN_FAIL+=1"
    )
  )
)

echo =========================================================
echo Total rows scanned   : %TOTAL_ROWS%
echo Timeout rows detected: %TIMEOUT_ROWS%
echo Rerun success        : %RERUN_OK%
echo Rerun failed         : %RERUN_FAIL%
echo Wrote                : %SUMMARY_OUT%
echo =========================================================

exit /b 0

:restart_carla
call :stop_carla
echo Starting CARLA: "%CARLA_EXE%"
start "" /high "%CARLA_EXE%"
timeout /t %CARLA_BOOT_WAIT% /nobreak >nul
exit /b 0

:stop_carla
taskkill /IM CarlaUnreal.exe /T /F >nul 2>&1
timeout /t 5 /nobreak >nul
exit /b 0

:wait_carla_ready
setlocal
set "H=%~1"
set "P=%~2"
set "T=%~3"
if "%T%"=="" set "T=120"
echo Waiting for CARLA to be ready... (timeout=%T%s host=%H% port=%P%)
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$h='%H%'; $p=[int]%P%; $deadline=(Get-Date).AddSeconds([int]%T%);" ^
  "while((Get-Date) -lt $deadline) {" ^
  "  $py = ('import carla; c=carla.Client(r''{0}'',{1}); c.set_timeout(5.0); w=c.get_world(); w.get_snapshot()' -f $h,$p);" ^
  "  & python -c $py > $null 2>&1;" ^
  "  if ($LASTEXITCODE -eq 0) { Write-Host ''CARLA READY''; exit 0 }" ^
  "  Start-Sleep -Seconds 2" ^
  "}" ^
  "Write-Host ''CARLA NOT READY (timeout)''; exit 1"
if errorlevel 1 (
  endlocal & exit /b 1
)
endlocal & exit /b 0

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
