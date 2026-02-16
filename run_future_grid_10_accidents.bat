@echo off
cd /d "%~dp0"
setlocal enabledelayedexpansion

if "%~1"=="" (
  set "CSV_PATH=%~dp0send_data\exp_accident.csv"
  set "OUTDIR=%~dp0results_grid_accident_multi"
) else (
  set "CSV_PATH=%~1"
  set "OUTDIR=%~2"
  if "%OUTDIR%"=="" set "OUTDIR=results_grid_accident_multi"
)

set "REPLAY_SCRIPT=scripts\udp_replay\replay_from_udp_future_exp.py"
set "SENDER_SCRIPT=send_data\send_udp_frames_from_csv.py"
set "LSTM_MODEL=scripts\udp_replay\traj_lstm.pt"
set "LSTM_DEVICE=cpu"
set "META_TOOL=scripts\udp_replay\meta_tools.py"
set "ACC_REF=%~dp0exp_future\collisions_exp_accident.csv"

set "FIXED_DELTA=0.1"
set "PRE_SEC=60"
set "POST_SEC=30"
set "PF_PER_SEC=10"
set "POLL_INTERVAL=0.1"
set "TRACKING_SEC=30"
set "FUTURE_SEC=10"
set "BUFFER_PF_BEFORE=2"
set "BUFFER_PF_AFTER=2"
set "WINDOW_SEC_BEFORE=60"
set "WINDOW_SEC_AFTER=30"
set "LEAD_MIN=1"
set "LEAD_MAX=10"
set "REPS=10"
set "BASE_SEED=20009"
set "STARTUP_DELAY=2"

REM ==== 事故フレーム10件（必要に応じて編集） ====
set "ACCIDENT_PF_LIST=25411 25041 25082 25447 25241 25276 25548 25287 25290 25309"

REM ==== CARLA server restart settings ====
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

set "MAX_RUNTIME=100"
set /a "WAIT_BASE=%MAX_RUNTIME%+30"
set /a "WAIT_SEC=%WAIT_BASE%"
if not defined CALIB_MAX_RUNTIME set "CALIB_MAX_RUNTIME=%MAX_RUNTIME%"
set /a "CALIB_WAIT_SEC=%CALIB_MAX_RUNTIME%+30"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set "SUMMARY=%OUTDIR%\summary_grid_multi_accidents.csv"
echo accident_idx,accident_payload_frame_ref,accident_tag,method,lead_sec,rep,seed,switch_payload_frame,ran_ok,accident_after_switch,first_accident_payload_frame,status> "%SUMMARY%"

echo ACCIDENT_PF_LIST=%ACCIDENT_PF_LIST%
echo LEADS=%LEAD_MIN%..%LEAD_MAX% REPS=%REPS% METHODS=autopilot,lstm

set /a "ACC_IDX=0"
for %%A in (%ACCIDENT_PF_LIST%) do (
  set /a "ACC_IDX+=1"
  set "ACCIDENT_PF=%%A"
  set "ACC_TAG=accident_!ACC_IDX!_pf_!ACCIDENT_PF!"
  set "ACC_OUTDIR=%OUTDIR%\!ACC_TAG!"

  if not exist "!ACC_OUTDIR!" mkdir "!ACC_OUTDIR!"

  set /a "START_FRAME=ACCIDENT_PF-(PRE_SEC*PF_PER_SEC)"
  set /a "END_FRAME=ACCIDENT_PF+(POST_SEC*PF_PER_SEC)"
  if !START_FRAME! LSS 0 set "START_FRAME=0"

  echo =========================================================
  echo [ACCIDENT !ACC_IDX!/10] PF=!ACCIDENT_PF! TAG=!ACC_TAG!
  echo [SENDER RANGE] !START_FRAME!..!END_FRAME!  (pre=%PRE_SEC%s post=%POST_SEC%s)

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

  for /L %%L in (%LEAD_MIN%,1,%LEAD_MAX%) do (

    for %%M in (autopilot lstm) do (

      echo =========================================================
      echo Restarting CARLA server for accident=!ACCIDENT_PF! lead=%%L method=%%M
      echo =========================================================
      call :restart_carla
      call :wait_carla_ready %CARLA_HOST% %CARLA_PORT% %CARLA_BOOT_TIMEOUT%
      if errorlevel 1 (
        echo CARLA did not become ready. Abort accident=!ACCIDENT_PF! lead=%%L method=%%M
        exit /b 1
      )

      echo Warmup wait %CARLA_WARMUP_SEC%s ...
      timeout /t %CARLA_WARMUP_SEC% /nobreak >nul

      for /f %%s in ('python "%META_TOOL%" switch_pf "!ACCIDENT_PF!" "%%L" "%FIXED_DELTA%"') do set "SWITCH_PF=%%s"
      set /a "SWITCH_PF_EVAL=!SWITCH_PF!+BUFFER_PF_AFTER"

      for /L %%R in (1,1,%REPS%) do (
        set /a "SEED=%BASE_SEED%+%%R"

        set "RUN_DIR=!ACC_OUTDIR!\%%M\lead_%%L\rep_%%R"
        set "RUN_LOGS=!RUN_DIR!\logs"
        set "RUN_META=!RUN_LOGS!\meta.json"
        set "RUN_COLL=!RUN_LOGS!\collisions.csv"
        set "RUN_ACTOR=!RUN_LOGS!\actor.csv"
        set "RUN_IDMAP=!RUN_LOGS!\id_map.csv"

        mkdir "!RUN_LOGS!" >nul 2>&1

        set "RAN_OK=1"
        set "STATUS=ok"

        set "RECV_OUT=!RUN_LOGS!\receiver.out.log"
        set "RECV_ERR=!RUN_LOGS!\receiver.err.log"
        set "PID_VALID=1"

        set "PID_FILE=!RUN_LOGS!\replay.pid"
        del /q "!PID_FILE!" >nul 2>&1

        powershell -NoProfile -Command ^
          "$ErrorActionPreference='Stop';" ^
          "$argsList=@('%REPLAY_SCRIPT%','--carla-host','%CARLA_HOST%','--carla-port','%CARLA_PORT%','--listen-host','%LISTEN_HOST%','--listen-port','%LISTEN_PORT%','--poll-interval','%POLL_INTERVAL%','--fixed-delta','%FIXED_DELTA%','--max-runtime','%MAX_RUNTIME%','--tm-seed','!SEED!','--future-mode','%%M','--switch-payload-frame','!SWITCH_PF!','--metadata-output','!RUN_META!','--collision-log','!RUN_COLL!','--actor-log','!RUN_ACTOR!','--id-map-file','!RUN_IDMAP!','--enable-completion');" ^
          "if ('%%M' -eq 'lstm') { $argsList += @('--lstm-model','%LSTM_MODEL%','--lstm-device','%LSTM_DEVICE%','--lstm-sample-interval','%FIXED_DELTA%') };" ^
          "$p=Start-Process -FilePath '%PY%' -ArgumentList $argsList -RedirectStandardOutput '!RECV_OUT!' -RedirectStandardError '!RECV_ERR!' -NoNewWindow -PassThru;" ^
          "Set-Content -Path '!PID_FILE!' -Value $p.Id -NoNewline;"

        set "REPLAY_PID="
        if exist "!PID_FILE!" set /p REPLAY_PID=<"!PID_FILE!"

        echo(!REPLAY_PID!| findstr /r "^[0-9][0-9]*$" >nul
        if errorlevel 1 (
          echo Failed: REPLAY_PID invalid: "!REPLAY_PID!"
          if exist "!RECV_OUT!" type "!RECV_OUT!"
          if exist "!RECV_ERR!" type "!RECV_ERR!"
          set "RAN_OK=0"
          set "STATUS=pid_invalid"
          set "PID_VALID=0"
        )

        if "!PID_VALID!"=="1" (
          timeout /t %STARTUP_DELAY% /nobreak >nul
          python "%SENDER_SCRIPT%" "%CSV_PATH%" --host "%SENDER_HOST%" --port "%SENDER_PORT%" --interval "%FIXED_DELTA%" --start-frame "!START_FRAME!" --end-frame "!END_FRAME!" --log-level INFO
          call :wait_for_pid !REPLAY_PID! %WAIT_SEC%
          if errorlevel 1 (
            set "RAN_OK=0"
            set "STATUS=timeout"
            taskkill /PID !REPLAY_PID! /T /F >nul 2>&1
          )
        )

        set "FIRST_ACC_PF="
        set "AFTER_SWITCH=0"
        if exist "!RUN_META!" (
          for /f %%f in ('python "%META_TOOL%" first_accident_pf_after_switch "!RUN_META!" "!SWITCH_PF_EVAL!"') do set "FIRST_ACC_PF=%%f"
          for /f %%a in ('python "%META_TOOL%" accident_after_switch "!RUN_META!" "!SWITCH_PF_EVAL!"') do set "AFTER_SWITCH=%%a"
        )

        if "!AFTER_SWITCH!"=="" set "AFTER_SWITCH=0"

        echo !ACC_IDX!,!ACCIDENT_PF!,!ACC_TAG!,%%M,%%L,%%R,!SEED!,!SWITCH_PF!,!RAN_OK!,!AFTER_SWITCH!,!FIRST_ACC_PF!,!STATUS!>> "%SUMMARY%"
      )

      REM method（10回）終了ごとに明示的に落としておく
      call :stop_carla
    )
  )
)

echo Wrote: %SUMMARY%
exit /b 0

:restart_carla
call :stop_carla
echo Starting CARLA: "%CARLA_EXE%"
start "" /high "%CARLA_EXE%"
timeout /t %CARLA_BOOT_WAIT% /nobreak >nul
exit /b 0

:stop_carla
REM 既存 CARLA を確実に落とす（複数起動していてもまとめて落とす）
taskkill /IM CarlaUnreal.exe /T /F >nul 2>&1
timeout /t 5 /nobreak >nul
exit /b 0

:wait_carla_ready
REM usage: call :wait_carla_ready <HOST> <PORT> <TIMEOUT_SEC>
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
