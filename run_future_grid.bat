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
set "ACC_REF=%~dp0exp_future\collisions_exp_accident.csv"

set "FIXED_DELTA=0.1"
set "POLL_INTERVAL=0.1"
set "TRACKING_SEC=30"
set "FUTURE_SEC=10"
set "BUFFER_PF_BEFORE=2"
set "BUFFER_PF_AFTER=2"
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
set "PY=python"

set /a "MAX_RUNTIME=%TRACKING_SEC%+%FUTURE_SEC%"
set /a "WAIT_SEC=%MAX_RUNTIME%+30"
if not defined CALIB_MAX_RUNTIME set "CALIB_MAX_RUNTIME=%MAX_RUNTIME%"
set /a "CALIB_WAIT_SEC=%CALIB_MAX_RUNTIME%+30"

for /f %%a in ('python "%META_TOOL%" accident_pf_from_collisions "%ACC_REF%"') do set "ACCIDENT_PF=%%a"

if "%ACCIDENT_PF%"=="" (
  echo Failed: no accident frame found in %ACC_REF%
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

      set "START_FRAME="
      set "END_FRAME="
      set "RUN_WAIT_SEC=%WAIT_SEC%"

      mkdir "!RUN_LOGS!" >nul 2>&1

      set "RAN_OK=1"
      set "STATUS=ok"

      set "RECV_LOG=!RUN_LOGS!\receiver.log"
      set "PID_VALID=1"

      for /f "tokens=1,2,3" %%u in ('
        powershell -NoProfile -Command ^
          "$ErrorActionPreference='Stop';" ^
          "$tracking=[double]$env:TRACKING_SEC;" ^
          "$future=[double]$env:FUTURE_SEC;" ^
          "$delta=[double]$env:FIXED_DELTA;" ^
          "$switch=[int]$env:SWITCH_PF;" ^
          "$accident=[int]$env:ACCIDENT_PF;" ^
          "$bufferBefore=[int]$env:BUFFER_PF_BEFORE;" ^
          "$bufferAfter=[int]$env:BUFFER_PF_AFTER;" ^
          "$trackingFrames=[int][math]::Ceiling($tracking / $delta);" ^
          "$futureFrames=[int][math]::Ceiling($future / $delta);" ^
          "$start=[math]::Max($switch - $trackingFrames - $bufferBefore, 0);" ^
          "$end=[math]::Max($switch + $futureFrames + $bufferAfter, $accident + $bufferAfter);" ^
          "$sendDuration=($end - $start + 1) * $delta;" ^
          "$wait=[int][math]::Ceiling($sendDuration + [double]$env:STARTUP_DELAY + 5);" ^
          "$waitBound=[int][math]::Max($wait,[int]$env:WAIT_SEC);" ^
          "Write-Output \"$start $end $waitBound\""
      ') do (
        set "START_FRAME=%%u"
        set "END_FRAME=%%v"
        set "RUN_WAIT_SEC=%%w"
      )

      for /f %%p in ('
        powershell -NoProfile -Command ^
          "$ErrorActionPreference='Stop';" ^
          "$argsList=@($env:REPLAY_SCRIPT,'--carla-host',$env:CARLA_HOST,'--carla-port',$env:CARLA_PORT,'--listen-host',$env:LISTEN_HOST,'--listen-port',$env:LISTEN_PORT,'--poll-interval',$env:POLL_INTERVAL,'--fixed-delta',$env:FIXED_DELTA,'--max-runtime',$env:MAX_RUNTIME,'--tm-seed',$env:SEED,'--future-mode','%%M','--switch-payload-frame',$env:SWITCH_PF,'--metadata-output',$env:RUN_META,'--collision-log',$env:RUN_COLL,'--actor-log',$env:RUN_ACTOR,'--id-map-file',$env:RUN_IDMAP);" ^
          "if ('%%M' -eq 'lstm') { $argsList += @('--lstm-model', $env:LSTM_MODEL, '--lstm-device', $env:LSTM_DEVICE, '--lstm-sample-interval', $env:FIXED_DELTA) };" ^
          "$p=Start-Process -FilePath $env:PY -ArgumentList $argsList -RedirectStandardOutput $env:RECV_LOG -RedirectStandardError $env:RECV_LOG -NoNewWindow -PassThru;" ^
          "$p.Id"
      ') do set "REPLAY_PID=%%p"

      timeout /t %STARTUP_DELAY% /nobreak >nul
      python "%SENDER_SCRIPT%" "%CSV_PATH%" --host "%SENDER_HOST%" --port "%SENDER_PORT%" --interval "%FIXED_DELTA%" --start-frame "!START_FRAME!" --end-frame "!END_FRAME!"
      call :wait_for_pid !REPLAY_PID! !RUN_WAIT_SEC!
      if errorlevel 1 (
        echo Failed: invalid replay PID "!REPLAY_PID!"
        if exist "!RECV_LOG!" type "!RECV_LOG!"
        set "RAN_OK=0"
        set "STATUS=pid_error"
        set "PID_VALID=0"
      )

      if "!PID_VALID!"=="1" (
        timeout /t %STARTUP_DELAY% /nobreak >nul
        python "%SENDER_SCRIPT%" "%CSV_PATH%" --host "%SENDER_HOST%" --port "%SENDER_PORT%" --interval "%FIXED_DELTA%"
        call :wait_for_pid !REPLAY_PID! %WAIT_SEC%
        if errorlevel 1 (
          set "RAN_OK=0"
          set "STATUS=timeout"
          taskkill /PID !REPLAY_PID! /T /F >nul 2>&1
        )
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
