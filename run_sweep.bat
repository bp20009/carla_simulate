@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ==========================================================
REM User config
REM ==========================================================
set "CARLA_HOST=127.0.0.1"
set "CARLA_PORT=2000"

set "UDP_HOST=127.0.0.1"
set "UDP_PORT=5005"

REM Input CSV (sender reads this)
REM   (e.g. output from scripts/convert_vehicle_state_csv.py)
set "CSV_PATH=send_data\exp_300.csv"

REM Script paths
set "SENDER=send_data\send_udp_frames_from_csv.py"
set "RECEIVER=scripts\udp_replay\replay_from_udp.py"
set "STREAMER=scripts\vehicle_state_stream.py"

REM Receiver fixed delta (do not sweep)
set "FIXED_DELTA=0.05"

REM Warmup frames and delay (receiver stays alive)
set "WARMUP_START_FRAME=0"
set "WARMUP_END_FRAME=1800"
set "WARMUP_INTERVAL=0.1"
set "WARMUP_WAIT_SEC=10"
set "WARMUP_CHECK_TIMEOUT_SEC=180"
set "WARMUP_CHECK_INTERVAL_SEC=5"
set "WARMUP_MAX_ATTEMPTS=3"

REM Cooldown to allow stale actor cleanup (seconds)
set "STALE_TIMEOUT=2.0"
set "COOLDOWN_SEC=3"

REM Output root dir
for /f "tokens=1-3 delims=/- " %%a in ("%date%") do set "DATE_TAG=%%a%%b%%c"
for /f "tokens=1-3 delims=:." %%a in ("%time%") do set "TIME_TAG=%%a%%b%%c"
set "OUTDIR=sweep_results_%DATE_TAG%_%TIME_TAG%"
mkdir "%OUTDIR%" 2>nul

set "RECEIVER_LOG=%OUTDIR%\receiver.log"
set "RECEIVER_TIMING_CSV=%OUTDIR%\update_timings_all.csv"
set "RECEIVER_EVAL_CSV=%OUTDIR%\eval_all.csv"
set "RECEIVER_PID_FILE=%OUTDIR%\receiver_pid.txt"

set "STREAMER_PID_FILE=%OUTDIR%\streamer_pid.txt"

echo [INFO] OUTDIR=%OUTDIR%
echo [INFO] CSV=%CSV_PATH%

REM ==========================================================
REM Sweep lists
REM   NS: 10..100 step 10
REM   TS_LIST: 0.10 1.00
REM ==========================================================
set "TS_LIST=0.10 1.00"

REM ==========================================================
REM Start RECEIVER once (keep alive for entire sweep)
REM ==========================================================
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p = Start-Process -PassThru -NoNewWindow -FilePath 'python' -ArgumentList @(" ^
  + "'%RECEIVER%'," ^
  + "'--carla-host','%CARLA_HOST%','--carla-port','%CARLA_PORT%'," ^
  + "'--listen-host','0.0.0.0','--listen-port','%UDP_PORT%'," ^
  + "'--fixed-delta','%FIXED_DELTA%'," ^
  + "'--stale-timeout','%STALE_TIMEOUT%'," ^
  + "'--measure-update-times'," ^
  + "'--timing-output','%RECEIVER_TIMING_CSV%'," ^
  + "'--eval-output','%RECEIVER_EVAL_CSV%'" ^
  + ") -RedirectStandardOutput '%RECEIVER_LOG%' -RedirectStandardError '%RECEIVER_LOG%';" ^
  + "$p.Id | Out-File -Encoding ascii '%RECEIVER_PID_FILE%'" ^
  >nul

REM Give receiver time to boot
timeout /t 2 /nobreak >nul

REM ==========================================================
REM Warmup: send a short segment to trigger initialization
REM ==========================================================
REM Warmup attempts until receiver logs a spawn
for /L %%A in (1,1,%WARMUP_MAX_ATTEMPTS%) do (
  echo [INFO] Warmup attempt %%A/%WARMUP_MAX_ATTEMPTS%
  type nul > "%RECEIVER_LOG%"
  echo [INFO] Warmup frames %WARMUP_START_FRAME%..%WARMUP_END_FRAME% interval=%WARMUP_INTERVAL%
  python "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %WARMUP_INTERVAL% --start-frame %WARMUP_START_FRAME% --end-frame %WARMUP_END_FRAME% > "%OUTDIR%\warmup_sender.log" 2>&1

  REM Optional wait for heavy initialization
  if %WARMUP_WAIT_SEC% GTR 0 (
    timeout /t %WARMUP_WAIT_SEC% /nobreak >nul
  )

  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$log='%RECEIVER_LOG%';" ^
    + "$timeout=%WARMUP_CHECK_TIMEOUT_SEC%; $interval=%WARMUP_CHECK_INTERVAL_SEC%;" ^
    + "$sw=[Diagnostics.Stopwatch]::StartNew();" ^
    + "while($sw.Elapsed.TotalSeconds -lt $timeout){" ^
    + "if(Select-String -Path $log -Pattern 'Spawned ' -Quiet){ exit 0 };" ^
    + "Start-Sleep -Seconds $interval" ^
    + "}; exit 1" ^
    >nul
  if !errorlevel! EQU 0 (
    echo [INFO] Warmup succeeded (spawn detected).
    goto :warmup_done
  )
  echo [WARN] No spawn detected after warmup attempt %%A. Retrying...
)

echo [ERROR] Warmup failed to detect spawn after %WARMUP_MAX_ATTEMPTS% attempts.
goto :cleanup

:warmup_done

REM ==========================================================
REM Sweep runs (restart only streamer + sender)
REM ==========================================================
for /L %%N in (10,10,100) do (
  for %%T in (%TS_LIST%) do (

    set "TAG=N%%N_Ts%%T"
    set "RUNDIR=%OUTDIR%\!TAG!"
    mkdir "!RUNDIR!" 2>nul

    echo ============================================================
    echo [RUN] !TAG!
    echo [DIR] !RUNDIR!

    set "STREAM_CSV=!RUNDIR!\replay_state_!TAG!.csv"
    set "STREAM_TIMING_CSV=!RUNDIR!\stream_timing_!TAG!.csv"

    set "SENDER_LOG=!RUNDIR!\sender_!TAG!.log"
    set "STREAMER_LOG=!RUNDIR!\streamer_!TAG!.log"

    REM ------------------------------------------------------------
    REM 1) Start STREAMER for this run (background)
    REM ------------------------------------------------------------
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$p = Start-Process -PassThru -NoNewWindow -FilePath 'python' -ArgumentList @(" ^
      + "'%STREAMER%'," ^
      + "'--host','%CARLA_HOST%','--port','%CARLA_PORT%'," ^
      + "'--mode','wait'," ^
      + "'--role-prefix','udp_replay:'," ^
      + "'--include-velocity','--frame-elapsed','--wall-clock','--include-object-id'," ^
      + "'--include-monotonic','--include-tick-wall-dt'," ^
      + "'--output','!STREAM_CSV!'," ^
      + "'--timing-output','!STREAM_TIMING_CSV!'," ^
      + "'--timing-flush-every','10'" ^
      + ") -RedirectStandardOutput '!STREAMER_LOG!' -RedirectStandardError '!STREAMER_LOG!';" ^
      + "$p.Id | Out-File -Encoding ascii '%STREAMER_PID_FILE%'" ^
      >nul

    REM Give streamer time to start
    timeout /t 1 /nobreak >nul

    REM ------------------------------------------------------------
    REM 2) Run SENDER (blocking)
    REM ------------------------------------------------------------
    set "FRAME_STRIDE=1"
    if "%%T"=="1.00" set "FRAME_STRIDE=10"

    echo [INFO] Sending... N=%%N Ts=%%T stride=!FRAME_STRIDE!
    python "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %%T --frame-stride !FRAME_STRIDE! --max-actors %%N > "!SENDER_LOG!" 2>&1

    REM ------------------------------------------------------------
    REM 3) Stop STREAMER for this run
    REM ------------------------------------------------------------
    for /f %%P in ('type "%STREAMER_PID_FILE%"') do taskkill /PID %%P /T /F >nul 2>&1

    REM Cooldown so stale actors are cleared before next run
    timeout /t %COOLDOWN_SEC% /nobreak >nul

    echo [DONE] !TAG!
  )
)

REM ==========================================================
REM Stop RECEIVER
REM ==========================================================
for /f %%P in ('type "%RECEIVER_PID_FILE%"') do taskkill /PID %%P /T /F >nul 2>&1

echo [ALL DONE] %OUTDIR%
endlocal
exit /b 0

:cleanup
for /f %%P in ('type "%RECEIVER_PID_FILE%"') do taskkill /PID %%P /T /F >nul 2>&1
endlocal
exit /b 1
