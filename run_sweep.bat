@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0"

REM ==========================================================
REM User config
REM ==========================================================
set "CARLA_HOST=127.0.0.1"
set "CARLA_PORT=2000"

set "UDP_HOST=127.0.0.1"
set "UDP_PORT=5005"
set "LISTEN_HOST=0.0.0.0"

set "ROOT=%~dp0"
set "CSV_PATH=%ROOT%send_data\exp_300.csv"
set "SENDER=%ROOT%send_data\send_udp_frames_from_csv.py"
set "RECEIVER=%ROOT%scripts\udp_replay\replay_from_udp.py"
set "STREAMER=%ROOT%scripts\vehicle_state_stream.py"

set "PYTHON_EXE=python"
where "%PYTHON_EXE%" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] python not found in PATH. Set PYTHON_EXE=py or full path.
  exit /b 1
)

REM Receiver fixed delta (do not sweep)
set "FIXED_DELTA=0.05"

REM Warmup
set "WARMUP_START_FRAME=0"
set "WARMUP_END_FRAME=1800"
set "WARMUP_INTERVAL=0.1"
set "WARMUP_WAIT_SEC=10"
set "WARMUP_CHECK_TIMEOUT_SEC=180"
set "WARMUP_CHECK_INTERVAL_SEC=5"
set "WARMUP_MAX_ATTEMPTS=3"

REM Stale/Cooldown
set "STALE_TIMEOUT=2.0"
set "COOLDOWN_SEC=5"

REM ==========================================================
REM OUTDIR (bat-only, unique timestamp)
REM ==========================================================
set "DT_TAG=%DATE%"
set "DT_TAG=%DT_TAG:/=%"
set "DT_TAG=%DT_TAG:-=%"
set "DT_TAG=%DT_TAG:.=%"
set "DT_TAG=%DT_TAG: =%"
set "TM_TAG=%TIME: =0%"
set "TM_TAG=%TM_TAG::=%"
set "TM_TAG=%TM_TAG:.=%"
set "OUTDIR=%ROOT%sweep_results_%DT_TAG%_%TM_TAG%"
mkdir "%OUTDIR%" 2>nul

set "RECEIVER_LOG_OUT=%OUTDIR%\receiver_stdout.log"
set "RECEIVER_LOG_ERR=%OUTDIR%\receiver_stderr.log"
set "RECEIVER_TIMING_CSV=%OUTDIR%\update_timings_all.csv"
set "RECEIVER_EVAL_CSV=%OUTDIR%\eval_all.csv"
set "RECEIVER_PID_FILE=%OUTDIR%\receiver_pid.txt"

echo [INFO] OUTDIR=%OUTDIR%
echo [INFO] CSV=%CSV_PATH%

REM ==========================================================
REM Sweep lists
REM ==========================================================
set "TS_LIST=0.10 1.00"

REM ==========================================================
REM Start RECEIVER once (keep alive)
REM ==========================================================
REM --- Kill previous receiver if PID file exists ---
call :kill_by_pidfile "%RECEIVER_PID_FILE%"
timeout /t 1 /nobreak >nul
echo [INFO] Starting receiver...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$args=@('%RECEIVER%','--carla-host','%CARLA_HOST%','--carla-port','%CARLA_PORT%','--listen-host','%LISTEN_HOST%','--listen-port','%UDP_PORT%','--fixed-delta','%FIXED_DELTA%','--stale-timeout','%STALE_TIMEOUT%','--measure-update-times','--timing-output','%RECEIVER_TIMING_CSV%','--eval-output','%RECEIVER_EVAL_CSV%');" ^
  "$p=Start-Process -PassThru -WindowStyle Hidden -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '%RECEIVER_LOG_OUT%' -RedirectStandardError '%RECEIVER_LOG_ERR%';" ^
  "$p.Id | Out-File -Encoding ascii '%RECEIVER_PID_FILE%'" ^
  >nul
set "RECEIVER_PID="
set /p RECEIVER_PID=<"%RECEIVER_PID_FILE%"
if not defined RECEIVER_PID (
  echo [ERROR] Failed to start receiver. Check logs:
  if exist "%RECEIVER_LOG_ERR%" type "%RECEIVER_LOG_ERR%"
  goto :cleanup
)
echo [INFO] receiver PID=%RECEIVER_PID%

REM Give receiver time to boot
timeout /t 2 /nobreak >nul

REM ==========================================================
REM Warmup (monitor timing CSV appended actor lines)
REM ==========================================================
for /L %%A in (1,1,%WARMUP_MAX_ATTEMPTS%) do (
  echo [INFO] Warmup attempt %%A/%WARMUP_MAX_ATTEMPTS%

  REM record current line count as offset
  set "TIMING_OFFSET_LINES=0"
  if exist "%RECEIVER_TIMING_CSV%" (
    call :count_lines "%RECEIVER_TIMING_CSV%" TIMING_OFFSET_LINES
  )

  echo [INFO] Warmup send frames %WARMUP_START_FRAME%..%WARMUP_END_FRAME% interval=%WARMUP_INTERVAL%
  "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %WARMUP_INTERVAL% --start-frame %WARMUP_START_FRAME% --end-frame %WARMUP_END_FRAME% > "%OUTDIR%\warmup_sender.log" 2>&1

  if %WARMUP_WAIT_SEC% GTR 0 timeout /t %WARMUP_WAIT_SEC% /nobreak >nul

  REM wait until timing CSV has appended actor update line
  call :wait_for_actor_update "%RECEIVER_TIMING_CSV%" !TIMING_OFFSET_LINES! %WARMUP_CHECK_TIMEOUT_SEC% %WARMUP_CHECK_INTERVAL_SEC%
  if !errorlevel! EQU 0 (
    echo [INFO] Warmup succeeded (actor update line detected in timing CSV).
    goto :warmup_done
  )

  echo [WARN] Warmup not confirmed (no actor update detected). Retrying...
)

echo [ERROR] Warmup failed after %WARMUP_MAX_ATTEMPTS% attempts.
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

    set "STREAM_CSV=!RUNDIR!\replay_state_!TAG!.csv"
    set "STREAM_TIMING_CSV=!RUNDIR!\stream_timing_!TAG!.csv"
    set "SENDER_LOG=!RUNDIR!\sender_!TAG!.log"
    set "STREAMER_LOG_OUT=!RUNDIR!\streamer_stdout.log"
    set "STREAMER_LOG_ERR=!RUNDIR!\streamer_stderr.log"
    set "STREAMER_PID_FILE=!RUNDIR!\streamer_pid.txt"

    echo ============================================================
    echo [RUN] !TAG!
    echo [DIR] !RUNDIR!

    REM 1) Start STREAMER bg
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$args=@('%STREAMER%','--host','%CARLA_HOST%','--port','%CARLA_PORT%','--mode','wait','--role-prefix','udp_replay:','--include-velocity','--frame-elapsed','--wall-clock','--include-object-id','--include-monotonic','--include-tick-wall-dt','--output','!STREAM_CSV!','--timing-output','!STREAM_TIMING_CSV!','--timing-flush-every','10');" ^
      "$p=Start-Process -PassThru -WindowStyle Hidden -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '!STREAMER_LOG_OUT!' -RedirectStandardError '!STREAMER_LOG_ERR!';" ^
      "$p.Id | Out-File -Encoding ascii '!STREAMER_PID_FILE!'" ^
      >nul
    set "STREAMER_PID="
    set /p STREAMER_PID=<"!STREAMER_PID_FILE!"
    if not defined STREAMER_PID (
      echo [ERROR] Failed to start streamer. Check logs:
      if exist "!STREAMER_LOG_ERR!" type "!STREAMER_LOG_ERR!"
      goto :cleanup
    )

    timeout /t 1 /nobreak >nul

    REM 2) Run SENDER (blocking)
    set "FRAME_STRIDE=1"
    if "%%T"=="1.00" set "FRAME_STRIDE=10"

    echo [INFO] Sending... N=%%N Ts=%%T stride=!FRAME_STRIDE!
    "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %%T --frame-stride !FRAME_STRIDE! --max-actors %%N > "!SENDER_LOG!" 2>&1

    REM 3) Stop STREAMER
    taskkill /PID !STREAMER_PID! /T /F >nul 2>&1

    timeout /t %COOLDOWN_SEC% /nobreak >nul
    echo [DONE] !TAG!
  )
)

REM ==========================================================
REM Stop RECEIVER
REM ==========================================================
call :kill_by_pidfile "%RECEIVER_PID_FILE%"

echo [ALL DONE] %OUTDIR%
popd
endlocal
exit /b 0

:cleanup
echo [CLEANUP]
call :kill_by_pidfile "%RECEIVER_PID_FILE%"
popd
endlocal
exit /b 1


REM ==========================================================
REM Subroutines
REM ==========================================================

:kill_by_pidfile
setlocal
set "F=%~1"
if not exist "%F%" ( endlocal & exit /b 0 )
set "PID="
set /p PID=<"%F%"
if defined PID taskkill /PID %PID% /T /F >nul 2>&1
del /q "%F%" >nul 2>&1
endlocal & exit /b 0

:count_lines
REM %1 file, %2 out var
set "%~2=0"
for /f %%A in ('find /v /c "" ^< "%~1"') do set "%~2=%%A"
exit /b 0

:check_appended_actor_lines
REM %1 file, %2 skipLines, %3 out(0/1)
set "%~3=0"
set "FILE=%~1"
set "SKIP=%~2"
REM 新規行の中に ,,,,, で始まる行があれば成功（actor更新行）
for /f "usebackq skip=%SKIP% delims=" %%L in ("%FILE%") do (
  echo %%L | findstr /b /c:",,,,," >nul && (set "%~3=1" & goto :app_done)
)
:app_done
exit /b 0

:wait_for_actor_update
REM %1 file, %2 offsetLines, %3 timeoutSec, %4 intervalSec
setlocal EnableDelayedExpansion
set "FILE=%~1"
set "OFFSET=%~2"
set "TIMEOUT=%~3"
set "INTERVAL=%~4"

set /a "ELAPSED=0"
:wait_loop
if !ELAPSED! GEQ !TIMEOUT! (
  endlocal & exit /b 1
)

if exist "!FILE!" (
  call :check_appended_actor_lines "!FILE!" !OFFSET! FOUND
  if "!FOUND!"=="1" (
    endlocal & exit /b 0
  )
)

timeout /t !INTERVAL! /nobreak >nul
set /a "ELAPSED+=INTERVAL"
goto :wait_loop
