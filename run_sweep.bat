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

set "RECEIVER_LOG=%OUTDIR%\receiver.log"
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
REM --- Kill previous receiver if still running ---
call :find_pid_by_script "replay_from_udp.py" OLD_RECEIVER_PID
if defined OLD_RECEIVER_PID (
  echo [WARN] Found existing receiver PID=!OLD_RECEIVER_PID! . Killing it...
  taskkill /PID !OLD_RECEIVER_PID! /T /F >nul 2>&1
  timeout /t 1 /nobreak >nul
)
echo [INFO] Starting receiver...
set "RECV_ARGS=--carla-host %CARLA_HOST% --carla-port %CARLA_PORT% --listen-host %LISTEN_HOST% --listen-port %UDP_PORT% --fixed-delta %FIXED_DELTA% --stale-timeout %STALE_TIMEOUT% --measure-update-times --timing-output %RECEIVER_TIMING_CSV% --eval-output %RECEIVER_EVAL_CSV%"
call :start_bg_python "%PYTHON_EXE%" "%RECEIVER_LOG%" "%RECEIVER%" "%RECV_ARGS%"

REM PIDを特定して保存（コマンドラインに replay_from_udp.py が含まれる python を拾う）
call :find_pid_by_script "replay_from_udp.py" RECEIVER_PID
if not defined RECEIVER_PID (
  echo [ERROR] receiver PID not found. Check log:
  if exist "%RECEIVER_LOG%" type "%RECEIVER_LOG%"
  goto :cleanup
)
echo %RECEIVER_PID% > "%RECEIVER_PID_FILE%"
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

  REM poll until timeout: look for newly appended lines that begin with ,,,,,
  set /a ELAPSED=0
  set "WARMUP_OK=0"

  :warmup_poll
  if !ELAPSED! GEQ %WARMUP_CHECK_TIMEOUT_SEC% goto :warmup_poll_done

  if exist "%RECEIVER_TIMING_CSV%" (
    call :check_appended_actor_lines "%RECEIVER_TIMING_CSV%" !TIMING_OFFSET_LINES! WARMUP_OK
    if "!WARMUP_OK!"=="1" (
      echo [INFO] Warmup succeeded (actor update line detected in timing CSV).
      goto :warmup_done
    )
  )

  timeout /t %WARMUP_CHECK_INTERVAL_SEC% /nobreak >nul
  set /a ELAPSED+=%WARMUP_CHECK_INTERVAL_SEC%
  goto :warmup_poll

  :warmup_poll_done
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
    set "STREAMER_LOG=!RUNDIR!\streamer_!TAG!.log"

    echo ============================================================
    echo [RUN] !TAG!
    echo [DIR] !RUNDIR!

    REM 1) Start STREAMER bg
    set "STR_ARGS=--host %CARLA_HOST% --port %CARLA_PORT% --mode wait --role-prefix udp_replay: --include-velocity --frame-elapsed --wall-clock --include-object-id --include-monotonic --include-tick-wall-dt --output !STREAM_CSV! --timing-output !STREAM_TIMING_CSV! --timing-flush-every 10"
    call :start_bg_python "%PYTHON_EXE%" "!STREAMER_LOG!" "%STREAMER%" "!STR_ARGS!"

    call :find_pid_by_script "vehicle_state_stream.py" STREAMER_PID
    if not defined STREAMER_PID (
      echo [ERROR] streamer PID not found. Check log:
      if exist "!STREAMER_LOG!" type "!STREAMER_LOG!"
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
if exist "%RECEIVER_PID_FILE%" (
  for /f %%P in ("%RECEIVER_PID_FILE%") do taskkill /PID %%P /T /F >nul 2>&1
)

echo [ALL DONE] %OUTDIR%
popd
endlocal
exit /b 0

:cleanup
echo [CLEANUP]
if exist "%RECEIVER_PID_FILE%" (
  for /f %%P in ("%RECEIVER_PID_FILE%") do taskkill /PID %%P /T /F >nul 2>&1
)
popd
endlocal
exit /b 1


REM ==========================================================
REM Subroutines
REM ==========================================================

:start_bg_python
REM usage: call :start_bg_python <python_exe> <log_path> <script_path> <arg_string>
set "PYEXE=%~1"
set "LOG=%~2"
set "SCRIPT=%~3"
set "ARGSTR=%~4"
REM start uses a new cmd so redirection is reliable
start "" /b "%ComSpec%" /s /c ""%PYEXE%" "%SCRIPT%" %ARGSTR% >>"%LOG%" 2>&1"
exit /b 0

:find_pid_by_script
REM %1 = script substring, %2 = out var
set "%~2="
for %%N in (python.exe pythonw.exe) do (
  for /f "tokens=2 delims== " %%P in ('
    wmic process where "Name='%%N' and CommandLine like '%%%%%~1%%%%'" get ProcessId /value 2^>nul ^| findstr /i "ProcessId="
  ') do (
    set "%~2=%%P"
    goto :pid_done
  )
)
:pid_done
exit /b 0

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
