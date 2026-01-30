@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM ==========================================================
REM Executables
REM ==========================================================
set "ROOT=%~dp0"
set "PY=python"

where "%PY%" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] python not found in PATH. Set PY=py or full path.
  goto :cleanup
)

set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" (
  echo [ERROR] PowerShell not found: %PS%
  goto :cleanup
)

echo [INFO] PS_EXE=%PS%

REM ==========================================================
REM User config
REM ==========================================================
set "CARLA_HOST=127.0.0.1"
set "CARLA_PORT=2000"

set "UDP_HOST=127.0.0.1"
set "UDP_PORT=5005"
set "LISTEN_HOST=0.0.0.0"

set "CSV_PATH=%ROOT%send_data\exp_300.csv"
set "SENDER=%ROOT%send_data\send_udp_frames_from_csv.py"
set "RECEIVER=%ROOT%scripts\udp_replay\replay_from_udp.py"
set "STREAMER=%ROOT%scripts\vehicle_state_stream.py"

REM strip accidental quotes
set "CSV_PATH=%CSV_PATH:"=%"
set "SENDER=%SENDER:"=%"
set "RECEIVER=%RECEIVER:"=%"
set "STREAMER=%STREAMER:"=%"

if not exist "%CSV_PATH%"  ( echo [ERROR] CSV not found: %CSV_PATH% & goto :cleanup )
if not exist "%SENDER%"    ( echo [ERROR] sender not found: %SENDER% & goto :cleanup )
if not exist "%RECEIVER%"  ( echo [ERROR] receiver not found: %RECEIVER% & goto :cleanup )
if not exist "%STREAMER%"  ( echo [ERROR] streamer not found: %STREAMER% & goto :cleanup )

REM ==========================================================
REM Policy
REM ==========================================================
set "FIXED_DELTA=0.05"
set "STALE_TIMEOUT=2.0"

REM Warmup: map compile can be 1-2 min
set "WARMUP_INTERVAL=0.1"
set "WARMUP_MAX_ATTEMPTS=2"
set "WARMUP_CHECK_TIMEOUT_SEC=900"
set "WARMUP_CHECK_INTERVAL_SEC=5"

REM Sweep
set "TS_LIST=0.10 1.00"
set "N_MIN=10"
set "N_MAX=100"
set "N_STEP=10"
set "COOLDOWN_SEC=3"

REM If CSV is 10Hz and Ts=1.00 means 1Hz, skip frames by 10
set "CSV_HZ=10"

REM ==========================================================
REM OUTDIR (safe timestamp from %DATE%/%TIME%)
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

set "RECV_OUT=%OUTDIR%\receiver_stdout.log"
set "RECV_ERR=%OUTDIR%\receiver_stderr.log"
set "RECV_PID_FILE=%OUTDIR%\receiver_pid.txt"
set "RECV_TIMING=%OUTDIR%\update_timings_all.csv"
set "RECV_EVAL=%OUTDIR%\eval_all.csv"

echo [INFO] OUTDIR=%OUTDIR%
echo [INFO] CSV=%CSV_PATH%

REM ==========================================================
REM Pre-clean: free UDP port (10048対策)
REM ==========================================================
echo [INFO] Freeing UDP port %UDP_PORT% if occupied...
call :free_udp_port_netstat %UDP_PORT%

REM ==========================================================
REM Start receiver (Start-Process + separate stdout/stderr)
REM ==========================================================
echo [INFO] Starting receiver...

type nul > "%RECV_OUT%"
type nul > "%RECV_ERR%"
del /q "%RECV_PID_FILE%" >nul 2>&1

"%PS%" -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$argsList=@('%RECEIVER%','--carla-host','%CARLA_HOST%','--carla-port','%CARLA_PORT%','--listen-host','%LISTEN_HOST%','--listen-port','%UDP_PORT%','--fixed-delta','%FIXED_DELTA%','--stale-timeout','%STALE_TIMEOUT%','--measure-update-times','--timing-output','%RECV_TIMING%','--eval-output','%RECV_EVAL%','--enable-completion');" ^
  "$p=Start-Process -FilePath '%PY%' -ArgumentList $argsList -RedirectStandardOutput '%RECV_OUT%' -RedirectStandardError '%RECV_ERR%' -NoNewWindow -WorkingDirectory '%ROOT%' -PassThru;" ^
  "Start-Sleep -Milliseconds 300;" ^
  "if($p.HasExited){ throw ('receiver exited early. ExitCode=' + $p.ExitCode) }" ^
  "Set-Content -Path '%RECV_PID_FILE%' -Value $p.Id -NoNewline;"

if errorlevel 1 (
  echo [ERROR] Failed to start receiver. receiver_stderr tail:
  call :tail_file "%RECV_ERR%" 120
  goto :cleanup
)

if not exist "%RECV_PID_FILE%" (
  echo [ERROR] receiver_pid.txt not created. receiver_stderr tail:
  call :tail_file "%RECV_ERR%" 120
  goto :cleanup
)

set "RECV_PID="
set /p RECV_PID=<"%RECV_PID_FILE%"
echo %RECV_PID%| findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 (
  echo [ERROR] Invalid receiver PID: %RECV_PID%
  call :tail_file "%RECV_ERR%" 120
  goto :cleanup
)
echo [INFO] receiver PID=%RECV_PID%

timeout /t 2 /nobreak >nul

REM ==========================================================
REM Warmup (discard): wait until first complete tracking update
REM   “Received first complete tracking update” が出るまで待つ
REM ==========================================================
for /L %%A in (1,1,%WARMUP_MAX_ATTEMPTS%) do (
  echo [INFO] Warmup attempt %%A/%WARMUP_MAX_ATTEMPTS%
  echo [INFO] Warmup send interval=%WARMUP_INTERVAL% [discard]

  "%PY%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %WARMUP_INTERVAL% > "%OUTDIR%\warmup_sender.log" 2>&1

  findstr /i /c:"Sent 0 frames" "%OUTDIR%\warmup_sender.log" >nul 2>&1
  if !errorlevel! EQU 0 (
    echo [ERROR] Warmup sender sent 0 frames. warmup_sender.log:
    type "%OUTDIR%\warmup_sender.log"
    goto :cleanup
  )

  call :wait_for_string "%RECV_ERR%" "Received first complete tracking update" %WARMUP_CHECK_TIMEOUT_SEC% %WARMUP_CHECK_INTERVAL_SEC%
  if !errorlevel! EQU 0 (
    echo [INFO] Warmup confirmed: first complete tracking update detected.
    goto :warmup_done
  )

  echo [WARN] Warmup not confirmed yet. receiver_stderr tail:
  call :tail_file "%RECV_ERR%" 60
)

echo [WARN] Warmup not confirmed, continue anyway.
:warmup_done

REM ==========================================================
REM Sweep: N=10..100 step10 x Ts
REM   receiverは生かしたまま．各runで streamerを起動 -> sender -> streamer停止
REM ==========================================================
for /L %%N in (%N_MIN%,%N_STEP%,%N_MAX%) do (
  for %%T in (%TS_LIST%) do (

    set "TAG=N%%N_Ts%%T"
    set "RUNDIR=%OUTDIR%\!TAG!"
    mkdir "!RUNDIR!" 2>nul

    set "STR_OUT=!RUNDIR!\streamer_stdout.log"
    set "STR_ERR=!RUNDIR!\streamer_stderr.log"
    set "STR_PID_FILE=!RUNDIR!\streamer_pid.txt"

    set "STREAM_CSV=!RUNDIR!\replay_state.csv"
    set "STREAM_TIMING=!RUNDIR!\stream_timing.csv"
    set "SEND_LOG=!RUNDIR!\sender.log"

    echo ============================================================
    echo [RUN] !TAG!
    echo [DIR] !RUNDIR!

    type nul > "!STR_OUT!"
    type nul > "!STR_ERR!"
    del /q "!STR_PID_FILE!" >nul 2>&1

    "%PS%" -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ErrorActionPreference='Stop';" ^
      "$argsList=@('%STREAMER%','--host','%CARLA_HOST%','--port','%CARLA_PORT%','--mode','wait','--role-prefix','udp_replay:','--include-velocity','--frame-elapsed','--wall-clock','--include-object-id','--include-monotonic','--include-tick-wall-dt','--output','!STREAM_CSV!','--timing-output','!STREAM_TIMING!','--timing-flush-every','10');" ^
      "$p=Start-Process -FilePath '%PY%' -ArgumentList $argsList -RedirectStandardOutput '!STR_OUT!' -RedirectStandardError '!STR_ERR!' -NoNewWindow -WorkingDirectory '%ROOT%' -PassThru;" ^
      "Start-Sleep -Milliseconds 300;" ^
      "if($p.HasExited){ throw ('streamer exited early. ExitCode=' + $p.ExitCode) }" ^
      "Set-Content -Path '!STR_PID_FILE!' -Value $p.Id -NoNewline;"

    if errorlevel 1 (
      echo [ERROR] Failed to start streamer. streamer_stderr tail:
      call :tail_file "!STR_ERR!" 120
      goto :cleanup
    )

    if not exist "!STR_PID_FILE!" (
      echo [ERROR] streamer_pid not created. streamer_stderr tail:
      call :tail_file "!STR_ERR!" 120
      goto :cleanup
    )

    set "STR_PID="
    set /p STR_PID=<"!STR_PID_FILE!"
    echo !STR_PID!| findstr /r "^[0-9][0-9]*$" >nul
    if errorlevel 1 (
      echo [ERROR] Invalid streamer PID: !STR_PID!
      call :tail_file "!STR_ERR!" 120
      goto :cleanup
    )

    timeout /t 1 /nobreak >nul

    REM decide stride: Ts==1.00 => stride=CSV_HZ (10) else 1
    set "FRAME_STRIDE=1"
    if "%%T"=="1.00" set "FRAME_STRIDE=%CSV_HZ%"

    echo [INFO] Sending... N=%%N Ts=%%T stride=!FRAME_STRIDE!
    "%PY%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %%T --frame-stride !FRAME_STRIDE! --max-actors %%N > "!SEND_LOG!" 2>&1

    findstr /i /c:"Sent 0 frames" "!SEND_LOG!" >nul 2>&1
    if !errorlevel! EQU 0 (
      echo [ERROR] sender sent 0 frames. sender log tail:
      call :tail_file "!SEND_LOG!" 80
      goto :cleanup
    )

    taskkill /PID !STR_PID! /T /F >nul 2>&1
    timeout /t %COOLDOWN_SEC% /nobreak >nul
    echo [DONE] !TAG!
  )
)

echo [INFO] Stopping receiver...
if defined RECV_PID taskkill /PID %RECV_PID% /T /F >nul 2>&1

echo [ALL DONE] %OUTDIR%
exit /b 0

:cleanup
echo [CLEANUP]
if defined STR_PID taskkill /PID %STR_PID% /T /F >nul 2>&1
if defined RECV_PID taskkill /PID %RECV_PID% /T /F >nul 2>&1
exit /b 1

REM ==========================================================
REM Subroutines
REM ==========================================================

:free_udp_port_netstat
REM %1 port
setlocal EnableExtensions EnableDelayedExpansion
set "PORT=%~1"
for /L %%K in (1,1,10) do (
  set "FOUND=0"
  for /f "tokens=1-5" %%a in ('netstat -ano -p udp ^| findstr /r /c:":%PORT% *"') do (
    set "FOUND=1"
    set "PID=%%e"
    echo [INFO] UDP :%PORT% owned by PID=!PID!. Killing...
    taskkill /PID !PID! /T /F >nul 2>&1
  )
  if "!FOUND!"=="0" goto :freed_done
  timeout /t 1 /nobreak >nul
)
:freed_done
endlocal & exit /b 0

:wait_for_string
REM %1 log_file, %2 substring, %3 timeout_sec, %4 interval_sec
setlocal EnableExtensions EnableDelayedExpansion
set "LOG=%~1"
set "PAT=%~2"
set "TMO=%~3"
set "INT=%~4"
set /a ELAPSED=0

:ws_loop
if %ELAPSED% GEQ %TMO% ( endlocal & exit /b 1 )
if exist "%LOG%" (
  findstr /i /c:"%PAT%" "%LOG%" >nul 2>&1
  if !errorlevel! EQU 0 ( endlocal & exit /b 0 )
)
timeout /t %INT% /nobreak >nul
set /a ELAPSED+=%INT%
goto :ws_loop

:tail_file
REM %1 file, %2 lines
setlocal EnableExtensions
set "F=%~1"
set "N=%~2"
if not exist "%F%" ( endlocal & exit /b 0 )
"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -LiteralPath '%F%' -Tail %N%" 2>nul
endlocal & exit /b 0
