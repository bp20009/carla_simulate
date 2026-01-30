@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0"

REM ==========================================================
REM Executables
REM ==========================================================
set "ROOT=%~dp0"
set "PYTHON_EXE=python"

where "%PYTHON_EXE%" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] python not found in PATH. Set PYTHON_EXE=py or full path.
  goto :cleanup
)

set "PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS_EXE%" (
  echo [ERROR] PowerShell not found: %PS_EXE%
  goto :cleanup
)
echo [INFO] PS_EXE=%PS_EXE%

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

REM Warmup: map compile waits can be long
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

REM ==========================================================
REM OUTDIR (safe tag from %DATE%/%TIME%)
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

set "RECEIVER_STDOUT=%OUTDIR%\receiver_stdout.log"
set "RECEIVER_STDERR=%OUTDIR%\receiver_stderr.log"
set "RECEIVER_LAUNCHER=%OUTDIR%\receiver_launcher.log"
set "RECEIVER_PID_FILE=%OUTDIR%\receiver_pid.txt"
set "RECEIVER_TIMING_CSV=%OUTDIR%\update_timings_all.csv"
set "RECEIVER_EVAL_CSV=%OUTDIR%\eval_all.csv"

echo [INFO] OUTDIR=%OUTDIR%
echo [INFO] CSV=%CSV_PATH%

REM ==========================================================
REM Pre-clean: free UDP port
REM ==========================================================
echo [INFO] Freeing UDP port %UDP_PORT% if occupied...
call :free_udp_port_netstat %UDP_PORT%

REM ==========================================================
REM Start receiver (cmd.exe wrapper + redirection)
REM ==========================================================
echo [INFO] Starting receiver...

type nul > "%RECEIVER_STDOUT%"
type nul > "%RECEIVER_STDERR%"
type nul > "%RECEIVER_LAUNCHER%"
del /q "%RECEIVER_PID_FILE%" >nul 2>&1

set "RECEIVER_CMD="%PYTHON_EXE%" "%RECEIVER%" --carla-host %CARLA_HOST% --carla-port %CARLA_PORT% --listen-host %LISTEN_HOST% --listen-port %UDP_PORT% --fixed-delta %FIXED_DELTA% --stale-timeout %STALE_TIMEOUT% --measure-update-times --timing-output "%RECEIVER_TIMING_CSV%" --eval-output "%RECEIVER_EVAL_CSV%" --enable-completion"

call :start_cmd_wrapper "%RECEIVER_PID_FILE%" "%RECEIVER_LAUNCHER%" "%RECEIVER_CMD%" "%RECEIVER_STDOUT%" "%RECEIVER_STDERR%"
if errorlevel 1 (
  echo [ERROR] Failed to start receiver. launcher tail:
  call :tail_file "%RECEIVER_LAUNCHER%" 120
  echo [ERROR] receiver_stderr tail:
  call :tail_file "%RECEIVER_STDERR%" 120
  goto :cleanup
)

set "RECEIVER_PID="
set /p RECEIVER_PID=<"%RECEIVER_PID_FILE%"
echo %RECEIVER_PID%| findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 (
  echo [ERROR] Invalid receiver PID: %RECEIVER_PID%
  call :tail_file "%RECEIVER_LAUNCHER%" 120
  call :tail_file "%RECEIVER_STDERR%" 120
  goto :cleanup
)
echo [INFO] receiver PID=%RECEIVER_PID%

timeout /t 2 /nobreak >nul

REM ==========================================================
REM Warmup (discard): wait until first complete tracking update
REM ==========================================================
for /L %%A in (1,1,%WARMUP_MAX_ATTEMPTS%) do (
  echo [INFO] Warmup attempt %%A/%WARMUP_MAX_ATTEMPTS%
  echo [INFO] Warmup send interval=%WARMUP_INTERVAL% [discard]

  "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %WARMUP_INTERVAL% > "%OUTDIR%\warmup_sender.log" 2>&1

  findstr /i /c:"Sent 0 frames" "%OUTDIR%\warmup_sender.log" >nul 2>&1
  if !errorlevel! EQU 0 (
    echo [ERROR] Warmup sender sent 0 frames. warmup_sender.log:
    type "%OUTDIR%\warmup_sender.log"
    goto :cleanup
  )

  call :wait_warmup_ready "%RECEIVER_STDERR%" %WARMUP_CHECK_TIMEOUT_SEC% %WARMUP_CHECK_INTERVAL_SEC%
  if !errorlevel! EQU 0 (
    echo [INFO] Warmup confirmed: first complete tracking update detected.
    goto :warmup_done
  )

  echo [WARN] Warmup not confirmed yet. receiver_stderr tail:
  call :tail_file "%RECEIVER_STDERR%" 60
)

echo [WARN] Warmup not confirmed, continue anyway.
:warmup_done

REM ==========================================================
REM Sweep: N=10..100 step10 x Ts
REM ==========================================================
for /L %%N in (%N_MIN%,%N_STEP%,%N_MAX%) do (
  for %%T in (%TS_LIST%) do (
    set "TAG=N%%N_Ts%%T"
    set "RUNDIR=%OUTDIR%\!TAG!"
    mkdir "!RUNDIR!" 2>nul

    set "STREAM_CSV=!RUNDIR!\replay_state_!TAG!.csv"
    set "STREAM_TIMING_CSV=!RUNDIR!\stream_timing_!TAG!.csv"

    set "SENDER_LOG=!RUNDIR!\sender_!TAG!.log"

    set "STREAMER_STDOUT=!RUNDIR!\streamer_stdout_!TAG!.log"
    set "STREAMER_STDERR=!RUNDIR!\streamer_stderr_!TAG!.log"
    set "STREAMER_LAUNCHER=!RUNDIR!\streamer_launcher_!TAG!.log"
    set "STREAMER_PID_FILE=!RUNDIR!\streamer_pid_!TAG!.txt"

    echo ============================================================
    echo [RUN] !TAG!
    echo [DIR] !RUNDIR!

    type nul > "!STREAMER_STDOUT!"
    type nul > "!STREAMER_STDERR!"
    type nul > "!STREAMER_LAUNCHER!"
    del /q "!STREAMER_PID_FILE!" >nul 2>&1

    set "STREAMER_CMD="%PYTHON_EXE%" "%STREAMER%" --host %CARLA_HOST% --port %CARLA_PORT% --mode wait --role-prefix udp_replay: --include-velocity --frame-elapsed --wall-clock --include-object-id --include-monotonic --include-tick-wall-dt --output "!STREAM_CSV!" --timing-output "!STREAM_TIMING_CSV!" --timing-flush-every 10"

    call :start_cmd_wrapper "!STREAMER_PID_FILE!" "!STREAMER_LAUNCHER!" "!STREAMER_CMD!" "!STREAMER_STDOUT!" "!STREAMER_STDERR!"
    if errorlevel 1 (
      echo [ERROR] Failed to start streamer. launcher tail:
      call :tail_file "!STREAMER_LAUNCHER!" 120
      echo [ERROR] streamer_stderr tail:
      call :tail_file "!STREAMER_STDERR!" 120
      goto :cleanup
    )

    set "STREAMER_PID="
    set /p STREAMER_PID=<"!STREAMER_PID_FILE!"
    echo !STREAMER_PID!| findstr /r "^[0-9][0-9]*$" >nul
    if errorlevel 1 (
      echo [ERROR] Invalid streamer PID: !STREAMER_PID!
      call :tail_file "!STREAMER_LAUNCHER!" 120
      goto :cleanup
    )

    timeout /t 1 /nobreak >nul

    echo [INFO] Sending... N=%%N Ts=%%T
    "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %%T --max-actors %%N > "!SENDER_LOG!" 2>&1

    findstr /i /c:"Sent 0 frames" "!SENDER_LOG!" >nul 2>&1
    if !errorlevel! EQU 0 (
      echo [ERROR] sender sent 0 frames. sender log tail:
      call :tail_file "!SENDER_LOG!" 80
      goto :cleanup
    )

    taskkill /PID !STREAMER_PID! /T /F >nul 2>&1
    timeout /t %COOLDOWN_SEC% /nobreak >nul
    echo [DONE] !TAG!
  )
)

echo [INFO] Stopping receiver...
if defined RECEIVER_PID taskkill /PID %RECEIVER_PID% /T /F >nul 2>&1

echo [ALL DONE] %OUTDIR%
popd
endlocal
exit /b 0

:cleanup
echo [CLEANUP]
if defined RECEIVER_PID taskkill /PID %RECEIVER_PID% /T /F >nul 2>&1
popd
endlocal
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

:start_cmd_wrapper
REM usage: call :start_cmd_wrapper <pid_file> <launcher_log> <cmdline> <stdout_file> <stderr_file>
setlocal EnableExtensions DisableDelayedExpansion
set "PID_FILE=%~1"
set "LAUNCHER=%~2"
set "CMDLINE=%~3"
set "OUT=%~4"
set "ERR=%~5"

REM NOTE:
REM - Start-Process(cmd.exe /c) + cmd redirection for logs
REM - Save PID of cmd wrapper. taskkill /T will stop python child too.

"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$cmd = %CMDLINE% + ' 1>>' + '""%OUT%""' + ' 2>>' + '""%ERR%""';" ^
  "$p = Start-Process -PassThru -NoNewWindow -WorkingDirectory '%ROOT%' -FilePath 'cmd.exe' -ArgumentList @('/c',$cmd);" ^
  "Start-Sleep -Milliseconds 800;" ^
  "if($p.HasExited){ Write-Output ('START_FAILED ExitCode=' + $p.ExitCode); exit 2 }" ^
  "Set-Content -NoNewline -Encoding ascii -Path '%PID_FILE%' -Value $p.Id;" ^
  "Write-Output ('START_OK PID=' + $p.Id);" ^
  "exit 0" ^
  1>>"%LAUNCHER%" 2>>"%LAUNCHER%"

set "RC=%errorlevel%"
endlocal & exit /b %RC%

:wait_warmup_ready
REM %1 log_file, %2 timeout_sec, %3 interval_sec
setlocal EnableExtensions EnableDelayedExpansion
set "LOG=%~1"
set "TMO=%~2"
set "INT=%~3"
set /a ELAPSED=0
:wr_loop
if %ELAPSED% GEQ %TMO% ( endlocal & exit /b 1 )
if exist "%LOG%" (
  findstr /i /c:"Received first complete tracking update" "%LOG%" >nul 2>&1
  if !errorlevel! EQU 0 ( endlocal & exit /b 0 )
)
timeout /t %INT% /nobreak >nul
set /a ELAPSED+=%INT%
goto :wr_loop

:tail_file
REM %1 file, %2 lines
setlocal EnableExtensions
set "F=%~1"
set "N=%~2"
if not exist "%F%" ( endlocal & exit /b 0 )
"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -LiteralPath '%F%' -Tail %N%" 2>nul
endlocal & exit /b 0
