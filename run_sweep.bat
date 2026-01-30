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
  goto :cleanup
)

if not exist "%CSV_PATH%" (
  echo [ERROR] CSV not found: %CSV_PATH%
  goto :cleanup
)
if not exist "%SENDER%" (
  echo [ERROR] sender not found: %SENDER%
  goto :cleanup
)
if not exist "%RECEIVER%" (
  echo [ERROR] receiver not found: %RECEIVER%
  goto :cleanup
)
if not exist "%STREAMER%" (
  echo [ERROR] streamer not found: %STREAMER%
  goto :cleanup
)

REM ==========================================================
REM Experiment policy
REM ==========================================================
set "FIXED_DELTA=0.05"
set "STALE_TIMEOUT=2.0"

REM map compileを考慮して長め
set "WARMUP_INTERVAL=0.1"
set "WARMUP_WAIT_SEC=2"
set "WARMUP_TIMEOUT_SEC=600"
set "WARMUP_POLL_SEC=5"

set "TS_LIST=0.10 1.00"
set "COOLDOWN_SEC=3"

set "N_START=10"
set "N_STEP=10"
set "N_END=100"

REM ==========================================================
REM OUTDIR
REM ==========================================================
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"') do set "DT_TAG=%%i"
set "OUTDIR=%ROOT%sweep_results_%DT_TAG%"
mkdir "%OUTDIR%" 2>nul

set "RECEIVER_STDOUT=%OUTDIR%\receiver_stdout.log"
set "RECEIVER_STDERR=%OUTDIR%\receiver_stderr.log"
set "RECEIVER_PID_FILE=%OUTDIR%\receiver_pid.txt"
set "RECEIVER_TIMING_CSV=%OUTDIR%\update_timings_all.csv"
set "RECEIVER_EVAL_CSV=%OUTDIR%\eval_all.csv"

echo [INFO] OUTDIR=%OUTDIR%
echo [INFO] CSV=%CSV_PATH%

REM ==========================================================
REM Pre-clean
REM ==========================================================
echo [INFO] Cleaning stale python processes (receiver/streamer)...
call :kill_python_by_cmd "vehicle_state_stream.py"
call :kill_python_by_cmd "replay_from_udp.py"

echo [INFO] Freeing UDP port %UDP_PORT% if occupied...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p=%UDP_PORT%; $eps=Get-NetUDPEndpoint -LocalPort $p -ErrorAction SilentlyContinue; " ^
  "if($eps){ $pids=$eps | Select-Object -Expand OwningProcess -Unique; " ^
  "foreach($pid in $pids){ try{ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue }catch{} } }" >nul 2>nul

REM ==========================================================
REM Start receiver (PIDはStart-Processから確実に取ってファイルへ)
REM ==========================================================
echo [INFO] Starting receiver...
type nul > "%RECEIVER_STDOUT%"
type nul > "%RECEIVER_STDERR%"
del /q "%RECEIVER_PID_FILE%" >nul 2>&1

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$args=@(" ^
  "  '-u'," ^
  "  '%RECEIVER%'," ^
  "  '--carla-host','%CARLA_HOST%'," ^
  "  '--carla-port','%CARLA_PORT%'," ^
  "  '--listen-host','%LISTEN_HOST%'," ^
  "  '--listen-port','%UDP_PORT%'," ^
  "  '--fixed-delta','%FIXED_DELTA%'," ^
  "  '--stale-timeout','%STALE_TIMEOUT%'," ^
  "  '--measure-update-times'," ^
  "  '--timing-output','%RECEIVER_TIMING_CSV%'," ^
  "  '--eval-output','%RECEIVER_EVAL_CSV%'" ^
  ");" ^
  "$p=Start-Process -PassThru -NoNewWindow -WorkingDirectory '%ROOT%' -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '%RECEIVER_STDOUT%' -RedirectStandardError '%RECEIVER_STDERR%';" ^
  "Start-Sleep -Milliseconds 500;" ^
  "if($p.HasExited){ throw ('receiver exited early. ExitCode=' + $p.ExitCode) }" ^
  "$p.Id | Out-File -Encoding ascii '%RECEIVER_PID_FILE%';" ^
  "exit 0" ^
  1>>"%RECEIVER_STDOUT%" 2>>"%RECEIVER_STDERR%"

if errorlevel 1 (
  echo [ERROR] Failed to start receiver. receiver_stderr tail:
  powershell -NoProfile -Command "if(Test-Path '%RECEIVER_STDERR%'){ Get-Content '%RECEIVER_STDERR%' -Tail 50 }"
  goto :cleanup
)

if not exist "%RECEIVER_PID_FILE%" (
  echo [ERROR] receiver PID file not created. receiver_stderr tail:
  powershell -NoProfile -Command "if(Test-Path '%RECEIVER_STDERR%'){ Get-Content '%RECEIVER_STDERR%' -Tail 50 }"
  goto :cleanup
)

set "RECEIVER_PID="
set /p RECEIVER_PID=<"%RECEIVER_PID_FILE%"
echo %RECEIVER_PID%| findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 (
  echo [ERROR] Invalid receiver PID: %RECEIVER_PID%
  echo [INFO] receiver_stderr tail:
  powershell -NoProfile -Command "if(Test-Path '%RECEIVER_STDERR%'){ Get-Content '%RECEIVER_STDERR%' -Tail 50 }"
  goto :cleanup
)
echo [INFO] receiver PID=%RECEIVER_PID%

REM ==========================================================
REM Warmup (discard-first)
REM ==========================================================
echo [INFO] Warmup (discard-first) start. interval=%WARMUP_INTERVAL% timeout=%WARMUP_TIMEOUT_SEC%s

"%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" ^
  --host "%UDP_HOST%" --port "%UDP_PORT%" ^
  --interval %WARMUP_INTERVAL% ^
  > "%OUTDIR%\warmup_sender.log" 2>&1

findstr /i /c:"Sent 0 frames" "%OUTDIR%\warmup_sender.log" >nul 2>&1
if !errorlevel! EQU 0 (
  echo [ERROR] Warmup sender sent 0 frames. warmup_sender.log:
  type "%OUTDIR%\warmup_sender.log"
  goto :cleanup
)

if %WARMUP_WAIT_SEC% GTR 0 timeout /t %WARMUP_WAIT_SEC% /nobreak >nul

call :wait_for_actor_updates "%RECEIVER_STDERR%" %WARMUP_TIMEOUT_SEC% %WARMUP_POLL_SEC%
if errorlevel 1 (
  echo [WARN] Warmup did not confirm actor updates within timeout.
  echo [WARN] Proceeding anyway (first measured run might be contaminated).
  echo [INFO] receiver_stderr tail:
  powershell -NoProfile -Command "if(Test-Path '%RECEIVER_STDERR%'){ Get-Content '%RECEIVER_STDERR%' -Tail 50 }"
) else (
  echo [INFO] Warmup confirmed (nonzero actor updates detected).
)

REM ==========================================================
REM Sweep runs (receiver stays)
REM ==========================================================
for /L %%N in (%N_START%,%N_STEP%,%N_END%) do (
  for %%T in (%TS_LIST%) do (

    set "TAG=N%%N_Ts%%T"
    set "RUNDIR=%OUTDIR%\!TAG!"
    mkdir "!RUNDIR!" 2>nul

    set "STREAM_CSV=!RUNDIR!\replay_state_!TAG!.csv"
    set "STREAM_TIMING_CSV=!RUNDIR!\stream_timing_!TAG!.csv"
    set "SENDER_LOG=!RUNDIR!\sender.log"
    set "STREAMER_STDOUT=!RUNDIR!\streamer_stdout.log"
    set "STREAMER_STDERR=!RUNDIR!\streamer_stderr.log"

    echo ============================================================
    echo [RUN] !TAG!

    type nul > "!STREAMER_STDOUT!"
    type nul > "!STREAMER_STDERR!"

    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ErrorActionPreference='Stop';" ^
      "$args=@(" ^
      "  '-u'," ^
      "  '%STREAMER%'," ^
      "  '--host','%CARLA_HOST%'," ^
      "  '--port','%CARLA_PORT%'," ^
      "  '--mode','wait'," ^
      "  '--role-prefix','udp_replay:'," ^
      "  '--include-velocity'," ^
      "  '--frame-elapsed'," ^
      "  '--wall-clock'," ^
      "  '--include-object-id'," ^
      "  '--include-monotonic'," ^
      "  '--include-tick-wall-dt'," ^
      "  '--output','!STREAM_CSV!'," ^
      "  '--timing-output','!STREAM_TIMING_CSV!'," ^
      "  '--timing-flush-every','10'" ^
      ");" ^
      "$p=Start-Process -PassThru -NoNewWindow -WorkingDirectory '%ROOT%' -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '!STREAMER_STDOUT!' -RedirectStandardError '!STREAMER_STDERR%';" ^
      "Start-Sleep -Milliseconds 500;" ^
      "if($p.HasExited){ throw ('streamer exited early. ExitCode=' + $p.ExitCode) }" ^
      "exit 0" ^
      1>>"!STREAMER_STDOUT!" 2>>"!STREAMER_STDERR!"

    if errorlevel 1 (
      echo [ERROR] Failed to start streamer. streamer_stderr tail:
      powershell -NoProfile -Command "if(Test-Path '!STREAMER_STDERR!'){ Get-Content '!STREAMER_STDERR!' -Tail 50 }"
      goto :cleanup
    )

    timeout /t 1 /nobreak >nul

    echo [INFO] Sending... N=%%N Ts=%%T
    "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" ^
      --host "%UDP_HOST%" --port "%UDP_PORT%" ^
      --interval %%T ^
      --max-actors %%N ^
      > "!SENDER_LOG!" 2>&1

    findstr /i /c:"Sent 0 frames" "!SENDER_LOG!" >nul 2>&1
    if !errorlevel! EQU 0 (
      echo [ERROR] Sender sent 0 frames for !TAG!. sender.log:
      type "!SENDER_LOG!"
      goto :cleanup
    )

    REM streamer停止（PID不要）
    call :kill_python_by_cmd "vehicle_state_stream.py"

    timeout /t %COOLDOWN_SEC% /nobreak >nul
    echo [DONE] !TAG!
  )
)

echo [INFO] Stopping receiver...
call :kill_python_by_cmd "replay_from_udp.py"

echo [ALL DONE] %OUTDIR%
popd
endlocal
exit /b 0

:cleanup
echo [CLEANUP]
call :kill_python_by_cmd "vehicle_state_stream.py"
call :kill_python_by_cmd "replay_from_udp.py"
popd
endlocal
exit /b 1

REM ==========================================================
REM Subroutines
REM ==========================================================

:kill_python_by_cmd
REM %1 substring in CommandLine
setlocal EnableExtensions
set "SUB=%~1"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$sub='%SUB%';" ^
  "$procs=Get-CimInstance Win32_Process -Filter ""Name='python.exe'"" | Where-Object { $_.CommandLine -like ('*' + $sub + '*') };" ^
  "foreach($p in $procs){ try{ Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }catch{} }" >nul 2>nul
endlocal & exit /b 0

:wait_for_actor_updates
REM %1 log_path, %2 timeout_sec, %3 poll_sec
setlocal
set "LOG=%~1"
set "TMO=%~2"
set "POLL=%~3"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$path='%LOG%'; $timeout=[int]%TMO%; $poll=[int]%POLL%; $sw=[Diagnostics.Stopwatch]::StartNew();" ^
  "while($sw.Elapsed.TotalSeconds -lt $timeout){" ^
  "  if(Test-Path $path){" ^
  "    $tail = Get-Content -LiteralPath $path -Tail 400 -ErrorAction SilentlyContinue;" ^
  "    foreach($line in $tail){" ^
  "      if($line -match '\(([1-9][0-9]*) actor updates\)'){ exit 0 }" ^
  "    }" ^
  "  }" ^
  "  Start-Sleep -Seconds $poll" ^
  "}" ^
  "exit 1" >nul
set "RC=%errorlevel%"
endlocal & exit /b %RC%
