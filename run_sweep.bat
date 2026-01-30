@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0"

REM ----------------------------------------------------------
REM Encoding (avoid mojibake in console as much as possible)
REM ----------------------------------------------------------
chcp 65001 >nul

REM ----------------------------------------------------------
REM Paths / executables
REM ----------------------------------------------------------
set "ROOT=%~dp0"
set "PYTHON_EXE=python"

where "%PYTHON_EXE%" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] python not found in PATH. Set PYTHON_EXE=py or full path.
  goto :cleanup
)

REM PowerShell path (IMPORTANT: no stray quotes inside the variable)
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" (
  REM fallback to pwsh if WindowsPowerShell is missing
  set "PS=pwsh"
  where "%PS%" >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] PowerShell not found. (powershell.exe / pwsh)
    goto :cleanup
  )
)

REM ----------------------------------------------------------
REM User config
REM ----------------------------------------------------------
set "CARLA_HOST=127.0.0.1"
set "CARLA_PORT=2000"

set "UDP_HOST=127.0.0.1"
set "UDP_PORT=5005"
set "LISTEN_HOST=0.0.0.0"

set "CSV_PATH=%ROOT%send_data\exp_300.csv"
set "SENDER=%ROOT%send_data\send_udp_frames_from_csv.py"
set "RECEIVER=%ROOT%scripts\udp_replay\replay_from_udp.py"
set "STREAMER=%ROOT%scripts\vehicle_state_stream.py"

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

REM ----------------------------------------------------------
REM Experiment policy
REM ----------------------------------------------------------
set "FIXED_DELTA=0.05"
set "STALE_TIMEOUT=2.0"

REM warmup: map compile can stall 1-2 min, so longer timeout
set "WARMUP_INTERVAL=0.1"
set "WARMUP_WAIT_SEC=2"
set "WARMUP_CHECK_TIMEOUT_SEC=900"
set "WARMUP_CHECK_INTERVAL_SEC=5"
set "WARMUP_MAX_ATTEMPTS=2"

REM sweep
set "TS_LIST=0.10 1.00"
set "COOLDOWN_SEC=3"

REM actor count sweep (10..100 step 10)
set "N_MIN=10"
set "N_MAX=100"
set "N_STEP=10"

REM ----------------------------------------------------------
REM OUTDIR (PowerShell Get-Date, but robust quoting)
REM ----------------------------------------------------------
set "DT_TAG="
for /f "usebackq delims=" %%i in (`"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"`) do set "DT_TAG=%%i"

if not defined DT_TAG (
  REM fallback (DATE/TIME can be locale dependent; used only if PS fails)
  set "DT_TAG=%DATE%_%TIME%"
  set "DT_TAG=%DT_TAG:/=%"
  set "DT_TAG=%DT_TAG::=%"
  set "DT_TAG=%DT_TAG:.=%"
  set "DT_TAG=%DT_TAG: =%"
)

set "OUTDIR=%ROOT%sweep_results_%DT_TAG%"
mkdir "%OUTDIR%" 2>nul

set "RECEIVER_STDOUT=%OUTDIR%\receiver_stdout.log"
set "RECEIVER_STDERR=%OUTDIR%\receiver_stderr.log"
set "RECEIVER_PID_FILE=%OUTDIR%\receiver_pid.txt"
set "RECEIVER_TIMING_CSV=%OUTDIR%\update_timings_all.csv"
set "RECEIVER_EVAL_CSV=%OUTDIR%\eval_all.csv"

echo [INFO] OUTDIR=%OUTDIR%
echo [INFO] CSV=%CSV_PATH%
echo [INFO] PS=%PS%

REM ----------------------------------------------------------
REM Pre-clean: kill stale processes + free UDP port
REM ----------------------------------------------------------
echo [INFO] Cleaning stale python processes (receiver/streamer)...
call :kill_python_by_cmd "replay_from_udp.py"
call :kill_python_by_cmd "vehicle_state_stream.py"

echo [INFO] Freeing UDP port %UDP_PORT% if occupied...
"%PS%" -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p=%UDP_PORT%; $eps=Get-NetUDPEndpoint -LocalPort $p -ErrorAction SilentlyContinue; " ^
  "if($eps){ $pids=$eps | Select-Object -Expand OwningProcess -Unique; " ^
  "foreach($pid in $pids){ try{ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue }catch{} } }" >nul 2>nul

REM ----------------------------------------------------------
REM Start receiver (keep alive)
REM ----------------------------------------------------------
echo [INFO] Starting receiver...
type nul > "%RECEIVER_STDOUT%"
type nul > "%RECEIVER_STDERR%"
del /q "%RECEIVER_PID_FILE%" >nul 2>&1

"%PS%" -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$args=@(" ^
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
  "if($p.HasExited){ 'receiver exited early. ExitCode=' + $p.ExitCode | Out-File -Encoding utf8 '%RECEIVER_STDERR%' -Append; exit 2 }" ^
  "$p.Id | Out-File -Encoding ascii '%RECEIVER_PID_FILE%'; exit 0" ^
  1>>"%RECEIVER_STDOUT%" 2>>"%RECEIVER_STDERR%"

if errorlevel 1 (
  echo [ERROR] Failed to start receiver. receiver_stderr tail:
  "%PS%" -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '%RECEIVER_STDERR%'){ Get-Content '%RECEIVER_STDERR%' -Tail 30 }"
  goto :cleanup
)

if not exist "%RECEIVER_PID_FILE%" (
  echo [ERROR] receiver_pid.txt not created. receiver_stderr tail:
  "%PS%" -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '%RECEIVER_STDERR%'){ Get-Content '%RECEIVER_STDERR%' -Tail 30 }"
  goto :cleanup
)

set /p RECEIVER_PID=<"%RECEIVER_PID_FILE%"
echo %RECEIVER_PID%| findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 (
  echo [ERROR] Invalid receiver PID: %RECEIVER_PID%
  echo [ERROR] receiver_stderr tail:
  "%PS%" -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '%RECEIVER_STDERR%'){ Get-Content '%RECEIVER_STDERR%' -Tail 30 }"
  goto :cleanup
)
echo [INFO] receiver PID=%RECEIVER_PID%

REM ----------------------------------------------------------
REM Warmup: trigger map compile once, then wait until actor updates appear
REM ----------------------------------------------------------
for /L %%A in (1,1,%WARMUP_MAX_ATTEMPTS%) do (
  echo [INFO] Warmup attempt %%A/%WARMUP_MAX_ATTEMPTS%
  echo [INFO] Warmup send interval=%WARMUP_INTERVAL%

  REM record receiver_stderr byte offset
  set "LOG_OFFSET=0"
  for /f "usebackq delims=" %%S in (`"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "(Get-Item '%RECEIVER_STDERR%').Length"`) do set "LOG_OFFSET=%%S"

  REM warmup sender: send ALL (no start/end, no id range management)
  "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %WARMUP_INTERVAL% > "%OUTDIR%\warmup_sender.log" 2>&1

  findstr /i /c:"Sent 0 frames" "%OUTDIR%\warmup_sender.log" >nul 2>&1
  if !errorlevel! EQU 0 (
    echo [ERROR] Warmup sender sent 0 frames. Check warmup_sender.log
    type "%OUTDIR%\warmup_sender.log"
    goto :cleanup
  )

  if %WARMUP_WAIT_SEC% GTR 0 timeout /t %WARMUP_WAIT_SEC% /nobreak >nul

  call :wait_actor_updates_in_log "%RECEIVER_STDERR%" !LOG_OFFSET! %WARMUP_CHECK_TIMEOUT_SEC% %WARMUP_CHECK_INTERVAL_SEC%
  if !errorlevel! EQU 0 (
    echo [INFO] Warmup confirmed (actor updates detected).
    goto :warmup_done
  )

  echo [WARN] Warmup not confirmed. retry...
  echo [INFO] receiver_stderr tail:
  "%PS%" -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '%RECEIVER_STDERR%'){ Get-Content '%RECEIVER_STDERR%' -Tail 20 }"
)

echo [WARN] Warmup not confirmed, but continue anyway (map compile may still be running).
echo [WARN] First run results may be garbage; receiver stays alive so later runs should stabilize.

:warmup_done

REM ----------------------------------------------------------
REM Sweep runs
REM   - receiver stays alive
REM   - streamer per run
REM   - sender per (N, Ts) sends ALL frames, but limits active actors with --max-actors N
REM ----------------------------------------------------------
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
    set "STREAMER_PID_FILE=!RUNDIR!\streamer_pid.txt"

    echo ============================================================
    echo [RUN] !TAG!
    echo [DIR] !RUNDIR!

    type nul > "!STREAMER_STDOUT!"
    type nul > "!STREAMER_STDERR!"
    del /q "!STREAMER_PID_FILE!" >nul 2>&1

    REM start streamer and write pid to file
    "%PS%" -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ErrorActionPreference='Stop';" ^
      "$args=@(" ^
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
      "Start-Sleep -Milliseconds 400;" ^
      "if($p.HasExited){ 'streamer exited early. ExitCode=' + $p.ExitCode | Out-File -Encoding utf8 '!STREAMER_STDERR%' -Append; exit 2 }" ^
      "$p.Id | Out-File -Encoding ascii '!STREAMER_PID_FILE!'; exit 0" ^
      1>>"!STREAMER_STDOUT!" 2>>"!STREAMER_STDERR!"

    if errorlevel 1 (
      echo [ERROR] Failed to start streamer. streamer_stderr tail:
      "%PS%" -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '!STREAMER_STDERR!'){ Get-Content '!STREAMER_STDERR!' -Tail 30 }"
      goto :cleanup
    )

    if not exist "!STREAMER_PID_FILE!" (
      echo [ERROR] streamer pid file not created. streamer_stderr tail:
      "%PS%" -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '!STREAMER_STDERR!'){ Get-Content '!STREAMER_STDERR!' -Tail 30 }"
      goto :cleanup
    )

    set /p STREAMER_PID=<"!STREAMER_PID_FILE!"
    echo !STREAMER_PID!| findstr /r "^[0-9][0-9]*$" >nul
    if errorlevel 1 (
      echo [ERROR] Invalid streamer PID: !STREAMER_PID!
      "%PS%" -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '!STREAMER_STDERR!'){ Get-Content '!STREAMER_STDERR!' -Tail 30 }"
      goto :cleanup
    )

    timeout /t 1 /nobreak >nul

    echo [INFO] Sending... N=%%N Ts=%%T (sender sends ALL frames, limits actors with --max-actors)
    "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %%T --max-actors %%N > "!SENDER_LOG!" 2>&1

    findstr /i /c:"Sent 0 frames" "!SENDER_LOG!" >nul 2>&1
    if !errorlevel! EQU 0 (
      echo [ERROR] sender sent 0 frames. Showing sender log tail:
      "%PS%" -NoProfile -ExecutionPolicy Bypass -Command "Get-Content '!SENDER_LOG!' -Tail 30"
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
call :kill_python_by_cmd "vehicle_state_stream.py"
call :kill_python_by_cmd "replay_from_udp.py"
if defined RECEIVER_PID taskkill /PID %RECEIVER_PID% /T /F >nul 2>&1
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
"%PS%" -NoProfile -ExecutionPolicy Bypass -Command ^
  "$sub='%SUB%';" ^
  "$procs=Get-CimInstance Win32_Process -Filter ""Name='python.exe'"" | Where-Object { $_.CommandLine -like ('*' + $sub + '*') };" ^
  "foreach($p in $procs){ try{ Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }catch{} }" >nul 2>nul
endlocal & exit /b 0

:wait_actor_updates_in_log
REM args: %1 log_path, %2 offset_bytes, %3 timeout_sec, %4 interval_sec
setlocal EnableDelayedExpansion
set "LOG=%~1"
set "OFF=%~2"
set "TMO=%~3"
set "INT=%~4"

"%PS%" -NoProfile -ExecutionPolicy Bypass -Command ^
  "$path='%LOG%'; $off=[int64]%OFF%; $timeout=%TMO%; $interval=%INT%; $sw=[Diagnostics.Stopwatch]::StartNew();" ^
  "while($sw.Elapsed.TotalSeconds -lt $timeout){" ^
  "  if(Test-Path $path){" ^
  "    $len=(Get-Item $path).Length;" ^
  "    if($len -gt $off){" ^
  "      $fs=[System.IO.File]::Open($path,'Open','Read','ReadWrite');" ^
  "      try{ $fs.Seek($off,[System.IO.SeekOrigin]::Begin)|Out-Null; $buf=New-Object byte[] ($len-$off); [void]$fs.Read($buf,0,$buf.Length); $txt=[Text.Encoding]::UTF8.GetString($buf) } finally { $fs.Close() }" ^
  "      if($txt -match '\(([1-9][0-9]*) actor updates\)'){ exit 0 }" ^
  "      $off=$len" ^
  "    }" ^
  "  }" ^
  "  Start-Sleep -Seconds $interval" ^
  "}" ^
  "exit 1" >nul

set "RC=%errorlevel%"
endlocal & exit /b %RC%
