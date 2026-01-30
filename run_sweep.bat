@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0"

REM ==========================================================
REM Config
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

set "PWSH=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PWSH%" set "PWSH=powershell"

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

REM Warmup: receiver維持のまま「全送信」を繰り返して，(>=1 actor updates) を引くまで回す
set "WARMUP_INTERVAL=0.1"
set "WARMUP_POSTSEND_POLL_SEC=15"
set "WARMUP_POSTSEND_POLL_INTERVAL=2"
set "WARMUP_MAX_ATTEMPTS=10"

REM Sweep
set "TS_LIST=0.10 1.00"
set "ACTOR_LIST=10 20 30 40 50 60 70 80 90 100"
set "COOLDOWN_SEC=3"

REM ==========================================================
REM OUTDIR
REM ==========================================================
for /f "usebackq delims=" %%i in (`"%PWSH%" -NoProfile -ExecutionPolicy Bypass -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"`) do set "DT_TAG=%%i"
set "OUTDIR=%ROOT%sweep_results_%DT_TAG%"
mkdir "%OUTDIR%" 2>nul

set "RECEIVER_STDOUT=%OUTDIR%\receiver_stdout.log"
set "RECEIVER_STDERR=%OUTDIR%\receiver_stderr.log"
set "RECEIVER_TIMING_CSV=%OUTDIR%\update_timings_all.csv"
set "RECEIVER_EVAL_CSV=%OUTDIR%\eval_all.csv"

echo [INFO] OUTDIR=%OUTDIR%
echo [INFO] CSV=%CSV_PATH%

REM ==========================================================
REM Pre-clean
REM ==========================================================
echo [INFO] Cleaning stale python processes (receiver/streamer)...
call :kill_python_by_cmd "replay_from_udp.py"
call :kill_python_by_cmd "vehicle_state_stream.py"

echo [INFO] Freeing UDP port %UDP_PORT% if occupied...
"%PWSH%" -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p=%UDP_PORT%; $eps=Get-NetUDPEndpoint -LocalPort $p -ErrorAction SilentlyContinue; " ^
  "if($eps){ $pids=$eps | Select-Object -ExpandProperty OwningProcess -Unique; " ^
  "foreach($pid in $pids){ try{ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue }catch{} } }" ^
  >nul 2>nul

REM ==========================================================
REM Start receiver (PowerShell Start-Process, capture PID)
REM ==========================================================
echo [INFO] Starting receiver...
type nul > "%RECEIVER_STDOUT%"
type nul > "%RECEIVER_STDERR%"

set "RECEIVER_PID="
for /f "usebackq delims=" %%P in (`
  "%PWSH%" -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "$args=@(" ^
      "'%RECEIVER%'," ^
      "'--carla-host','%CARLA_HOST%'," ^
      "'--carla-port','%CARLA_PORT%'," ^
      "'--listen-host','%LISTEN_HOST%'," ^
      "'--listen-port','%UDP_PORT%'," ^
      "'--fixed-delta','%FIXED_DELTA%'," ^
      "'--stale-timeout','%STALE_TIMEOUT%'," ^
      "'--measure-update-times'," ^
      "'--timing-output','%RECEIVER_TIMING_CSV%'," ^
      "'--eval-output','%RECEIVER_EVAL_CSV%'" ^
    ");" ^
    "$p=Start-Process -PassThru -NoNewWindow -WorkingDirectory '%ROOT%' -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '%RECEIVER_STDOUT%' -RedirectStandardError '%RECEIVER_STDERR%';" ^
    "Start-Sleep -Milliseconds 400;" ^
    "if($p.HasExited){ 'START_FAILED'; exit 0 };" ^
    "$p.Id"
`) do set "RECEIVER_PID=%%P"

echo %RECEIVER_PID%| findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 (
  echo [ERROR] Failed to start receiver. receiver_stderr tail:
  "%PWSH%" -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '%RECEIVER_STDERR%'){ Get-Content '%RECEIVER_STDERR%' -Tail 30 }"
  goto :cleanup
)

echo [INFO] receiver PID=%RECEIVER_PID%
timeout /t 2 /nobreak >nul

REM ==========================================================
REM Warmup: send full CSV repeatedly until receiver sees (>=1 actor updates)
REM   コンパイル中は更新が出ないので「送って→短時間だけログ確認→ダメならまた送る」
REM ==========================================================
for /L %%A in (1,1,%WARMUP_MAX_ATTEMPTS%) do (
  echo [INFO] Warmup attempt %%A/%WARMUP_MAX_ATTEMPTS%
  echo [INFO] Warmup send interval=%WARMUP_INTERVAL%

  REM offset bytes for receiver stderr
  set "LOG_OFFSET=0"
  for /f "usebackq delims=" %%S in (`"%PWSH%" -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '%RECEIVER_STDERR%'){ (Get-Item '%RECEIVER_STDERR%').Length } else { 0 }"`) do set "LOG_OFFSET=%%S"

  "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %WARMUP_INTERVAL% > "%OUTDIR%\warmup_sender.log" 2>&1

  findstr /i /c:"Sent 0 frames" "%OUTDIR%\warmup_sender.log" >nul 2>&1
  if !errorlevel! EQU 0 (
    echo [ERROR] Warmup sender sent 0 frames. See: %OUTDIR%\warmup_sender.log
    type "%OUTDIR%\warmup_sender.log"
    goto :cleanup
  )

  call :wait_actor_updates_in_log "%RECEIVER_STDERR%" !LOG_OFFSET! %WARMUP_POSTSEND_POLL_SEC% %WARMUP_POSTSEND_POLL_INTERVAL%
  if !errorlevel! EQU 0 (
    echo [INFO] Warmup succeeded.
    goto :warmup_done
  )

  echo [WARN] Warmup not confirmed. retry...
  echo [INFO] receiver_stderr tail:
  "%PWSH%" -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '%RECEIVER_STDERR%'){ Get-Content '%RECEIVER_STDERR%' -Tail 10 }"
)

echo [ERROR] Warmup failed.
goto :cleanup

:warmup_done

REM ==========================================================
REM Sweep: actor count 10..100 step10, Ts in list
REM   receiver stays alive. streamer per run.
REM ==========================================================
for %%N in (%ACTOR_LIST%) do (
  for %%T in (%TS_LIST%) do (
    set "TAG=N%%N_Ts%%T"
    set "RUNDIR=%OUTDIR%\!TAG!"
    mkdir "!RUNDIR!" 2>nul

    set "STREAM_CSV=!RUNDIR!\replay_state_!TAG!.csv"
    set "STREAM_TIMING_CSV=!RUNDIR!\stream_timing_!TAG!.csv"
    set "SENDER_LOG=!RUNDIR!\sender_!TAG!.log"
    set "STREAMER_STDOUT=!RUNDIR!\streamer_stdout_!TAG!.log"
    set "STREAMER_STDERR=!RUNDIR!\streamer_stderr_!TAG!.log"

    echo ============================================================
    echo [RUN] !TAG!
    echo [DIR] !RUNDIR!

    type nul > "!STREAMER_STDOUT!"
    type nul > "!STREAMER_STDERR!"

    set "STREAMER_PID="
    for /f "usebackq delims=" %%P in (`
      "%PWSH%" -NoProfile -ExecutionPolicy Bypass -Command ^
        "$ErrorActionPreference='Stop';" ^
        "$args=@(" ^
          "'%STREAMER%'," ^
          "'--host','%CARLA_HOST%'," ^
          "'--port','%CARLA_PORT%'," ^
          "'--mode','wait'," ^
          "'--role-prefix','udp_replay:'," ^
          "'--include-velocity'," ^
          "'--frame-elapsed'," ^
          "'--wall-clock'," ^
          "'--include-object-id'," ^
          "'--include-monotonic'," ^
          "'--include-tick-wall-dt'," ^
          "'--output','!STREAM_CSV!'," ^
          "'--timing-output','!STREAM_TIMING_CSV!'," ^
          "'--timing-flush-every','10'" ^
        ");" ^
        "$p=Start-Process -PassThru -NoNewWindow -WorkingDirectory '%ROOT%' -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '!STREAMER_STDOUT!' -RedirectStandardError '!STREAMER_STDERR%';" ^
        "Start-Sleep -Milliseconds 300;" ^
        "if($p.HasExited){ 'START_FAILED'; exit 0 };" ^
        "$p.Id"
    `) do set "STREAMER_PID=%%P"

    echo !STREAMER_PID!| findstr /r "^[0-9][0-9]*$" >nul
    if errorlevel 1 (
      echo [ERROR] Failed to start streamer. streamer_stderr tail:
      "%PWSH%" -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '!STREAMER_STDERR!'){ Get-Content '!STREAMER_STDERR!' -Tail 30 }"
      goto :cleanup
    )

    timeout /t 1 /nobreak >nul

    echo [INFO] Sending... N=%%N Ts=%%T
    "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %%T --max-actors %%N > "!SENDER_LOG!" 2>&1

    findstr /i /c:"Sent 0 frames" "!SENDER_LOG!" >nul 2>&1
    if !errorlevel! EQU 0 (
      echo [ERROR] Sender sent 0 frames. Check: !SENDER_LOG!
      type "!SENDER_LOG!"
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
"%PWSH%" -NoProfile -ExecutionPolicy Bypass -Command ^
  "$sub='%SUB%';" ^
  "$procs=Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -and ($_.CommandLine -like ('*'+$sub+'*')) };" ^
  "foreach($p in $procs){ try{ Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }catch{} }" ^
  >nul 2>nul
endlocal & exit /b 0

:wait_actor_updates_in_log
REM args: %1 log_path, %2 offset_bytes, %3 timeout_sec, %4 interval_sec
setlocal EnableDelayedExpansion
set "LOG=%~1"
set "OFF=%~2"
set "TMO=%~3"
set "INT=%~4"

"%PWSH%" -NoProfile -ExecutionPolicy Bypass -Command ^
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
