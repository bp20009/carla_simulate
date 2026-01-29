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
REM Receiver params (fixed)
set "FIXED_DELTA=0.05"
set "STALE_TIMEOUT=2.0"

REM Warmup (frame range OK. id管理ではないので残す)
set "WARMUP_START_FRAME=0"
set "WARMUP_END_FRAME=1800"
set "WARMUP_INTERVAL=0.1"
set "WARMUP_WAIT_SEC=10"
set "WARMUP_CHECK_TIMEOUT_SEC=180"
set "WARMUP_CHECK_INTERVAL_SEC=5"
set "WARMUP_MAX_ATTEMPTS=3"

REM Sweep (senderは常に全送信。ここではTsだけ振る)
set "TS_LIST=0.10 1.00"
set "COOLDOWN_SEC=5"

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
REM Pre-clean (kill stale processes + free UDP port)
REM ==========================================================
echo [INFO] Cleaning stale python processes (receiver/streamer)...
call :kill_python_by_cmd "replay_from_udp.py"
call :kill_python_by_cmd "vehicle_state_stream.py"

echo [INFO] Freeing UDP port %UDP_PORT% if occupied...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p=%UDP_PORT%; $eps=Get-NetUDPEndpoint -LocalPort $p -ErrorAction SilentlyContinue; " ^
  "if($eps){ $pids=$eps | Select-Object -Expand OwningProcess -Unique; " ^
  "foreach($pid in $pids){ try{ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue }catch{} } }" >nul 2>nul

REM ==========================================================
REM Start receiver (write PID to file, avoid for /f hanging)
REM ==========================================================
echo [INFO] Starting receiver...

set "RECEIVER_STDOUT=%OUTDIR%\receiver_stdout.log"
set "RECEIVER_STDERR=%OUTDIR%\receiver_stderr.log"
set "RECEIVER_PID_FILE=%OUTDIR%\receiver_pid.txt"

type nul > "%RECEIVER_STDOUT%"
type nul > "%RECEIVER_STDERR%"
del /q "%RECEIVER_PID_FILE%" >nul 2>&1

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
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
  "Start-Sleep -Milliseconds 300;" ^
  "if($p.HasExited){ 'START_FAILED: exited early. ExitCode=' + $p.ExitCode | Out-File -Encoding utf8 '%RECEIVER_STDERR%' -Append; exit 2 }" ^
  "$p.Id | Out-File -Encoding ascii '%RECEIVER_PID_FILE%'; exit 0" ^
  1>>"%RECEIVER_STDOUT%" 2>>"%RECEIVER_STDERR%"

if errorlevel 1 (
  echo [ERROR] Failed to start receiver. Check logs:
  if exist "%RECEIVER_STDERR%" type "%RECEIVER_STDERR%"
  if exist "%RECEIVER_STDOUT%" type "%RECEIVER_STDOUT%"
  goto :cleanup
)

if not exist "%RECEIVER_PID_FILE%" (
  echo [ERROR] receiver_pid.txt not created. Check logs:
  if exist "%RECEIVER_STDERR%" type "%RECEIVER_STDERR%"
  if exist "%RECEIVER_STDOUT%" type "%RECEIVER_STDOUT%"
  goto :cleanup
)

set /p RECEIVER_PID=<"%RECEIVER_PID_FILE%"
echo %RECEIVER_PID%| findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 (
  echo [ERROR] Invalid receiver PID: %RECEIVER_PID%
  if exist "%RECEIVER_STDERR%" type "%RECEIVER_STDERR%"
  if exist "%RECEIVER_STDOUT%" type "%RECEIVER_STDOUT%"
  goto :cleanup
)

echo [INFO] receiver PID=%RECEIVER_PID%
timeout /t 2 /nobreak >nul

REM ==========================================================
REM Warmup: sender短区間送信 + receiverログに "(>=1 actor updates)" が出るのを待つ
REM   INFOログはstderrに出ることが多いので両方監視
REM ==========================================================
for /L %%A in (1,1,%WARMUP_MAX_ATTEMPTS%) do (
  echo [INFO] Warmup attempt %%A/%WARMUP_MAX_ATTEMPTS%

  set "OFF_OUT=0"
  set "OFF_ERR=0"
  if exist "%RECEIVER_STDOUT%" for /f %%S in ('powershell -NoProfile -Command "(Get-Item ''%RECEIVER_STDOUT%'').Length"') do set "OFF_OUT=%%S"
  if exist "%RECEIVER_STDERR%" for /f %%S in ('powershell -NoProfile -Command "(Get-Item ''%RECEIVER_STDERR%'').Length"') do set "OFF_ERR=%%S"

  set "WARMUP_SENDER_LOG=%OUTDIR%\warmup_sender_%%A.log"
  type nul > "!WARMUP_SENDER_LOG!"

  echo [INFO] Warmup send frames %WARMUP_START_FRAME%..%WARMUP_END_FRAME% interval=%WARMUP_INTERVAL%
  "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" ^
    --host "%UDP_HOST%" --port "%UDP_PORT%" ^
    --interval %WARMUP_INTERVAL% ^
    --start-frame %WARMUP_START_FRAME% --end-frame %WARMUP_END_FRAME% ^
    > "!WARMUP_SENDER_LOG!" 2>&1

  if %WARMUP_WAIT_SEC% GTR 0 timeout /t %WARMUP_WAIT_SEC% /nobreak >nul

  call :wait_actor_updates_in_two_logs "%RECEIVER_STDOUT%" !OFF_OUT! "%RECEIVER_STDERR%" !OFF_ERR! %WARMUP_CHECK_TIMEOUT_SEC% %WARMUP_CHECK_INTERVAL_SEC%
  if !errorlevel! EQU 0 (
    echo [INFO] Warmup succeeded.
    goto :warmup_done
  )
  echo [WARN] Warmup not confirmed. retry...
)

echo [ERROR] Warmup failed.
goto :cleanup

:warmup_done

REM ==========================================================
REM Sweep runs (receiver stays. streamer + sender per Ts)
REM   senderは全送信（max-actors等は使わない）
REM ==========================================================
for %%T in (%TS_LIST%) do (
  set "TAG=Ts%%T"
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
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ErrorActionPreference='Stop';" ^
      "try{" ^
      "  $args=@(" ^
      "    '%STREAMER%'," ^
      "    '--host','%CARLA_HOST%'," ^
      "    '--port','%CARLA_PORT%'," ^
      "    '--mode','wait'," ^
      "    '--role-prefix','udp_replay:'," ^
      "    '--include-velocity'," ^
      "    '--frame-elapsed'," ^
      "    '--wall-clock'," ^
      "    '--include-object-id'," ^
      "    '--include-monotonic'," ^
      "    '--include-tick-wall-dt'," ^
      "    '--output','!STREAM_CSV!'," ^
      "    '--timing-output','!STREAM_TIMING_CSV!'," ^
      "    '--timing-flush-every','10'" ^
      "  );" ^
      "  $p=Start-Process -PassThru -NoNewWindow -WorkingDirectory '%ROOT%' -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '!STREAMER_STDOUT!' -RedirectStandardError '!STREAMER_STDERR!';" ^
      "  Start-Sleep -Milliseconds 200;" ^
      "  if($p.HasExited){ 'START_FAILED: exited early. ExitCode=' + $p.ExitCode; exit 0 }" ^
      "  $p.Id" ^
      "} catch { 'START_FAILED: ' + $_.Exception.Message }"
  `) do set "STREAMER_PID=%%P"

  echo [INFO] streamer start result: !STREAMER_PID!
  echo !STREAMER_PID!| findstr /r "^[0-9][0-9]*$" >nul
  if errorlevel 1 (
    echo [ERROR] Failed to start streamer. Reason:
    echo !STREAMER_PID!
    echo [ERROR] streamer_stderr:
    if exist "!STREAMER_STDERR!" type "!STREAMER_STDERR!"
    echo [ERROR] streamer_stdout:
    if exist "!STREAMER_STDOUT!" type "!STREAMER_STDOUT!"
    goto :cleanup
  )

  timeout /t 1 /nobreak >nul

  echo [INFO] Sending... Ts=%%T
  "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" ^
    --host "%UDP_HOST%" --port "%UDP_PORT%" ^
    --interval %%T ^
    > "!SENDER_LOG!" 2>&1

  taskkill /PID !STREAMER_PID! /T /F >nul 2>&1

  timeout /t %COOLDOWN_SEC% /nobreak >nul
  echo [DONE] !TAG!
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
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$sub='%SUB%';" ^
  "$procs=Get-CimInstance Win32_Process -Filter ""Name='python.exe'"" | Where-Object { $_.CommandLine -like ('*' + $sub + '*') };" ^
  "foreach($p in $procs){ try{ Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }catch{} }" >nul 2>nul
endlocal & exit /b 0

:wait_actor_updates_in_two_logs
REM args:
REM   %1 log1_path %2 off1_bytes %3 log2_path %4 off2_bytes %5 timeout_sec %6 interval_sec
setlocal EnableDelayedExpansion
set "L1=%~1"
set "O1=%~2"
set "L2=%~3"
set "O2=%~4"
set "TMO=%~5"
set "INT=%~6"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$l1='%L1%'; $o1=[int64]%O1%; $l2='%L2%'; $o2=[int64]%O2%; $timeout=%TMO%; $interval=%INT%;" ^
  "$re='\(([1-9][0-9]*) actor updates\)';" ^
  "$sw=[Diagnostics.Stopwatch]::StartNew();" ^
  "function ReadNew([string]$p,[ref]$off){" ^
  "  if(-not (Test-Path $p)){ return '' }" ^
  "  $len=(Get-Item $p).Length; if($len -le $off.Value){ return '' }" ^
  "  $fs=[System.IO.File]::Open($p,'Open','Read','ReadWrite');" ^
  "  try{ $fs.Seek($off.Value,[System.IO.SeekOrigin]::Begin)|Out-Null; $buf=New-Object byte[] ($len-$off.Value); [void]$fs.Read($buf,0,$buf.Length); } finally { $fs.Close() }" ^
  "  $off.Value=$len; return [Text.Encoding]::UTF8.GetString($buf)" ^
  "}" ^
  "while($sw.Elapsed.TotalSeconds -lt $timeout){" ^
  "  $t1=ReadNew $l1 ([ref]$o1); if($t1 -match $re){ exit 0 }" ^
  "  $t2=ReadNew $l2 ([ref]$o2); if($t2 -match $re){ exit 0 }" ^
  "  Start-Sleep -Seconds $interval" ^
  "}" ^
  "exit 1" >nul

set "RC=%errorlevel%"
endlocal & exit /b %RC%
