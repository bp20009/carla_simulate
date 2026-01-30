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

REM ==========================================================
REM Checks
REM ==========================================================
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
set "ENABLE_COMPLETION=1"

REM Warmup
REM map compile が 1-2 分走る前提で長めに待つ
set "WARMUP_INTERVAL=0.1"
set "WARMUP_WAIT_SEC=0"
set "WARMUP_CHECK_TIMEOUT_SEC=420"
set "WARMUP_CHECK_INTERVAL_SEC=5"
set "WARMUP_MAX_ATTEMPTS=2"

REM Sweep (Ts only). sender is ALWAYS full-send.
set "TS_LIST=0.10 1.00"
set "COOLDOWN_SEC=3"

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
REM Start receiver (PID -> file, avoid for /f hanging)
REM ==========================================================
echo [INFO] Starting receiver...

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
  "if(%ENABLE_COMPLETION% -eq 1){ $args += '--enable-completion' }" ^
  "$p=Start-Process -PassThru -NoNewWindow -WorkingDirectory '%ROOT%' -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '%RECEIVER_STDOUT%' -RedirectStandardError '%RECEIVER_STDERR%';" ^
  "Start-Sleep -Milliseconds 400;" ^
  "if($p.HasExited){ 'START_FAILED: exited early. ExitCode=' + $p.ExitCode | Out-File -Encoding utf8 '%RECEIVER_STDERR%' -Append; exit 2 }" ^
  "$p.Id | Out-File -Encoding ascii '%RECEIVER_PID_FILE%'; exit 0" ^
  1>>"%RECEIVER_STDOUT%" 2>>"%RECEIVER_STDERR%"

if errorlevel 1 (
  echo [ERROR] Failed to start receiver. Check logs:
  call :tail "%RECEIVER_STDERR%" 80
  call :tail "%RECEIVER_STDOUT%" 40
  goto :cleanup
)

if not exist "%RECEIVER_PID_FILE%" (
  echo [ERROR] receiver_pid.txt not created. Check logs:
  call :tail "%RECEIVER_STDERR%" 80
  call :tail "%RECEIVER_STDOUT%" 40
  goto :cleanup
)

set /p RECEIVER_PID=<"%RECEIVER_PID_FILE%"
echo %RECEIVER_PID%| findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 (
  echo [ERROR] Invalid receiver PID: %RECEIVER_PID%
  call :tail "%RECEIVER_STDERR%" 80
  goto :cleanup
)

echo [INFO] receiver PID=%RECEIVER_PID%
timeout /t 2 /nobreak >nul

REM ==========================================================
REM Warmup:
REM   - sender: full send (start/end 使わない)
REM   - confirm: receiver stdout/stderr の tail に "(>=1 actor updates)" が出るまで待つ
REM ==========================================================
for /L %%A in (1,1,%WARMUP_MAX_ATTEMPTS%) do (
  echo [INFO] Warmup attempt %%A/%WARMUP_MAX_ATTEMPTS%
  echo [INFO] Warmup send interval=%WARMUP_INTERVAL%

  "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" ^
    --host "%UDP_HOST%" --port "%UDP_PORT%" ^
    --interval %WARMUP_INTERVAL% ^
    > "%OUTDIR%\warmup_sender.log" 2>&1

  if errorlevel 1 (
    echo [ERROR] Warmup sender failed. Tail:
    call :tail "%OUTDIR%\warmup_sender.log" 120
    goto :cleanup
  )

  findstr /i /c:"Sent 0 frames" "%OUTDIR%\warmup_sender.log" >nul 2>&1
  if !errorlevel! EQU 0 (
    echo [ERROR] Warmup sender sent 0 frames. Tail:
    call :tail "%OUTDIR%\warmup_sender.log" 120
    goto :cleanup
  )

  if %WARMUP_WAIT_SEC% GTR 0 timeout /t %WARMUP_WAIT_SEC% /nobreak >nul

  call :wait_actor_updates_tail "%RECEIVER_STDOUT%" "%RECEIVER_STDERR%" %WARMUP_CHECK_TIMEOUT_SEC% %WARMUP_CHECK_INTERVAL_SEC%
  if !errorlevel! EQU 0 (
    echo [INFO] Warmup confirmed (nonzero actor updates).
    goto :warmup_done
  )

  echo [WARN] Warmup not confirmed yet. (map compile may be running) retry...
  echo [INFO] receiver_stderr tail:
  call :tail "%RECEIVER_STDERR%" 30
)

echo [ERROR] Warmup failed (no actor updates observed).
echo [INFO] receiver_stderr tail:
call :tail "%RECEIVER_STDERR%" 80
echo [INFO] receiver_stdout tail:
call :tail "%RECEIVER_STDOUT%" 40
goto :cleanup

:warmup_done

REM ==========================================================
REM Sweep runs (receiver stays; streamer+sender per Ts)
REM   sender: ALWAYS full send
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
      "  Start-Sleep -Milliseconds 300;" ^
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
    call :tail "!STREAMER_STDERR!" 120
    echo [ERROR] streamer_stdout:
    call :tail "!STREAMER_STDOUT!" 80
    goto :cleanup
  )

  timeout /t 1 /nobreak >nul

  echo [INFO] Sending... Ts=%%T
  "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" ^
    --host "%UDP_HOST%" --port "%UDP_PORT%" ^
    --interval %%T ^
    > "!SENDER_LOG!" 2>&1

  if errorlevel 1 (
    echo [ERROR] sender failed. Tail:
    call :tail "!SENDER_LOG!" 200
    goto :cleanup
  )

  findstr /i /c:"Sent 0 frames" "!SENDER_LOG!" >nul 2>&1
  if !errorlevel! EQU 0 (
    echo [ERROR] sender sent 0 frames. Tail:
    call :tail "!SENDER_LOG!" 120
    goto :cleanup
  )

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

:tail
REM %1 path, %2 lines
setlocal
set "P=%~1"
set "N=%~2"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "if(Test-Path '%P%'){ Get-Content -Tail %N% '%P%' }" 2>nul
endlocal & exit /b 0

:wait_actor_updates_tail
REM args: %1 stdout_path, %2 stderr_path, %3 timeout_sec, %4 interval_sec
setlocal EnableDelayedExpansion
set "OUTP=%~1"
set "ERRP=%~2"
set "TMO=%~3"
set "INT=%~4"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$out='%OUTP%'; $err='%ERRP%'; $timeout=%TMO%; $interval=%INT%;" ^
  "$sw=[Diagnostics.Stopwatch]::StartNew();" ^
  "while($sw.Elapsed.TotalSeconds -lt $timeout){" ^
  "  $txt='';" ^
  "  if(Test-Path $out){ $txt += (Get-Content -Tail 200 -Raw -ErrorAction SilentlyContinue $out) }" ^
  "  if(Test-Path $err){ $txt += (Get-Content -Tail 200 -Raw -ErrorAction SilentlyContinue $err) }" ^
  "  if($txt -match '\(([1-9][0-9]*) actor updates\)'){ exit 0 }" ^
  "  Start-Sleep -Seconds $interval" ^
  "}" ^
  "exit 1" >nul

set "RC=%errorlevel%"
endlocal & exit /b %RC%
