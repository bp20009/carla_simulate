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

REM Receiver params
set "FIXED_DELTA=0.05"
set "STALE_TIMEOUT=2.0"

REM Warmup: map compile can take 1-2 minutes; give enough time
set "WARMUP_INTERVAL=0.10"
set "WARMUP_WAIT_SEC=0"
set "WARMUP_TIMEOUT_SEC=240"
set "WARMUP_POLL_SEC=5"
set "WARMUP_MAX_ATTEMPTS=2"

REM Sweep: you can adjust TS_LIST only; N is handled externally (CSV generation etc.)
set "TS_LIST=0.10 1.00"
set "COOLDOWN_SEC=3"

REM Sender should ALWAYS send everything
set "SEND_START_FRAME=0"
set "SEND_END_FRAME=999999999"

REM ==========================================================
REM Basic checks
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
  echo [ERROR] SENDER not found: %SENDER%
  goto :cleanup
)
if not exist "%RECEIVER%" (
  echo [ERROR] RECEIVER not found: %RECEIVER%
  goto :cleanup
)
if not exist "%STREAMER%" (
  echo [ERROR] STREAMER not found: %STREAMER%
  goto :cleanup
)

REM ==========================================================
REM OUTDIR
REM ==========================================================
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"') do set "DT_TAG=%%i"
set "OUTDIR=%ROOT%sweep_results_%DT_TAG%"
mkdir "%OUTDIR%" 2>nul

set "RECEIVER_PID_FILE=%OUTDIR%\receiver_pid.txt"
set "RECEIVER_STDOUT=%OUTDIR%\receiver_stdout.log"
set "RECEIVER_STDERR=%OUTDIR%\receiver_stderr.log"
set "RECEIVER_TIMING_CSV=%OUTDIR%\update_timings_all.csv"
set "RECEIVER_EVAL_CSV=%OUTDIR%\eval_all.csv"

set "WARMUP_SENDER_STDOUT=%OUTDIR%\warmup_sender_stdout.log"
set "WARMUP_SENDER_STDERR=%OUTDIR%\warmup_sender_stderr.log"

echo [INFO] OUTDIR=%OUTDIR%
echo [INFO] CSV=%CSV_PATH%

REM ==========================================================
REM Pre-clean: kill stale receiver/streamer (no wmic)
REM ==========================================================
echo [INFO] Cleaning stale python processes (receiver/streamer)...
call :kill_python_by_cmd "replay_from_udp.py"
call :kill_python_by_cmd "vehicle_state_stream.py"

echo [INFO] Freeing UDP port %UDP_PORT% if occupied...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p=%UDP_PORT%;" ^
  "$eps=Get-NetUDPEndpoint -LocalPort $p -ErrorAction SilentlyContinue;" ^
  "if($eps){ $pids=$eps|Select-Object -Expand OwningProcess -Unique; foreach($pid in $pids){ try{ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue }catch{} } }" ^
  >nul 2>nul

REM ==========================================================
REM Start receiver (keep alive to avoid repeated map compile)
REM   NOTE: stdout/stderr must be different files on PS Start-Process
REM ==========================================================
echo [INFO] Starting receiver...
del /q "%RECEIVER_STDOUT%" "%RECEIVER_STDERR%" >nul 2>&1
set "RECEIVER_PID="

for /f %%P in ('
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
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
    "$p=Start-Process -PassThru -NoNewWindow -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '%RECEIVER_STDOUT%' -RedirectStandardError '%RECEIVER_STDERR%';" ^
    "Start-Sleep -Milliseconds 300;" ^
    "if($p.HasExited){ Write-Output 'START_FAILED'; exit 0 };" ^
    "Write-Output $p.Id"
') do set "RECEIVER_PID=%%P"

if not defined RECEIVER_PID (
  echo [ERROR] Failed to start receiver (PID empty).
  goto :cleanup
)
if /i "%RECEIVER_PID%"=="START_FAILED" (
  echo [ERROR] Receiver exited immediately. Tail stderr:
  call :tail "%RECEIVER_STDERR%" 80
  goto :cleanup
)

echo %RECEIVER_PID%> "%RECEIVER_PID_FILE%"
echo [INFO] receiver PID=%RECEIVER_PID%

timeout /t 1 /nobreak >nul

REM ==========================================================
REM Warmup: trigger first actor spawn (map compile may happen here)
REM   Success condition: receiver log contains "(>=1 actor updates)"
REM ==========================================================
for /L %%A in (1,1,%WARMUP_MAX_ATTEMPTS%) do (
  echo [INFO] Warmup attempt %%A/%WARMUP_MAX_ATTEMPTS%
  del /q "%WARMUP_SENDER_STDOUT%" "%WARMUP_SENDER_STDERR%" >nul 2>&1

  REM Offsets (bytes) for incremental scan
  set "OFF_OUT=0"
  set "OFF_ERR=0"
  for /f %%S in ('powershell -NoProfile -Command "if(Test-Path ''%RECEIVER_STDOUT%''){(Get-Item ''%RECEIVER_STDOUT%'').Length}else{0}"') do set "OFF_OUT=%%S"
  for /f %%S in ('powershell -NoProfile -Command "if(Test-Path ''%RECEIVER_STDERR%''){(Get-Item ''%RECEIVER_STDERR%'').Length}else{0}"') do set "OFF_ERR=%%S"

  echo [INFO] Warmup send interval=%WARMUP_INTERVAL% (send all frames)
  "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" ^
    --host "%UDP_HOST%" --port "%UDP_PORT%" ^
    --interval %WARMUP_INTERVAL% ^
    --start-frame %SEND_START_FRAME% --end-frame %SEND_END_FRAME% ^
    1> "%WARMUP_SENDER_STDOUT%" 2> "%WARMUP_SENDER_STDERR%"

  if errorlevel 1 (
    echo [ERROR] Warmup sender failed. Tail stderr:
    call :tail "%WARMUP_SENDER_STDERR%" 120
    goto :cleanup
  )

  REM If sender says "Sent 0 frames", fail fast
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$t=''; if(Test-Path '%WARMUP_SENDER_STDOUT%'){ $t+=Get-Content -Raw '%WARMUP_SENDER_STDOUT%' }" ^
    "if(Test-Path '%WARMUP_SENDER_STDERR%'){ $t+=Get-Content -Raw '%WARMUP_SENDER_STDERR%' }" ^
    "if($t -match 'Sent\s+0\s+frames'){ exit 2 } else { exit 0 }" >nul
  if !errorlevel! EQU 2 (
    echo [ERROR] Warmup sender sent 0 frames. Tail sender logs:
    call :tail "%WARMUP_SENDER_STDOUT%" 60
    call :tail "%WARMUP_SENDER_STDERR%" 60
    goto :cleanup
  )

  if %WARMUP_WAIT_SEC% GTR 0 timeout /t %WARMUP_WAIT_SEC% /nobreak >nul

  call :wait_actor_updates "%RECEIVER_STDOUT%" "%RECEIVER_STDERR%" !OFF_OUT! !OFF_ERR! %WARMUP_TIMEOUT_SEC% %WARMUP_POLL_SEC%
  if !errorlevel! EQU 0 (
    echo [INFO] Warmup confirmed (nonzero actor updates observed).
    goto :warmup_done
  )

  echo [WARN] Warmup not confirmed. retry...
  echo [INFO] receiver_stderr tail:
  call :tail "%RECEIVER_STDERR%" 40
)

echo [ERROR] Warmup failed.
goto :cleanup

:warmup_done

REM ==========================================================
REM Sweep runs (receiver stays alive; restart streamer only)
REM ==========================================================
for %%T in (%TS_LIST%) do (
  set "TAG=Ts%%T"
  set "RUNDIR=%OUTDIR%\!TAG!"
  mkdir "!RUNDIR!" 2>nul

  set "STREAM_CSV=!RUNDIR!\replay_state_!TAG!.csv"
  set "STREAM_TIMING_CSV=!RUNDIR!\stream_timing_!TAG!.csv"
  set "STREAMER_STDOUT=!RUNDIR!\streamer_stdout.log"
  set "STREAMER_STDERR=!RUNDIR!\streamer_stderr.log"
  set "SENDER_STDOUT=!RUNDIR!\sender_stdout.log"
  set "SENDER_STDERR=!RUNDIR!\sender_stderr.log"

  echo ============================================================
  echo [RUN] !TAG!
  echo [DIR] !RUNDIR!

  del /q "!STREAMER_STDOUT!" "!STREAMER_STDERR!" >nul 2>&1
  set "STREAMER_PID="

  for /f %%P in ('
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
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
      "$p=Start-Process -PassThru -NoNewWindow -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '!STREAMER_STDOUT!' -RedirectStandardError '!STREAMER_STDERR%';" ^
      "Start-Sleep -Milliseconds 200;" ^
      "if($p.HasExited){ Write-Output 'START_FAILED'; exit 0 };" ^
      "Write-Output $p.Id"
  ') do set "STREAMER_PID=%%P"

  if not defined STREAMER_PID (
    echo [ERROR] Failed to start streamer (PID empty).
    goto :cleanup
  )
  if /i "!STREAMER_PID!"=="START_FAILED" (
    echo [ERROR] Streamer exited immediately. Tail stderr:
    call :tail "!STREAMER_STDERR!" 120
    goto :cleanup
  )

  timeout /t 1 /nobreak >nul

  del /q "!SENDER_STDOUT!" "!SENDER_STDERR!" >nul 2>&1
  echo [INFO] Sending interval=%%T (send all frames)
  "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" ^
    --host "%UDP_HOST%" --port "%UDP_PORT%" ^
    --interval %%T ^
    --start-frame %SEND_START_FRAME% --end-frame %SEND_END_FRAME% ^
    1> "!SENDER_STDOUT!" 2> "!SENDER_STDERR!"

  if errorlevel 1 (
    echo [ERROR] sender failed. Tail stderr:
    call :tail "!SENDER_STDERR!" 120
    goto :cleanup
  )

  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$t=''; if(Test-Path '!SENDER_STDOUT!'){ $t+=Get-Content -Raw '!SENDER_STDOUT!' }" ^
    "if(Test-Path '!SENDER_STDERR!'){ $t+=Get-Content -Raw '!SENDER_STDERR!' }" ^
    "if($t -match 'Sent\s+0\s+frames'){ exit 2 } else { exit 0 }" >nul
  if !errorlevel! EQU 2 (
    echo [ERROR] sender reports Sent 0 frames. Tail logs:
    call :tail "!SENDER_STDOUT!" 80
    call :tail "!SENDER_STDERR!" 80
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

REM ==========================================================
REM Cleanup
REM ==========================================================
:cleanup
echo [CLEANUP]
if defined RECEIVER_PID taskkill /PID %RECEIVER_PID% /T /F >nul 2>&1
call :kill_python_by_cmd "vehicle_state_stream.py"
call :kill_python_by_cmd "replay_from_udp.py"
popd
endlocal
exit /b 1

REM ==========================================================
REM Subroutines (no wmic, no ps1 generation)
REM ==========================================================

:kill_python_by_cmd
REM %1 substring in CommandLine
setlocal EnableExtensions
set "SUB=%~1"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$sub='%SUB%';" ^
  "$procs=Get-CimInstance Win32_Process -Filter ""Name='python.exe'"" | Where-Object { $_.CommandLine -like ('*' + $sub + '*') };" ^
  "foreach($p in $procs){ try{ Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }catch{} }" ^
  >nul 2>nul
endlocal & exit /b 0

:tail
REM %1 path, %2 lines
setlocal
set "P=%~1"
set "N=%~2"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "if(Test-Path '%P%'){ Get-Content -Tail %N% '%P%' }" ^
  2>nul
endlocal & exit /b 0

:wait_actor_updates
REM args: %1 out_log, %2 err_log, %3 out_off_bytes, %4 err_off_bytes, %5 timeout_sec, %6 poll_sec
setlocal EnableDelayedExpansion
set "OUTL=%~1"
set "ERRL=%~2"
set "OOFF=%~3"
set "EOFF=%~4"
set "TMO=%~5"
set "POLL=%~6"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$out='%OUTL%'; $err='%ERRL%'; $ooff=[int64]%OOFF%; $eoff=[int64]%EOFF%; $timeout=%TMO%; $poll=%POLL%;" ^
  "$sw=[Diagnostics.Stopwatch]::StartNew();" ^
  "function ReadNew([string]$path,[ref]$off){" ^
  "  if(!(Test-Path $path)){ return '' }" ^
  "  $len=(Get-Item $path).Length; if($len -le $off.Value){ return '' }" ^
  "  $fs=[System.IO.File]::Open($path,'Open','Read','ReadWrite');" ^
  "  try{ $fs.Seek($off.Value,[System.IO.SeekOrigin]::Begin)|Out-Null; $buf=New-Object byte[] ($len-$off.Value); [void]$fs.Read($buf,0,$buf.Length); } finally { $fs.Close() }" ^
  "  $off.Value=$len; return [Text.Encoding]::UTF8.GetString($buf)" ^
  "}" ^
  "while($sw.Elapsed.TotalSeconds -lt $timeout){" ^
  "  $txt = (ReadNew $out ([ref]$ooff)) + (ReadNew $err ([ref]$eoff));" ^
  "  if($txt -match '\(([1-9][0-9]*) actor updates\)'){ exit 0 }" ^
  "  Start-Sleep -Seconds $poll" ^
  "}" ^
  "exit 1" ^
  >nul

set "RC=%errorlevel%"
endlocal & exit /b %RC%
