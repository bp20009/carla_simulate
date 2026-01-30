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

REM Receiver params
set "FIXED_DELTA=0.05"
set "STALE_TIMEOUT=2.0"

REM Warmup
set "WARMUP_INTERVAL=0.1"
set "WARMUP_MAX_ATTEMPTS=2"
set "WARMUP_CHECK_TIMEOUT_SEC=420"
set "WARMUP_CHECK_INTERVAL_SEC=2"

REM Sweep
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

set "WARMUP_SENDER_LOG=%OUTDIR%\warmup_sender.log"

echo [INFO] OUTDIR=%OUTDIR%
echo [INFO] CSV=%CSV_PATH%

REM ==========================================================
REM Pre-clean (kill stale + free UDP port)
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
REM Start receiver ONCE (never restart)
REM ==========================================================
echo [INFO] Starting receiver...
del /q "%RECEIVER_STDOUT%" "%RECEIVER_STDERR%" >nul 2>&1

set "RECEIVER_PID="
for /f "usebackq delims=" %%P in (`
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
      "'--eval-output','%RECEIVER_EVAL_CSV%'," ^
      "'--enable-completion'" ^
    ");" ^
    "$p=Start-Process -PassThru -NoNewWindow -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '%RECEIVER_STDOUT%' -RedirectStandardError '%RECEIVER_STDERR%';" ^
    "Start-Sleep -Milliseconds 400;" ^
    "if($p.HasExited){ Write-Output ('START_FAILED ExitCode='+$p.ExitCode); exit 2 }" ^
    "Write-Output $p.Id"
`) do set "RECEIVER_PID=%%P"

echo %RECEIVER_PID% | findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 (
  echo [ERROR] Failed to start receiver. Result=%RECEIVER_PID%
  echo [ERROR] receiver_stderr:
  if exist "%RECEIVER_STDERR%" type "%RECEIVER_STDERR%"
  goto :cleanup
)

echo %RECEIVER_PID% > "%RECEIVER_PID_FILE%"
echo [INFO] receiver PID=%RECEIVER_PID%

timeout /t 1 /nobreak >nul

REM ==========================================================
REM Warmup: sender sends ALL frames, wait until "(>=1 actor updates)" appears
REM Then record baseline offsets (bytes) to discard warmup portion later.
REM ==========================================================
echo [INFO] Warmup phase (compile may take 1-2 minutes)...

set "WARMUP_OK=0"
for /L %%A in (1,1,%WARMUP_MAX_ATTEMPTS%) do (
  echo [INFO] Warmup attempt %%A/%WARMUP_MAX_ATTEMPTS%

  REM record current stderr size as offset for tail scan
  set "LOG_OFF=0"
  call :get_file_len "%RECEIVER_STDERR%" LOG_OFF

  REM sender: send ALL frames (no id/max range management)
  "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %WARMUP_INTERVAL% > "%WARMUP_SENDER_LOG%" 2>&1

  REM wait for actor updates log
  call :wait_log_regex "%RECEIVER_STDERR%" !LOG_OFF! "\(([1-9][0-9]*) actor updates\)" %WARMUP_CHECK_TIMEOUT_SEC% %WARMUP_CHECK_INTERVAL_SEC%
  if !errorlevel! EQU 0 (
    set "WARMUP_OK=1"
    goto :warmup_done
  )

  echo [WARN] Warmup not confirmed yet. receiver_stderr tail:
  powershell -NoProfile -Command "if(Test-Path '%RECEIVER_STDERR%'){ Get-Content -Path '%RECEIVER_STDERR%' -Tail 30 }"
)

:warmup_done
if "%WARMUP_OK%" NEQ "1" (
  echo [ERROR] Warmup failed (actor updates not detected within timeout).
  goto :cleanup
)
echo [INFO] Warmup succeeded. Recording baseline offsets (discard warmup data).

set "BASE_TIMING_OFF=0"
set "BASE_EVAL_OFF=0"
call :get_file_len "%RECEIVER_TIMING_CSV%" BASE_TIMING_OFF
call :get_file_len "%RECEIVER_EVAL_CSV%" BASE_EVAL_OFF

echo [INFO] baseline timing bytes=%BASE_TIMING_OFF%
echo [INFO] baseline eval   bytes=%BASE_EVAL_OFF%

REM ==========================================================
REM Sweep runs (receiver stays alive)
REM - For each run, slice [offset_before, offset_after) from master CSVs
REM - Save per-run CSVs under RUNDIR
REM ==========================================================
for %%T in (%TS_LIST%) do (
  set "TAG=Ts%%T"
  set "RUNDIR=%OUTDIR%\!TAG!"
  mkdir "!RUNDIR!" 2>nul

  set "STREAM_CSV=!RUNDIR!\replay_state_!TAG!.csv"
  set "STREAM_TIMING_CSV=!RUNDIR!\stream_timing_!TAG!.csv"
  set "STREAMER_STDOUT=!RUNDIR!\streamer_stdout.log"
  set "STREAMER_STDERR=!RUNDIR!\streamer_stderr.log"
  set "SENDER_LOG=!RUNDIR!\sender_!TAG!.log"

  set "RUN_TIMING_MASTER_BEFORE=0"
  set "RUN_EVAL_MASTER_BEFORE=0"
  set "RUN_TIMING_MASTER_AFTER=0"
  set "RUN_EVAL_MASTER_AFTER=0"

  echo ============================================================
  echo [RUN] !TAG!
  echo [DIR] !RUNDIR!

  REM capture offsets BEFORE run, but never earlier than baseline
  call :get_file_len "%RECEIVER_TIMING_CSV%" RUN_TIMING_MASTER_BEFORE
  call :get_file_len "%RECEIVER_EVAL_CSV%"   RUN_EVAL_MASTER_BEFORE
  if !RUN_TIMING_MASTER_BEFORE! LSS %BASE_TIMING_OFF% set "RUN_TIMING_MASTER_BEFORE=%BASE_TIMING_OFF%"
  if !RUN_EVAL_MASTER_BEFORE!   LSS %BASE_EVAL_OFF%   set "RUN_EVAL_MASTER_BEFORE=%BASE_EVAL_OFF%"

  REM start streamer
  del /q "!STREAMER_STDOUT!" "!STREAMER_STDERR!" >nul 2>&1
  set "STREAMER_PID="
  for /f "usebackq delims=" %%P in (`
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
      "Start-Sleep -Milliseconds 300;" ^
      "if($p.HasExited){ Write-Output ('START_FAILED ExitCode='+$p.ExitCode); exit 2 }" ^
      "Write-Output $p.Id"
  `) do set "STREAMER_PID=%%P"

  echo !STREAMER_PID! | findstr /r "^[0-9][0-9]*$" >nul
  if errorlevel 1 (
    echo [ERROR] Failed to start streamer. Result=!STREAMER_PID!
    if exist "!STREAMER_STDERR!" type "!STREAMER_STDERR!"
    goto :cleanup
  )

  timeout /t 1 /nobreak >nul

  REM sender: send ALL frames (no id/max)
  echo [INFO] Sending... interval=%%T (sender sends ALL frames)
  "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %%T > "!SENDER_LOG!" 2>&1

  REM stop streamer
  taskkill /PID !STREAMER_PID! /T /F >nul 2>&1

  REM cooldown
  timeout /t %COOLDOWN_SEC% /nobreak >nul

  REM capture offsets AFTER run
  call :get_file_len "%RECEIVER_TIMING_CSV%" RUN_TIMING_MASTER_AFTER
  call :get_file_len "%RECEIVER_EVAL_CSV%"   RUN_EVAL_MASTER_AFTER

  REM slice master deltas into run files
  call :slice_csv "%RECEIVER_TIMING_CSV%" !RUN_TIMING_MASTER_BEFORE! !RUN_TIMING_MASTER_AFTER! "!RUNDIR!\update_timings_!TAG!.csv"
  call :slice_csv "%RECEIVER_EVAL_CSV%"   !RUN_EVAL_MASTER_BEFORE!   !RUN_EVAL_MASTER_AFTER!   "!RUNDIR!\eval_!TAG!.csv"

  echo [DONE] !TAG!
)

REM ==========================================================
REM Stop receiver
REM ==========================================================
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

:get_file_len
REM args: file_path out_var
set "%~2=0"
if exist "%~1" (
  for /f %%S in ('powershell -NoProfile -Command "(Get-Item ''%~1'').Length"') do set "%~2=%%S"
)
exit /b 0

:wait_log_regex
REM args: log_path offset_bytes regex timeout_sec interval_sec
setlocal EnableDelayedExpansion
set "LOG=%~1"
set "OFF=%~2"
set "PAT=%~3"
set "TMO=%~4"
set "INT=%~5"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$path='%LOG%'; $off=[int64]%OFF%; $pat='%PAT%'; $timeout=%TMO%; $interval=%INT%; $sw=[Diagnostics.Stopwatch]::StartNew();" ^
  "while($sw.Elapsed.TotalSeconds -lt $timeout){" ^
  "  if(Test-Path $path){" ^
  "    $len=(Get-Item $path).Length;" ^
  "    if($len -gt $off){" ^
  "      $fs=[System.IO.File]::Open($path,'Open','Read','ReadWrite');" ^
  "      try{ $fs.Seek($off,[System.IO.SeekOrigin]::Begin)|Out-Null; $buf=New-Object byte[] ($len-$off); [void]$fs.Read($buf,0,$buf.Length); $txt=[Text.Encoding]::UTF8.GetString($buf) } finally { $fs.Close() }" ^
  "      if($txt -match $pat){ exit 0 }" ^
  "      $off=$len" ^
  "    }" ^
  "  }" ^
  "  Start-Sleep -Seconds $interval" ^
  "}" ^
  "exit 1" >nul
set "RC=%errorlevel%"
endlocal & exit /b %RC%

:slice_csv
REM args: master_csv start_byte end_byte out_csv
setlocal EnableDelayedExpansion
set "IN=%~1"
set "S=%~2"
set "E=%~3"
set "OUT=%~4"

if not exist "%IN%" (
  (echo) > "%OUT%"
  endlocal & exit /b 0
)

set /a "NBYTES=E-S"
if !NBYTES! LEQ 0 (
  (echo) > "%OUT%"
  endlocal & exit /b 0
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$in='%IN%'; $out='%OUT%'; $s=[int64]%S%; $e=[int64]%E%; $n=$e-$s;" ^
  "$fs=[System.IO.File]::Open($in,'Open','Read','ReadWrite');" ^
  "try{ $fs.Seek($s,[System.IO.SeekOrigin]::Begin)|Out-Null; $buf=New-Object byte[] $n; $r=$fs.Read($buf,0,$n); $txt=[Text.Encoding]::UTF8.GetString($buf,0,$r) } finally { $fs.Close() }" ^
  "Set-Content -Path $out -Value $txt -Encoding utf8" >nul 2>nul

endlocal & exit /b 0
