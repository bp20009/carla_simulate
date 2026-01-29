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

REM ==========================================================
REM Receiver params
REM ==========================================================
set "FIXED_DELTA=0.05"
set "STALE_TIMEOUT=2.0"

REM ==========================================================
REM Warmup settings
REM   sender is ALWAYS "from start, send all", but warmup stops early
REM ==========================================================
set "WARMUP_INTERVAL=0.1"
set "WARMUP_MAX_ATTEMPTS=3"
set "WARMUP_CHECK_TIMEOUT_SEC=180"
set "WARMUP_CHECK_INTERVAL_SEC=2"

REM ==========================================================
REM Sweep settings
REM ==========================================================
set "TS_LIST=0.10 1.00"
set "COOLDOWN_SEC=5"

REM ==========================================================
REM OUTDIR
REM ==========================================================
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"') do set "DT_TAG=%%i"
set "OUTDIR=%ROOT%sweep_results_%DT_TAG%"
mkdir "%OUTDIR%" 2>nul

set "RECEIVER_LOG=%OUTDIR%\receiver.log"
set "RECEIVER_TIMING_CSV=%OUTDIR%\update_timings_all.csv"
set "RECEIVER_EVAL_CSV=%OUTDIR%\eval_all.csv"
set "RECEIVER_PID_FILE=%OUTDIR%\receiver_pid.txt"

echo [INFO] OUTDIR=%OUTDIR%
echo [INFO] CSV=%CSV_PATH%

REM ==========================================================
REM Pre-clean (kill stale receiver/streamer + free UDP port)
REM ==========================================================
echo [INFO] Cleaning stale python processes (receiver/streamer)...
call :kill_python_by_cmd "replay_from_udp.py"
call :kill_python_by_cmd "vehicle_state_stream.py"

echo [INFO] Freeing UDP port %UDP_PORT% if occupied...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p=%UDP_PORT%; " ^
  "$eps=Get-NetUDPEndpoint -LocalPort $p -ErrorAction SilentlyContinue; " ^
  "if($eps){ $pids=$eps | Select-Object -Expand OwningProcess -Unique; " ^
  "foreach($pid in $pids){ try{ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue }catch{} } }" ^
  >nul 2>nul

REM ==========================================================
REM Start RECEIVER (background), capture PID
REM ==========================================================
echo [INFO] Starting receiver...
del /q "%RECEIVER_LOG%" >nul 2>&1
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
    "$p=Start-Process -PassThru -NoNewWindow -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '%RECEIVER_LOG%' -RedirectStandardError '%RECEIVER_LOG%';" ^
    "Start-Sleep -Milliseconds 300;" ^
    "if($p.HasExited){ exit 2 };" ^
    "Write-Output $p.Id"
') do set "RECEIVER_PID=%%P"

if not defined RECEIVER_PID (
  echo [ERROR] Failed to start receiver. Check log:
  if exist "%RECEIVER_LOG%" type "%RECEIVER_LOG%"
  goto :cleanup
)

echo %RECEIVER_PID% > "%RECEIVER_PID_FILE%"
echo [INFO] receiver PID=%RECEIVER_PID%

timeout /t 1 /nobreak >nul

REM ==========================================================
REM Warmup
REM   sender: ALWAYS from start, send all
REM   but we stop sender early when receiver reports "(>=1 actor updates)"
REM ==========================================================
for /L %%A in (1,1,%WARMUP_MAX_ATTEMPTS%) do (
  echo [INFO] Warmup attempt %%A/%WARMUP_MAX_ATTEMPTS%

  set "LOG_OFFSET=0"
  if exist "%RECEIVER_LOG%" (
    for /f %%S in ('powershell -NoProfile -Command "(Get-Item ''%RECEIVER_LOG%'').Length"') do set "LOG_OFFSET=%%S"
  )

  set "WARMUP_SENDER_LOG=%OUTDIR%\warmup_sender_%%A.log"
  del /q "!WARMUP_SENDER_LOG!" >nul 2>&1
  set "WARMUP_SENDER_PID="

  REM start sender in background (from start, send all)
  for /f %%P in ('
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ErrorActionPreference='Stop';" ^
      "$args=@(" ^
        "'%SENDER%'," ^
        "'%CSV_PATH%'," ^
        "'--host','%UDP_HOST%'," ^
        "'--port','%UDP_PORT%'," ^
        "'--interval','%WARMUP_INTERVAL%'" ^
      ");" ^
      "$p=Start-Process -PassThru -NoNewWindow -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '!WARMUP_SENDER_LOG!' -RedirectStandardError '!WARMUP_SENDER_LOG!';" ^
      "Start-Sleep -Milliseconds 200;" ^
      "Write-Output $p.Id"
  ') do set "WARMUP_SENDER_PID=%%P"

  if not defined WARMUP_SENDER_PID (
    echo [ERROR] Warmup sender failed to start. log:
    if exist "!WARMUP_SENDER_LOG!" type "!WARMUP_SENDER_LOG!"
    goto :cleanup
  )

  REM wait receiver log update
  call :wait_actor_updates_in_log "%RECEIVER_LOG%" !LOG_OFFSET! %WARMUP_CHECK_TIMEOUT_SEC% %WARMUP_CHECK_INTERVAL_SEC%
  if !errorlevel! EQU 0 (
    echo [INFO] Warmup succeeded. Killing warmup sender PID=!WARMUP_SENDER_PID!
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "try{ Stop-Process -Id !WARMUP_SENDER_PID! -Force -ErrorAction SilentlyContinue }catch{}" >nul 2>nul
    goto :warmup_done
  )

  echo [WARN] Warmup not confirmed. Killing warmup sender and retry...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "try{ Stop-Process -Id !WARMUP_SENDER_PID! -Force -ErrorAction SilentlyContinue }catch{}" >nul 2>nul
)

echo [ERROR] Warmup failed.
goto :cleanup

:warmup_done

REM ==========================================================
REM Sweep runs
REM   receiver stays alive
REM   streamer restarts each run
REM   sender ALWAYS from start, send all (no frame range). we only vary interval/stride/max-actors
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

    del /q "!STREAMER_LOG!" >nul 2>&1
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
        "$p=Start-Process -PassThru -NoNewWindow -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '!STREAMER_LOG!' -RedirectStandardError '!STREAMER_LOG!';" ^
        "Start-Sleep -Milliseconds 200;" ^
        "if($p.HasExited){ exit 2 };" ^
        "Write-Output $p.Id"
    ') do set "STREAMER_PID=%%P"

    if not defined STREAMER_PID (
      echo [ERROR] Failed to start streamer.
      if exist "!STREAMER_LOG!" type "!STREAMER_LOG!"
      goto :cleanup
    )

    timeout /t 1 /nobreak >nul

    set "FRAME_STRIDE=1"
    if "%%T"=="1.00" set "FRAME_STRIDE=10"

    echo [INFO] Sending... N=%%N Ts=%%T stride=!FRAME_STRIDE!
    "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" ^
      --host "%UDP_HOST%" --port "%UDP_PORT%" ^
      --interval %%T --frame-stride !FRAME_STRIDE! --max-actors %%N ^
      > "!SENDER_LOG!" 2>&1

    REM if sender sent 0 frames, stop early with logs
    findstr /i /c:"Sent 0 frames" "!SENDER_LOG!" >nul && (
      echo [ERROR] Sender sent 0 frames. Check CSV format/columns and sender filters.
      echo [ERROR] See: !SENDER_LOG!
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
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$sub='%SUB%';" ^
  "$procs=Get-CimInstance Win32_Process -Filter ""Name='python.exe'"" | Where-Object { $_.CommandLine -and ($_.CommandLine -like ('*' + $sub + '*')) };" ^
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

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$path='%LOG%'; $off=[int64]%OFF%; $timeout=[int]%TMO%; $interval=[int]%INT%; $sw=[Diagnostics.Stopwatch]::StartNew();" ^
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
