@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0"

REM ==============================
REM Config
REM ==============================
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
where "%PYTHON_EXE%" >nul 2>&1 || (
  echo [ERROR] python not found in PATH. set PYTHON_EXE=py or full path.
  exit /b 1
)

set "FIXED_DELTA=0.05"
set "STALE_TIMEOUT=2.0"
set "COOLDOWN_SEC=5"

REM Warmup
set "WARMUP_START_FRAME=0"
set "WARMUP_END_FRAME=1800"
set "WARMUP_INTERVAL=0.1"
set "WARMUP_WAIT_SEC=10"
set "WARMUP_CHECK_TIMEOUT_SEC=300"
set "WARMUP_CHECK_INTERVAL_SEC=5"
set "WARMUP_MAX_ATTEMPTS=3"

REM Sweep
set "TS_LIST=0.10 1.00"

REM ==============================
REM OUTDIR
REM ==============================
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"') do set "DT_TAG=%%i"
set "OUTDIR=%ROOT%sweep_results_%DT_TAG%"
mkdir "%OUTDIR%" 2>nul

set "RECEIVER_LOG=%OUTDIR%\receiver.log"
set "RECEIVER_TIMING_CSV=%OUTDIR%\update_timings_all.csv"
set "RECEIVER_EVAL_CSV=%OUTDIR%\eval_all.csv"

echo [INFO] OUTDIR=%OUTDIR%
echo [INFO] CSV=%CSV_PATH%

REM ==============================
REM Ensure UDP port is free
REM ==============================
echo [INFO] Freeing UDP port %UDP_PORT% if occupied...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p=(Get-NetUDPEndpoint -LocalPort %UDP_PORT% -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess); if($p){ Write-Host ('[WARN] Port %UDP_PORT% used by PID=' + $p); Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }" >nul

REM ==============================
REM Start receiver (once)
REM ==============================
echo [INFO] Starting receiver...
set "RECEIVER_PID="
for /f %%P in ('powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$args=@('%RECEIVER%','--carla-host','%CARLA_HOST%','--carla-port','%CARLA_PORT%','--listen-host','%LISTEN_HOST%','--listen-port','%UDP_PORT%','--fixed-delta','%FIXED_DELTA%','--stale-timeout','%STALE_TIMEOUT%','--measure-update-times','--timing-output','%RECEIVER_TIMING_CSV%','--eval-output','%RECEIVER_EVAL_CSV%');" ^
  "$p=Start-Process -PassThru -NoNewWindow -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '%RECEIVER_LOG%' -RedirectStandardError '%RECEIVER_LOG%';" ^
  "$p.Id"') do set "RECEIVER_PID=%%P"

if not defined RECEIVER_PID (
  echo [ERROR] receiver start failed. check log:
  if exist "%RECEIVER_LOG%" type "%RECEIVER_LOG%"
  goto :cleanup
)
echo [INFO] receiver PID=%RECEIVER_PID%

timeout /t 2 /nobreak >nul

REM ==============================
REM Warmup (watch timing CSV for actor lines)
REM actor line pattern: frame_sequence,carla_frame,,,,actor_id,actor_duration_ms
REM ==============================
set "WARMUP_OK=0"
for /L %%A in (1,1,%WARMUP_MAX_ATTEMPTS%) do (
  echo [INFO] Warmup attempt %%A/%WARMUP_MAX_ATTEMPTS%

  REM record current file length (bytes) as offset
  set "WARMUP_OFFSET=0"
  if exist "%RECEIVER_TIMING_CSV%" (
    for /f %%S in ('powershell -NoProfile -Command "(Get-Item ''%RECEIVER_TIMING_CSV%'').Length"') do set "WARMUP_OFFSET=%%S"
  )

  "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %WARMUP_INTERVAL% --start-frame %WARMUP_START_FRAME% --end-frame %WARMUP_END_FRAME% > "%OUTDIR%\warmup_sender.log" 2>&1

  if %WARMUP_WAIT_SEC% GTR 0 timeout /t %WARMUP_WAIT_SEC% /nobreak >nul

  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$path='%RECEIVER_TIMING_CSV%'; $off=[int64]%WARMUP_OFFSET%; $timeout=%WARMUP_CHECK_TIMEOUT_SEC%; $interval=%WARMUP_CHECK_INTERVAL_SEC%;" ^
    "$sw=[Diagnostics.Stopwatch]::StartNew();" ^
    "while($sw.Elapsed.TotalSeconds -lt $timeout){" ^
    "  if(Test-Path $path){" ^
    "    $len=(Get-Item $path).Length;" ^
    "    if($len -gt $off){" ^
    "      $fs=[System.IO.File]::Open($path,'Open','Read','ReadWrite');" ^
    "      try{ $fs.Seek($off,[System.IO.SeekOrigin]::Begin) | Out-Null; $buf=New-Object byte[] ($len-$off); [void]$fs.Read($buf,0,$buf.Length); $text=[Text.Encoding]::UTF8.GetString($buf) } finally { $fs.Close() }" ^
    "      if($text -match '^[^,]+,[^,]+,,,,[^,]+,' ){ exit 0 }" ^
    "    }" ^
    "  }" ^
    "  Start-Sleep -Seconds $interval" ^
    "}" ^
    "exit 1" >nul

  if !errorlevel! EQU 0 (
    echo [INFO] Warmup succeeded.
    set "WARMUP_OK=1"
    goto :warmup_done
  )

  echo [WARN] Warmup not confirmed. retry...
)

:warmup_done
if "%WARMUP_OK%" NEQ "1" (
  echo [ERROR] Warmup failed.
  goto :cleanup
)

REM ==============================
REM Sweep runs (restart streamer + sender only)
REM ==============================
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

    REM start streamer and capture PID
    set "STREAMER_PID="
    for /f %%P in ('powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$args=@('%STREAMER%','--host','%CARLA_HOST%','--port','%CARLA_PORT%','--mode','wait','--role-prefix','udp_replay:','--include-velocity','--frame-elapsed','--wall-clock','--include-object-id','--include-monotonic','--include-tick-wall-dt','--output','!STREAM_CSV!','--timing-output','!STREAM_TIMING_CSV!','--timing-flush-every','10');" ^
      "$p=Start-Process -PassThru -NoNewWindow -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '!STREAMER_LOG!' -RedirectStandardError '!STREAMER_LOG!';" ^
      "$p.Id"') do set "STREAMER_PID=%%P"

    if not defined STREAMER_PID (
      echo [ERROR] streamer start failed. check log:
      if exist "!STREAMER_LOG!" type "!STREAMER_LOG!"
      goto :cleanup
    )

    timeout /t 1 /nobreak >nul

    set "FRAME_STRIDE=1"
    if "%%T"=="1.00" set "FRAME_STRIDE=10"

    echo [INFO] Sending... N=%%N Ts=%%T stride=!FRAME_STRIDE!
    "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %%T --frame-stride !FRAME_STRIDE! --max-actors %%N > "!SENDER_LOG!" 2>&1

    powershell -NoProfile -ExecutionPolicy Bypass -Command "Stop-Process -Id %STREAMER_PID% -Force -ErrorAction SilentlyContinue" >nul

    timeout /t %COOLDOWN_SEC% /nobreak >nul
    echo [DONE] !TAG!
  )
)

echo [INFO] Stopping receiver...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Stop-Process -Id %RECEIVER_PID% -Force -ErrorAction SilentlyContinue" >nul

echo [ALL DONE] %OUTDIR%
popd
endlocal
exit /b 0

:cleanup
echo [CLEANUP]
if defined RECEIVER_PID (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Stop-Process -Id %RECEIVER_PID% -Force -ErrorAction SilentlyContinue" >nul
)
popd
endlocal
exit /b 1
