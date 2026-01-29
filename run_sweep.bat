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

REM Receiver params
set "FIXED_DELTA=0.05"
set "STALE_TIMEOUT=2.0"

REM Warmup
set "WARMUP_START_FRAME=0"
set "WARMUP_END_FRAME=1800"
set "WARMUP_INTERVAL=0.1"
set "WARMUP_WAIT_SEC=10"
set "WARMUP_CHECK_TIMEOUT_SEC=180"
set "WARMUP_CHECK_INTERVAL_SEC=5"
set "WARMUP_MAX_ATTEMPTS=3"

REM Sweep
set "TS_LIST=0.10 1.00"
set "COOLDOWN_SEC=5"

REM ==========================================================
REM OUTDIR
REM ==========================================================
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"') do set "DT_TAG=%%i"
set "OUTDIR=%ROOT%sweep_results_%DT_TAG%"
mkdir "%OUTDIR%" 2>nul

set "RECEIVER_LOG=%OUTDIR%\receiver.log"
set "RECEIVER_START_ERR=%OUTDIR%\start_receiver.err"
set "RECEIVER_TIMING_CSV=%OUTDIR%\update_timings_all.csv"
set "RECEIVER_EVAL_CSV=%OUTDIR%\eval_all.csv"
set "RECEIVER_PID_FILE=%OUTDIR%\receiver_pid.txt"

echo [INFO] OUTDIR=%OUTDIR%
echo [INFO] CSV=%CSV_PATH%

if not exist "%CSV_PATH%" (
  echo [ERROR] CSV not found: %CSV_PATH%
  goto :cleanup
)

REM ==========================================================
REM Free UDP port if occupied (no wmic)
REM ==========================================================
echo [INFO] Freeing UDP port %UDP_PORT% if occupied...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p=%UDP_PORT%; $eps=Get-NetUDPEndpoint -LocalPort $p -ErrorAction SilentlyContinue; " ^
  "if($eps){ $pids=$eps | Select-Object -Expand OwningProcess -Unique; " ^
  "foreach($pid in $pids){ try{ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue }catch{} } }" >nul 2>nul

REM ==========================================================
REM Kill previous receiver by PID file (best-effort)
REM ==========================================================
if exist "%RECEIVER_PID_FILE%" (
  set "OLD_RPID="
  set /p OLD_RPID=<"%RECEIVER_PID_FILE%"
  if defined OLD_RPID (
    echo [WARN] Killing previous receiver PID=%OLD_RPID%
    taskkill /PID %OLD_RPID% /T /F >nul 2>&1
    timeout /t 1 /nobreak >nul
  )
)

REM ==========================================================
REM Start RECEIVER once (PID returned as digits only)
REM ==========================================================
echo [INFO] Starting receiver...

del /q "%RECEIVER_START_ERR%" >nul 2>&1

set "RECEIVER_PID="
for /f %%P in ('
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue';" ^
    "$py='%PYTHON_EXE%';" ^
    "$args=@('%RECEIVER%','--carla-host','%CARLA_HOST%','--carla-port','%CARLA_PORT%','--listen-host','%LISTEN_HOST%','--listen-port','%UDP_PORT%','--fixed-delta','%FIXED_DELTA%','--stale-timeout','%STALE_TIMEOUT%','--measure-update-times','--timing-output','%RECEIVER_TIMING_CSV%','--eval-output','%RECEIVER_EVAL_CSV%');" ^
    "try{" ^
    "  $p=Start-Process -PassThru -NoNewWindow -FilePath $py -ArgumentList $args -RedirectStandardOutput '%RECEIVER_LOG%' -RedirectStandardError '%RECEIVER_LOG%';" ^
    "  Start-Sleep -Milliseconds 300;" ^
    "  if($p.HasExited){ exit 2 }" ^
    "  Write-Output $p.Id" ^
    "}catch{ $_ | Out-String | Write-Error; exit 3 }"
') do set "RECEIVER_PID=%%P"

if errorlevel 1 (
  echo [ERROR] PowerShell failed to start receiver.
  echo [ERROR] See: %RECEIVER_START_ERR%
)

REM もし PowerShell 側で例外が出ているなら stderr を保存して見せる
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "if($Error.Count -gt 0){ exit 1 } else { exit 0 }" >nul 2>nul

REM 上の for /f の stderr は取り込めないので，別途 start_receiver.err にも落とす版が必要なら下の1行に置換して使う:
REM (環境によってはこの行の方が確実)
REM for /f %%P in ('powershell ... 2^> "%RECEIVER_START_ERR%"') do ...

if not defined RECEIVER_PID (
  echo [ERROR] Failed to start receiver (PID empty).
  echo [ERROR] receiver.log:
  if exist "%RECEIVER_LOG%" type "%RECEIVER_LOG%"
  goto :cleanup
)

echo %RECEIVER_PID% > "%RECEIVER_PID_FILE%"
echo [INFO] receiver PID=%RECEIVER_PID%

timeout /t 2 /nobreak >nul

REM ==========================================================
REM Warmup: watch receiver.log for "(>=1 actor updates)"
REM ==========================================================
for /L %%A in (1,1,%WARMUP_MAX_ATTEMPTS%) do (
  echo [INFO] Warmup attempt %%A/%WARMUP_MAX_ATTEMPTS%

  set "LOG_OFFSET=0"
  if exist "%RECEIVER_LOG%" (
    for /f %%S in ('powershell -NoProfile -Command "(Get-Item ''%RECEIVER_LOG%'').Length"') do set "LOG_OFFSET=%%S"
  )

  echo [INFO] Warmup send frames %WARMUP_START_FRAME%..%WARMUP_END_FRAME% interval=%WARMUP_INTERVAL%
  "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %WARMUP_INTERVAL% --start-frame %WARMUP_START_FRAME% --end-frame %WARMUP_END_FRAME% > "%OUTDIR%\warmup_sender.log" 2>&1

  if %WARMUP_WAIT_SEC% GTR 0 timeout /t %WARMUP_WAIT_SEC% /nobreak >nul

  call :wait_for_actor_updates_in_log "%RECEIVER_LOG%" %LOG_OFFSET% %WARMUP_CHECK_TIMEOUT_SEC% %WARMUP_CHECK_INTERVAL_SEC%
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
REM Sweep runs (restart only streamer + sender)
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
    set "STREAMER_START_ERR=!RUNDIR!\start_streamer.err"

    echo ============================================================
    echo [RUN] !TAG!
    echo [DIR] !RUNDIR!

    del /q "!STREAMER_START_ERR!" >nul 2>&1

    set "STREAMER_PID="
    for /f %%P in ('
      powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue';" ^
        "$py='%PYTHON_EXE%';" ^
        "$args=@('%STREAMER%','--host','%CARLA_HOST%','--port','%CARLA_PORT%','--mode','wait','--role-prefix','udp_replay:','--include-velocity','--frame-elapsed','--wall-clock','--include-object-id','--include-monotonic','--include-tick-wall-dt','--output','!STREAM_CSV!','--timing-output','!STREAM_TIMING_CSV!','--timing-flush-every','10');" ^
        "try{" ^
        "  $p=Start-Process -PassThru -NoNewWindow -FilePath $py -ArgumentList $args -RedirectStandardOutput '!STREAMER_LOG!' -RedirectStandardError '!STREAMER_LOG%';" ^
        "  Start-Sleep -Milliseconds 200;" ^
        "  if($p.HasExited){ exit 2 }" ^
        "  Write-Output $p.Id" ^
        "}catch{ $_ | Out-String | Write-Error; exit 3 }"
    ') do set "STREAMER_PID=%%P"

    if not defined STREAMER_PID (
      echo [ERROR] Failed to start streamer (PID empty).
      if exist "!STREAMER_LOG!" type "!STREAMER_LOG!"
      goto :cleanup
    )

    timeout /t 1 /nobreak >nul

    set "FRAME_STRIDE=1"
    if "%%T"=="1.00" set "FRAME_STRIDE=10"

    echo [INFO] Sending... N=%%N Ts=%%T stride=!FRAME_STRIDE!
    "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %%T --frame-stride !FRAME_STRIDE! --max-actors %%N > "!SENDER_LOG!" 2>&1

    taskkill /PID !STREAMER_PID! /T /F >nul 2>&1

    timeout /t %COOLDOWN_SEC% /nobreak >nul
    echo [DONE] !TAG!
  )
)

echo [INFO] Stopping receiver...
taskkill /PID %RECEIVER_PID% /T /F >nul 2>&1

echo [ALL DONE] %OUTDIR%
popd
endlocal
exit /b 0

:cleanup
echo [CLEANUP]
if defined RECEIVER_PID (
  taskkill /PID %RECEIVER_PID% /T /F >nul 2>&1
)
popd
endlocal
exit /b 1

REM ==========================================================
REM Subroutines
REM ==========================================================

:wait_for_actor_updates_in_log
REM args: %1 log_path, %2 offset_bytes, %3 timeout_sec, %4 interval_sec
setlocal EnableExtensions EnableDelayedExpansion
set "LOG=%~1"
set "OFF=%~2"
set "TMO=%~3"
set "INT=%~4"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
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
