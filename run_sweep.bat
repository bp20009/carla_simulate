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

REM Input CSV (sender reads this)
REM   (e.g. output from scripts/convert_vehicle_state_csv.py)
set "CSV_PATH=send_data\exp_300.csv"

REM Script paths
set "ROOT=%~dp0"
set "SENDER=%ROOT%send_data\send_udp_frames_from_csv.py"
set "RECEIVER=%ROOT%scripts\udp_replay\replay_from_udp.py"
set "STREAMER=%ROOT%scripts\vehicle_state_stream.py"

REM Receiver fixed delta (do not sweep)
set "FIXED_DELTA=0.05"

REM Warmup frames and delay (receiver stays alive)
set "WARMUP_START_FRAME=0"
set "WARMUP_END_FRAME=1800"
set "WARMUP_INTERVAL=0.1"
set "WARMUP_WAIT_SEC=10"
set "WARMUP_CHECK_TIMEOUT_SEC=180"
set "WARMUP_CHECK_INTERVAL_SEC=5"
set "WARMUP_MAX_ATTEMPTS=3"

REM Cooldown to allow stale actor cleanup (seconds)
set "STALE_TIMEOUT=2.0"
set "COOLDOWN_SEC=3"

REM Output root dir
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "DT_TAG=%%i"
set "OUTDIR=sweep_results_%DT_TAG%"
mkdir "%OUTDIR%" 2>nul

set "RECEIVER_LOG=%OUTDIR%\receiver.log"
set "RECEIVER_TIMING_CSV=%OUTDIR%\update_timings_all.csv"
set "RECEIVER_EVAL_CSV=%OUTDIR%\eval_all.csv"
set "RECEIVER_PID_FILE=%OUTDIR%\receiver_pid.txt"

echo [INFO] OUTDIR=%OUTDIR%
echo [INFO] CSV=%CSV_PATH%

REM ==========================================================
REM Sweep lists
REM   NS: 10..100 step 10
REM   TS_LIST: 0.10 1.00
REM ==========================================================
set "TS_LIST=0.10 1.00"

REM ==========================================================
REM Start RECEIVER once (keep alive for entire sweep) - robust
REM ==========================================================
set "PS_START_RECEIVER=%OUTDIR%\start_receiver.ps1"
> "%PS_START_RECEIVER%" (
  echo $ErrorActionPreference = 'Stop'
  echo $wd = '%CD%'
  echo $python = 'python'
  echo $argsList = @(
  echo   '%RECEIVER%',
  echo   '--carla-host','%CARLA_HOST%',
  echo   '--carla-port','%CARLA_PORT%',
  echo   '--listen-host','%LISTEN_HOST%',
  echo   '--listen-port','%UDP_PORT%',
  echo   '--fixed-delta','%FIXED_DELTA%',
  echo   '--stale-timeout','%STALE_TIMEOUT%',
  echo   '--measure-update-times',
  echo   '--timing-output','%RECEIVER_TIMING_CSV%',
  echo   '--eval-output','%RECEIVER_EVAL_CSV%'
  echo ^)
  echo $p = Start-Process -PassThru -NoNewWindow -WorkingDirectory $wd -FilePath $python -ArgumentList $argsList -RedirectStandardOutput '%RECEIVER_LOG%' -RedirectStandardError '%RECEIVER_LOG%'
  echo Start-Sleep -Seconds 1
  echo if($p.HasExited){
  echo   Add-Content -Path '%RECEIVER_LOG%' -Value ("[BAT] receiver exited immediately. ExitCode=" + $p.ExitCode)
  echo   exit 2
  echo }
  echo Set-Content -Encoding ascii -Path '%RECEIVER_PID_FILE%' -Value $p.Id
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_START_RECEIVER%" >nul
if errorlevel 1 (
  echo [ERROR] Failed to start receiver. Check %RECEIVER_LOG%
  type "%RECEIVER_LOG%"
  goto :cleanup
)

REM Give receiver time to boot
timeout /t 2 /nobreak >nul

REM ==========================================================
REM Warmup: send a short segment to trigger initialization
REM ==========================================================
REM Warmup attempts until receiver logs a spawn
for /L %%A in (1,1,%WARMUP_MAX_ATTEMPTS%) do (
  echo [INFO] Warmup attempt %%A/%WARMUP_MAX_ATTEMPTS%
  set "TIMING_OFFSET=0"
  if exist "%RECEIVER_TIMING_CSV%" (
    for /f %%S in ('powershell -NoProfile -Command "(Get-Item ''%RECEIVER_TIMING_CSV%'').Length"') do set "TIMING_OFFSET=%%S"
  )
  echo [INFO] Warmup frames %WARMUP_START_FRAME%..%WARMUP_END_FRAME% interval=%WARMUP_INTERVAL%
  python "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %WARMUP_INTERVAL% --start-frame %WARMUP_START_FRAME% --end-frame %WARMUP_END_FRAME% > "%OUTDIR%\warmup_sender.log" 2>&1

  REM Optional wait for heavy initialization
  if %WARMUP_WAIT_SEC% GTR 0 (
    timeout /t %WARMUP_WAIT_SEC% /nobreak >nul
  )

  powershell -NoProfile -ExecutionPolicy Bypass -Command "$path='%RECEIVER_TIMING_CSV%'; $offset=[int64]%TIMING_OFFSET%; $timeout=%WARMUP_CHECK_TIMEOUT_SEC%; $interval=%WARMUP_CHECK_INTERVAL_SEC%; $sw=[Diagnostics.Stopwatch]::StartNew(); while($sw.Elapsed.TotalSeconds -lt $timeout){ if(Test-Path $path){ $len=(Get-Item $path).Length; if($len -gt $offset){ $fs=[System.IO.File]::Open($path,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,[System.IO.FileShare]::ReadWrite); try{ $fs.Seek($offset,[System.IO.SeekOrigin]::Begin) | Out-Null; $buf=New-Object byte[] ($len-$offset); [void]$fs.Read($buf,0,$buf.Length); $text=[System.Text.Encoding]::UTF8.GetString($buf); } finally { $fs.Close() } $offset=$len; foreach($line in ($text -split \"`r?`n\")){ if(-not $line){ continue } $cols=$line.Split(','); if($cols.Length -ge 7 -and $cols[5] -ne ''){ exit 0 } } } } Start-Sleep -Seconds $interval }; exit 1" >nul
  if !errorlevel! EQU 0 (
    echo [INFO] Warmup succeeded (actor update detected).
    goto :warmup_done
  )
  echo [WARN] Warmup not confirmed (no actor updates detected). Retrying...
)

echo [ERROR] Warmup failed to detect spawn after %WARMUP_MAX_ATTEMPTS% attempts.
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

    echo ============================================================
    echo [RUN] !TAG!
    echo [DIR] !RUNDIR!

    set "STREAM_CSV=!RUNDIR!\replay_state_!TAG!.csv"
    set "STREAM_TIMING_CSV=!RUNDIR!\stream_timing_!TAG!.csv"

    set "SENDER_LOG=!RUNDIR!\sender_!TAG!.log"
    set "STREAMER_LOG=!RUNDIR!\streamer_!TAG!.log"
    set "STREAMER_PID_FILE=!RUNDIR!\streamer_pid.txt"

    REM ------------------------------------------------------------
    REM 1) Start STREAMER for this run (background)
    REM ------------------------------------------------------------
    set "PS_START_STREAMER=!RUNDIR!\start_streamer.ps1"
    > "!PS_START_STREAMER!" (
      echo $ErrorActionPreference = 'Stop'
      echo $argsList = @(
      echo   '%STREAMER%',
      echo   '--host','%CARLA_HOST%',
      echo   '--port','%CARLA_PORT%',
      echo   '--mode','wait',
      echo   '--role-prefix','udp_replay:',
      echo   '--include-velocity',
      echo   '--frame-elapsed',
      echo   '--wall-clock',
      echo   '--include-object-id',
      echo   '--include-monotonic',
      echo   '--include-tick-wall-dt',
      echo   '--output','!STREAM_CSV!',
      echo   '--timing-output','!STREAM_TIMING_CSV!',
      echo   '--timing-flush-every','10'
      echo ^)
      echo $p = Start-Process -PassThru -NoNewWindow -FilePath 'python' -ArgumentList $argsList -RedirectStandardOutput '!STREAMER_LOG!' -RedirectStandardError '!STREAMER_LOG!'
      echo Set-Content -Encoding ascii -Path '!STREAMER_PID_FILE!' -Value $p.Id
    )
    powershell -NoProfile -ExecutionPolicy Bypass -File "!PS_START_STREAMER!" >nul

    REM Give streamer time to start
    timeout /t 1 /nobreak >nul

    REM ------------------------------------------------------------
    REM 2) Run SENDER (blocking)
    REM ------------------------------------------------------------
    set "FRAME_STRIDE=1"
    if "%%T"=="1.00" set "FRAME_STRIDE=10"

    echo [INFO] Sending... N=%%N Ts=%%T stride=!FRAME_STRIDE!
    python "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %%T --frame-stride !FRAME_STRIDE! --max-actors %%N > "!SENDER_LOG!" 2>&1

    REM ------------------------------------------------------------
    REM 3) Stop STREAMER for this run
    REM ------------------------------------------------------------
    for /f %%P in ('type "%STREAMER_PID_FILE%"') do taskkill /PID %%P /T /F >nul 2>&1

    REM Cooldown so stale actors are cleared before next run
    timeout /t %COOLDOWN_SEC% /nobreak >nul

    echo [DONE] !TAG!
  )
)

REM ==========================================================
REM Stop RECEIVER
REM ==========================================================
for /f %%P in ('type "%RECEIVER_PID_FILE%"') do taskkill /PID %%P /T /F >nul 2>&1

echo [ALL DONE] %OUTDIR%
popd
endlocal
exit /b 0

:cleanup
for /f %%P in ('type "%RECEIVER_PID_FILE%"') do taskkill /PID %%P /T /F >nul 2>&1
popd
endlocal
exit /b 1
