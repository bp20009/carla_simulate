@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0"

REM ==========================================================
REM Executables
REM ==========================================================
set "ROOT=%~dp0"
set "PYTHON_EXE=python"

where "%PYTHON_EXE%" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] python not found in PATH. Set PYTHON_EXE=py or full path.
  goto :cleanup
)

set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" (
  set "PS=pwsh"
  where "%PS%" >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] PowerShell not found. (powershell.exe / pwsh)
    goto :cleanup
  )
)

REM ==========================================================
REM User config
REM ==========================================================
set "CARLA_HOST=127.0.0.1"
set "CARLA_PORT=2000"

set "UDP_HOST=127.0.0.1"
set "UDP_PORT=5005"
set "LISTEN_HOST=0.0.0.0"

set "CSV_PATH=%ROOT%send_data\exp_300.csv"
set "SENDER=%ROOT%send_data\send_udp_frames_from_csv.py"
set "RECEIVER=%ROOT%scripts\udp_replay\replay_from_udp.py"
set "STREAMER=%ROOT%scripts\vehicle_state_stream.py"

REM normalize accidental quotes (safety)
set "CSV_PATH=%CSV_PATH:"=%"
set "SENDER=%SENDER:"=%"
set "RECEIVER=%RECEIVER:"=%"
set "STREAMER=%STREAMER:"=%"

if not exist "%CSV_PATH%"   ( echo [ERROR] CSV not found: %CSV_PATH% & goto :cleanup )
if not exist "%SENDER%"     ( echo [ERROR] sender not found: %SENDER% & goto :cleanup )
if not exist "%RECEIVER%"   ( echo [ERROR] receiver not found: %RECEIVER% & goto :cleanup )
if not exist "%STREAMER%"   ( echo [ERROR] streamer not found: %STREAMER% & goto :cleanup )

REM ==========================================================
REM Policy
REM ==========================================================
set "FIXED_DELTA=0.05"
set "STALE_TIMEOUT=2.0"

REM Warmup: first actor spawn can trigger map compile (long)
set "WARMUP_INTERVAL=0.1"
set "WARMUP_WAIT_SEC=2"
set "WARMUP_CHECK_TIMEOUT_SEC=900"
set "WARMUP_CHECK_INTERVAL_SEC=5"
set "WARMUP_MAX_ATTEMPTS=2"

REM Sweep
set "TS_LIST=0.10 1.00"
set "N_MIN=10"
set "N_MAX=100"
set "N_STEP=10"
set "COOLDOWN_SEC=3"

REM ==========================================================
REM OUTDIR (use PowerShell to avoid locale DATE/TIME junk)
REM ==========================================================
for /f "usebackq delims=" %%i in (`"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"`) do set "DT_TAG=%%i"
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

REM ==========================================================
REM Pre-clean
REM ==========================================================
echo [INFO] Cleaning stale processes (receiver/streamer)...
call :kill_by_cmdline "replay_from_udp.py"
call :kill_by_cmdline "vehicle_state_stream.py"

echo [INFO] Freeing UDP port %UDP_PORT% if occupied...
call :free_udp_port %UDP_PORT%

REM ==========================================================
REM Start receiver (python direct, separate stdout/stderr)
REM ==========================================================
echo [INFO] Starting receiver...

type nul > "%RECEIVER_STDOUT%"
type nul > "%RECEIVER_STDERR%"
del /q "%RECEIVER_PID_FILE%" >nul 2>&1

set "RECEIVER_PID="
call :ps_start_python_pid RECEIVER_PID "%RECEIVER_STDOUT%" "%RECEIVER_STDERR%" "%ROOT%" ^
  "%RECEIVER%" ^
  "--carla-host|%CARLA_HOST%|--carla-port|%CARLA_PORT%|--listen-host|%LISTEN_HOST%|--listen-port|%UDP_PORT%|--fixed-delta|%FIXED_DELTA%|--stale-timeout|%STALE_TIMEOUT%|--measure-update-times|--timing-output|%RECEIVER_TIMING_CSV%|--eval-output|%RECEIVER_EVAL_CSV%"

if not defined RECEIVER_PID (
  echo [ERROR] Failed to start receiver. receiver_stderr tail:
  "%PS%" -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '%RECEIVER_STDERR%'){ Get-Content '%RECEIVER_STDERR%' -Tail 80 }"
  goto :cleanup
)

echo %RECEIVER_PID%> "%RECEIVER_PID_FILE%"
echo [INFO] receiver PID=%RECEIVER_PID%
timeout /t 2 /nobreak >nul

REM ==========================================================
REM Warmup (discard): send once to trigger compile, wait actor updates
REM ==========================================================
for /L %%A in (1,1,%WARMUP_MAX_ATTEMPTS%) do (
  echo [INFO] Warmup attempt %%A/%WARMUP_MAX_ATTEMPTS%
  echo [INFO] Warmup send interval=%WARMUP_INTERVAL% (discard)

  set "LOG_OFFSET=0"
  for /f "usebackq delims=" %%S in (`"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '%RECEIVER_STDERR%'){ (Get-Item '%RECEIVER_STDERR%').Length } else { 0 }"`) do set "LOG_OFFSET=%%S"

  "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %WARMUP_INTERVAL% > "%OUTDIR%\warmup_sender.log" 2>&1

  findstr /i /c:"Sent 0 frames" "%OUTDIR%\warmup_sender.log" >nul 2>&1
  if !errorlevel! EQU 0 (
    echo [ERROR] Warmup sender sent 0 frames. warmup_sender.log:
    type "%OUTDIR%\warmup_sender.log"
    goto :cleanup
  )

  if %WARMUP_WAIT_SEC% GTR 0 timeout /t %WARMUP_WAIT_SEC% /nobreak >nul

  call :wait_actor_updates_in_log "%RECEIVER_STDERR%" !LOG_OFFSET! %WARMUP_CHECK_TIMEOUT_SEC% %WARMUP_CHECK_INTERVAL_SEC%
  if !errorlevel! EQU 0 (
    echo [INFO] Warmup confirmed (actor updates detected).
    goto :warmup_done
  )

  echo [WARN] Warmup not confirmed yet. receiver_stderr tail:
  "%PS%" -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '%RECEIVER_STDERR%'){ Get-Content '%RECEIVER_STDERR%' -Tail 30 }"
)

echo [WARN] Warmup not confirmed, but continue anyway.
echo [WARN] First sweep run may include map compile overhead.
:warmup_done

REM ==========================================================
REM Sweep: N=10..100 step10 x Ts
REM ==========================================================
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
    set "STREAMER_PID_FILE=!RUNDIR!\streamer_pid_!TAG!.txt"

    echo ============================================================
    echo [RUN] !TAG!
    echo [DIR] !RUNDIR!

    type nul > "!STREAMER_STDOUT!"
    type nul > "!STREAMER_STDERR!"
    del /q "!STREAMER_PID_FILE!" >nul 2>&1

    set "STREAMER_PID="
    call :ps_start_python_pid STREAMER_PID "!STREAMER_STDOUT!" "!STREAMER_STDERR!" "%ROOT%" ^
      "%STREAMER%" ^
      "--host|%CARLA_HOST%|--port|%CARLA_PORT%|--mode|wait|--role-prefix|udp_replay:|--include-velocity|--frame-elapsed|--wall-clock|--include-object-id|--include-monotonic|--include-tick-wall-dt|--output|!STREAM_CSV!|--timing-output|!STREAM_TIMING_CSV!|--timing-flush-every|10"

    if not defined STREAMER_PID (
      echo [ERROR] Failed to start streamer. streamer_stderr tail:
      "%PS%" -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '!STREAMER_STDERR!'){ Get-Content '!STREAMER_STDERR!' -Tail 80 }"
      goto :cleanup
    )
    echo !STREAMER_PID!> "!STREAMER_PID_FILE!"

    timeout /t 1 /nobreak >nul

    echo [INFO] Sending... N=%%N Ts=%%T
    "%PYTHON_EXE%" "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %%T --max-actors %%N > "!SENDER_LOG!" 2>&1

    findstr /i /c:"Sent 0 frames" "!SENDER_LOG!" >nul 2>&1
    if !errorlevel! EQU 0 (
      echo [ERROR] sender sent 0 frames. sender log tail:
      "%PS%" -NoProfile -ExecutionPolicy Bypass -Command "Get-Content '!SENDER_LOG!' -Tail 60"
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
call :kill_by_cmdline "vehicle_state_stream.py"
call :kill_by_cmdline "replay_from_udp.py"
if defined RECEIVER_PID taskkill /PID %RECEIVER_PID% /T /F >nul 2>&1
popd
endlocal
exit /b 1

REM ==========================================================
REM Subroutines
REM ==========================================================

:kill_by_cmdline
REM %1 substring in CommandLine; kill python and wrappers
setlocal EnableExtensions
set "SUB=%~1"
"%PS%" -NoProfile -ExecutionPolicy Bypass -Command ^
  "$sub='%SUB%';" ^
  "$procs=Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -like ('*'+$sub+'*')) };" ^
  "foreach($p in $procs){ try{ Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }catch{} }" >nul 2>nul
endlocal & exit /b 0

:free_udp_port
REM %1 port
setlocal EnableExtensions
set "PORT=%~1"
"%PS%" -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p=[int]%PORT%; for($i=0;$i -lt 5;$i++){" ^
  "  $eps=Get-NetUDPEndpoint -LocalPort $p -ErrorAction SilentlyContinue;" ^
  "  if(-not $eps){ exit 0 }" ^
  "  $pids=$eps | Select-Object -ExpandProperty OwningProcess -Unique;" ^
  "  foreach($pid in $pids){ try{ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue }catch{} }" ^
  "  Start-Sleep -Seconds 1" ^
  "} exit 0" >nul 2>nul
endlocal & exit /b 0

:ps_start_python_pid
REM args:
REM   %1 outvar(PID) %2 stdout %3 stderr %4 workdir %5 script %6 argpipe(--a|v|--b|v|flag)
setlocal EnableExtensions DisableDelayedExpansion
set "OUTVAR=%~1"
set "%OUTVAR%="
set "STDOUT=%~2"
set "STDERR=%~3"
set "WORK=%~4"
set "SCRIPT=%~5"
set "ARGPIPE=%~6"

REM Capture PID by printing only PID from PowerShell
for /f "usebackq delims=" %%P in (`
  "%PS%" -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "$args = @('%SCRIPT%') + ('%ARGPIPE%'.Split('|') | Where-Object { $_ -ne '' });" ^
    "$p = Start-Process -PassThru -NoNewWindow -WorkingDirectory '%WORK%' -FilePath '%PYTHON_EXE%' -ArgumentList $args -RedirectStandardOutput '%STDOUT%' -RedirectStandardError '%STDERR%';" ^
    "Start-Sleep -Milliseconds 400;" ^
    "if($p.HasExited){ throw ('exited early. ExitCode=' + $p.ExitCode) }" ^
    "$p.Id"
`) do (
  endlocal & set "%OUTVAR=%%P" & exit /b 0
)

endlocal & exit /b 1

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
  "      try{ $fs.Seek($off,[System.IO.SeekOrigin]::Begin)|Out-Null; $buf=New-Object byte[] ($len-$off); [void]$fs.Read($buf,0,$buf.Length); $txt=[Text.Encoding]::ASCII.GetString($buf) } finally { $fs.Close() }" ^
  "      if($txt -match '\(([1-9][0-9]*) actor updates\)'){ exit 0 }" ^
  "      $off=$len" ^
  "    }" ^
  "  }" ^
  "  Start-Sleep -Seconds $interval" ^
  "}" ^
  "exit 1" >nul

set "RC=%errorlevel%"
endlocal & exit /b %RC%
