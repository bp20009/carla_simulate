@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ==========================================================
REM User config
REM ==========================================================
set "CARLA_HOST=127.0.0.1"
set "CARLA_PORT=2000"

set "UDP_HOST=127.0.0.1"
set "UDP_PORT=5005"

REM Input CSV (sender reads this)
REM   (e.g. output from scripts/convert_vehicle_state_csv.py)
set "CSV_PATH=send_data\vehicle_states_reduced.csv"

REM Script paths
set "SENDER=send_data\send_udp_frames_from_csv.py"
set "RECEIVER=scripts\udp_replay\replay_from_udp.py"
set "STREAMER=scripts\vehicle_state_stream.py"

REM Receiver/Streamer max runtime [seconds] (hard stop)
set "RECEIVER_MAX_RUNTIME=120"
set "STREAMER_MAX_RUNTIME=120"

REM Output root dir
for /f "tokens=1-3 delims=/- " %%a in ("%date%") do set "DATE_TAG=%%a%%b%%c"
for /f "tokens=1-3 delims=:." %%a in ("%time%") do set "TIME_TAG=%%a%%b%%c"
set "OUTDIR=sweep_results_%DATE_TAG%_%TIME_TAG%"
mkdir "%OUTDIR%" 2>nul

echo [INFO] OUTDIR=%OUTDIR%
echo [INFO] CSV=%CSV_PATH%

REM ==========================================================
REM Sweep lists
REM   NS: 10..100 step 10
REM   TS_LIST: 0.05 0.10 0.20
REM   DT_LIST: 0.05 0.10
REM ==========================================================
set "TS_LIST=0.05 0.10 0.20"
set "DT_LIST=0.05 0.10"

for /L %%N in (10,10,100) do (
  for %%TS in (%TS_LIST%) do (
    for %%DT in (%DT_LIST%) do (

      set "TAG=N%%N_Ts%%TS_dt%%DT"
      set "RUNDIR=%OUTDIR%\!TAG!"
      mkdir "!RUNDIR!" 2>nul

      echo ============================================================
      echo [RUN] !TAG!
      echo [DIR] !RUNDIR!

      set "TIMING_CSV=!RUNDIR!\timings_!TAG!.csv"
      set "EVAL_CSV=!RUNDIR!\eval_!TAG!.csv"
      set "STREAM_CSV=!RUNDIR!\replay_state_!TAG!.csv"

      set "SENDER_LOG=!RUNDIR!\sender_!TAG!.log"
      set "RECEIVER_LOG=!RUNDIR!\receiver_!TAG!.log"
      set "STREAMER_LOG=!RUNDIR!\streamer_!TAG!.log"

      REM ------------------------------------------------------------
      REM 1) Start RECEIVER (UDP replay) in background, hard-stop after timeout
      REM    Using PowerShell Start-Process because cmd has no good bg+kill+timeout
      REM ------------------------------------------------------------
      powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$p = Start-Process -PassThru -NoNewWindow -FilePath 'python' -ArgumentList @(" ^
        + "'%RECEIVER%'," ^
        + "'--carla-host','%CARLA_HOST%','--carla-port','%CARLA_PORT%'," ^
        + "'--listen-host','0.0.0.0','--listen-port','%UDP_PORT%'," ^
        + "'--fixed-delta','%%DT'," ^
        + "'--measure-update-times'," ^
        + "'--timing-output','!TIMING_CSV!'," ^
        + "'--eval-output','!EVAL_CSV!'" ^
        + ") -RedirectStandardOutput '!RECEIVER_LOG!' -RedirectStandardError '!RECEIVER_LOG!';" ^
        + "Start-Sleep -Seconds %RECEIVER_MAX_RUNTIME%; if(!$p.HasExited){ $p.Kill() }" ^
        >nul

      REM ------------------------------------------------------------
      REM 2) Start STREAMER in background, hard-stop after timeout
      REM ------------------------------------------------------------
      powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$p = Start-Process -PassThru -NoNewWindow -FilePath 'python' -ArgumentList @(" ^
        + "'%STREAMER%'," ^
        + "'--host','%CARLA_HOST%','--port','%CARLA_PORT%'," ^
        + "'--mode','wait'," ^
        + "'--role-prefix','udp_replay:'," ^
        + "'--include-velocity','--frame-elapsed','--wall-clock'," ^
        + "'--include-monotonic','--include-tick-wall-dt'," ^
        + "'--output','!STREAM_CSV!'" ^
        + ") -RedirectStandardOutput '!STREAMER_LOG!' -RedirectStandardError '!STREAMER_LOG!';" ^
        + "Start-Sleep -Seconds %STREAMER_MAX_RUNTIME%; if(!$p.HasExited){ $p.Kill() }" ^
        >nul

      REM Give receiver/streamer time to start
      timeout /t 1 /nobreak >nul

      REM ------------------------------------------------------------
      REM 3) Run SENDER (blocking)
      REM ------------------------------------------------------------
      echo [INFO] Sending... N=%%N Ts=%%TS
      python "%SENDER%" "%CSV_PATH%" --host "%UDP_HOST%" --port "%UDP_PORT%" --interval %%TS --max-actors %%N > "!SENDER_LOG!" 2>&1

      REM Wait a bit so receiver ticks and flushes logs
      timeout /t 2 /nobreak >nul

      echo [DONE] !TAG!
      timeout /t 1 /nobreak >nul

    )
  )
)

echo [ALL DONE] %OUTDIR%
endlocal
