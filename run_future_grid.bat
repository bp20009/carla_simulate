@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
  echo Usage: %~nx0 path\to\reduced.csv [outdir]
  exit /b 1
)

set "CSV_PATH=%~1"
set "OUTDIR=%~2"
if "%OUTDIR%"=="" set "OUTDIR=results_grid"

set "REPLAY_SCRIPT=scripts\udp_replay\replay_from_udp_future_exp.py"
set "SENDER_SCRIPT=send_data\send_udp_frames_from_csv.py"

set "FIXED_DELTA=0.1"
set "POLL_INTERVAL=0.1"
set "TRACKING_SEC=30"
set "FUTURE_SEC=10"
set "LEAD_MIN=1"
set "LEAD_MAX=10"
set "REPS=5"
set "BASE_SEED=20009"
set "STARTUP_DELAY=1"

set "CARLA_HOST=127.0.0.1"
set "CARLA_PORT=2000"
set "LISTEN_HOST=0.0.0.0"
set "LISTEN_PORT=5005"
set "SENDER_HOST=127.0.0.1"
set "SENDER_PORT=5005"

set /a "MAX_RUNTIME=%TRACKING_SEC%+%FUTURE_SEC%"
set /a "WAIT_SEC=%MAX_RUNTIME%+30"

set "CALIB_LOGS=%OUTDIR%\calibration\logs"
set "CALIB_META=%CALIB_LOGS%\meta.json"
set "CALIB_COLL=%CALIB_LOGS%\collisions.csv"

mkdir "%CALIB_LOGS%" >nul 2>&1

echo [calibration] starting...
for /f %%p in ('powershell -NoProfile -Command ^
  "$p = Start-Process -FilePath python -ArgumentList @('%REPLAY_SCRIPT%','--carla-host','%CARLA_HOST%','--carla-port','%CARLA_PORT%','--listen-host','%LISTEN_HOST%','--listen-port','%LISTEN_PORT%','--poll-interval','%POLL_INTERVAL%','--fixed-delta','%FIXED_DELTA%','--max-runtime','%MAX_RUNTIME%','--tm-seed','%BASE_SEED%','--future-mode','none','--metadata-output','%CALIB_META%','--collision-log','%CALIB_COLL%') -PassThru; $p.Id"') do set "REPLAY_PID=%%p"

timeout /t %STARTUP_DELAY% /nobreak >nul
python "%SENDER_SCRIPT%" "%CSV_PATH%" --host %SENDER_HOST% --port %SENDER_PORT% --interval %FIXED_DELTA%
powershell -NoProfile -Command "Wait-Process -Id %REPLAY_PID% -Timeout %WAIT_SEC%" >nul

for /f %%a in ('powershell -NoProfile -Command ^
  "$m = Get-Content '%CALIB_META%' | ConvertFrom-Json; if ($m.accidents -and $m.accidents.Count -gt 0) { $m.accidents[0].payload_frame }"') do set "ACCIDENT_PF=%%a"

if "%ACCIDENT_PF%"=="" (
  echo Calibration failed: no accidents in %CALIB_META%
  exit /b 1
)

set "SUMMARY=%OUTDIR%\summary_grid.csv"
echo method,lead_sec,rep,seed,switch_payload_frame,ran_ok,accident_after_switch,first_accident_payload_frame,status,accident_payload_frame_ref > "%SUMMARY%"

for %%M in (autopilot lstm) do (
  for /L %%L in (%LEAD_MIN%,1,%LEAD_MAX%) do (
    for /f %%s in ('powershell -NoProfile -Command ^
      "$lead = [int]%%L; $delta = [double]%FIXED_DELTA%; $frames = [int][math]::Round($lead / $delta); $sw = [int]([math]::Max(%ACCIDENT_PF% - $frames, 0)); $sw"') do set "SWITCH_PF=%%s"

    for /L %%R in (1,1,%REPS%) do (
      set /a "SEED=%BASE_SEED%+%%R"
      set "RUN_DIR=%OUTDIR%\%%M\lead_%%L\rep_%%R"
      set "RUN_LOGS=!RUN_DIR!\logs"
      set "RUN_META=!RUN_LOGS!\meta.json"
      set "RUN_COLL=!RUN_LOGS!\collisions.csv"
      set "RUN_ACTOR=!RUN_LOGS!\actor.csv"
      set "RUN_IDMAP=!RUN_LOGS!\id_map.csv"

      mkdir "!RUN_LOGS!" >nul 2>&1

      set "RAN_OK=1"
      set "STATUS=ok"

      for /f %%p in ('powershell -NoProfile -Command ^
        "$p = Start-Process -FilePath python -ArgumentList @('%REPLAY_SCRIPT%','--carla-host','%CARLA_HOST%','--carla-port','%CARLA_PORT%','--listen-host','%LISTEN_HOST%','--listen-port','%LISTEN_PORT%','--poll-interval','%POLL_INTERVAL%','--fixed-delta','%FIXED_DELTA%','--max-runtime','%MAX_RUNTIME%','--tm-seed','!SEED!','--future-mode','%%M','--switch-payload-frame','!SWITCH_PF!','--metadata-output','!RUN_META!','--collision-log','!RUN_COLL!','--actor-log','!RUN_ACTOR!','--id-map-file','!RUN_IDMAP!') -PassThru; $p.Id"') do set "REPLAY_PID=%%p"

      timeout /t %STARTUP_DELAY% /nobreak >nul
      python "%SENDER_SCRIPT%" "%CSV_PATH%" --host %SENDER_HOST% --port %SENDER_PORT% --interval %FIXED_DELTA%
      powershell -NoProfile -Command "Wait-Process -Id !REPLAY_PID! -Timeout %WAIT_SEC%" >nul
      if errorlevel 1 (
        set "RAN_OK=0"
        set "STATUS=timeout"
      )

      for /f %%f in ('powershell -NoProfile -Command ^
        "$m = Get-Content '!RUN_META!' | ConvertFrom-Json; if ($m.accidents -and $m.accidents.Count -gt 0) { $m.accidents[0].payload_frame }"') do set "FIRST_ACC_PF=%%f"

      for /f %%a in ('powershell -NoProfile -Command ^
        "$m = Get-Content '!RUN_META!' | ConvertFrom-Json; $sw = [int]!SWITCH_PF!; $hit = 0; if ($m.accidents) { foreach ($e in $m.accidents) { if ([int]$e.payload_frame -ge $sw) { $hit = 1; break } } }; $hit"') do set "AFTER_SWITCH=%%a"

      if "!FIRST_ACC_PF!"=="" set "FIRST_ACC_PF="
      if "!AFTER_SWITCH!"=="" set "AFTER_SWITCH=0"

      echo %%M,%%L,%%R,!SEED!,!SWITCH_PF!,!RAN_OK!,!AFTER_SWITCH!,!FIRST_ACC_PF!,!STATUS!,%ACCIDENT_PF%>> "%SUMMARY%"
    )
  )
)

echo Wrote: %SUMMARY%
exit /b 0
