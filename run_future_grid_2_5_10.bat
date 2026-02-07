了解です。「lead=2,5,10 だけ回す」ように、ループを `/L` から “リスト指定” に変えるのが一番安全です。加えて、`summary_grid.csv` を上書きしないように（再実行で既存結果を消さないように）**別ファイルに出す**のを推奨します。

下の `.bat` をそのままコピペして、元の `run_future_grid.bat` とは別名（例：`run_future_grid_rerun_2_5_10.bat`）で保存して実行してください。

```bat
@echo off
cd /d "%~dp0"
setlocal enabledelayedexpansion

if "%~1"=="" (
  set "CSV_PATH=%~dp0send_data\exp_accident.csv"
  set "OUTDIR=%~dp0results_grid_accident"
) else (
  set "CSV_PATH=%~1"
  set "OUTDIR=%~2"
  if "%OUTDIR%"=="" set "OUTDIR=results_grid"
)

set "REPLAY_SCRIPT=scripts\udp_replay\replay_from_udp_future_exp.py"
set "SENDER_SCRIPT=send_data\send_udp_frames_from_csv.py"
set "LSTM_MODEL=scripts\udp_replay\traj_lstm.pt"
set "LSTM_DEVICE=cpu"
set "META_TOOL=scripts\udp_replay\meta_tools.py"

set "FIXED_DELTA=0.1"
set "PRE_SEC=60"
set "POST_SEC=30"
set "PF_PER_SEC=10"
set "POLL_INTERVAL=0.1"
set "BUFFER_PF_AFTER=2"
set "REPS=1"
set "BASE_SEED=20009"
set "STARTUP_DELAY=2"

REM ==== RERUN target leads (only these) ====
set "LEAD_LIST=7"

REM ==== methods to run ====
set "METHODS=autopilot lstm"
REM LSTM を回したくないなら次行に変更:
REM set "METHODS=autopilot"

REM ==== CARLA server restart settings ====
set "CARLA_ROOT=D:\Carla-0.10.0-Win64-Shipping"
set "CARLA_EXE=%CARLA_ROOT%\CarlaUnreal.exe"
set "CARLA_BOOT_WAIT=60"
set "CARLA_BOOT_TIMEOUT=300"
set "CARLA_WARMUP_SEC=90"

set "CARLA_HOST=127.0.0.1"
set "CARLA_PORT=2000"
set "LISTEN_HOST=0.0.0.0"
set "LISTEN_PORT=5005"
set "SENDER_HOST=127.0.0.1"
set "SENDER_PORT=5005"
set "PY=python"
set "BASE_PF=25411"
set "ACCIDENT_PF=%BASE_PF%"

set "MAX_RUNTIME=100"
set /a "WAIT_BASE=%MAX_RUNTIME%+30"
set /a "WAIT_SEC=%WAIT_BASE%"

set /a "START_FRAME=BASE_PF-(PRE_SEC*PF_PER_SEC)"
set /a "END_FRAME=BASE_PF+(POST_SEC*PF_PER_SEC)"
if %START_FRAME% LSS 0 set "START_FRAME=0"

echo BASE_PF=%BASE_PF%
echo SENDER_RANGE=%START_FRAME%..%END_FRAME%  (pre=%PRE_SEC%s post=%POST_SEC%s)
echo RERUN_LEADS=%LEAD_LIST%
echo METHODS=%METHODS%

for /f %%w in ('
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "$delta=%FIXED_DELTA%;" ^
    "$start=%START_FRAME%;" ^
    "$end=%END_FRAME%;" ^
    "$sendDuration=($end - $start + 1) * $delta;" ^
    "$wait=[int][math]::Ceiling($sendDuration + %STARTUP_DELAY% + 5);" ^
    "$waitBound=[int][math]::Max($wait,%WAIT_BASE%);" ^
    "Write-Output $waitBound"
') do set "WAIT_SEC=%%w"

REM ==== write rerun summary separately (recommended) ====
set "SUMMARY=%OUTDIR%\summary_rerun_leads_2_5_10.csv"
echo method,lead_sec,rep,seed,switch_payload_frame,ran_ok,accident_after_switch,first_accident_payload_frame,status,accident_payload_frame_ref > "%SUMMARY%"

for %%L in (%LEAD_LIST%) do (

  echo =========================================================
  echo Restarting CARLA server for lead=%%L
  echo =========================================================
  call :restart_carla
  call :wait_carla_ready %CARLA_HOST% %CARLA_PORT% %CARLA_BOOT_TIMEOUT%
  if errorlevel 1 (
    echo CARLA did not become ready. Abort lead=%%L
    exit /b 1
  )

  echo Warmup wait %CARLA_WARMUP_SEC%s ...
  timeout /t %CARLA_WARMUP_SEC% /nobreak >nul

  for %%M in (%METHODS%) do (

    for /f %%s in ('python "%META_TOOL%" switch_pf "%ACCIDENT_PF%" "%%L" "%FIXED_DELTA%"') do set "SWITCH_PF=%%s"
    set /a "SWITCH_PF_EVAL=!SWITCH_PF!+BUFFER_PF_AFTER"

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

      set "RECV_OUT=!RUN_LOGS!\receiver.out.log"
      set "RECV_ERR=!RUN_LOGS!\receiver.err.log"
      set "PID_VALID=1"

      set "PID_FILE=!RUN_LOGS!\replay.pid"
      del /q "!PID_FILE!" >nul 2>&1

      powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$ErrorActionPreference='Stop';" ^
        "$argsList=@('%REPLAY_SCRIPT%','--carla-host','%CARLA_HOST%','--carla-port','%CARLA_PORT%','--listen-host','%LISTEN_HOST%','--listen-port','%LISTEN_PORT%','--poll-interval','%POLL_INTERVAL%','--fixed-delta','%FIXED_DELTA%','--max-runtime','%MAX_RUNTIME%','--tm-seed','!SEED!','--future-mode','%%M','--switch-payload-frame','!SWITCH_PF!','--metadata-output','!RUN_META!','--collision-log','!RUN_COLL!','--actor-log','!RUN_ACTOR!','--id-map-file','!RUN_IDMAP!','--enable-completion');" ^
        "if ('%%M' -eq 'lstm') { $argsList += @('--lstm-model','%LSTM_MODEL%','--lstm-device','%LSTM_DEVICE%','--lstm-sample-interval','%FIXED_DELTA%') };" ^
        "$p=Start-Process -FilePath '%PY%' -ArgumentList $argsList -RedirectStandardOutput '!RECV_OUT!' -RedirectStandardError '!RECV_ERR!' -NoNewWindow -PassThru;" ^
        "Set-Content -Path '!PID_FILE!' -Value $p.Id -NoNewline;"

      set "REPLAY_PID="
      if exist "!PID_FILE!" set /p REPLAY_PID=<"!PID_FILE!"

      echo(!REPLAY_PID!| findstr /r "^[0-9][0-9]*$" >nul
      if errorlevel 1 (
        echo Failed: REPLAY_PID invalid: "!REPLAY_PID!"
        if exist "!RECV_OUT!" type "!RECV_OUT!"
        if exist "!RECV_ERR!" type "!RECV_ERR!"
        set "RAN_OK=0"
        set "STATUS=pid_invalid"
        set "PID_VALID=0"
      )

      if "!PID_VALID!"=="1" (
        timeout /t %STARTUP_DELAY% /nobreak >nul
        python "%SENDER_SCRIPT%" "%CSV_PATH%" --host "%SENDER_HOST%" --port "%SENDER_PORT%" --interval "%FIXED_DELTA%" --start-frame "!START_FRAME!" --end-frame "!END_FRAME!" --log-level INFO
        call :wait_for_pid !REPLAY_PID! %WAIT_SEC%
        if errorlevel 1 (
          set "RAN_OK=0"
          set "STATUS=timeout"
          taskkill /PID !REPLAY_PID! /T /F >nul 2>&1
        )
      )

      set "FIRST_ACC_PF="
      set "AFTER_SWITCH=0"
      if exist "!RUN_META!" (
        for /f %%f in ('python "%META_TOOL%" first_accident_pf_after_switch "!RUN_META!" "!SWITCH_PF_EVAL!"') do set "FIRST_ACC_PF=%%f"
        for /f %%a in ('python "%META_TOOL%" accident_after_switch "!RUN_META!" "!SWITCH_PF_EVAL!"') do set "AFTER_SWITCH=%%a"
      )

      if "!AFTER_SWITCH!"=="" set "AFTER_SWITCH=0"

      echo %%M,%%L,%%R,!SEED!,!SWITCH_PF!,!RAN_OK!,!AFTER_SWITCH!,!FIRST_ACC_PF!,!STATUS!,%ACCIDENT_PF%>> "%SUMMARY%"
    )
  )

  call :stop_carla
)

echo Wrote: %SUMMARY%
exit /b 0

:restart_carla
call :stop_carla
echo Starting CARLA: "%CARLA_EXE%"
start "" /high "%CARLA_EXE%"
timeout /t %CARLA_BOOT_WAIT% /nobreak >nul
exit /b 0

:stop_carla
taskkill /IM CarlaUnreal.exe /T /F >nul 2>&1
timeout /t 5 /nobreak >nul
exit /b 0

:wait_carla_ready
setlocal
set "H=%~1"
set "P=%~2"
set "T=%~3"
if "%T%"=="" set "T=120"

echo Waiting for CARLA to be ready... (timeout=%T%s host=%H% port=%P%)
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$h='%H%'; $p=[int]%P%; $deadline=(Get-Date).AddSeconds([int]%T%);" ^
  "while((Get-Date) -lt $deadline) {" ^
  "  $py = ('import carla; c=carla.Client(r''{0}'',{1}); c.set_timeout(5.0); w=c.get_world(); w.get_snapshot()' -f $h,$p);" ^
  "  & python -c $py > $null 2>&1;" ^
  "  if ($LASTEXITCODE -eq 0) { Write-Host 'CARLA READY'; exit 0 }" ^
  "  Start-Sleep -Seconds 2" ^
  "}" ^
  "Write-Host 'CARLA NOT READY (timeout)'; exit 1"
if errorlevel 1 ( endlocal & exit /b 1 )
endlocal & exit /b 0

:wait_for_pid
setlocal
set "PID=%~1"
set /a "REMAIN=%~2"
:wait_loop
tasklist /fi "PID eq %PID%" | findstr /i "%PID%" >nul
if errorlevel 1 ( endlocal & exit /b 0 )
if %REMAIN% LEQ 0 ( endlocal & exit /b 1 )
set /a REMAIN-=1
timeout /t 1 /nobreak >nul
goto wait_loop
```

### これでどうなるか

* lead は **2,5,10 だけ**回ります。
* 結果は `results_grid_accident\summary_rerun_leads_2_5_10.csv` に出ます（既存の `summary_grid.csv` を壊しません）。

### 既存 summary と結合したい場合

一番ミスが少ないのは「元の summary から lead=2,5,10 の行を削除して、rerun の行を足す」です。必要なら、その結合用の `merge_summary.py`（1ファイル）もこちらで即作ります。
