@echo off
rem Run future-accident extraction in one command.
cd /d "%~dp0"
setlocal

rem Arguments:
rem   %1: root directory containing method/lead/rep/logs (default: results_grid_accident)
rem   %2: output CSV path (default: <root>\future_accidents_ge1000.csv)
rem   %3: intensity threshold (default: 1000)
rem   %4: set to "noacc" to skip --require-is-accident (default: require)

set "ROOT_DIR=%~1"
if "%ROOT_DIR%"=="" set "ROOT_DIR=results_grid_accident"

set "OUT_FILE=%~2"
if "%OUT_FILE%"=="" set "OUT_FILE=%ROOT_DIR%\future_accidents_ge1000.csv"

set "THRESHOLD=%~3"
if "%THRESHOLD%"=="" set "THRESHOLD=1000"

set "ACC_FLAG=--require-is-accident"
if /I "%~4"=="noacc" set "ACC_FLAG="

echo Running extraction...
echo   root      : %ROOT_DIR%
echo   out       : %OUT_FILE%
echo   threshold : %THRESHOLD%
if defined ACC_FLAG (
  echo   flag      : %ACC_FLAG%
) else (
  echo   flag      : (none)
)

python scripts\extract_future_accidents.py "%ROOT_DIR%" --threshold %THRESHOLD% --out "%OUT_FILE%" %ACC_FLAG%

endlocal
