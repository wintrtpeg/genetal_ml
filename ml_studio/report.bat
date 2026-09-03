@echo off
setlocal
cd /d "%~dp0"
title Diagnostic Report

set "LOG=%~dp0report_console.log"
> "%LOG%" echo === report.bat console log ===

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
  where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
  where py >nul 2>&1 && set "PY=py -3"
)

if not defined PY (
  echo.
  echo [X] Python not found.
  echo     Install Python 3.12 from python.org,
  echo     and check "Add python.exe to PATH" on the first screen.
  echo.
  >>"%LOG%" echo [X] Python not found.
  goto :done
)

echo Using: %PY%
>>"%LOG%" echo Using: %PY%
echo.
echo Collecting diagnostic report. This takes 3-15 minutes.
echo.

%PY% "scripts\collect_report.py" %* 2>&1
set "RC=%ERRORLEVEL%"
>>"%LOG%" echo [collect_report.py exit code %RC%]

if not "%RC%"=="0" (
  echo.
  echo [!] Collector exited with code %RC%.
)

:done
echo.
echo ----------------------------------------------------------------
echo  Send these files from this folder (whichever exist):
echo.
echo     diagnostic_report.txt   ^(the diagnostic report^)
echo     report_console.log      ^(this console log^)
echo.
echo ----------------------------------------------------------------
echo  This window STAYS OPEN. Close it with the X button when done.
echo ----------------------------------------------------------------
echo.
cmd /k
