@echo off
setlocal
cd /d "%~dp0"
title Timeseries ML Studio

REM  Keep this file DUMB. No if/else blocks, no parentheses around commands.
REM  cmd.exe parses a whole ( ... ) block before running it, so an empty
REM  variable inside one turns into a syntax error -- ") was unexpected at
REM  this time." -- and the window closes before anyone can read it.
REM  All argument handling lives in scripts/setup.py instead.
REM
REM    run.bat          normal run  (fast after the first time)
REM    run.bat --full   redo the install and the checks
REM    run.bat 3.12     use that exact Python version (first run only)

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"

if defined PY goto :go

echo.
echo [X] Python not found.
echo     Install Python 3.10-3.12 from python.org
echo     and CHECK "Add python.exe to PATH" on the first install screen.
echo.
pause
exit /b 1

:go
%PY% "scripts\setup.py" %*
echo.
pause
