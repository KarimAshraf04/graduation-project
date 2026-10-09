@echo off
REM Usage:
REM   run.bat            -> webcam 0
REM   run.bat 1          -> webcam 1
REM   run.bat video.mp4  -> video file
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo venv not found. Run setup first:
    echo    powershell -ExecutionPolicy Bypass -File .\setup.ps1
    pause
    exit /b 1
)
set SRC=%1
if "%SRC%"=="" set SRC=0
"venv\Scripts\python.exe" vision_pipeline.py --source %SRC%
pause
