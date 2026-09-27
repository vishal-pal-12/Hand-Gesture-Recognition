@echo off
title DeafVoice AI - Assistive Communicator for Deaf & Mute
cls
cd /d "%~dp0"
echo ======================================================================
echo   DEAFVOICE AI: TWO-WAY SIGN COMMUNICATOR FOR THE DEAF & MUTE
echo   Translating Sign Language to Spoken Voice & Text in Real-Time
echo ======================================================================
echo.
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\venv!
    echo Please make sure the virtual environment is installed.
    pause
    exit /b 1
)

echo Select an option:
echo   [1] Launch Live Sign-to-Speech Camera (Webcam Communicator)
echo   [2] Translate Single Sign Image (sample_images\posture_a_sample.jpg)
echo   [3] Batch Translate All Sample Sign Images
echo   [4] Run Comprehensive DeafVoice System Verification
echo   [5] Evaluate Model Accuracy on 400 Test Sign Images
echo   [6] Train CNN on Deaf Sign Dataset
echo   [Q] Exit
echo.
set /p choice="Enter choice [1-6, Q]: "

if /i "%choice%"=="1" (
    .\venv\Scripts\python.exe run.py deaf-assist
) else if /i "%choice%"=="2" (
    .\venv\Scripts\python.exe run.py predict --image sample_images\posture_a_sample.jpg --speak
) else if /i "%choice%"=="3" (
    .\venv\Scripts\python.exe run.py predict --dir sample_images
) else if /i "%choice%"=="4" (
    .\venv\Scripts\python.exe run.py verify
) else if /i "%choice%"=="5" (
    .\venv\Scripts\python.exe run.py evaluate
) else if /i "%choice%"=="6" (
    .\venv\Scripts\python.exe run.py train
) else (
    echo Exiting DeafVoice AI.
)
pause
