@echo off
title DeafVoice AI - Multi-Hand Assistive Communicator for Deaf & Mute
cls
cd /d "%~dp0"
echo ======================================================================
echo   DEAFVOICE AI: MULTI-HAND SIGN COMMUNICATOR FOR THE DEAF & MUTE
echo   Finger-Wise (Index=Hello, Fist=No, Palm=Yes, Thumbs=Fine, etc.)
echo   Supports Both Left and Right Hands & Multiple Hands in Real-Time
echo ======================================================================
echo.
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\venv!
    pause
    exit /b 1
)

echo Select an option:
echo   [1] Launch Live Multi-Hand Camera (Webcam Communicator)
echo   [2] Translate Single Sign Image (sample_images\index_hello_sample.jpg)
echo   [3] Batch Translate All Sample Sign Images
echo   [4] Run Comprehensive Multi-Hand System Verification
echo   [5] Evaluate Model Accuracy on 400 Test Samples (100.0%% Benchmark)
echo   [6] Train Finger-Wise Multi-Hand Model
echo   [Q] Exit
echo.
set /p choice="Enter choice [1-6, Q]: "

if /i "%choice%"=="1" (
    .\venv\Scripts\python.exe run.py deaf-assist
) else if /i "%choice%"=="2" (
    .\venv\Scripts\python.exe run.py predict --image sample_images\index_hello_sample.jpg --speak
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
