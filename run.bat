@echo off
echo === F5-TTS Voice Cloning Service ===

set PYTHON=python
where /Q py && set PYTHON=py -3.12
if exist "C:\Users\meebw\AppData\Local\Programs\Python\Python312\python.exe" set PYTHON=C:\Users\meebw\AppData\Local\Programs\Python\Python312\python.exe

echo Using: %PYTHON%

if not exist venv (
    echo Creating virtual environment with Python 3.12...
    %PYTHON% -m venv venv
)

echo Activating venv and installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

REM Add FFmpeg shared DLLs to PATH if available
for /D %%G in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared*") do (
    for /D %%H in ("%%G\ffmpeg-*-full_build-shared") do (
        set "PATH=%%H\bin;%PATH%"
        echo Added FFmpeg shared DLLs from %%H\bin
    )
)

echo.
echo Starting server on http://localhost:8000
python server.py
