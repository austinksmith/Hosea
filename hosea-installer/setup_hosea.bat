@echo off
echo ===== Real-Time Audio Analysis Setup =====
echo.

where python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please install Python 3.10 manually from https://www.python.org/downloads/release/python-3100/
    pause
    exit /b
)

echo Creating virtual environment...
python -m venv hosea-env
call hosea-env\Scripts\activate

echo Upgrading pip...
pip install --upgrade pip

echo Installing required Python modules...
pip install scipy SpeechRecognition noisereduce pyaudio torch git+https://github.com/openai/whisper.git

echo Downloading sample CSV...
curl -O https://hamsters.io/airdrop/detected_speech_25_log.csv

echo.
echo Installation complete!
echo To run the program, type:
echo     call hosea-env\Scripts\activate
echo     python hosea.py
pause
