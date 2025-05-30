#!/bin/bash
echo "===== Real-Time Audio Analysis Setup ====="

if ! command -v python3.10 &> /dev/null
then
    echo "Python 3.10 not found. Please install it first:"
    echo "https://www.python.org/downloads/release/python-3100/"
    exit 1
fi

echo "Creating virtual environment..."
python3.10 -m venv hosea-env
source hosea-env/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install scipy SpeechRecognition noisereduce pyaudio torch git+https://github.com/openai/whisper.git

echo "Downloading sample CSV..."
curl -O https://hamsters.io/airdrop/detected_speech_25_log.csv

echo
echo "Setup complete!"
echo "To run the program:"
echo "source hosea-env/bin/activate"
echo "python hosea.py"
