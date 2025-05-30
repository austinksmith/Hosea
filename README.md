Hosea Real-Time Audio Harassment Detection Tool
================================================


Description:

I've also put together a python program that takes the raw audio feed from a microphone and performs several transformations including binaural audio phase shifting, FM, AM, SBB demodulation, and runs all audio versions through a FFT frequency analysis as well as a speech to text transcription that converts audio into text and logs them to a CSV file, a downloadable version of this program is available below. As well as a downloadable CSV file detailing a small sample of the audio recorded and trasncribed.

After downloading the Python Program you'll need several python modules installed in order to make use of the program. As well as python 3.10, versions higher than 3.10 aren't guaranteed to work properly with pyaudio. A quick installer script is available in this repository which will automatically install the required depenencies for you to run the program and start recording!

Required Python3 Modules:

1. python-venv
2. Scipy
3. SpeechRecognition
4. MP (MultiProcessing)
5. noisereduce
6. pyaudio
7. torch (use torch gpu for cuda support)
8. whisper-ai

In order to make use of GPU processing you'll need a Nvidia CUDA capable GPU as well as nvidia cuda toolkit installed and a gpu enabled version of pytorch installed, if you dont install these the program will use CPU only mode which is slower but will still work.

This program is configured to use 4 CPU worker threads along with a CUDA enabled GPU to provide real time audio transcription and transformations to extract speech from LRAD or Microwave Auditory Effect devices, 4 worker threads should work on an nvidia GPU with 4GB of dedicated VRAM, if you have lower vram than 4GB simply reduce the number of workers (NUM_WORKERS) in the program on line 35, using 1 worker will mean somewhat slower processing but lower VRAM requirements. Alternatively if you have more VRAM available, increase the NUM_WORKERS variable value to say 6 or 8 workers for even faster processing.

Installation Instructions:

1. Install Python 3.10 from https://www.python.org/downloads/release/python-3100/

2. Unzip this package.

3. Run the appropriate installer script:

   - Windows: Double-click 
   ```sh
   setup_hosea.bat
   ```
   - macOS/Linux: Open terminal and run:
     ```sh
     chmod +x setup_hosea.sh
     ```
     ```sh
     ./setup_hosea.sh
     ```

4. Once installed, activate the environment:

   - Windows: 
        ```sh 
            call hosea-env\Scripts\activate
        ```
   - macOS/Linux: 

        ```sh
            source hosea-env/bin/activate
        ```

5. Run the program:
   
   ```sh
    python hosea.py
   ```

   or if you have multiple python versions installed specify python3.10

   ```sh
    python3.10 hosea.py
   ```

Resources:
- Program: hosea.py
- Sample Log: detected_speech_25_log.csv
- Installer ZIP: hosea-installer.zip


**Hosea 4:6**

***My people are destroyed for lack of knowledge: because thou hast rejected knowledge, I will also reject thee, that thou shalt be no priest to me: seeing thou hast forgotten the law of thy God, I will also forget thy children.***