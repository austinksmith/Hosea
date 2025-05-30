import speech_recognition as sr
import whisper
import numpy as np
import wave
import os
import datetime
import multiprocessing as mp
import csv
import time
import tempfile
import shutil
import traceback
from scipy.fft import fft
import pyaudio
import torch

# Set multiprocessing start method to 'spawn'
# This fixes "Cannot re-initialize CUDA in forked subprocess" error
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# Check if CUDA is available and set device accordingly
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Set microphone index (update as needed)
MIC_INDEX = 4

# Directory for saving logs and recordings
AUDIO_LOG_DIR = "radio_audio_logs"
os.makedirs(AUDIO_LOG_DIR, exist_ok=True)
SPEECH_LOG_FILE = "detected_speech_log.csv"

# Number of parallel workers for processing audio
NUM_WORKERS = 4

# Initialize CSV log if it doesn't exist
if not os.path.exists(SPEECH_LOG_FILE):
    with open(SPEECH_LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Whisper Raw", "Whisper Phase", "Whisper AM", "Whisper FM", "Whisper SSB", 
                         "Frequencies (Hz)"])

# Shared Whisper model instance
def init_global_model():
    global model
    if 'model' not in globals():
        download_dir = "./whisper_models"
        whisper_model_path = "whisper_models"
        os.makedirs(download_dir, exist_ok=True)
        if not os.path.exists(whisper_model_path):
            raise FileNotFoundError("Whisper model not found!")
        # Load model on the same device as the rest of the processing
        model = whisper.load_model("base", download_root=download_dir, device=DEVICE)

recognizer = sr.Recognizer()

def analyze_frequencies(audio_data, sample_rate):
    n = len(audio_data)
    fft_result = fft(audio_data)
    magnitude = np.abs(fft_result[:n//2]) / n
    frequencies = np.linspace(0, sample_rate/2, n//2)
    top_frequencies = frequencies[np.argsort(magnitude)[-10:]]
    freq_list = [f'{freq:.1f}' for freq in top_frequencies]
    print("📡 Top 10 Frequencies:", freq_list)
    return freq_list

def recognize_whisper(audio_data, sample_rate, transformation_type):
    try:
        # Make sure audio data is contiguous
        if isinstance(audio_data, np.ndarray) and not audio_data.flags.c_contiguous:
            print(f"⚠️ Converting non-contiguous array to contiguous for {transformation_type}")
            audio_data = np.ascontiguousarray(audio_data)
            
        # Save the audio data to a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_name = os.path.join(temp_dir, f"temp_{transformation_type}.wav")
        
        with wave.open(temp_name, 'wb') as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())  # Ensure proper bytes conversion

        print(f"🔍 Transcribing audio for transformation: {transformation_type}, from file: {temp_name}")
        
        # Use Whisper to transcribe the audio from the temporary file
        result = model.transcribe(temp_name, language="en")
        
        print(f"✅ Whisper Result for {transformation_type}: {result['text']}")  # Log the result
        
        # Clean up
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"⚠️ Could not remove temporary directory: {str(e)}")
            
        return result['text']
    
    except Exception as e:
        print(f"❌ Whisper Error: {str(e)}")
        traceback.print_exc()  # Print the full traceback
        return f"Whisper Error: {str(e)}"

# Apply binaural phase shifting
def apply_binaural_shift(audio_data, phase_shift=0.1):
    # Make sure audio data is contiguous
    if isinstance(audio_data, np.ndarray) and not audio_data.flags.c_contiguous:
        audio_data = np.ascontiguousarray(audio_data)
        
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32, device=DEVICE)
    shift_index = torch.tensor(int(len(audio_tensor) * phase_shift), device=DEVICE)
    shifted_tensor = torch.roll(audio_tensor, shifts=shift_index.item(), dims=0)
    result = shifted_tensor.cpu().numpy().astype(np.int16)
    return np.ascontiguousarray(result)  # Ensure result is contiguous

# Apply AM demodulation
def am_demodulate(audio_data):
    if isinstance(audio_data, np.ndarray) and not audio_data.flags.c_contiguous:
        audio_data = np.ascontiguousarray(audio_data)
        
    # Run entirely on CPU
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32, device='cpu')
    N = audio_tensor.shape[0]
    f = torch.fft.fft(audio_tensor)

    h = torch.zeros(N, dtype=torch.complex64)
    if N % 2 == 0:
        h[0] = 1
        h[1:N//2] = 2
        h[N//2] = 1
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2

    analytic = torch.fft.ifft(f * h)
    envelope = torch.abs(analytic)

    result = envelope.numpy().astype(np.int16)
    return np.ascontiguousarray(result)

def fm_demodulate(audio_data):
    # Make sure audio data is contiguous
    if isinstance(audio_data, np.ndarray) and not audio_data.flags.c_contiguous:
        audio_data = np.ascontiguousarray(audio_data)
        
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32, device="cpu")

    # Hilbert transform via FFT
    N = audio_tensor.shape[0]
    f = torch.fft.fft(audio_tensor)
    
    h = torch.zeros(N, device="cpu", dtype=torch.complex64)
    if N % 2 == 0:
        h[0] = 1
        h[1:N//2] = 2
        h[N//2] = 1
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2

    analytic = torch.fft.ifft(f * h)

    # Phase -> unwrap -> derivative
    phase = torch.angle(analytic)
    unwrapped = torch_unwrap(phase)
    fm = torch.diff(unwrapped)

    fm = (fm * (180.0 / np.pi))  # Scale to degrees if needed
    result = fm.cpu().numpy().astype(np.int16)
    return np.ascontiguousarray(result)  # Ensure result is contiguous

# Apply SSB demodulation
def ssb_demodulate(audio_data):
    # Make sure audio data is contiguous
    if isinstance(audio_data, np.ndarray) and not audio_data.flags.c_contiguous:
        audio_data = np.ascontiguousarray(audio_data)
        
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32, device="cpu")

    # Construct analytic signal via Hilbert-like transform
    N = audio_tensor.shape[0]
    f = torch.fft.fft(audio_tensor)
    
    h = torch.zeros(N, device="cpu", dtype=torch.complex64)
    if N % 2 == 0:
        h[0] = 1
        h[1:N//2] = 2
        h[N//2] = 1
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2

    analytic = torch.fft.ifft(f * h)

    # Take real or imaginary part to isolate SSB-like signal
    ssb_signal = analytic.real  # crude approximation

    result = ssb_signal.cpu().numpy().astype(np.int16)
    return np.ascontiguousarray(result)  # Ensure result is contiguous

def torch_unwrap(p, discont=3.14159265, dim=-1):
    """
    Version of np.unwrap for PyTorch.
    """
    dd = torch.diff(p, dim=dim)
    ddmod = (dd + np.pi) % (2 * np.pi) - np.pi
    ddmod = torch.where((ddmod == -np.pi) & (dd > 0), np.pi, ddmod)

    ph_correct = ddmod - dd
    ph_correct = torch.where(torch.abs(dd) >= discont, ph_correct, torch.zeros_like(ph_correct))

    correction = torch.cumsum(ph_correct, dim=dim)
    p_unwrapped = p.clone()
    p_unwrapped[..., 1:] += correction

    return p_unwrapped

def process_audio(audio_filename):
    print(f"🔍 Processing file: {audio_filename}")
    with wave.open(audio_filename, 'rb') as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(-1)
        audio_data = np.frombuffer(frames, dtype=np.int16)
        # Extract first channel only if stereo
        if wf.getnchannels() == 2:
            audio_data = audio_data[::2]
        
    # Ensure the data is contiguous
    audio_data = np.ascontiguousarray(audio_data)
    
    # Ensure the file is closed before processing further
    time.sleep(0.1)
    
    print("🔄 Applying transformations...")

    phase_shifted = apply_binaural_shift(audio_data)
    am_demod = am_demodulate(audio_data)
    fm_demod = fm_demodulate(audio_data)
    ssb_demod = ssb_demodulate(audio_data)

    freqs = analyze_frequencies(audio_data, sample_rate)

    print("🗣 Running speech recognition...")
    whisper_raw = recognize_whisper(audio_data, sample_rate, "raw")
    whisper_phase = recognize_whisper(phase_shifted, sample_rate, "phase")
    whisper_am = recognize_whisper(am_demod, sample_rate, "AM")
    whisper_fm = recognize_whisper(fm_demod, sample_rate, "FM")
    whisper_ssb = recognize_whisper(ssb_demod, sample_rate, "SSB")

    print("📊 Detected Speech:")
    print(f"Whisper Raw: {whisper_raw}")
    print(f"Whisper Phase: {whisper_phase}")
    print(f"Whisper AM: {whisper_am}")
    print(f"Whisper FM: {whisper_fm}")
    print(f"Whisper SSB: {whisper_ssb}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(SPEECH_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, whisper_raw, whisper_phase, whisper_am, whisper_fm, whisper_ssb,
                ', '.join(freqs)
            ])
    except Exception as e:
        print(f"❌ Failed to log speech data: {e}")

    print(f"✅ Processed {audio_filename} and logged results.")
    
    # Ensure the file is not locked before deleting
    for _ in range(5):
        try:
            os.remove(audio_filename)
            break
        except PermissionError:
            print(f"⚠️ File {audio_filename} is still in use, retrying...")
            time.sleep(1.0)

def process_audio_worker(queue):
    init_global_model()
    while True:
        audio_filename = queue.get()
        if audio_filename is None:
            break
        process_audio(audio_filename)
        time.sleep(0.1)

def record_audio(queue):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=2, rate=44100,
                        input=True, frames_per_buffer=1024, input_device_index=MIC_INDEX)
    print("🎤 Recording started...")
    try:
        while True:
            frames = []
            for _ in range(0, int(44100 / 1024 * 5)):
              data = stream.read(1024, exception_on_overflow=False)
              frames.append(data)
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(AUDIO_LOG_DIR, f"radio_{timestamp}.wav")
            
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(2)                
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b"".join(frames))
            
            queue.put(filename)
            print(f"🔊 Saved audio file: {filename}")
    except KeyboardInterrupt:
        print("🎤 Recording stopped.")
        stream.stop_stream()
        stream.close()
        audio.terminate()

def main():
    print("🎤 Starting audio recording and processing...")

    # Initialize model in main process before spawning children
    init_global_model()
    audio_queue = mp.Queue()
    recorder_process = mp.Process(target=record_audio, args=(audio_queue,))
    processor_processes = [mp.Process(target=process_audio_worker, args=(audio_queue,)) for _ in range(NUM_WORKERS)]
    
    recorder_process.start()
    for p in processor_processes:
        p.start()
    
    try:
        recorder_process.join()
    except KeyboardInterrupt:
        print("Stopping processes...")
    finally:
        # Make sure to send termination signals
        for _ in range(NUM_WORKERS):
            audio_queue.put(None)
        for p in processor_processes:
            p.join()

if __name__ == "__main__":
    # Main entry point is now at the bottom to ensure mp.set_start_method is called first
    main()