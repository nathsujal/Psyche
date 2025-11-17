import os
import wave
import pyaudio
from pynput import keyboard
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4096

audio = pyaudio.PyAudio()
frames = []
recording = True  # Will stop when space is pressed

def on_press(key):
    global recording
    try:
        if hasattr(key, "char") and key.char == "q":
            print("\n🔵 Stopping recording...")
            recording = False
            return False  # Stop keyboard listener
    except:
        pass

def record_audio():
    global frames, recording

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("🎤 Recording... Press SPACE to stop.\n")

    while recording:
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()

def save_wav(filename="full_recording.wav"):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

# --- Main ---
listener = keyboard.Listener(on_press=on_press)
listener.start()

record_audio()  # Wait until spacebar stops it

wav_file = save_wav()

print("🔁 Sending audio to Groq Whisper...")

with open(wav_file, "rb") as f:
    result = client.audio.transcriptions.create(
        file=(wav_file, f.read()),
        model="whisper-large-v3-turbo",
        temperature=0,
        response_format="verbose_json",
    )

print("\n🟢 Transcription:")
print(result.text)

audio.terminate()
