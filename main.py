import os
import pyaudio
import numpy as np
import json
import tempfile
import logging
import zipfile
import urllib.request
import subprocess
from vosk import Model, KaldiRecognizer
from langchain_ollama import ChatOllama

from src.ttsprocessor import TTS
from src.sttprocessor import STT
from src.ragprocessor import RAGProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

csv_file_path = "data/data.csv"
chroma_directory = "chroma_db"
model_path = "models/vosk-model"

VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"

def download_vosk_model():
    if os.path.exists(model_path) and os.listdir(model_path):
        return
    print(f"Vosk model not found at '{model_path}'. Downloading...")
    os.makedirs("models", exist_ok=True)
    zip_path = os.path.join("models", f"{VOSK_MODEL_NAME}.zip")
    urllib.request.urlretrieve(VOSK_MODEL_URL, zip_path)
    print("Extracting model...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("models")
    os.rename(os.path.join("models", VOSK_MODEL_NAME), model_path)
    os.remove(zip_path)
    print("Vosk model downloaded and ready.")

OLLAMA_MODEL = "qwen2.5:0.5b-instruct-q4_0"

def ensure_ollama_model():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if OLLAMA_MODEL not in result.stdout:
            print(f"Ollama model '{OLLAMA_MODEL}' not found. Pulling...")
            subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
            print(f"Ollama model '{OLLAMA_MODEL}' pulled successfully.")
    except FileNotFoundError:
        print("ERROR: Ollama is not installed. Please install it from https://ollama.com")
        raise

print("Loading models into memory...")

print("Loading RAG processor...")
rag_processor = RAGProcessor(csv_file_path, chroma_directory)

print("Loading LLM...")
ensure_ollama_model()
llm = ChatOllama(model=OLLAMA_MODEL)

print("Loading Vosk model...")
download_vosk_model()
vosk_model = Model(model_path)

print("Loading optimized Whisper model...")
whisper_transcriber = STT(model_size="base",device=None,compute_type=None,beam_size=1,sample_rate=16000)

print("Loading Kokoro TTS model into memory...")
tts_client = TTS(lang_code='a',voice="af_sky",sample_rate=24000,speed=1.0)

print("All models loaded successfully!")

print("Initializing audio system...")
pa = pyaudio.PyAudio()
RATE = 16000 
CHUNK_SIZE = 2048

def init_audio_stream():
    # List available input devices
    default_idx = None
    print("Available input devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  [{i}] {info['name']} (inputs: {info['maxInputChannels']})")
            if default_idx is None:
                default_idx = i
    if default_idx is None:
        raise RuntimeError("No audio input device found. Please connect a microphone.")
    
    try:
        stream = pa.open(rate=RATE, channels=1, format=pyaudio.paInt16,
                         input=True, frames_per_buffer=CHUNK_SIZE,
                         input_device_index=default_idx)
        return stream
    except OSError as e:
        print(f"Failed to open device [{default_idx}]: {e}")
        print("Retrying with system default device...")
        stream = pa.open(rate=RATE, channels=1, format=pyaudio.paInt16,
                         input=True, frames_per_buffer=CHUNK_SIZE)
        return stream

audio_stream = init_audio_stream()

print("Audio system initialized.")
print("Vector database ready.")
print("All systems initialized. Voice assistant is ready!")

def speak(text):
    try:
        audio_data = tts_client.text_to_speech(text)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_data)
        
        os.system(f"afplay {temp_path}")
        os.unlink(temp_path)
        
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        print(f"TTS Error: {e}")

def query_inventory(question):
    context = rag_processor.query_inventory(question)
    
    prompt = f"""Inventory data:
    {context}

    Question: {question}

    Provide a concise one-line answer about the inventory item's location, quantity, or specifications. If not related to inventory, respond: "I can't help with that." """
        
    response = llm.invoke(prompt)
    answer = response.content.strip()
    return answer

def listen_for_wake_word():
    print("Listening for wake word ('sandy')...")
    rec = KaldiRecognizer(vosk_model, RATE)
    rec.SetWords(True)
    
    while True:
        data = audio_stream.read(4096, exception_on_overflow=False)
        if len(data) == 0:
            continue
            
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if result.get("text", ""):
                text = result["text"].lower()
                print(f"Recognized: {text}")
                if "sandy" in text:
                    print("Wake word detected!")
                    return

def record_command(duration=5, rate=16000):
    print("Listening for your question...")
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=1024)
    frames = []
    for _ in range(0, int(rate / 1024 * duration)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
    stream.stop_stream()
    stream.close()

    audio_data = b''.join(frames)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    try:
        text, metadata = whisper_transcriber.transcribe(audio_np)
        logger.info(f"Transcription metadata: {metadata}")
        print(f"User Query: {text}")
        return text
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        print(f"Transcription error: {e}")
        return ""

try:
    while True:
        listen_for_wake_word()
        user_question = record_command()
        if not user_question:
            speak("Sorry, I didn't catch that. Please try again.")
            continue
        answer = query_inventory(user_question)
        print(f"Answer: {answer}")
        speak(answer)

except KeyboardInterrupt:
    print("Exiting program...")

finally:
    try:
        audio_stream.stop_stream()
        audio_stream.close()
    except OSError:
        pass
    pa.terminate()