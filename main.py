import os
import pyaudio
import numpy as np
import json
import tempfile
import logging
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

print("Loading models into memory...")

print("1/5 Loading RAG processor...")
rag_processor = RAGProcessor(csv_file_path, chroma_directory)

print("2/5 Loading LLM model...")
llm = ChatOllama(model="qwen2.5:0.5b-instruct-q4_0")

print("3/5 Loading Vosk model...")
vosk_model = Model(model_path)

print("4/5 Loading optimized Whisper model...")
whisper_transcriber = STT(
    model_size="base",
    device=None,  
    compute_type=None,
    beam_size=1,
    sample_rate=16000
)

print("5/5 Loading Kokoro TTS model into memory...")
tts_client = TTS(
    lang_code='a',
    voice="af_sky",
    sample_rate=24000,
    speed=1.0
)

print("All models loaded successfully!")

print("Initializing audio system...")
pa = pyaudio.PyAudio()
RATE = 16000 
CHUNK_SIZE = 2048
audio_stream = pa.open(
    rate=RATE,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=CHUNK_SIZE
)

print("Audio system initialized.")
print("Vector database ready.")
print("All systems initialized. Voice assistant is ready!")

def speak(text):
    """Use optimized TTS client for text-to-speech."""
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
    # Get inventory data from RAG processor
    context = rag_processor.query_inventory(question)
    
    # Create prompt for LLM
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
    audio_stream.stop_stream()
    audio_stream.close()
    pa.terminate()