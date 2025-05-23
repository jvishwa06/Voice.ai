import os
import warnings
import shutil
import pyaudio
import pyttsx3
import whisper
import numpy as np
import pandas as pd
import soundfile as sf
import json
from vosk import Model, KaldiRecognizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from kokoro import KPipeline

warnings.filterwarnings("ignore")

csv_file_path = "data/inv_data.csv"
chroma_directory = "chroma_db"
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def rebuild_chroma_index():
    print("Rebuilding Chroma index from CSV data...")
    if os.path.exists(chroma_directory):
        shutil.rmtree(chroma_directory)
    df = pd.read_csv(csv_file_path)
    df = df.fillna('')
    documents = []
    for _, row in df.iterrows():
        metadata = {k: str(v) for k, v in row.items() if v and str(v).strip() and k != "Category"}
        content_parts = []
        for col, val in row.items():
            if col != "Category" and pd.notna(val) and str(val).strip():
                content_parts.append(f"{col}: {val}")
        content = "\n".join(content_parts)
        documents.append(Document(page_content=content, metadata=metadata))
    vector_store = Chroma.from_documents(documents,embedding_model,persist_directory=chroma_directory)
    vector_store.persist()
    print(f"Created and saved new Chroma index with {len(documents)} documents.")
    return vector_store

if os.path.exists(chroma_directory) and os.listdir(chroma_directory):
    try:
        vector_store = Chroma(persist_directory=chroma_directory, embedding_function=embedding_model)
        print("Loaded existing Chroma vector database.")
    except Exception as e:
        print(f"Error loading existing database: {e}")
        vector_store = rebuild_chroma_index()
else:
    vector_store = rebuild_chroma_index()

llm = ChatOllama(model="llama3.2:latest")
pipeline = KPipeline(lang_code='a')

tts_engine = pyttsx3.init()

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def query_inventory(question):
    docs = vector_store.max_marginal_relevance_search(question,k=3,fetch_k=8)
    
    components = []
    for i, doc in enumerate(docs):
        component_data = {"ComponentID": f"Item {i+1}","S.No": doc.metadata.get("S.No", "N/A")}
        
        for line in doc.page_content.strip().split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key, value = parts
                    component_data[key.strip()] = value.strip()
        
        components.append(component_data)
    
    formatted_components = []
    for i, comp in enumerate(components):
        lines = []
        for field in ["Name of the component", "Location", "Quantity", "Category"]:
            if field in comp and comp[field]:
                lines.append(f"- {field}: {comp[field]}")
        
        for k, v in comp.items():
            if k not in ["Name of the component", "Location", "Quantity", "Category", "ComponentID", "S.No"] and v:
                lines.append(f"- {k}: {v}")
                
        formatted_comp = f"Component {comp['S.No']}:\n" + "\n".join(lines)
        formatted_components.append(formatted_comp)

    context = "\n\n".join(formatted_components)

    prompt = f"""Based on the following inventory data:

            {context}

            Answer the question: {question}

            You are an intelligent assistant for the inventory management system of a lab or organization. Your job is to provide accurate and concise one-line answers to queries about inventory items, including their current location, available quantity, specifications, and technical functions related to electronic components.
            When stating quantities, always provide the units as integers (e.g., 2 units, NOT 2.0 units).
            
            Answer clearly, referring to the inventory data, and keep the answer strictly to one line only — no extra explanations or sentences beyond that single line.
            
            If the question is NOT related to context provided, respond ONLY with:

            "I can't help with that."

            Do NOT add any explanations, reasoning, or extra text."""
        
    response = llm.invoke(prompt)
    answer = response.content.strip()
    
    generator = pipeline(answer, voice='af_heart')
    for i, (gs, ps, audio) in enumerate(generator):
        output_file = os.path.join(output_folder, f"answer_{i}.wav")
        sf.write(output_file, audio, 24000)
        print(f"Audio saved to {output_file}")
    return answer

model_path = "models/vosk-model"

model = Model(model_path)
pa = pyaudio.PyAudio()
RATE = 16000 
audio_stream = pa.open(rate=RATE,channels=1,format=pyaudio.paInt16,input=True,frames_per_buffer=4096)

whisper_model = whisper.load_model("tiny")

def listen_for_wake_word():
    print("Listening for wake word ('sandy')...")
    rec = KaldiRecognizer(model, RATE)
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

    audio_np = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0
    result = whisper_model.transcribe(audio_np, language='en')
    text = result.get("text", "").strip()
    print(f"Recognized command: {text}")
    return text

print("Inventory voice assistant started. Say 'sandy' to wake.")

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