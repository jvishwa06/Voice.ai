import os
import pandas as pd
import soundfile as sf
import warnings
from kokoro import KPipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

warnings.filterwarnings("ignore")

csv_file_path = "data/inv_data.csv"
chroma_directory = "chroma_db"

def rebuild_chroma_index():
    print("Rebuilding Chroma index from CSV data...")
    if os.path.exists(chroma_directory):
        import shutil
        shutil.rmtree(chroma_directory)
    
    df = pd.read_csv(csv_file_path)
    df = df.fillna('')
    
    documents = []
    for i, row in df.iterrows():
        metadata = {k: str(v) for k, v in row.items() if v and str(v).strip()}
        
        content_parts = []
        for col, val in row.items():
            if pd.notna(val) and str(val).strip(): 
                content_parts.append(f"{col}: {val}")
        
        content = "\n".join(content_parts)
        documents.append(Document(page_content=content, metadata=metadata))
    
    vector_store = Chroma.from_documents(
        documents, 
        embedding_model, 
        persist_directory=chroma_directory
    )
    vector_store.persist()
    print(f"Created and saved new Chroma index with {len(documents)} documents.")
    return vector_store

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(chroma_directory) and os.listdir(chroma_directory):
    try:
        vector_store = Chroma(persist_directory=chroma_directory, embedding_function=embedding_model)
        print("Loaded existing Chroma vector database.")
    except Exception as e:
        print(f"Error loading existing database: {e}")
        vector_store = rebuild_chroma_index()
else:
    vector_store = rebuild_chroma_index()


llm = ChatOllama(model="qwen2.5:0.5b")

pipeline = KPipeline(lang_code='a')

output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

def query_inventory(question):
    """Query the inventory data using Chroma and return only direct answers."""
    docs = vector_store.max_marginal_relevance_search(
        question, 
        k=3, 
        fetch_k=8  
    )
    
    components = []
    for i, doc in enumerate(docs):
        component_data = {
            "ComponentID": f"Item {i+1}",
            "S.No": doc.metadata.get("S.No", "N/A")
        }
        
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

Important: 
1. Your response must be extremely concise and brief - maximum 1-2 sentences.
2. Only mention the exact location (e.g., SC3, SA1) and quantity of the specific item asked about.
3. Do not mention any other components unless directly asked for them.
4. Format your answer like: "The [item name] is located in [location] with quantity [number]."
5. Do not include any explanations, additional information, or items that weren't specifically requested.
6. Do not mention component numbers or line numbers in your answer."""
    
    response = llm.invoke(prompt)
    answer = response.content.strip()
    print(answer)

    generator = pipeline(answer, voice='af_heart')  
    for i, (gs, ps, audio) in enumerate(generator):
        output_file = os.path.join(output_folder, f"answer_{i}.wav")
        sf.write(output_file, audio, 24000)
        print(f"Audio saved to {output_file}")
        os.system(f"open {output_file}")

def rebuild_index_command():
    """Manually rebuild the vector index."""
    print("Rebuilding vector index...")
    return rebuild_chroma_index()

if __name__ == "__main__":
    print("Voice Assistant Inventory System")
    print("Type 'exit' or 'quit' to end the program")
    print("Type 'rebuild' to rebuild the vector index")
    
    while True:
        user_question = input("\nEnter your question about the inventory: ")
        
        if user_question.lower() in ['exit', 'quit']:
            print("Exiting the voice assistant. Goodbye!")
            break
        elif user_question.lower() == 'rebuild':
            vector_store = rebuild_index_command()
            continue
            
        query_inventory(user_question)
