import os
import shutil
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class RAGProcessor:
    """
    RAG (Retrieval-Augmented Generation) processor for inventory data.
    Handles vector database operations and document retrieval.
    """
    
    def __init__(self, csv_file_path, chroma_directory, embedding_model_name="all-MiniLM-L6-v2"):
        """
        Initialize the RAG processor.
        
        Args:
            csv_file_path (str): Path to the CSV file containing inventory data
            chroma_directory (str): Directory path for Chroma vector database
            embedding_model_name (str): Name of the HuggingFace embedding model
        """
        self.csv_file_path = csv_file_path
        self.chroma_directory = chroma_directory
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_store = None
        
        # Initialize the vector store
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or load the vector store."""
        if os.path.exists(self.chroma_directory) and os.listdir(self.chroma_directory):
            try:
                self.vector_store = Chroma(
                    persist_directory=self.chroma_directory, 
                    embedding_function=self.embedding_model
                )
                print("Loaded existing Chroma vector database.")
            except Exception as e:
                print(f"Error loading existing database: {e}")
                self.vector_store = self._rebuild_chroma_index()
        else:
            self.vector_store = self._rebuild_chroma_index()
    
    def _rebuild_chroma_index(self):
        """
        Rebuild the Chroma index from CSV data.
        
        Returns:
            Chroma: The newly created vector store
        """
        print("Rebuilding Chroma index from CSV data...")
        
        # Remove existing directory if it exists
        if os.path.exists(self.chroma_directory):
            shutil.rmtree(self.chroma_directory)
        
        # Load and process CSV data
        df = pd.read_csv(self.csv_file_path)
        df = df.fillna('')
        
        documents = []
        for _, row in df.iterrows():
            # Create metadata with essential fields
            metadata = {
                "component_name": str(row.get("Name of the component", "")),
                "location": str(row.get("Location", "")),
                "quantity": str(row.get("Quantity", ""))
            }
            
            # Create content with essential information
            content_parts = [
                f"Component: {row.get('Name of the component', '')}",
                f"Location: {row.get('Location', '')}",
                f"Quantity: {row.get('Quantity', '')}"
            ]
            
            content = "\n".join(content_parts)
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Create and persist the vector store
        vector_store = Chroma.from_documents(
            documents, 
            self.embedding_model, 
            persist_directory=self.chroma_directory
        )
        
        print(f"Created and saved new Chroma index with {len(documents)} documents.")
        return vector_store
    
    def query_inventory(self, question, k=2, fetch_k=3):
        """
        Query the inventory using vector similarity search.
        
        Args:
            question (str): The user's question about inventory
            k (int): Number of documents to return
            fetch_k (int): Number of documents to fetch before filtering
            
        Returns:
            str: Formatted response with inventory information
        """
        if not self.vector_store:
            return "Vector store not initialized."
        
        # Perform similarity search
        docs = self.vector_store.max_marginal_relevance_search(
            question, k=k, fetch_k=fetch_k
        )
        
        # Process and format the results
        components = []
        for i, doc in enumerate(docs):
            # Extract information directly from metadata
            component_data = {
                "component_name": doc.metadata.get("component_name", "N/A"),
                "location": doc.metadata.get("location", "N/A"),
                "quantity": doc.metadata.get("quantity", "N/A")
            }
            components.append(component_data)
        
        # Format components for display
        formatted_components = []
        for i, comp in enumerate(components):
            formatted_comp = f"Component {i+1}:\n"
            formatted_comp += f"- Name: {comp['component_name']}\n"
            formatted_comp += f"- Location: {comp['location']}\n"
            formatted_comp += f"- Quantity: {comp['quantity']}"
            formatted_components.append(formatted_comp)
        
        return "\n\n".join(formatted_components)
    
    def rebuild_index(self):
        """
        Manually trigger a rebuild of the vector index.
        
        Returns:
            bool: True if rebuild was successful, False otherwise
        """
        try:
            self.vector_store = self._rebuild_chroma_index()
            return True
        except Exception as e:
            print(f"Error rebuilding index: {e}")
            return False
    
    def get_vector_store(self):
        """
        Get the underlying vector store object.
        
        Returns:
            Chroma: The vector store instance
        """
        return self.vector_store
