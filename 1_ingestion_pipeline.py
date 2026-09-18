import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

print("Current directory:", os.getcwd())
print("API key loaded:", bool(os.getenv("OPENAI_API_KEY")))





def load_documents(docs_path="docs"):
    """ Load all text files from the docs directory"""
    print(f"Loading documents from  {docs_path}...")

    # Check if the specified directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The specified directory '{docs_path}' does not exist. Please create it and add your company files")
    
    # Load all text files from the specified directory
    loader = DirectoryLoader(
        path=docs_path, 
        glob="*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No documents found in the specified directory '{docs_path}'. Please add your company documents")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" Metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """ Split documents into smaller chunks for processing"""
    print("Splitting documents into chunks of size")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )

    chunks=text_splitter.split_documents(documents)
    #if chunks:
        #for i, chunk in enumerate(chunks[:5]):
            #print(f"\n--- Chunk {i+1} ---")
            #print(f" Source: {chunk.metadata} [source]")
            #print(f" Length: {len(chunk.page_content)} characters")
            #print(f" Content:")
            #print(f" chunk.page_content")
            #print("-"*50)
        #if len(chunks) > 5:
         #   print(f"\n...and {len(chunks) - 5} more chunks.")    

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """ Create and persist ChromaDB vector store"""
    print("Creating Embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Create ChromaDB vector store
    print("--- Creating ChromaDB vector store ---")

    # Create a Chroma vector store from the document chunks and embeddings
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished Creating Vector store ---")

    # Persist the vector store to disk
    print(f"Vector store created and persisted to '{persist_directory}'.")
    return vector_store


def main():
    print("Main function started")

    # Load documents from the specified directory
    documents = load_documents(docs_path="docs")

    # Split documents into smaller chunks for processing
    chunks=split_documents(documents)
    # Create and persist ChromaDB vector store
    vector_store = create_vector_store(chunks)
    print("Original chunks:", len(chunks))    

if __name__ == "__main__":
    main()    