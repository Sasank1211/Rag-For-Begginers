from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search for relevant documents based on a query
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

# Display results
print("---Context---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}"""

# Create a ChatOllama instance
model = ChatOllama(
    model="llama3.2",
    temperature=0
)

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the model's response
print("\n---Generated Response---")
# print("Full Result")
# print(result)
print("Content Only:")
print(result.content)