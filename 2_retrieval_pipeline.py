from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search for relevant documents based on a query
query = "What was Microsoft's first hardware product release?"

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

# Display results
print("---Context---")

for i, doc in enumerate(relevant_docs,1):
    print(f"Document{i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: "{query}"   

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful and answer using only the information from the documents above. If you can't find the answer in the documents, please respond with "I don't know".
"""

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

# Display the full result and content only
print("\n--- Generated Responce ---")

# print("Full Result:")
# print(result)
print("Content Only:")
print(result.content)

# Synthetic Questions

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomus spaceport drone ship that achieved the first successfull sea leading?"
# 8. "What was the original name of Microsoft before it became Microsoft?"
