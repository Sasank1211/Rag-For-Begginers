from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

# setup
persist_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOllama(model="llama3.2",temperature=0)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}    
)

# pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str]

# MAIN EXECUTION

# Original query
original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}\n")

# Step 1: Genarate multiple query variations

llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""Genarate 3 different variations of this query that would help retrieve relevant documents:
Original query: {original_query}
Return 3 alternative quries that rephrase or approach the same question from different angles."""

responce = llm_with_tools.invoke(prompt)
query_variations = responce.queries

print(f"Genarated Query Variations:")
for i, variation in enumerate(query_variations):
    print(f"{i+1}. {variation}")

print("\n" + "="*60)


# step 2: Search with Each Query Variation & Store Results

retriever = db.as_retriever(search_kwargs={"k": 5}) # Get more docs for better RRF
all_retrieved_results = []

for i, query in enumerate(query_variations,1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")

    docs = retriever.invoke(query)
    all_retrieved_results.append(docs) # Store for RRF calculations

    print(f"Retrieved {len(docs)} documents:\n")

    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")
    print("-" * 50)
print("\n" + "="*60)
print("Multi-Query Retrieval Complete!")
        
print("Documents in Chroma:", db._collection.count())

#print("Original chunks:", len(chunks))
#print("Processed chunks:", len(processed_chunks))
