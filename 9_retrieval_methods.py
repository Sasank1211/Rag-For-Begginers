from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# setup
persist_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Query to test

query = "How much did microsoft pay to acquire GitHub?"
# query = "How do you plant tomatoes in a garden?"
print(f"Query: {query}")

# Method 1: Basic similarity search
# Returns the top k most similar documents
#print("=== Method 1: Basic similarity search ===")
#retriever = db.as_retriever(searchkwargs={"k": 3})

#docs = retriever.invoke(query)
#print(f"Retrieved {len(docs)} documents:\n")

#for i, doc in enumerate(docs):
 #   print(f"Document {i}:")
  #  print(f"{doc.page_content}\n")
#print("-" * 60)

# Method 2: Similarity search with score threshold
# Only Returns above a certain similarity score
#print("=== Method 2: Similarity search with score threshold ===")
#retriever = db.as_retriever(
 #   search_type = "similarity_score_threshold",
  #  search_kwargs={
   #     "k": 3,
    #    "score_threshold": 0.3 # only return docs with similarity  >= 0.3
    #}
#)

#docs = retriever.invoke(query)
#print(f"Retrieved {len(docs)} documents:\n")

#for i, doc in enumerate(docs):
 #   print(f"Document {i}:")
  #  print(f"{doc.page_content}\n")

#print("-" * 60)

# Method 3: MMR (Maximal Marginal Relevance)
# Balances relevance and diversity - avoids redundant results

print("=== Method 3: MMR (Maximal Marginal Relevance) ===")
retriever = db.as_retriever(
    search_type = "mmr",
    search_kwargs={
       "k": 3,
       "fetch_k": 10, # fetch 10 docs and return the top 3 after MMR
       "lambda_mult": 0.5 # balance between relevance and diversity
    } 
)

docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents:\n")

for i, doc in enumerate(docs):
   print(f"Document {i}:")
   print(f"{doc.page_content}\n")

print("-" * 60)
print("Done! Try different queries or parameters to see differences.")    