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

# Step 3: Apply Reciprocal Rank Fusion (RRF)

def reciprocal_rank_fusion(all_retrieved_results, k=60, verbose=True):

    if verbose:
        print("\n" + "-"*60)
        print("Applying Reciprocal Rank Fusion (RRF)")
        print("-"*60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")

        # Data structures for RRF calculation
        rrf_scores = {} # Will store : {chunk_content: rrf_score}
        all_unique_chunks = {} # Will store all unique chunks across all queries

        # For verbose output - track chunks ID
        chunk_id_map = {}
        chunk_counter = 1
    
    # Go through each retrieval result
    for query_idx, chunks in enumerate(all_retrieved_results, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")
        
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content
            
            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            
            chunk_id = chunk_id_map[chunk_content]
            
            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)

             # Initialize score for a new chunk
            if chunk_content not in rrf_scores:
                rrf_scores[chunk_content] = 0
            
            # Add to RRF score
            rrf_scores[chunk_content] += position_score
            
            if verbose:
                print(f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                print(f"    Preview: {chunk_content[:80]}...")
        
        if verbose:
            print()
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )
    
    if verbose:
        print(f" RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(all_retrieved_results)} queries.")
    
    return sorted_chunks

# Apply RRF to our retrival results
fused_results = reciprocal_rank_fusion(all_retrieved_results, k=60, verbose=True)

import re


def is_reference_chunk(text):
    text_lower = text.lower()

    reference_words = [
        "references",
        "bibliography",
        "retrieved from",
        "doi.org",
        "https://",
        "http://"
    ]

    return any(word in text_lower for word in reference_words)


def is_noisy_chunk(text):

    words = text.split()

    if not words:
        return True

    numbers = re.findall(r'\b\d[\d,.]*\b', text)

    number_ratio = len(numbers) / len(words)

    return number_ratio > 0.40


def clean_chunks(fused_results):

    cleaned_results = []

    for chunk, score in fused_results:

        text = chunk.page_content.strip()

        # Remove empty chunks
        if not text:
            continue

        # Remove very short chunks
        if len(text) < 100:
            continue

        # Remove reference/citation chunks
        if is_reference_chunk(text):
            continue

        # Remove chart/numerical garbage
        if is_noisy_chunk(text):
            continue

        cleaned_results.append((chunk, score))

    return cleaned_results


# Clean the RRF results
cleaned_results = clean_chunks(fused_results)


# Select Top 7 after cleaning
top_results = cleaned_results[:7]


print("\n" + "=" * 60)
print("CLEANED TOP 7 RESULTS")
print("=" * 60)

for i, (chunk, score) in enumerate(top_results, 1):

    print(f"\n--- Result {i} ---")
    print(f"RRF Score: {score:.4f}")
    print(f"Content:\n{chunk.page_content[:500]}")
    print("-" * 60)

