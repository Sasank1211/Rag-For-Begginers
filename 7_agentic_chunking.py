from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)
# Tesla text to chunk
tesla_text="""Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

# Create the prompt
prompt = f"""
You are a text chunking expert.

Split the text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less.
- Split only at natural topic boundaries.
- Keep related information together.
- Do NOT add any explanation.
- Do NOT add an introduction.
- Do NOT add "Here is the text..."
- Do NOT change, summarize, or rewrite the original text.
- Return ONLY the original text with <<<SPLIT>>> inserted between chunks.
- The response MUST start with the original text, not with <<<SPLIT>>> or any explanation.

Text:
{tesla_text}

Return ONLY the chunked text:
"""


# Get AI Response
print("Asking AI to chunk the text...")
response = llm.invoke(prompt)
marked_text = response.content

# Split the text at markers
chunks = marked_text.split("<<<SPLIT>>>")

# Clean up the chunks(remove extra whitespace)
clean_chunks = []
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned:
        clean_chunks.append(cleaned)

# Show results
print("\n AGENTIC CHUNKING RESULTS:")
print("=" * 50)

for i, chunk in enumerate(clean_chunks):
    print(f"Chunk{i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()

