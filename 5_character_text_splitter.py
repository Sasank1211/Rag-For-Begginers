from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
Mode Y performance
The Model Y become the best-selling vehicle globality, with 350,000 units sold.
Production challenges
Supply chain issues caused a 12% increase in production costs.
This is one very long paragraph that definately exceeds our 100 character limit and has no double newlines inside it whatsover making it impossible to split properly."""

splitter1 = CharacterTextSplitter(
    separator=" ",
    chunk_size=100,
    chunk_overlap=0
)

chunks1 = splitter1.split_text(tesla_text)
for i,chunk in enumerate(chunks1, 1):
    print(f"Chunk{i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()


# Example 2 RecursiveCharacterTextSplitter fixes this
print("\n" + "=" * 60)
print("2. RCURSIVE CHARACTER TEXT SPLITTER SOLUTION")
print("=" * 60)

recursive_splitter=RecursiveCharacterTextSplitter(
    separators=["\n\n","\n",". "," ",""],
    chunk_size=100,
    chunk_overlap=0
)

chunks2 = recursive_splitter.split_text(tesla_text)
print(f"Same problem text, but with RecursiveCharacterTextSplitter:")
for i,chunk in enumerate(chunks2,1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()
