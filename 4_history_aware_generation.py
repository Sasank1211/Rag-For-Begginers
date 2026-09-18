from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv  
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv()

# Connect to the Chroma vector store
persistent_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model)
# Set up AI model
model = ChatOllama(
    model="llama3.2",
    temperature=0
)

# Store our conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # step1 : make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),

        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # step2 : retrieve relevant documents
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    relevant_docs = retriever.invoke(search_question)

    print(f"Found {len(relevant_docs)} relevant documents:")
    for i, doc in enumerate(relevant_docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.splitlines()
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")    

    # step 3: Create final prompt 
    combined_input = f"""Based on the following documents, please answer this question: {user_question}
    Documents:
    {"\n".join([f" - {doc.page_content}" for doc in relevant_docs])}
    Please provide a clear, helpful and answer using only the information from the documents above. If you can't find the answer in the documents, please respond with "I don't know".
    """

    # step 4: Get the Answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    # step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer

# simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            print("Goodbye!")
            break
        ask_question(question)

if __name__ == "__main__":
    start_chat()        

