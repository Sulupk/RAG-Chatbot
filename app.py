import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

st.title("Document RAG Chatbot")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
query = st.text_input("Ask a question")

# Initialize Groq client
client = Groq()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Store DB in session (IMPORTANT)
if "db" not in st.session_state:
    st.session_state.db = None

if uploaded_file:
    st.success("File uploaded successfully!")

    # Only process once
    if st.session_state.db is None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # Split text
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create FAISS DB
        st.session_state.db = FAISS.from_documents(docs, embeddings)

        st.success("Document processed successfully!")
print(query)
# Run query only if DB exists
if query and st.session_state.db is not None:

    # Retrieve top chunks
    results = st.session_state.db.similarity_search(query, k=3)

    if results:
        # Combine context
        context = "\n\n".join([doc.page_content for doc in results])

        # Send to Groq LLM
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based only on the provided document"
                },
                {
                    "role": "user",
                    "content": f"""
Context:
{context}

Question:
{query}

Answer clearly and concisely:
"""
                }
            ],
            temperature=0.3,
            max_completion_tokens=500,
        )

        answer = response.choices[0].message.content

        st.subheader("Answer:")
        st.write(answer)

    else:
        st.write("No relevant answer found.")