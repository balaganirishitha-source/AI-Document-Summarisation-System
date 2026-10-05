import streamlit as st
import ollama
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


st.set_page_config(
    page_title="AI Document Summarisation",
    page_icon="📄"
)

st.title("AI-Based Document Summarisation System")

st.write(
    "Upload a PDF document and generate an AI-powered summary."
)


# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


if uploaded_file is not None:

    # Read PDF
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"


    # Show document preview
    st.subheader("Document Preview")

    st.write(text[:2000])


    # Create text chunks
    words = text.split()

    chunks = []

    chunk_size = 300

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)


    st.info(f"Document divided into {len(chunks)} text chunks.")


    if st.button("Summarize Document"):

        if text.strip():

            # Create ChromaDB client
            client = chromadb.Client()

            collection = client.get_or_create_collection(
                name="document_collection"
            )


            # Create embeddings
            embeddings = embedding_model.encode(
                chunks
            ).tolist()


            # Store chunks in ChromaDB
            ids = []

            for i in range(len(chunks)):
                ids.append(f"chunk_{i}")


            collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=ids
            )


            st.success(
                "Document processed and stored successfully!"
            )


            # AI summarisation
            prompt = f"""
            You are an AI document summarisation assistant.

            Summarize the following document in simple and
            clear language.

            Give the answer in this format:

            Overall Summary:
            Main Important Points:
            Conclusion:

            Document:
            {text}
            """


            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )


            summary = response["message"]["content"]


            # Display summary
            st.subheader("AI Generated Summary")

            st.write(summary)


        else:

            st.error(
                "Could not extract text from this PDF."
            )