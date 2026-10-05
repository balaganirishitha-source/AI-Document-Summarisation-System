import streamlit as st
import ollama
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="AI Document Summarisation",
    page_icon="📄",
    layout="wide"
)


# ---------------- TITLE ----------------

st.title("📄 AI-Based Document Summarisation System")

st.write(
    "Upload a PDF document and generate an AI-powered summary "
    "using Llama 3.2."
)


# ---------------- LOAD EMBEDDING MODEL ----------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# ---------------- PDF UPLOAD ----------------

uploaded_file = st.file_uploader(
    "📤 Upload your PDF document",
    type=["pdf"]
)


if uploaded_file is not None:

    # ---------------- READ PDF ----------------

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"


    # ---------------- DOCUMENT INFORMATION ----------------

    pages = len(reader.pages)

    words = text.split()

    word_count = len(words)


    # ---------------- CREATE CHUNKS ----------------

    chunks = []

    chunk_size = 300

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk.strip():

            chunks.append(chunk)


    chunk_count = len(chunks)


    # ---------------- DOCUMENT STATISTICS ----------------

    st.subheader("📊 Document Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Pages",
            pages
        )

    with col2:
        st.metric(
            "Words",
            word_count
        )

    with col3:
        st.metric(
            "Text Chunks",
            chunk_count
        )


    # ---------------- DOCUMENT PREVIEW ----------------

    st.subheader("📖 Document Preview")

    if text.strip():

        st.code(
            text[:3000],
            language="text"
        )

    else:

        st.warning(
            "⚠️ No readable text found in this PDF."
        )


    # ---------------- SUMMARIZE BUTTON ----------------

    if st.button(
        "🤖 Summarize Document",
        use_container_width=True
    ):

        if text.strip():

            with st.spinner(
                "Processing document and generating summary..."
            ):

                # ---------------- CHROMADB ----------------

                client = chromadb.Client()

                collection = client.get_or_create_collection(
                    name="document_collection"
                )


                # ---------------- EMBEDDINGS ----------------

                embeddings = embedding_model.encode(
                    chunks
                ).tolist()


                # ---------------- CREATE UNIQUE IDS ----------------

                ids = []

                for i in range(len(chunks)):

                    ids.append(
                        f"chunk_{i}"
                    )


                # ---------------- STORE DOCUMENT ----------------

                collection.add(
                    documents=chunks,
                    embeddings=embeddings,
                    ids=ids
                )


                # ---------------- AI PROMPT ----------------

                prompt = f"""
You are an AI document summarisation assistant.

Read the document carefully and provide a concise summary.

Use this format:

Overall Summary:
Write a short summary of the complete document.

Main Important Points:
- Give the most important points.
- Keep them clear and simple.

Conclusion:
Give the conclusion of the document.

Do not unnecessarily repeat the document.

Document:
{text}
"""


                # ---------------- OLLAMA ----------------

                response = ollama.chat(

                    model="llama3.2",

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )


                summary = response[
                    "message"
                ][
                    "content"
                ]


            # ---------------- SUCCESS MESSAGE ----------------

            st.success(
                "✅ Document processed successfully!"
            )


            # ---------------- SUMMARY ----------------

            st.subheader(
                "🤖 AI Generated Summary"
            )

            st.write(summary)


            # ---------------- DOWNLOAD SUMMARY ----------------

            st.download_button(

                label="📥 Download Summary",

                data=summary,

                file_name="document_summary.txt",

                mime="text/plain",

                use_container_width=True
            )


        else:

            st.error(
                "❌ Could not extract text from this PDF."
            )