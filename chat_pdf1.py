import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

st.set_page_config(
    page_title="PDF Q&A",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF Question Answering")

TOP_K = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

CHROMA_DIR = "./chroma_db"



@st.cache_resource
def load_embedding_model():
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    return HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-base-en-v1.5"
    )


@st.cache_resource
def load_llm():
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-3n-E4B-it",
        max_new_tokens=512,
        temperature=0.1,
    )

    return ChatHuggingFace(llm=llm)



if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file:

    if st.session_state.processed_file != uploaded_file.name:

        with st.spinner("📚 Processing PDF..."):

            try:
                
                from langchain_community.document_loaders import PyPDFLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                from langchain_chroma import Chroma

                
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    pdf_path = tmp.name

                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )

                chunks = splitter.split_documents(documents)

                
                embedding_model = load_embedding_model()

                
                vector_store = Chroma(
                    collection_name="pdf_docs",
                    embedding_function=embedding_model,
                    persist_directory=CHROMA_DIR,
                )

                vector_store.add_documents(chunks)

                st.session_state.vector_store = vector_store
                st.session_state.processed_file = uploaded_file.name

                os.unlink(pdf_path)

                st.success(
                    f"✅ Indexed {len(chunks)} chunks from {len(documents)} pages."
                )

            except Exception as e:
                st.error(f"❌ PDF Processing Error: {e}")

    else:
        st.info(
            f"📎 Using already indexed file: {uploaded_file.name}"
        )




if st.session_state.vector_store:

    st.markdown("---")

    question = st.text_input(
        "💬 Ask a question about your PDF",
        placeholder="What is the main topic?"
    )

    if st.button(
        "Ask",
        type="primary",
        use_container_width=True
    ):

        if question:

            with st.spinner("🤔 Thinking..."):

                try:
                    relevant_docs = (
                        st.session_state.vector_store
                        .similarity_search(question, k=TOP_K)
                    )

                    context = "\n\n".join(
                        doc.page_content
                        for doc in relevant_docs
                    )

                    from langchain_core.prompts import PromptTemplate

                    prompt = PromptTemplate.from_template(
                        """
You are a helpful assistant.

Answer ONLY from the provided context.

If the answer is not available in the context, reply:

"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
                    )

                    chat_model = load_llm()

                    chain = prompt | chat_model

                    response = chain.invoke(
                        {
                            "context": context,
                            "question": question
                        }
                    )

                    st.markdown("### 📝 Answer")
                    st.write(response.content)

                    with st.expander("📎 Source Chunks"):

                        for idx, doc in enumerate(
                            relevant_docs,
                            start=1
                        ):

                            page = (
                                doc.metadata.get("page", 0)
                                + 1
                            )

                            st.markdown(
                                f"**Chunk {idx} (Page {page})**"
                            )

                            st.caption(
                                doc.page_content[:500]
                            )

                            st.divider()

                except Exception as e:
                    st.error(
                        f"❌ Error generating answer: {e}"
                    )