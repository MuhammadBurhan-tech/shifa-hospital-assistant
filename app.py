import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ---- Config ----
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4

st.set_page_config(page_title="Hospital Policy Assistant", page_icon="🏥", layout="centered")


# ---- Load resources (cached so they only load once per session) ----
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


@st.cache_resource
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in secrets. Add it to .streamlit/secrets.toml")
        st.stop()
    return Groq(api_key=api_key)


def retrieve_chunks(vectorstore, query, k=TOP_K):
    results = vectorstore.similarity_search(query, k=k)
    return results


def build_context(chunks):
    context_parts = []
    for i, chunk in enumerate(chunks):
        dept = chunk.metadata.get("department", "Unknown")
        source = chunk.metadata.get("source_file", "Unknown")
        page = chunk.metadata.get("page", "?")
        context_parts.append(
            f"[Chunk {i+1} | Department: {dept} | Source: {source} | Page: {page}]\n{chunk.page_content}"
        )
    return "\n\n".join(context_parts)


def generate_answer(client, query, context):
    system_prompt = (
        "You are a hospital policy assistant. Answer the user's question using ONLY the "
        "provided context chunks. If the answer isn't in the context, say you don't have "
        "that information in the knowledge base. Be concise and clear."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---- UI ----
st.title("🏥 Hospital Policy Assistant")
st.caption("Ask a question about hospital policy. Answers are grounded in your uploaded documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

vectorstore = load_vectorstore()
client = load_groq_client()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source_file']}** (Dept: {s['department']}, Page: {s['page']})")

# Chat input
query = st.chat_input("Ask a question about hospital policy...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            chunks = retrieve_chunks(vectorstore, query)
            context = build_context(chunks)
            answer = generate_answer(client, query, context)

        st.markdown(answer)

        sources = [
            {
                "source_file": c.metadata.get("source_file", "Unknown"),
                "department": c.metadata.get("department", "Unknown"),
                "page": c.metadata.get("page", "?"),
            }
            for c in chunks
        ]
        # de-duplicate sources by file+page
        seen = set()
        unique_sources = []
        for s in sources:
            key = (s["source_file"], s["page"])
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        with st.expander("📄 Sources"):
            for s in unique_sources:
                st.markdown(f"- **{s['source_file']}** (Dept: {s['department']}, Page: {s['page']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": unique_sources
    })
