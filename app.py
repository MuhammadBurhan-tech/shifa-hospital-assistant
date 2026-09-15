import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ---- Config ----
FAISS_INDEX_PATH = "index_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4

st.set_page_config(page_title="Hospital Policy Assistant", page_icon="🏥", layout="centered")


# ---- Load resources (cached so they only load once per session) ----
@st.cache_resource
def load_vectorstore():
    # --- Diagnostics: check the index exists and is readable BEFORE calling FAISS ---
    abs_path = os.path.abspath(FAISS_INDEX_PATH)

    if not os.path.isdir(FAISS_INDEX_PATH):
        st.error(
            f"FAISS index folder not found at: `{abs_path}`\n\n"
            f"Current working directory contents: {os.listdir('.')}\n\n"
            "This means `faiss_index/` was not deployed with your app — check that it's "
            "committed to your GitHub repo (not excluded by .gitignore, and not stuck as a "
            "Git LFS pointer)."
        )
        st.stop()

    files_found = os.listdir(FAISS_INDEX_PATH)
    required = ["index.faiss", "index.pkl"]
    missing = [f for f in required if f not in files_found]

    if missing:
        st.error(
            f"`faiss_index/` exists at `{abs_path}` but is missing: {missing}\n\n"
            f"Files actually found there: {files_found}\n\n"
            "Re-check that both index.faiss and index.pkl were committed and pushed to GitHub."
        )
        st.stop()

    # Show file sizes — a 0-byte or few-KB index.faiss usually means a bad Git LFS pointer
    # instead of the real binary file.
    sizes = {f: os.path.getsize(os.path.join(FAISS_INDEX_PATH, f)) for f in required}
    tiny_files = [f for f, sz in sizes.items() if sz < 1024]
    if tiny_files:
        st.error(
            f"These index files are suspiciously small (likely broken/placeholder, not the "
            f"real binary): {tiny_files} — sizes in bytes: {sizes}\n\n"
            "This is the classic symptom of a Git LFS pointer file being committed instead "
            "of the actual file content. Re-upload these files directly (not via LFS) or "
            "check your repo's LFS settings."
        )
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    try:
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"FAISS failed to read the index despite files being present.\n\nError: {e}")
        st.stop()


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
