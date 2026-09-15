import os
import json
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

FAISS_INDEX_PATH = "index_faiss"
FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 4


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hospital Knowledge Assistant",
    page_icon="🏥",
    layout="centered"
)


# ============================================================
# LOAD FAISS INDEX + METADATA + EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_resources():

    # --------------------------------------------------------
    # Check FAISS folder
    # --------------------------------------------------------

    abs_folder_path = os.path.abspath(FAISS_INDEX_PATH)

    if not os.path.isdir(FAISS_INDEX_PATH):

        st.error(
            f"""
            ❌ FAISS folder not found.

            Expected folder:
            {abs_folder_path}

            Current directory:
            {os.getcwd()}

            Files available:
            {os.listdir('.')}
            """
        )

        st.stop()


    # --------------------------------------------------------
    # Define actual files
    # --------------------------------------------------------

    index_path = os.path.join(
        FAISS_INDEX_PATH,
        FAISS_INDEX_FILE
    )

    metadata_path = os.path.join(
        FAISS_INDEX_PATH,
        METADATA_FILE
    )


    # --------------------------------------------------------
    # Check FAISS index
    # --------------------------------------------------------

    if not os.path.isfile(index_path):

        st.error(
            f"""
            ❌ FAISS index file not found.

            Expected:
            {index_path}

            Files inside {FAISS_INDEX_PATH}:
            {os.listdir(FAISS_INDEX_PATH)}
            """
        )

        st.stop()


    # --------------------------------------------------------
    # Check metadata
    # --------------------------------------------------------

    if not os.path.isfile(metadata_path):

        st.error(
            f"""
            ❌ Metadata file not found.

            Expected:
            {metadata_path}

            Files inside {FAISS_INDEX_PATH}:
            {os.listdir(FAISS_INDEX_PATH)}
            """
        )

        st.stop()


    # --------------------------------------------------------
    # Load everything
    # --------------------------------------------------------

    try:

        # Load FAISS index
        index = faiss.read_index(index_path)


        # Load metadata
        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)


        # Load embedding model
        embeddings = SentenceTransformer(
            EMBEDDING_MODEL
        )


        return index, metadata, embeddings


    except Exception as e:

        st.error(
            f"""
            ❌ Failed to load RAG resources.

            Error:
            {str(e)}
            """
        )

        st.stop()


# ============================================================
# LOAD GROQ CLIENT
# ============================================================

@st.cache_resource
def load_groq_client():

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:

        st.error(
            """
            ❌ GROQ_API_KEY was not found.

            Please add your Groq API key to:

            .streamlit/secrets.toml

            Example:

            GROQ_API_KEY = "your_api_key"
            """
        )

        st.stop()


    return Groq(
        api_key=api_key
    )


# ============================================================
# EXTRACT TEXT FROM METADATA
# ============================================================

def extract_text(item):

    """
    Handles different possible metadata formats.

    It checks common field names such as:
    text
    page_content
    content
    chunk
    document
    """

    if isinstance(item, str):

        return item


    if not isinstance(item, dict):

        return str(item)


    possible_text_fields = [
        "text",
        "page_content",
        "content",
        "chunk",
        "document",
        "text_content"
    ]


    for field in possible_text_fields:

        if field in item:

            value = item[field]

            if value is not None:

                return str(value)


    # --------------------------------------------------------
    # Some metadata files may store the text inside metadata
    # --------------------------------------------------------

    if "metadata" in item:

        nested = item["metadata"]

        if isinstance(nested, dict):

            for field in possible_text_fields:

                if field in nested:

                    return str(
                        nested[field]
                    )


    return ""


# ============================================================
# GET METADATA VALUE
# ============================================================

def get_metadata_value(
    item,
    possible_keys,
    default="Unknown"
):

    """
    Safely gets metadata values from different
    possible JSON structures.
    """

    if not isinstance(item, dict):

        return default


    # Check directly
    for key in possible_keys:

        if key in item:

            value = item[key]

            if value is not None:

                return value


    # Check nested metadata
    if isinstance(
        item.get("metadata"),
        dict
    ):

        nested = item["metadata"]

        for key in possible_keys:

            if key in nested:

                value = nested[key]

                if value is not None:

                    return value


    return default


# ============================================================
# NORMALIZE METADATA
# ============================================================

def normalize_metadata(metadata):

    """
    Converts different possible metadata.json formats
    into a simple list.

    Supported examples:

    [
        {...},
        {...}
    ]

    or:

    {
        "0": {...},
        "1": {...}
    }

    or:

    {
        "documents": [...]
    }

    or:

    {
        "chunks": [...]
    }
    """

    # --------------------------------------------------------
    # Case 1: Metadata is already a list
    # --------------------------------------------------------

    if isinstance(metadata, list):

        return metadata


    # --------------------------------------------------------
    # Case 2: Metadata is a dictionary
    # --------------------------------------------------------

    if isinstance(metadata, dict):

        # Common container keys
        for key in [
            "documents",
            "chunks",
            "data",
            "items",
            "metadata"
        ]:

            value = metadata.get(key)

            if isinstance(value, list):

                return value


        # Dictionary indexed by FAISS vector ID
        numeric_keys = []

        for key in metadata.keys():

            try:

                numeric_keys.append(
                    (int(key), key)
                )

            except (ValueError, TypeError):

                pass


        if numeric_keys:

            numeric_keys.sort(
                key=lambda x: x[0]
            )

            return [
                metadata[key]
                for _, key in numeric_keys
            ]


        # If dictionary contains one document
        return [metadata]


    return []


# ============================================================
# RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_chunks(
    index,
    metadata,
    embeddings,
    query,
    k=TOP_K
):

    # --------------------------------------------------------
    # Normalize metadata
    # --------------------------------------------------------

    metadata_list = normalize_metadata(
        metadata
    )


    if len(metadata_list) == 0:

        return []


    # --------------------------------------------------------
    # Create embedding for user query
    # --------------------------------------------------------

    query_embedding = embeddings.encode(
        [query],
        convert_to_numpy=True
    )


    # FAISS normally expects float32
    query_embedding = query_embedding.astype(
        "float32"
    )


    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    actual_k = min(
        k,
        index.ntotal
    )


    if actual_k <= 0:

        return []


    distances, indices = index.search(
        query_embedding,
        actual_k
    )


    results = []


    # --------------------------------------------------------
    # Convert FAISS results into documents
    # --------------------------------------------------------

    for distance, index_id in zip(
        distances[0],
        indices[0]
    ):

        # Invalid FAISS result
        if index_id < 0:

            continue


        index_id = int(index_id)


        # Make sure metadata exists
        if index_id >= len(metadata_list):

            continue


        item = metadata_list[index_id]


        # Extract text
        text = extract_text(
            item
        )


        # Skip empty chunks
        if not text.strip():

            continue


        # Extract information
        department = get_metadata_value(
            item,
            [
                "department",
                "dept"
            ],
            "Unknown"
        )


        source_file = get_metadata_value(
            item,
            [
                "source_file",
                "source",
                "file",
                "filename",
                "file_name"
            ],
            "Unknown"
        )


        page = get_metadata_value(
            item,
            [
                "page",
                "page_number",
                "page_num"
            ],
            "?"
        )


        results.append({

            "content": text,

            "department": str(
                department
            ),

            "source_file": str(
                source_file
            ),

            "page": str(
                page
            ),

            "distance": float(
                distance
            ),

            "index": index_id

        })


    return results


# ============================================================
# BUILD CONTEXT FOR LLM
# ============================================================

def build_context(chunks):

    context_parts = []


    for i, chunk in enumerate(chunks):

        context_parts.append(

            f"""
[Retrieved Chunk {i + 1}]
Department: {chunk["department"]}
Source: {chunk["source_file"]}
Page: {chunk["page"]}

Content:
{chunk["content"]}
""".strip()

        )


    return "\n\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER USING GROQ
# ============================================================

def generate_answer(
    client,
    query,
    context
):

    system_prompt = """

You are a Hospital Knowledge Assistant.

Your job is to answer questions using ONLY
the information provided in the retrieved
hospital knowledge-base context.

IMPORTANT RULES:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the
   provided context, say:

   "I don't have this information in
   the hospital knowledge base."

4. Keep answers clear and concise.
5. Use simple professional language.
6. If the context contains multiple relevant
   policies, combine them accurately.
7. Do not provide unsupported medical or
   hospital policy claims.

"""


    user_prompt = f"""

Hospital Knowledge Base Context:

{context}


User Question:

{query}


Answer the question based ONLY on the
provided hospital knowledge base context.

"""


    response = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }

        ],

        temperature=0.2

    )


    return response.choices[0].message.content


# ============================================================
# APPLICATION UI
# ============================================================

st.title(
    "🏥 Hospital Knowledge Assistant"
)


st.caption(
    "Ask questions about hospital policies "
    "and get answers grounded in the hospital "
    "knowledge base."
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# LOAD RESOURCES
# ============================================================

index, metadata, embeddings = load_resources()

client = load_groq_client()


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        # Show sources for assistant messages
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📄 Sources"
            ):

                for source in message["sources"]:

                    st.markdown(

                        f"- **{source['source_file']}** "
                        f"(Department: "
                        f"{source['department']}, "
                        f"Page: {source['page']})"

                    )


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask a question about hospital policy..."
)


if query:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": query

    })


    with st.chat_message("user"):

        st.markdown(
            query
        )


    # --------------------------------------------------------
    # Generate assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching hospital knowledge base..."
        ):

            # Retrieve relevant chunks
            chunks = retrieve_chunks(

                index=index,

                metadata=metadata,

                embeddings=embeddings,

                query=query,

                k=TOP_K

            )


        # ----------------------------------------------------
        # If nothing was retrieved
        # ----------------------------------------------------

        if not chunks:

            answer = (
                "I don't have this information "
                "in the hospital knowledge base."
            )

            sources = []


        else:

            # ------------------------------------------------
            # Build RAG context
            # ------------------------------------------------

            context = build_context(
                chunks
            )


            # ------------------------------------------------
            # Generate answer
            # ------------------------------------------------

            with st.spinner(
                "🤖 Generating answer..."
            ):

                answer = generate_answer(

                    client=client,

                    query=query,

                    context=context

                )


            # ------------------------------------------------
            # Prepare sources
            # ------------------------------------------------

            sources = [

                {

                    "source_file":
                        chunk["source_file"],

                    "department":
                        chunk["department"],

                    "page":
                        chunk["page"]

                }

                for chunk in chunks

            ]


            # ------------------------------------------------
            # Remove duplicate sources
            # ------------------------------------------------

            seen = set()

            unique_sources = []


            for source in sources:

                key = (

                    source["source_file"],

                    source["page"]

                )


                if key not in seen:

                    seen.add(key)

                    unique_sources.append(
                        source
                    )


            sources = unique_sources


        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        st.markdown(
            answer
        )


        # ----------------------------------------------------
        # Display sources
        # ----------------------------------------------------

        if sources:

            with st.expander(
                "📄 Sources"
            ):

                for source in sources:

                    st.markdown(

                        f"- **{source['source_file']}** "
                        f"(Department: "
                        f"{source['department']}, "
                        f"Page: {source['page']})"

                    )


    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "sources": sources

    })
