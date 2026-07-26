import streamlit as st

from graph import graph
from chunk_retriever import index_documents, build_retriever


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.block-container{
    padding-top: 2rem;
}

.source-box{
    padding:12px;
    border-radius:10px;
    margin-bottom:10px;
    font-weight:bold;
}

.cache{
    background-color:#d4edda;
    color:#155724;
}

.rag{
    background-color:#d1ecf1;
    color:#0c5460;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.title("🤖 AI Assistant")

    st.markdown("""
    ### Features

    ✅ Semantic Cache

    ✅ Vector Database

    ✅ Hybrid Retrieval

    ✅ Cross Encoder Reranker

    ✅ RAG Pipeline

    ✅ Answer Evaluation

    ✅ LangGraph Workflow
    """)

    st.divider()


    # ------------------------------------------------
    # DOCUMENT INGESTION
    # ------------------------------------------------

    st.subheader("📚 Knowledge Base")

    st.caption(
        "Upload a document, provide a URL, "
        "or enter your own text."
    )


    # -----------------------------------------------
    # FILE UPLOAD
    # -----------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=[
            "pdf",
            "docx",
            "csv"
        ]
    )


    # -----------------------------------------------
    # URL INPUT
    # -----------------------------------------------

    url = st.text_input(
        "🌐 Website URL",
        placeholder="https://example.com"
    )


    # -----------------------------------------------
    # RAW TEXT
    # -----------------------------------------------

    raw_text = st.text_area(
        "📝 Enter Text",
        placeholder="Paste your text here...",
        height=150
    )


    # -----------------------------------------------
    # PROCESS DOCUMENT
    # -----------------------------------------------

    if st.button(
        "🚀 Process Knowledge",
        use_container_width=True
    ):

        # -------------------------------------------
        # VALIDATION
        # -------------------------------------------

        input_count = sum(
            bool(x)
            for x in [
                uploaded_file,
                url.strip(),
                raw_text.strip()
            ]
        )


        if input_count == 0:

            st.warning(
                "Please upload a file, enter a URL, "
                "or provide some text."
            )


        elif input_count > 1:

            st.warning(
                "Please provide only one source "
                "at a time."
            )


        else:

            try:

                with st.spinner(
                    "Processing knowledge base..."
                ):

                    # --------------------------------
                    # INGEST DOCUMENT
                    # --------------------------------

                    chunks = index_documents(
                        uploaded_file=uploaded_file,
                        url=url.strip() if url else None,
                        raw_text=raw_text.strip()
                        if raw_text else None
                    )


                    # --------------------------------
                    # BUILD RETRIEVER
                    # --------------------------------

                    retriever = build_retriever(
                        chunks
                    )


                    # --------------------------------
                    # SAVE RETRIEVER
                    # --------------------------------

                    st.session_state.retriever = (
                        retriever
                    )

                    st.session_state.document_processed = (
                        True
                    )


                st.success(
                    "Knowledge base processed successfully!"
                )


            except Exception as e:

                st.error(
                    f"Error while processing document: {e}"
                )


    # ------------------------------------------------
    # STATUS
    # ------------------------------------------------

    st.divider()

    if st.session_state.document_processed:

        st.success(
            "🟢 Knowledge Base Ready"
        )

    else:

        st.info(
            "🔵 No document processed yet"
        )


    # ------------------------------------------------
    # CLEAR CHAT
    # ------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("💬 AI Knowledge Assistant")

st.caption(
    "Ask questions from your knowledge base"
)


# ---------------------------------------------------
# DISPLAY OLD CHAT
# ---------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(
            msg["content"]
        )


# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

query = st.chat_input(
    "Ask a question..."
)


if query:

    # -----------------------------------------------
    # CHECK KNOWLEDGE BASE
    # -----------------------------------------------

    if st.session_state.retriever is None:

        st.warning(
            "Please process a document before "
            "asking a question."
        )

        st.stop()


    # -----------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    with st.chat_message("user"):

        st.markdown(
            query
        )


    # -----------------------------------------------
    # RUN GRAPH
    # -----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            result = graph.invoke(
                {
                    "query": query,

                    "document": [],

                    "answer": "",

                    "result": None,

                    "context": "",

                    "model_used": "",

                    "token_usage": 0,

                    # IMPORTANT
                    # Pass current retriever
                    "retriever":
                        st.session_state.retriever
                }
            )


        # -------------------------------------------
        # GET RESULT
        # -------------------------------------------

        answer = result[
            "answer"
        ][
            "answer"
        ]


        cache_hit = result.get(
            "cache_hit",
            False
        )


        evaluation = result.get(
            "result"
        )


        model_used = result.get(
            "model_used",
            "N/A"
        )


        token_usage = result.get(
            "token_usage",
            0
        )


        # -------------------------------------------
        # ANSWER SOURCE
        # -------------------------------------------

        if cache_hit:

            st.markdown(
                """
                <div class="source-box cache">
                ⚡ Answer Source: Semantic Cache
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="source-box rag">
                📚 Answer Source: Vector Database (RAG)
                </div>
                """,
                unsafe_allow_html=True
            )


        # -------------------------------------------
        # ANSWER
        # -------------------------------------------

        st.markdown(
            answer
        )


        # -------------------------------------------
        # SAVE ASSISTANT MESSAGE
        # -------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        st.divider()


        # -------------------------------------------
        # METRICS
        # -------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Source",
                "Cache"
                if cache_hit
                else "Vector DB"
            )


        with col2:

            st.metric(
                "Model",
                model_used
            )


        with col3:

            st.metric(
                "Tokens",
                token_usage
            )


        # -------------------------------------------
        # RETRIEVAL FLOW
        # -------------------------------------------

        with st.expander(
            "🔍 Pipeline Details"
        ):


            if cache_hit:

                st.success(
                    "Semantic Cache Hit"
                )

                st.write(
                    "✅ Similar query found"
                )

                st.write(
                    "✅ Returned cached answer"
                )

                st.write(
                    "❌ Vector search skipped"
                )

                st.write(
                    "❌ Evaluation skipped"
                )


            else:

                st.info(
                    "Cache Miss"
                )

                st.write(
                    "❌ No matching cached query"
                )

                st.write(
                    "✅ Retrieved documents from Vector DB"
                )

                st.write(
                    "✅ Generated answer using LLM"
                )

                st.write(
                    "✅ Evaluated response"
                )


        # -------------------------------------------
        # EVALUATION
        # -------------------------------------------

        if (
            not cache_hit
            and evaluation
        ):

            with st.expander(
                "📊 Evaluation Results"
            ):

                st.write(
                    f"**Relevant:** "
                    f"{evaluation.relevant}"
                )

                st.write(
                    f"**Grounded:** "
                    f"{evaluation.grounded}"
                )

                st.write(
                    f"**Answer Question:** "
                    f"{evaluation.answer_question}"
                )


                if (
                    evaluation.relevant
                    and evaluation.grounded
                    and evaluation.answer_question
                ):

                    st.success(
                        "Answer Passed Evaluation"
                    )

                else:

                    st.warning(
                        "Answer Needs Review"
                    )