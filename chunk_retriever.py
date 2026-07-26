import os
import hashlib

from dotenv import load_dotenv
load_dotenv()

from chunks import create_chunks
from ingestion import get_docs

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)

from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pinecone-hybrid-index"


# =====================================================
# PINECONE INITIALIZATION
# =====================================================

pc = Pinecone(api_key=PINECONE_API_KEY)


if INDEX_NAME not in pc.list_indexes().names():

    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


index = pc.Index(INDEX_NAME)


# =====================================================
# EMBEDDING MODEL
# =====================================================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={
        "normalize_embeddings": True
    }
)


# =====================================================
# VECTOR STORE
# =====================================================

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedding
)


# =====================================================
# RERANKER
# =====================================================

rerank_model = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)


reranker = CrossEncoderReranker(
    model=rerank_model,
    top_n=5
)


# =====================================================
# INDEX DOCUMENT
# =====================================================

def index_documents(
    uploaded_file=None,
    url=None,
    raw_text=None
):

    docs = get_docs(
        uploaded_file=uploaded_file,
        url=url,
        raw_text=raw_text
    )

    all_chunks = []

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "unknown"
        )

        print(f"Processing: {source}")


        # -----------------------------------------
        # DOCUMENT ID
        # -----------------------------------------

        doc_id = hashlib.md5(
            source.encode()
        ).hexdigest()


        # -----------------------------------------
        # CONTENT HASH
        # -----------------------------------------

        content_hash = hashlib.md5(
            doc.page_content.encode()
        ).hexdigest()


        # -----------------------------------------
        # CHUNK DOCUMENT
        # -----------------------------------------

        chunks = create_chunks([doc])


        # Always keep chunks for BM25
        all_chunks.extend(chunks)


        # -----------------------------------------
        # ADD METADATA
        # -----------------------------------------

        ids = []

        for i, chunk in enumerate(chunks):

            chunk.metadata["doc_id"] = doc_id

            chunk.metadata["content_hash"] = content_hash

            chunk_id = hashlib.md5(
                f"{doc_id}_{i}".encode()
            ).hexdigest()

            ids.append(chunk_id)


        # -----------------------------------------
        # CHECK EXISTING DOCUMENT
        # -----------------------------------------

        first_chunk_id = hashlib.md5(
            f"{doc_id}_0".encode()
        ).hexdigest()


        fetch_result = index.fetch(
            ids=[first_chunk_id]
        )


        already_indexed = (
            len(fetch_result.vectors) > 0
        )


        # -----------------------------------------
        # INDEX ONLY IF NOT EXISTS
        # -----------------------------------------

        if not already_indexed:

            print(
                "New content detected. Indexing..."
            )

            vectorstore.add_documents(
                documents=chunks,
                ids=ids
            )

            print(
                f"Indexed {len(chunks)} chunks"
            )

        else:

            print(
                "Document already indexed."
            )

    return all_chunks


# =====================================================
# BUILD RETRIEVER
# =====================================================

def build_retriever(chunks):

    pinecone_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 10
        }
    )


    bm25 = BM25Retriever.from_documents(
        chunks
    )

    bm25.k = 10


    hybrid_retriever = EnsembleRetriever(

        retrievers=[
            bm25,
            pinecone_retriever
        ],

        weights=[
            0.4,
            0.6
        ]
    )


    final_retriever = ContextualCompressionRetriever(

        base_retriever=hybrid_retriever,

        base_compressor=reranker
    )


    return final_retriever