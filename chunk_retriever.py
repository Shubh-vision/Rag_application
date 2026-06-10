from dotenv import load_dotenv
load_dotenv()
from chunks import create_chunks
from ingestion import get_docs
import os
import hashlib

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

# Embedding Model

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedding
)


# Load Document
docs_by_url = get_docs()

all_chunks = []   # ONLY for BM25 + retriever (not blindly filled)

for url, docs in docs_by_url.items():
    print(f"Processing {url}")

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        print(f"Source: {source}")


        # generate document ID
        doc_id = hashlib.md5(source.encode()).hexdigest()

        # generate content hash
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()

        print(f"DOC ID: {doc_id}")
        print(f"CONTENT HASH: {content_hash}")


        # Check if already indexed

        first_chunk_id = hashlib.md5(f"{doc_id}_0".encode()).hexdigest()
        fetch_result = index.fetch(ids=[first_chunk_id])
        already_indexed = (len(fetch_result.vectors) > 0)    


        #chunking for BM25....it always needed
        chunks = create_chunks([doc])
        all_chunks.extend(chunks)   # ALWAYS needed for BM25
        
        # ---------------- IMPORTANT FIX ----------------
        # DO NOT chunk unless needed
        # Index Only If Needed

        if not already_indexed:

            print("New content detected. Indexing...")

            ids = []

            for i, chunk in enumerate(chunks):

                chunk.metadata["doc_id"] = doc_id
                chunk.metadata["content_hash"] = content_hash

                chunk_id = hashlib.md5(
                    f"{doc_id}_{i}".encode()
                ).hexdigest()

                ids.append(chunk_id)

            vectorstore.add_documents(
                documents=chunks,
                ids=ids
            )

            print(f"Indexed {len(chunks)} chunks")

        else:
            print("Document already indexed. Skipping embedding.")

# ---------------- RERANKER ----------------

rerank_model = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = CrossEncoderReranker(
        model=rerank_model,
        top_n=5)



def build_retriever(chunks):

    pinecone_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 10

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            bm25,
            pinecone_retriever
        ],
        weights=[0.4, 0.6]
    )

    


    final_retriever = ContextualCompressionRetriever(
        base_retriever=hybrid_retriever,
        base_compressor=reranker
    )

    return final_retriever


retriever = build_retriever(all_chunks)
