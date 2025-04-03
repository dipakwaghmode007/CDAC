from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from elasticsearch import Elasticsearch
import os
import fitz  # PyMuPDF
import uvicorn

app = FastAPI()

# Load model & databases
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient("http://localhost:6333")  # Connect to Qdrant
es = Elasticsearch("http://localhost:9200")

INDEX_NAME = "pdf_documents"
CHUNK_SIZE = 256  # 🔹 Reduce chunk size for better search results


class QueryRequest(BaseModel):
    query: str


@app.post("/search")
def search_docs(request: QueryRequest):
    query_vector = embedding_model.encode(request.query).tolist()

    # 🔹 Search Qdrant for closest text chunks
    results = qdrant.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=3,  # Retrieve only top 3 matches
    )

    # Retrieve most relevant text chunks (not full PDFs)
    extracted_answers = []
    for hit in results:
        text_chunk = hit.payload["text"]
        extracted_answers.append(text_chunk[:400])  # Limit text to 400 characters

    return {"answers": extracted_answers}


def chunk_text(text, size=CHUNK_SIZE):
    """Splits text into smaller, overlapping chunks for better search accuracy."""
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def ensure_qdrant_collection():
    """Ensures that the Qdrant collection exists before inserting data."""
    collections = qdrant.get_collections()
    collection_names = [col.name for col in collections.collections]

    if "documents" not in collection_names:
        qdrant.create_collection(
            collection_name="documents",
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )


def process_pdfs():
    """Processes and indexes PDFs from the 'pdfs' folder."""
    # 🔹 Check if the Elasticsearch index exists before creating it
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body={"settings": {}})

    pdf_folder = "pdfs"
    documents = []
    doc_id = 0

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)

            # Extract text using PyMuPDF
            text = ""
            with fitz.open(filepath) as pdf_doc:
                for page in pdf_doc:
                    text += page.get_text("text") + "\n"

            # 🔹 Break text into small chunks for better precision
            text_chunks = chunk_text(text)

            for chunk in text_chunks:
                # Generate embedding for each chunk
                vector = embedding_model.encode(chunk).tolist()
                documents.append({"id": doc_id, "text": chunk, "vector": vector})

                # Store chunked data in Elasticsearch
                es.index(index=INDEX_NAME, body={"content": chunk})

                doc_id += 1

    # Ensure the Qdrant collection exists before inserting data
    ensure_qdrant_collection()

    # Store embeddings in Qdrant
    qdrant.upsert(
        collection_name="documents",
        points=[
            models.PointStruct(id=doc["id"], vector=doc["vector"], payload={"text": doc["text"]})
            for doc in documents
        ],
    )

    print("✅ PDFs processed & indexed successfully!")


if __name__ == "__main__":
    process_pdfs()  # Process PDFs before starting FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)
