from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from elasticsearch import Elasticsearch
import os
import fitz  # PyMuPDF

app = FastAPI()

# Load model & databases
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient("http://localhost:6333")  # Connect to Qdrant container
es = Elasticsearch("http://localhost:9200")

INDEX_NAME = "pdf_documents"

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
def search_docs(request: QueryRequest):
    query_vector = embedding_model.encode(request.query).tolist()

    # Search Qdrant for vector similarity
    results = qdrant.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=5,
    )

    # Retrieve relevant text
    responses = [hit.payload["text"] for hit in results]

    return {"answers": responses}

def process_pdfs():
    """Processes and indexes PDFs from the 'pdfs' folder."""
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body={"settings": {}})

    pdf_folder = "pdfs"
    documents = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)

            # Extract text using PyMuPDF
            text = ""
            with fitz.open(filepath) as pdf_doc:
                for page in pdf_doc:
                    text += page.get_text("text") + "\n"

            # Generate embedding
            vector = embedding_model.encode(text).tolist()
            documents.append({"text": text, "vector": vector})

            # Store in Elasticsearch
            es.index(index=INDEX_NAME, body={"content": text})

    # Store embeddings in Qdrant
    qdrant.create_collection(
        collection_name="documents",
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

    qdrant.upsert(
        collection_name="documents",
        points=[models.PointStruct(id=i, vector=doc["vector"], payload={"text": doc["text"]}) for i, doc in enumerate(documents)]
    )

    print("✅ PDFs processed & indexed successfully!")

if __name__ == "__main__":
    process_pdfs()