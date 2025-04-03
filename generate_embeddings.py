import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from elasticsearch import Elasticsearch

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Qdrant (in-memory vector database)
qdrant = QdrantClient(":memory:")

# Initialize Elasticsearch
es = Elasticsearch("http://localhost:9200")

INDEX_NAME = "pdf_documents"

# Create Elasticsearch index if not exists
if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(index=INDEX_NAME, ignore=400)

# Load PDFs from "pdfs" folder
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