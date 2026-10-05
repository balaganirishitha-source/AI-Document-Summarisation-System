from sentence_transformers import SentenceTransformer
import chromadb

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()

collection = client.create_collection(
    name="documents"
)

text = "Artificial Intelligence is used in healthcare, education and banking."

embedding = model.encode(text).tolist()

collection.add(
    documents=[text],
    embeddings=[embedding],
    ids=["doc1"]
)

print("Document stored successfully!")

results = collection.query(
    query_embeddings=[embedding],
    n_results=1
)

print("Search Result:")
print(results["documents"])