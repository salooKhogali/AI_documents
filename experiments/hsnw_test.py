import time
import numpy as np
import faiss

vectors = np.load("doc_vectors.npy")  # your exact file name
print("Vectors shape:", vectors.shape, "| type:", vectors.dtype)

d = vectors.shape[1]   # number of dimensions (768)
M = 32                 # connections per point in the graph

index = faiss.IndexHNSWFlat(d, M)

start = time.perf_counter()
index.add(vectors)
elapsed = time.perf_counter() - start

print("Chunks in index:", index.ntotal)
print(f"Build time: {elapsed * 1000:.1f} ms")

import json
import ollama

with open("Chunks.json", encoding="utf-8") as f:  # your exact file name
    chunks = json.load(f)

def embed(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return np.array(response["embedding"], dtype=np.float32)

query = "What are the requirements for a valid tax invoice?"
query_vector = embed(query).reshape(1, -1)   # FAISS wants a 2D table

k = 3
start = time.perf_counter()
distances, ids = index.search(query_vector, k)
elapsed = time.perf_counter() - start

print(f"\nHNSW search took {elapsed * 1000:.2f} ms\n")
for dist, i in zip(distances[0], ids[0]):
    print(f"Distance: {dist:.2f} | Source: {chunks[i]['source']}")
    print(chunks[i]["text"][:150].replace("\n", " "))
    print("-" * 40)