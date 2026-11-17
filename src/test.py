import ollama
import numpy as np
texts=[   "How to get a tax number",
    "Steps to obtain a tax registration number",
    "How do I register for VAT?",
    "What is the weather today",
    "Requirements for e-invoicing",
]

def embed(text):
    response=ollama.embeddings(model="nomic-embed-text",prompt=text)
    return np.array(response["embedding"])


def euclidean(a,b):
    return np.sqrt(np.sum((a-b)**2))

vector=[embed(text) for text in texts]

query = "How can I obtain my tax number?"
query_vector=embed(query)

distance=[euclidean(query_vector,v) for v in vector]
k=2
closet=np. argsort(distance)[:k]

for i  in closet:
     print(round(distance[i], 2), "->", texts[i])