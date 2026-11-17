import numpy as np
import  json
import ollama
import time


vector=np.load("doc_vectors.npy")
with open("chunks.json",encoding="utf-8") as f:
    chunks=json.load(f)



def embed(text):
   response=ollama.embeddings (model="nomic-embed-text", prompt=text )
   return np.array(response["embedding"],dtype=np.float32)


def uclidean(a,b):
   return np.sqrt(np.sum((a-b)**2))


query="What are the requirements for a valid tax invoice?"
q_vector=embed(query)

k=3
start=time.perf_counter()
distance=[uclidean(q_vector,v)for v in vector]
closest=np.argsort(distance)[:k]
elapsed=time.perf_counter()-start

print(f"\nSearch over {len(chunks)} chunks took {elapsed * 1000:.1f} ms\n")

for i in closest:
    print(f"Distance: {distance[i]:.2f} | Source: {chunks[i]['source']}")
    print(chunks[i]["text"][:300])
    print("-" * 40)
best = closest[0]
print("\nFULL TEXT OF BEST CHUNK:")
print(chunks[best]["text"])


def search(query, k=3):
    query_vector = embed(query)
    distances = [uclidean(query_vector, v) for v in vector]
    closest = np.argsort(distances)[:k]
    print("\nQUESTION:", query)
    for i in closest:
        print(f"  {distances[i]:.2f} | {chunks[i]['source']}")
        print("  ", chunks[i]["text"][:150].replace("\n", " "))

eval_set=[
    ("What are the requirements for a valid tax invoice?",
     "Tax-Invoicing-and-Records-under-VAT-Provisions.pdf"),
    ("How is VAT applied on imported goods?",
     "ImportsExportsVAT.pdf"),
    ("How does VAT apply to retail businesses?",
     "RetailSectorVAT.pdf"),
    ("What changed in the Implementing Regulation amendments?",
     "ImplementingRegulation(VAT).pdf"),
    ("What is Offsetting Refundable Amounts Against Other Amounts Owed to ZATCA?",
     "ImplementingRegulation(VAT).pdf"),
    ("What is a Qualified Voucher?",
     "RetailSectorVAT.pdf"),
    ("What does exportation include?",
     "ImportsExportsVAT.pdf"),
    ("What is a simplified tax invoice?",
     "Tax-Invoicing-and-Records-under-VAT-Provisions.pdf"),
]


def top_sources(query,k):
   query_vector =embed(query)
   distance=[uclidean(query_vector,v)for v in vector]
   idx=np.argsort(distance)[:k]
   return[chunks[i]["source"]for i in idx]



hits_top1 = 0
hits_top3 = 0
for question,expected in eval_set:

    sources = top_sources(question, 3)
    ok1 = sources[0] == expected
    ok3 = expected in sources
    hits_top1 += ok1
    hits_top3 += ok3
    print("OK" if ok1 else ("TOP3" if ok3 else "MISS"), "|", question[:60])

print(f"\nTop-1: {hits_top1}/{len(eval_set)}")
print(f"Top-3: {hits_top3}/{len(eval_set)}")






   
