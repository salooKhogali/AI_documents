import json
import ollama
import numpy as np


vectors=np.load("doc_vectors.npy")
with open("chunks.json",encoding="utf-8")as f:
    chunks=json.load(f)

MAX_DISTANCE = 16.65
def embed(text):
    response=ollama.embeddings(model="nomic-embed-text",prompt=text)
    return np.array(response["embedding"],dtype=np.float32)

def eulidean(a,b):
    return np.sqrt(np.sum((a-b)**2))


def retriver(question,k=3):
     query_vector=embed(question)
     distance=[eulidean(query_vector,v)for v in vectors]
     closet= np.argsort(distance)[:k]
     best=distance[closet[0]]
     return[chunks[i] for i in closet],best


def build_prompt(question,retrived_chunk):
    context=" "
    for c in retrived_chunk:
        context += f"[Source: {c['source']}]\n{c['text']}\n\n"

    prompt=(
         "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say: I don't know.\n\n"
        f"Context:\n{context}"
        f"Question: {question}\n"
        "Answer:"
    
    )
    return prompt


def ask(question,k=3):

 result, best = retriver(question, k)

 if best > MAX_DISTANCE:
        return "I don't know. I couldn't find this in the documents. Try rephrasing with specific VAT terms.", [],[]

 prompt=build_prompt(question,result)

 response=ollama.chat(model="llama3.2",
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0},
    )
  
  
 answer=response["message"]["content"]
 
 if answer.strip().lower().startswith("i don't know"):
        return answer, [], []

 sources = sorted(set(r["source"] for r in result))
 return answer, sources, result

if __name__ == "__main__":

    while True:
       q = input("\nYour question (or 'exit'): ")
       if q.strip().lower() == "exit":
             break
       answer, sources,_ = ask(q)
       print("ANSWER:", answer)
       print("SOURCES:", sources)
   
    