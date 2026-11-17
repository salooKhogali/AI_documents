from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import ollama
import numpy as np
import json



folder=Path("data")
splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
chunk=[]
for file in sorted(folder.glob("*.pdf")):
    page=PyPDFLoader(str(file)).load()
    for doc in splitter.split_documents(page):
        text=doc.page_content.strip()
        if text:
            chunk.append({"text":text,"source":file.name})
    print("load:",file.name)


print("no of chunk:" ,len(chunk))   



def embed(text):
    response=ollama.embeddings(model="nomic-embed-text",prompt=text)
    return np.array(response["embedding"],dtype=np.float32)

doc_vector=np.array([embed(c["text"]) for c in chunk])

print("Vectors shape:", doc_vector.shape)
    



np.save("doc_vectors.npy",doc_vector)
with open("chunks.json","w",encoding="utf-8") as f:
    json.dump(chunk,f,ensure_ascii=False)