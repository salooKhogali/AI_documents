import time
import numpy as np


def euclidean(a,b):
    return np.sqrt(np.sum((a-b)**2))

def brute_force_search(query_vector,doc_vector,k=3):
    distance=[euclidean(query_vector,d)for d in doc_vector]
    return np.argsort(distance)[:k]



dim =768
rng = np.random.default_rng(42)
query_vector = rng.random(dim, dtype=np.float32)
for n in [100, 1_000, 10_000, 50_000]:
    doc_vectors = rng.random((n, dim), dtype=np.float32)


    start = time.perf_counter()
    brute_force_search(query_vector, doc_vectors)
    elapsed = time.perf_counter() - start

    print(f"{n} docs -> {elapsed * 1000:.1f} ms")