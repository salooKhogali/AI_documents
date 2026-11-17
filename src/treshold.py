from rag import embed, eulidean, vectors

def best_distance(question):
    query_vector = embed(question)
    distance = [eulidean(query_vector, v) for v in vectors]
    return min(distance)

in_scope = [
    # ... your existing ones ...
    "When must a taxable person issue a tax invoice?",
    "What are the VAT rules on exports?",
    "How long must invoices be kept?",
    "What is a summary tax invoice?",
    "Explain VAT on retail sales",
]

out_of_scope = [
    # ... your existing ones ...
    "Tell me about Saudi Arabia",
    "What is the capital of France?",
    "write about jeddah",
    "Who is the king of Saudi Arabia?",
    "How do I invest in the stock market?",
]

in_results = []
print("IN SCOPE")
for q in in_scope:
    d = best_distance(q)
    in_results.append(d)
    print(f"  {d:.2f} | {q}")

out_results = []
print("OUT OF SCOPE")
for q in out_of_scope:
    d = best_distance(q)
    out_results.append(d)
    print(f"  {d:.2f} | {q}")

print(f"\nFarthest in-scope question:   {max(in_results):.2f}")
print(f"Closest out-of-scope question: {min(out_results):.2f}")