# ZATCA VAT Document Assistant

A local RAG (Retrieval-Augmented Generation) app that answers questions about
Saudi VAT guidelines from ZATCA PDF documents, and says "I don't know" when
the answer is not in the documents.

## How it works

1. **Load and chunk:** 4 ZATCA VAT PDFs are split into 703 chunks
   (800 characters, 100 overlap).
2. **Embed:** each chunk becomes a 768-number vector with `nomic-embed-text`.
3. **Retrieve:** the question is embedded, and the 3 closest chunks are found
   by Euclidean distance (brute-force KNN).
4. **Guard:** if the closest chunk is too far (distance > 16.65), the app
   answers "I don't know" without calling the model.
5. **Generate:** `llama3.2` answers using only the retrieved text.
6. **Show sources:** the app lists the source files and the exact text used.

Everything runs locally with Ollama. No API keys.

## Results

Evaluation set of 8 questions, each with an expected source file:

| Metric | Score |
|---|---|
| Top-1 (right file ranked first) | 7/8 |
| Top-3 (right file in top 3) | 8/8 |

The distance threshold (16.65) was measured on 9 in-scope and 14 out-of-scope
questions. Unrelated questions scored 21 or more, in-scope questions at most
16.49.

Results were measured on the guideline versions downloaded in November 2026.

## Known limitations

- Short, vague questions can retrieve the wrong file.
- The threshold margin is narrow (16.49 vs 16.83), so a vague but valid
  question may be rejected.
- Sources show the files that were retrieved, not only the ones the answer used.
- English only for now. The threshold is specific to `nomic-embed-text` and
  must be re-measured if the model changes.

## Data

The PDFs are not included in this repository. Download them from the official
ZATCA guidelines pages (https://zatca.gov.sa/en/HelpCenter/guidelines) and
save them in a `data/` folder with these names:

| Save as | Download |
|---|---|
| `Tax-Invoicing-and-Records-under-VAT-Provisions.pdf` | https://zatca.gov.sa/en/HelpCenter/guidelines/Documents/Guideline-for-Tax-Invoicing-and-Records-under-VAT-Provisions.pdf |
| `RetailSectorVAT.pdf` | https://zatca.gov.sa/en/HelpCenter/guidelines/Documents/Guideline-For-Retail-Sector-under-VAT-Provisions.pdf |
| `ImportsExportsVAT.pdf` | https://zatca.gov.sa/en/HelpCenter/guidelines/Documents/Guideline-on-Imports-and-Exports-under-VAT-Provision.pdf |
| `ImplementingRegulation(VAT).pdf` | https://zatca.gov.sa/en/HelpCenter/guidelines/Documents/Amendments-to-the-Implementing-Regulation-of-%28VAT%29.PDF |

## Run it

```
ollama pull nomic-embed-text
ollama pull llama3.2
pip install -r requirements.txt
python src/build_index.py
streamlit run src/app.py
```

Run both commands from the project root. The repository already includes the
index files (`doc_vectors.npy`, `chunks.json`), so `build_index.py` is only
needed if you change the PDFs.

## Project structure

```
src/        rag.py (retrieval, prompt, guard, generation), app.py (Streamlit UI),
            build_index.py (builds the index), plus threshold and timing scripts
experiments/  HNSW and timing experiments
```