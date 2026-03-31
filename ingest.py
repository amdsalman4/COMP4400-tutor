"""
ingest.py — Parse course PDFs, chunk, embed, and store in ChromaDB.

Usage:
    python ingest.py

Place your PDFs in data/pdfs/ before running.
The vector store is saved to data/vectorstore/ and only needs to be built once
(or re-run when PDFs change).
"""

import pathlib
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
RAW_TEXT_DIR = BASE_DIR / "data" / "raw_text"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"

# ── Chunking settings ──────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ── Embedding model (free, local) ──────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "comp4400"


def extract_text_from_pdf(pdf_path: pathlib.Path) -> list[dict]:
    """Return a list of {page, text} dicts for every page in the PDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append({"page": page_num, "text": text})
    doc.close()
    return pages


def save_raw_text(pdf_name: str, pages: list[dict]) -> None:
    RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_TEXT_DIR / (pdf_name + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(f"--- Page {p['page']} ---\n{p['text']}\n\n")
    print(f"  Saved raw text → {out_path.name}")


def chunk_pages(pdf_name: str, pages: list[dict]) -> list[dict]:
    """Split page text into overlapping chunks; keep source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for p in pages:
        splits = splitter.split_text(p["text"])
        for split in splits:
            chunks.append({
                "text": split,
                "source": pdf_name,
                "page": p["page"],
            })
    return chunks


def build_vectorstore(all_chunks: list[dict]) -> None:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    # Delete existing collection so re-runs start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(i) for i in range(len(all_chunks))]
    documents = [c["text"] for c in all_chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in all_chunks]

    BATCH = 500
    for start in range(0, len(ids), BATCH):
        collection.add(
            ids=ids[start: start + BATCH],
            documents=documents[start: start + BATCH],
            metadatas=metadatas[start: start + BATCH],
        )
        print(f"  Indexed chunks {start}–{min(start + BATCH, len(ids)) - 1}")

    print(f"\nVector store ready: {len(all_chunks)} chunks in '{COLLECTION_NAME}'.")


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}. Add your course PDFs there and re-run.")
        return

    all_chunks: list[dict] = []

    for pdf_path in pdfs:
        name = pdf_path.stem
        print(f"\nProcessing: {pdf_path.name}")
        pages = extract_text_from_pdf(pdf_path)
        print(f"  {len(pages)} pages with text")
        save_raw_text(name, pages)
        chunks = chunk_pages(name, pages)
        print(f"  {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all PDFs: {len(all_chunks)}")
    print("Building vector store…")
    build_vectorstore(all_chunks)


if __name__ == "__main__":
    main()
