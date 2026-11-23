"""Step 2: Chunk data/pages.json, embed locally, store in Chroma.

Run:   python ingest.py                      -> build the vector DB
Test:  python ingest.py "What does Gnani do?" -> build, then show top matches
       python ingest.py --query-only "..."    -> skip rebuild, just search
"""
import json
import re
import sys

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

PAGES_FILE = "data/pages.json"
DB_DIR = "chroma_db"
COLLECTION = "gnani"
EMBED_MODEL = "all-MiniLM-L6-v2"   # local, free, no API key
# MiniLM reads ~256 tokens max (~1000 chars), so keep chunks around that size
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH = 64

# Legal pages and locale duplicates (e.g. /ja-jp/...) crowd out useful chunks
SKIP_URL = re.compile(r"/(legal|privacy|terms|eula)(/|$)|/[a-z]{2}-[a-z]{2}(/|$)", re.I)


def load_chunks() -> list[dict]:
    with open(PAGES_FILE, encoding="utf-8") as f:
        pages = json.load(f)

    total = len(pages)
    pages = [p for p in pages if not SKIP_URL.search(p["url"])]
    print(f"Skipped {total - len(pages)} legal/locale pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        for i, text in enumerate(splitter.split_text(page["text"])):
            chunks.append({
                "id": f"{page['url']}#{i}",
                "text": text,
                "url": page["url"],
                "title": page["title"],
                "chunk": i,
            })
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    return chunks


def build_db() -> None:
    chunks = load_chunks()
    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)  # rebuild from scratch each run
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        embeddings = model.encode(
            [c["text"] for c in batch], normalize_embeddings=True
        ).tolist()
        col.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embeddings,
            metadatas=[{"url": c["url"], "title": c["title"], "chunk": c["chunk"]}
                       for c in batch],
        )
        print(f"  embedded {min(start + BATCH, len(chunks))}/{len(chunks)}")
    print(f"Saved vector DB to {DB_DIR}/ ({col.count()} chunks)")


def search(question: str, k: int = 4) -> None:
    model = SentenceTransformer(EMBED_MODEL)
    col = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    q = model.encode([question], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q, n_results=k)

    print(f"\nQuestion: {question}\n")
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                               res["distances"][0]):
        similarity = 1 - dist  # cosine similarity (higher = closer)
        print(f"[{similarity:.2f}] {meta['url']}")
        print(f"       {doc[:200].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--query-only":
        search(" ".join(args[1:]))
    else:
        build_db()
        if args:
            search(" ".join(args))
