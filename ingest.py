# simple_ingest.py
# Minimal ingestion supporting .txt, .md and .pdf files.
#
# - Put files under: docs/<SectionName>/ (supports .txt, .md, .pdf)
# - Install deps: pip install sentence-transformers faiss-cpu PyPDF2
# - Run: python simple_ingest.py
#
# Output: vectorstores/<SectionName>/index.faiss  and vectorstores/<SectionName>/meta.json

import os
import json
from pathlib import Path
import re

# simple deps (install before running)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    raise SystemExit(
        "Install sentence-transformers first: pip install sentence-transformers"
    )

try:
    import faiss
except Exception:
    raise SystemExit("Install faiss-cpu first: pip install faiss-cpu")

# PDF reader (optional dependency)
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = (
        None  # we'll raise a helpful error if a PDF is present and PyPDF2 missing
    )

try:
    from ftfy import fix_text as _ftfy_fix
except Exception:
    _ftfy_fix = None

import numpy as np

# ---- CONFIG ----
DOCS_DIR = Path("docs")
VECTOR_DIR = Path("vectorstores")
EMBED_MODEL = "all-MiniLM-L6-v2"  # small and fast
CHUNK_WORDS = 500  # keep chunks readable/simple


def clean_text(text: str) -> str:
    """
    Fix common PDF/text extraction issues:
    - remove hyphenation split across line breaks (e.g., 'exam-\nple' -> 'example')
    - collapse multiple newlines to a single space
    - collapse repeated whitespace
    - optionally apply ftfy if available
    """
    if not text:
        return text
    # 1) fix hyphenation across line breaks
    text = re.sub(r"-\s*\n\s*", "", text)
    # 2) replace remaining newlines with a space
    text = re.sub(r"\s*\n+\s*", " ", text)
    # 3) collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    # 4) optional ftfy cleanup
    if _ftfy_fix:
        try:
            text = _ftfy_fix(text)
        except Exception:
            pass
    return text


# ---- small helper functions ----
def read_text_file(path: Path) -> str:
    """Read .txt / .md / .pdf and return text."""
    suf = path.suffix.lower()
    if suf in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    elif suf == ".pdf":
        if PdfReader is None:
            raise RuntimeError(
                "PyPDF2 is required to read PDF files. Install with: pip install PyPDF2"
            )
        reader = PdfReader(str(path))
        pages = []
        for p in reader.pages:
            text = p.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)
    else:
        return ""


def split_into_chunks(text: str, words_per_chunk: int = CHUNK_WORDS):
    """Split a long text into list of chunks (approx `words_per_chunk` words)."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + words_per_chunk]
        chunks.append(" ".join(chunk))
        i += words_per_chunk
    return chunks


def sanitize(name: str) -> str:
    """Make safe folder name for saving."""
    return "".join(c if (c.isalnum() or c in " -_") else "_" for c in name).strip()


# ---- main ingestion ----
def ingest_all_sections():
    if not DOCS_DIR.exists():
        raise SystemExit(
            f"Create a folder named '{DOCS_DIR}' and inside it create section folders (e.g. 'Statistics & Probability')."
        )

    model = SentenceTransformer(EMBED_MODEL)  # loads model (downloads first time)

    section_dirs = [p for p in DOCS_DIR.iterdir() if p.is_dir()]
    if not section_dirs:
        raise SystemExit(
            f"No sections found inside {DOCS_DIR}. Create docs/<SectionName>/ and add files (.txt, .md, .pdf)."
        )

    for sec in section_dirs:
        sec_name = sec.name
        print(f"\nProcessing section: {sec_name}")

        texts = []
        sources = []

        # read each supported file
        for f in sorted(sec.iterdir()):
            if f.suffix.lower() not in (".txt", ".md", ".pdf"):
                print("  Skipping (unsupported):", f.name)
                continue
            print("  Reading:", f.name)
            raw = clean_text(read_text_file(f))
            if not raw:
                print("   (no text extracted, skipping file)")
                continue
            chunks = split_into_chunks(raw, CHUNK_WORDS)
            for i, c in enumerate(chunks):
                texts.append(c)
                sources.append(f"{f.name}::chunk{i}")

        if not texts:
            print("  No text chunks found for this section, skipping.")
            continue

        # create embeddings (simple: encode all at once)
        print(f"  Creating embeddings for {len(texts)} chunks ...")
        embs = model.encode(texts, convert_to_numpy=True)
        # normalize for cosine-similarity (we'll use inner-product index)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        embs = embs / norms
        embs = embs.astype("float32")

        # build FAISS index (inner-product = cosine on normalized vectors)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

        # save index + metadata
        save_folder = VECTOR_DIR / sanitize(sec_name)
        save_folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(save_folder / "index.faiss"))

        meta = [{"text": t, "source": s} for t, s in zip(texts, sources)]
        with open(save_folder / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"  Saved index and meta for section: {sec_name}")

    print("\nAll done. Vectorstores written to:", VECTOR_DIR)


if __name__ == "__main__":
    ingest_all_sections()
