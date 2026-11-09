"""High-performance RAG (Retrieval-Augmented Generation) retrieval service.

Indexes and retrieves domain-specific BFSI (Banking, Financial Services, and Insurance)
knowledge base documents, as well as authenticated customer uploaded documents (PDFs, text, markdown).
Partitioned by section headings and pages for high semantic precision.
"""

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import threading
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent / "data" / "knowledge_base"
USER_DOCS_DIR = Path(__file__).resolve().parent / "data" / "user_documents"

_index_lock = threading.Lock()
_cached_chunks: list[dict[str, Any]] = []
_cached_vectorizer: TfidfVectorizer | None = None
_cached_matrix: Any = None
_last_loaded_mtime: float = 0.0


def _build_base_index() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    if not ROOT.exists():
        return []

    for path in sorted(ROOT.glob("*.md")):
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            continue

        doc_title = path.stem.replace("_", " ").title()
        # Partition document by markdown headings (# or ##)
        raw_sections = re.split(r"\n(?=##?\s+)", content)
        for section in raw_sections:
            section_clean = section.strip()
            if not section_clean:
                continue

            lines = section_clean.split("\n")
            heading = lines[0].lstrip("#").strip()
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else section_clean

            search_text = f"{doc_title} - {heading}\n{body}"
            chunks.append(
                {
                    "title": doc_title,
                    "section": heading,
                    "source": path.name,
                    "text": section_clean,
                    "search_text": search_text,
                    "is_user_upload": False,
                }
            )

    return chunks


def _get_base_chunks() -> list[dict[str, Any]]:
    global _cached_chunks, _last_loaded_mtime

    current_mtime = 0.0
    if ROOT.exists():
        for p in ROOT.glob("*.md"):
            try:
                current_mtime = max(current_mtime, p.stat().st_mtime)
            except OSError:
                pass

    with _index_lock:
        if not _cached_chunks or current_mtime > _last_loaded_mtime:
            _cached_chunks = _build_base_index()
            _last_loaded_mtime = current_mtime
        return list(_cached_chunks)


def extract_text_from_file(filename: str, content_bytes: bytes) -> str:
    """Extract readable text from PDF, Markdown, Text, or CSV files."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            pages = []
            for idx, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"## Page {idx}\n{text.strip()}")
            if pages:
                return "\n\n".join(pages)
        except Exception as e:
            # Fallback if PDF parsing has issues
            return f"PDF content could not be fully extracted: {e}"

    # For text, markdown, csv, or json
    try:
        return content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return content_bytes.decode("latin-1", errors="replace")


def save_user_document(user_id: str, filename: str, content_bytes: bytes) -> tuple[str, int]:
    """Process an uploaded document, partition into RAG chunks, and index for the customer."""
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(filename).name)
    raw_text = extract_text_from_file(safe_name, content_bytes).strip()

    if not raw_text:
        raise ValueError("The uploaded document appears to be empty or unreadable.")

    user_folder = USER_DOCS_DIR / re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
    user_folder.mkdir(parents=True, exist_ok=True)

    # Partition document by headings or paragraph blocks
    sections = re.split(r"\n(?=##?\s+)", raw_text)
    if len(sections) == 1:
        # Split by double newlines if no headings
        sections = re.split(r"\n\s*\n", raw_text)

    doc_chunks: list[dict[str, Any]] = []
    doc_title = Path(safe_name).stem.replace("_", " ").title()

    for idx, sec in enumerate(sections, 1):
        clean_sec = sec.strip()
        if not clean_sec:
            continue
        lines = clean_sec.split("\n")
        heading = lines[0].lstrip("#").strip() if lines[0].startswith("#") else f"Section {idx}"
        body = "\n".join(lines[1:]).strip() if lines[0].startswith("#") and len(lines) > 1 else clean_sec

        search_text = f"Uploaded Document: {doc_title} ({safe_name}) - {heading}\n{body}"
        doc_chunks.append({
            "title": f"Uploaded: {safe_name}",
            "section": heading,
            "source": f"Uploaded: {safe_name}",
            "text": clean_sec,
            "search_text": search_text,
            "is_user_upload": True,
        })

    # Save processed document bundle
    meta_path = user_folder / f"{safe_name}.json"
    meta = {
        "filename": safe_name,
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunks_count": len(doc_chunks),
        "chunks": doc_chunks,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return safe_name, len(doc_chunks)


def get_user_document_chunks(user_id: str) -> list[dict[str, Any]]:
    """Retrieve all chunks belonging to documents uploaded by this customer."""
    user_folder = USER_DOCS_DIR / re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
    if not user_folder.exists():
        return []

    user_chunks: list[dict[str, Any]] = []
    for meta_file in sorted(user_folder.glob("*.json")):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            for chunk in data.get("chunks", []):
                user_chunks.append(chunk)
        except Exception:
            continue
    return user_chunks


def list_user_documents(user_id: str) -> list[dict[str, Any]]:
    """List active uploaded documents for a customer."""
    user_folder = USER_DOCS_DIR / re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
    if not user_folder.exists():
        return []

    docs = []
    for meta_file in sorted(user_folder.glob("*.json")):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            docs.append({
                "filename": data.get("filename", meta_file.stem),
                "uploaded_at": data.get("uploaded_at"),
                "chunks_count": data.get("chunks_count", 0),
            })
        except Exception:
            continue
    return docs


def clear_user_documents(user_id: str, filename: str | None = None) -> bool:
    """Clear a specific uploaded document or all uploaded documents for a user."""
    user_folder = USER_DOCS_DIR / re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
    if not user_folder.exists():
        return False

    if filename:
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(filename).name)
        target = user_folder / f"{safe_name}.json"
        if target.exists():
            target.unlink()
            return True
        return False

    for item in user_folder.glob("*.json"):
        try:
            item.unlink()
        except OSError:
            pass
    return True


def retrieve(
    question: str,
    limit: int = 4,
    threshold: float = 0.03,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top relevant BFSI knowledge base and uploaded document sections."""
    question_clean = question.strip()
    if not question_clean:
        return []

    # Combine base knowledge base chunks with any authenticated customer uploaded document chunks
    all_chunks = _get_base_chunks()
    user_chunks: list[dict[str, Any]] = []
    if user_id:
        user_chunks = get_user_document_chunks(user_id)
        all_chunks.extend(user_chunks)

    if not all_chunks:
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=6000,
        )
        matrix = vectorizer.fit_transform([chunk["search_text"] for chunk in all_chunks])
        q_vec = vectorizer.transform([question_clean])
        scores = cosine_similarity(q_vec, matrix).ravel()

        # Boost user document chunks if the user query explicitly refers to documents/policies/uploads
        lower_q = question_clean.lower()
        doc_intent = any(w in lower_q for w in ["document", "uploaded", "upload", "file", "policy", "statement", "pdf", "agreement", "attachment"])

        ranked_indices = []
        for i, score in enumerate(scores):
            adjusted_score = float(score)
            if doc_intent and all_chunks[i].get("is_user_upload"):
                adjusted_score = adjusted_score * 1.5 + 0.1
            ranked_indices.append((i, adjusted_score))

        ranked_indices.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked_indices:
            if score < threshold and len(results) >= 1:
                break
            if score > 0.001:
                chunk_copy = dict(all_chunks[idx])
                chunk_copy["score"] = f"{score:.3f}"
                chunk_copy["score_raw"] = score
                results.append(chunk_copy)
                if len(results) >= limit:
                    break
        return results
    except Exception:
        return []


def format_rag_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved knowledge chunks into structured prompt context."""
    if not chunks:
        return ""

    formatted_sections = []
    for i, chunk in enumerate(chunks, 1):
        source_label = chunk.get("source", "Knowledge Base")
        section_label = chunk.get("section", "Section")
        is_upload = chunk.get("is_user_upload", False)
        prefix = "CUSTOMER UPLOADED DOCUMENT" if is_upload else "SMARTBANK POLICY KNOWLEDGE BASE"

        formatted_sections.append(
            f"--- [{prefix} {i}: {source_label} | Section: {section_label}] ---\n"
            f"{chunk.get('text')}"
        )
    return "\n\n".join(formatted_sections)
