"""Small local TF-IDF retrieval over bundled banking guidance documents."""
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent / "data" / "knowledge_base"

def retrieve(question: str, limit: int = 3) -> list[dict[str, str]]:
    docs = []
    for path in sorted(ROOT.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append({"title": path.stem.replace("_", " ").title(), "source": path.name, "text": text})
    if not docs:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([d["text"] for d in docs] + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    best = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:limit]
    return [{**docs[i], "score": f"{scores[i]:.3f}"} for i in best if scores[i] > 0]
