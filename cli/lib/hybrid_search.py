import os
from typing import NotRequired, TypedDict

from lib.search_utils import Movie, SearchResult

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearchResult( TypedDict ):
    keyword_score: float
    semantic_score: float
    hybrid_score: float
    title: str
    document: str
    id: int

class RRFSearchResult( TypedDict ):
    keyword_rank: int
    semantic_rank: int
    hybrid_score: float
    title: str
    document: str
    id: int

class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[SearchResult]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[HybridSearchResult]:
        keyword_results = self._bm25_search(query, limit * 500)
        normalized_keyword_scores = normalize([result["score"] for result in keyword_results])
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        normalized_semantic_scores = normalize([result["score"] for result in semantic_results])
        movie_map: dict[int, HybridSearchResult] = {}
        for i, k_result in enumerate(keyword_results):
            movie_map[k_result["id"]] = {
                    "keyword_score": normalized_keyword_scores[i],
                    "semantic_score": 0.0,
                    "hybrid_score": hybrid_score(normalized_keyword_scores[i], 0.0, alpha),
                    "title": k_result["title"],
                    "document": k_result["document"],
                    "id": k_result["id"]
                    }
        for i, s_result in enumerate(semantic_results):
            existing_entry = movie_map.get(s_result["id"], None)
            if existing_entry is not None:
                existing_entry["semantic_score"] = normalized_semantic_scores[i]
                existing_entry["hybrid_score"] = hybrid_score(
                    existing_entry["keyword_score"],
                    normalized_semantic_scores[i],
                    alpha,
                )
            else:
                movie_map[s_result["id"]] = {
                        "keyword_score": 0.0,
                        "semantic_score": normalized_semantic_scores[i],
                        "hybrid_score": hybrid_score(0.0, normalized_semantic_scores[i], alpha),
                        "title": s_result["title"],
                        "document": s_result["document"],
                        "id": s_result["id"]
                        }
        return sorted(movie_map.values(), key=lambda item: item["hybrid_score"], reverse=True)[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 5) -> list[HybridSearchResult]:
        keyword_results = sorted(self._bm25_search(query, limit * 500), key=lambda item: item["score"], reverse=True)
        semantic_results = sorted(self.semantic_search.search_chunks(query, limit * 500), key=lambda item: item["score"], reverse=True)
        movie_map: dict[int, HybridSearchResult] = {}
        for i, k_result in enumerate(keyword_results, 1):
            movie_map[k_result["id"]] = {
                    "keyword_score": i,
                    "semantic_score": 0.0,
                    "hybrid_score": rrf_score(i, k),
                    "title": k_result["title"],
                    "document": k_result["document"],
                    "id": k_result["id"]
                    }
        for i, s_result in enumerate(semantic_results, 1):
            existing_entry = movie_map.get(s_result["id"], None)
            if existing_entry is not None:
                existing_entry["semantic_score"] = i
                existing_entry["hybrid_score"] = rrf_score(i, k) + rrf_score(int(existing_entry["keyword_score"]), k)
            else:
                movie_map[s_result["id"]] = {
                        "keyword_score": 0.0,
                        "semantic_score": i,
                        "hybrid_score": rrf_score(i, k),
                        "title": s_result["title"],
                        "document": s_result["document"],
                        "id": s_result["id"]
                        }
        return sorted(movie_map.values(), key=lambda item: item["hybrid_score"], reverse=True)[:limit]


def normalize(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return []
    maximum = max(scores)
    minimum = min(scores)
    result: list[float] = [0.0] * len(scores)
    for i, score in enumerate(scores):
        normalized_score = 1.0 if maximum is minimum else ((score - minimum) / (maximum - minimum))
        result[i] = normalized_score
    return result

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)
