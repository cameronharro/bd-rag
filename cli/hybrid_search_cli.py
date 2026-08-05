import argparse

from lib.search_utils import load_movies, DOCUMENT_PREVIEW_LENGTH
from lib import hybrid_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Accepts a list of scores and prints normalized values")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="The list of scores to be normalized")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Execute bm25 keyword search and semantic search in parallel, weighting with alpha")
    weighted_search_parser.add_argument("query", type=str, help="The query to be searched for")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="A float between 0 and 1 that adjusts the weight given to different search methods. Higher -> more keyword-weighted")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="The number of results to return")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = hybrid_search.normalize(args.scores)
            for _, normalized_score in enumerate(scores):
                print(f"* {normalized_score:.4f}")
        case "weighted-search":
            hyb_search = hybrid_search.HybridSearch(load_movies())
            results = hyb_search.weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(results):
                print(f"{i+1}. {result["title"]}\n  Hybrid Score: {result["hybrid_score"]:.3f}\n  BM25: {result["keyword_score"]:.3f}, Semantic: {result["semantic_score"]:.3f}\n  {result["document"][:DOCUMENT_PREVIEW_LENGTH]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
