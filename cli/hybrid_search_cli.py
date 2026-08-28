import argparse

from lib.search_utils import load_movies, DOCUMENT_PREVIEW_LENGTH
from lib import hybrid_search
from lib.llm import get_llm_client, enhancement_query

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Accepts a list of scores and prints normalized values")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="The list of scores to be normalized")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Execute bm25 keyword search and semantic search in parallel, weighting with alpha")
    weighted_search_parser.add_argument("query", type=str, help="The query to be searched for")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="A float between 0 and 1 that adjusts the weight given to different search methods. Higher -> more keyword-weighted")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="The number of results to return")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Execute bm25 keyword search and semantic search in parallel, weighting with alpha")
    rrf_search_parser.add_argument("query", type=str, help="The query to be searched for")
    rrf_search_parser.add_argument("--k", type=int, default=60, help="A positive constant used to shape how rankings are weighted. A lower number leads to a sharper initial decline/more weight on higher-ranked items")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="The number of results to return")
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell"],
        help="Query enhancement method",
    )

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
                print(f"{i+1}. {result["title"]}\n  Hybrid Score: {result["hybrid_score"]:.3f}\n  BM25: {result["keyword_score"]:.3f}, Semantic: {result["semantic_score"]:.3f}\n  {result["document"][:DOCUMENT_PREVIEW_LENGTH]}...")
        case "rrf-search":
            query = args.query
            if args.enhance != "" and args.enhance is not None:
                response = get_llm_client().chat.completions.create(
                    model="openrouter/free",
                    messages=[{"role": "user", "content": enhancement_query(query, args.enhance)}],
                )
                enhanced_query = response.choices[0].message.content
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'")
                query = enhanced_query

            hyb_search = hybrid_search.HybridSearch(load_movies())
            results = hyb_search.rrf_search(args.query, args.k, args.limit)
            for i, result in enumerate(results):
                print(f"{i+1}. {result["title"]}\n  RRF Score: {result["hybrid_score"]:.3f}\n  BM25 Rank: {result["keyword_score"]}, Semantic Rank: {result["semantic_score"]}\n  {result["document"][:DOCUMENT_PREVIEW_LENGTH]}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
